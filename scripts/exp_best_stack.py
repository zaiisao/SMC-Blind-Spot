"""The best NO-ORACLE stack we can assemble, measured cumulatively in ONE setup.

WHY THIS SCRIPT EXISTS. Every component below was measured in isolation, and this project has twice
been burned by adding such numbers together: TTA's published +0.0366 CMLt (peak-picked) shrank to
+0.0123 on the beat-only DBN, and TTA+tempo turned out SUB-additive on CMLt while SUPER-additive on
AMLc. So the stack is measured, not summed.

ROWS (each adds one component to the row above; all n=217, Beat This activations, mir_eval 5 s trim):
  0  madmom joint DBN            -- what a practitioner would actually run today
  1  peak-pick @0.5              -- the decoder-free baseline the paper quotes (0.6270)
  2  beat-only DBN               -- our decoder finding (+0.0255 F / +0.0570 CMLt vs joint)
  3  + weight rebalance          -- w=[1.6, 0, 0, 0.6], selected on CORPUS fold 0, leak-free
  4  + TTA {0.85, 1.2}           -- the audit's best subset; the r=1.0 pass contributes nothing
  5  + tempo head at beta=0.2    -- our best head (PESTO-equivariance loss, 57.6% track accuracy)

NO ORACLE ANYWHERE. The tempo head trains on corpus only; the weights were selected on corpus fold 0,
which is held out of that head's training; TTA needs no labels. Nothing here sees SMC in training or
selection.

Reported against BOTH baselines, because they answer different questions: row 1 (peak-pick) is what
the paper's own numbers use, row 0 (madmom joint) is what the field deploys.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import (FPS, track_ids, load_gt, ev, peakpick, make_dbn, dbn_beats,
                      bthis_beat_prob, bthis_downbeat_prob)
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_tempo_observation import load_net

CACHE = ROOT / "data" / "ts_tta_cache"
SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
HEAD = Path("/disk1/jaehoon/tempo_head_eq1.0.pt")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
TTA_RATES = [0.85, 1.2]
W_REBAL = [1.6, 0.0, 0.0, 0.6]
BETA = 0.2
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"


def tta_prob(tid):
    ls = []
    for r in TTA_RATES:
        f = CACHE / f"{tid}__r{r}.npz"
        if not f.exists():
            return None
        ls.append(np.asarray(np.load(f)["logits"], dtype=np.float64))
    n = min(len(x) for x in ls)
    return expit(np.stack([x[:n] for x in ls], 0).mean(0))


def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    bfr = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)
    s2c = m.state_to_classes[0].to(dev)
    net = load_net(str(HEAD), dev)
    marg = np.load(str(HEAD) + ".marginal.npy")
    logmarg = torch.tensor(np.log(np.clip(marg, 1e-12, None)), dtype=torch.float64, device=dev)
    joint = make_dbn(min_bpm=30.0, transition_lambda=100)

    def beatonly(act, w=None, tempo_lq=None, beta=0.0):
        got = emissions_beatonly(m, act)
        if got is None:
            return None
        lcd, dec, ff = got
        T = lcd.shape[0]
        w = w or [1.0, 0.0, 0.0, 1.0]
        em = w[0] * lcd[:, s2c]
        if beta > 0 and tempo_lq is not None:
            lq = tempo_lq
            if lq.shape[0] < ff + T:
                lq = torch.cat([lq, lq[-1:].repeat(ff + T - lq.shape[0], 1)], 0)
            em = em + beta * ((lq[ff:ff + T] - logmarg[None, :])[:, tmap] / bfr.unsqueeze(0))
        path, _ = m.dynamic_programs[0].viterbi(
            m.log_initial_distributions[0], w[3] * m.log_tempo_transition, em,
            state_to_class=None, return_log_score=True)
        return state_path_to_events(path.cpu().numpy(), space, FPS,
                                    snap_to_activations=dec, first_frame=ff)["beats"]

    ROWS = ("0 madmom joint", "1 peak-pick", "2 beat-only", "3 +rebalance", "4 +TTA", "5 +tempo")
    res = {r: {} for r in ROWS}
    for t in track_ids():
        raw = np.asarray(bthis_beat_prob(t), np.float64)
        tta = tta_prob(t)
        f = SMC_EMB / f"{t}.npz"
        if tta is None or not f.exists():
            continue
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(np.load(f, allow_pickle=True)["emb"],
                                            np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            lq = torch.log_softmax(net(x).float(), 1)[0].T.double()
        gt = load_gt(t)
        out = {
            "0 madmom joint": dbn_beats(raw, np.asarray(bthis_downbeat_prob(t), np.float64), joint),
            "1 peak-pick": peakpick(raw, 0.5, FPS),
            "2 beat-only": beatonly(raw),
            "3 +rebalance": beatonly(raw, W_REBAL),
            "4 +TTA": beatonly(tta, W_REBAL),
            "5 +tempo": beatonly(tta, W_REBAL, lq, BETA),
        }
        if any(v is None for v in out.values()):
            continue
        for k, v in out.items():
            res[k][t] = ev(gt, v)

    ids = sorted(set.intersection(*[set(res[r]) for r in ROWS]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}   NO ORACLE ANYWHERE")
    print(f"{'row':<16}" + "".join(f"{k:>9}" for k in METRICS))
    for r in ROWS:
        print(f"{r:<16}" + "".join(f"{A(res[r])[k]:>9.4f}" for k in METRICS))
    for base in ("1 peak-pick", "0 madmom joint"):
        print(f"\nfull stack (row 5) minus {base}, paired:")
        for k in METRICS:
            d = np.array([res["5 +tempo"][t][k] - res[base][t][k] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            print(f"    {k} {d.mean():+.4f} (p={p:.3g})")
    print("\nper-step increments (each row minus the one above), CMLt / F:")
    for a, b in zip(ROWS[1:], ROWS[2:]):
        dc = np.array([res[b][t]["CMLt"] - res[a][t]["CMLt"] for t in ids])
        df = np.array([res[b][t]["F"] - res[a][t]["F"] for t in ids])
        print(f"    {b:<14} CMLt {dc.mean():+.4f} (p={wilcoxon(dc).pvalue:.3g})   "
              f"F {df.mean():+.4f} (p={wilcoxon(df).pvalue:.3g})")
    json.dump({"n": len(ids), "summary": {r: A(res[r]) for r in ROWS}},
              open("/tmp/overnight/best_stack.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
