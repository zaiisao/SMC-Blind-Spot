"""GATE: the CRF must reproduce the baseline DBN exactly at w = [1, 0, 0, 1].

Nothing learned on top of a log-linear model means anything until the model with UNIT weights on the
emission and transition, and ZERO on everything else, decodes identically to the decoder it
generalises. The first CRF run's "null" row gave F 0.6215 / CMLt 0.5720 where the beat-only DBN
gives 0.6141 / 0.5685 -- close, but a discrepancy that was explained by hand-waving ("I used the
unmasked kernel") rather than reconciled to zero.

G1  w=[1,0,0,1] with the MASKED kernel must equal the beat-only DBN, per track, bit-exactly.
G2  the unmasked kernel is then measured separately, so the size of that choice is known rather
    than assumed.
G3  RECOVERY: the trainer must recover a KNOWN weight vector. Score every path under a known w*,
    label the argmax path as "truth", train from a different init, and check w converges back
    (direction, since the objective is invariant to positive rescaling of the whole vector).
    This proves the optimiser works independently of whether the features are useful -- the
    "successful case" the implementation has never had.

Only if all three pass is a learned-weight result interpretable.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_contrastive_dbn_fixed import log_tempo_transition, build_path, make_negatives
from train_crf_decoder import acf_logratio, crf_score

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
N_GATE = 40


def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    first_state = np.cumsum(np.r_[0, iv[:-1]]).astype(np.int64)
    s2c = m.state_to_classes[0].to(dev)
    iv_t = torch.from_numpy(iv).to(dev)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)

    # the decoder's own kernel, built by madmom's construction (masked at np.spacing(1))
    ref_logT = m.log_tempo_transition
    masked = log_tempo_transition(iv_t, torch.tensor(100.0, dtype=torch.float64, device=dev),
                                  mask_mode="madmom")
    unmasked = log_tempo_transition(iv_t, torch.tensor(100.0, dtype=torch.float64, device=dev),
                                    mask_mode="none")
    fin = torch.isfinite(ref_logT) & torch.isfinite(masked)
    print(f"kernel check: ref vs masked  max|diff| on finite entries = "
          f"{float((ref_logT[fin]-masked[fin]).abs().max()):.3e}; "
          f"-inf pattern identical = {bool((~torch.isfinite(ref_logT)).eq(~torch.isfinite(masked)).all())}")
    print(f"masked forbids {int((~torch.isfinite(masked)).sum())}/{masked.numel()} pairs; "
          f"unmasked forbids {int((~torch.isfinite(unmasked)).sum())}")
    print(f"max|masked-unmasked| on finite = "
          f"{float((masked[fin]-unmasked[fin]).abs().max()):.3f} nats\n")

    ids = [t for t in track_ids()][:N_GATE]
    same_m = same_u = 0
    dm, du = [], []
    for t in ids:
        got = emissions_beatonly(m, np.asarray(bthis_beat_prob(t), np.float64))
        if got is None:
            continue
        lcd, dec, ff = got
        gt = load_gt(t)

        def decode(logT, w_em, w_tr):
            em = w_em * lcd[:, s2c]
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], w_tr * logT, em,
                state_to_class=None, return_log_score=True)
            return path.cpu().numpy()

        ref_path, _ = m.dynamic_programs[0].viterbi(
            m.log_initial_distributions[0], ref_logT, lcd,
            state_to_class=s2c, return_log_score=True)
        ref_path = ref_path.cpu().numpy()
        pm = decode(masked, 1.0, 1.0)
        pu = decode(unmasked, 1.0, 1.0)
        same_m += int(np.array_equal(ref_path, pm))
        same_u += int(np.array_equal(ref_path, pu))
        ev_ref = ev(gt, state_path_to_events(ref_path, space, FPS, snap_to_activations=dec,
                                             first_frame=ff)["beats"])
        for p_, acc in ((pm, dm), (pu, du)):
            e = ev(gt, state_path_to_events(p_, space, FPS, snap_to_activations=dec,
                                            first_frame=ff)["beats"])
            acc.append(e["F"] - ev_ref["F"])
    n = len(dm)
    print(f"G1  w=[1,0,0,1], MASKED kernel   : identical path on {same_m}/{n} tracks, "
          f"mean dF {np.mean(dm):+.5f}   {'PASS' if same_m == n else 'FAIL'}")
    print(f"G2  w=[1,0,0,1], UNMASKED kernel : identical path on {same_u}/{n} tracks, "
          f"mean dF {np.mean(du):+.5f}   (the cost of mask_mode='none')")

    # ---------------- G3: can the trainer recover a KNOWN weight vector? ----------------
    print("\nG3  weight recovery on synthetic 'truth'")
    rng = np.random.default_rng(0)
    W_STAR = torch.tensor([1.0, 0.6, -0.3, 0.8], dtype=torch.float64, device=dev)
    items = []
    for t in ids[:20]:
        got = emissions_beatonly(m, np.asarray(bthis_beat_prob(t), np.float64))
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        F = {"em": lcd, "s2c": s2c,
             "tempo": torch.log_softmax(torch.randn(T, len(iv), dtype=torch.float64,
                                                    device=dev), 1)[:, tmap],
             "acf": acf_logratio(np.asarray(bthis_beat_prob(t), np.float64)[ff:ff + T],
                                 iv, 10.0, dev)[tmap]}
        gt = np.asarray(load_gt(t), float) * FPS - ff
        cands = {"pos": build_path(gt, T, iv, first_state)}
        for k, v in make_negatives(gt, rng).items():
            r = build_path(v, T, iv, first_state)
            if r is not None:
                cands[k] = r
        cands = {k: v for k, v in cands.items() if v is not None}
        if len(cands) < 3:
            continue
        tt = lambda p: (torch.as_tensor(p[0], device=dev), torch.as_tensor(p[1], device=dev),
                        torch.as_tensor(p[2], device=dev))
        items.append({"F": F, "cands": [tt(v) for v in cands.values()]})
    logT_star = log_tempo_transition(iv_t, torch.tensor(100.0, dtype=torch.float64, device=dev),
                                     mask_mode="none")
    # relabel: under w*, whichever candidate scores highest IS the truth
    for it in items:
        sc = torch.stack([crf_score(*c, it["F"], logT_star, W_STAR) for c in it["cands"]])
        it["y"] = int(sc.argmax())
    MARGIN = float(__import__("os").environ.get("MARGIN", "1.0"))
    L2 = float(__import__("os").environ.get("L2", "0.05"))
    print(f"   (margin={MARGIN}, L2={L2}: without them the problem is SEPARABLE -- loss reached"
          f" 1e-4 while cos(w,w*) DEGRADED 0.9467 -> 0.9415 and ||w|| inflated)")
    w = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for step in range(400):
        opt.zero_grad()
        tot = 0.0
        for it in items:
            sc = torch.stack([crf_score(*c, it["F"], logT_star, w) for c in it["cands"]])
            sc = sc + MARGIN                      # margin on every candidate...
            sc = sc - MARGIN * torch.nn.functional.one_hot(
                torch.tensor(it["y"], device=dev), len(sc)).double()   # ...except the true one
            loss = -(sc[it["y"]] - torch.logsumexp(sc, 0))
            loss.backward(retain_graph=True)
            tot += float(loss)
        if L2 > 0:
            (L2 * (w * w).sum()).backward()
        torch.nn.utils.clip_grad_norm_([w], 5.0)
        opt.step()
        if (step + 1) % 100 == 0:
            wd = w.detach()
            cos = float(torch.dot(wd, W_STAR) / (wd.norm() * W_STAR.norm()))
            print(f"   step {step+1}  loss {tot/max(len(items),1):.4f}  "
                  f"w=[{', '.join(f'{float(x):+.3f}' for x in wd)}]  cos(w,w*)={cos:.4f}",
                  flush=True)
    wd = w.detach()
    cos = float(torch.dot(wd, W_STAR) / (wd.norm() * W_STAR.norm()))
    print(f"   w*     = [{', '.join(f'{float(x):+.3f}' for x in W_STAR)}]")
    print(f"   learned= [{', '.join(f'{float(x):+.3f}' for x in wd)}]")
    print(f"   cosine similarity (objective is scale-invariant) = {cos:.4f}   "
          f"{'PASS' if cos > 0.95 else 'FAIL'}")
    print(f"\nn_items={len(items)}  (recovery uses SYNTHETIC labels, so it tests the OPTIMISER, "
          f"not whether the features are useful)")


if __name__ == "__main__":
    main()
