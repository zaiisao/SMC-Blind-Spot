"""Score Beat This training checkpoints on SMC, per epoch, with OUR validated decoding path.

WHY NOT USE THE TRAINER'S OWN VALIDATION. beat_this's pl_module logs metrics POOLED across all
validation datasets (log_metrics has no per-dataset breakdown), and it postprocesses with its
`minimal` peak-picker. The question here is specifically how SMC evolves under the beat-only DBN we
ship, so the checkpoints are scored separately with the same path every other number in this project
uses: cached SMC spectrograms -> model -> beat-only DBN (class-corrected, min_bpm 30, lambda 100)
-> mir_eval with the 5 s trim.

WHAT THIS IS FOR. The leave-one-dataset-out ablation: train from scratch with and without ASAP and
watch SMC. ASAP is the corpus's expressive-piano data (sigma_d 0.213 vs SMC's 0.152, 473 tracks) and
is already in Beat This's training set, so its marginal value cannot be measured by fine-tuning --
you cannot un-learn a dataset by fine-tuning on the others. From-scratch training is the only way to
ask what it is worth.
    removing ASAP costs SMC a lot  -> the regime matters, synthesising more of it is justified
    removing ASAP costs ~nothing   -> the premise is wrong, stop building on it

FOLD HONESTY. Both arms train with --fold 0, i.e. fold 0 is held out of training for EVERY dataset.
So SMC fold 0 (27 tracks) is the honest test set here, and the other SMC folds were trained on --
scoring all 217 would mix memorised and held-out tracks. Both are reported, clearly separated,
because the held-out number is the valid one and the trained-on number shows the memorisation gap
for reference.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events

SMC_NPZ = Path("/disk1/jaehoon/beat_this_spect/done/smc.npz")
# The repo was reorganised mid-session (submodules moved under third-party/), so resolve rather
# than hard-code -- the old top-level path still exists but is now an empty directory.
SPLIT = next((p for p in (ROOT / "third-party" / "beat_this_annotations" / "smc" / "8-folds.split",
                          ROOT / "beat_this_annotations" / "smc" / "8-folds.split")
              if p.exists()), None)
assert SPLIT is not None, "8-folds.split not found in either layout"
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def load_ckpt(path, device):
    """Lightning checkpoint -> a bare BeatThis in eval mode."""
    from beat_this.model.beat_tracker import BeatThis
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    # pl_module wraps the network as self.model, so keys are prefixed 'model.'
    inner = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    if not inner:
        inner = sd
    net = BeatThis()
    miss, unexp = net.load_state_dict(inner, strict=False)
    if miss or unexp:
        print(f"    (load: {len(miss)} missing, {len(unexp)} unexpected)", flush=True)
    return net.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="checkpoint directories to scan")
    ap.add_argument("--fold", type=int, default=0, help="fold held out of training")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default="/tmp/overnight/ablation_smc.json")
    args = ap.parse_args()
    dev = args.device

    splits = {}
    for line in open(SPLIT):
        p = line.split()
        if len(p) >= 2:
            splits[p[0]] = int(p[-1])
    heldout = [t for t in track_ids() if splits.get(t) == args.fold]
    trained = [t for t in track_ids() if splits.get(t) not in (None, args.fold)]

    m = make_beatonly_dbn(30.0, 16, 100, dev)
    space, dp0 = m.state_spaces[0], m.dynamic_programs[0]
    s2c = m.state_to_classes[0]
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    z = np.load(SMC_NPZ, mmap_mode="r")
    from beat_this.inference import split_predict_aggregate

    @torch.no_grad()
    def score(net, ids):
        out = {}
        for t in ids:
            key = f"{t}/track"
            if key not in z.files:
                continue
            sp = torch.from_numpy(np.asarray(z[key], np.float32)).to(dev)
            o = split_predict_aggregate(spect=sp, chunk_size=1500, overlap_mode="keep_first",
                                        border_size=6, model=net)
            prob = expit(np.asarray(o["beat"].float().cpu().numpy(), np.float64))
            got = emissions_beatonly(m, prob)
            if got is None:
                continue
            lcd, dec, ff = got
            path, _ = dp0.viterbi(logpi, logA, lcd, state_to_class=s2c, return_log_score=True)
            b = state_path_to_events(path.cpu().numpy(), space, FPS,
                                     snap_to_activations=dec, first_frame=ff)["beats"]
            out[t] = ev(np.asarray(load_gt(t), float), b)
        return out

    A = lambda r: {k: float(np.mean([v[k] for v in r.values()])) for k in METRICS}
    results = {}
    for d in args.dirs:
        arm = Path(d).parent.name
        cks = sorted(Path(d).glob("*.ckpt"))
        if not cks:
            print(f"{d}: no checkpoints yet", flush=True)
            continue
        for ck in cks:
            mt = re.search(r"epepoch=(\d+)", ck.name)
            ep = int(mt.group(1)) if mt else -1
            name = re.sub(r" S0.*", "", ck.name)
            net = load_ckpt(ck, dev)
            ho = score(net, heldout)
            tr = score(net, trained[:40])          # subsample: reference only, not the result
            results.setdefault(name, {})[ep] = {"heldout": A(ho), "trained_on": A(tr),
                                                "n_heldout": len(ho)}
            h, t2 = A(ho), A(tr)
            print(f"{name:<14} ep{ep:>3}  HELD-OUT F {h['F']:.4f} CMLt {h['CMLt']:.4f}"
                  f"   | trained-on F {t2['F']:.4f} (reference)", flush=True)
            del net
            torch.cuda.empty_cache()

    print(f"\nreference: peak-picked Beat This (released fold0 ckpt) F 0.6394 on this fold;"
          f" beat-only DBN F 0.6394 / CMLt 0.6273")
    json.dump(results, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
