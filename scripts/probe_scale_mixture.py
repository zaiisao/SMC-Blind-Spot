"""Quick probe: does a MIXTURE OVER TRANSITION SCALES beat the best single lambda?

Motivation. On SMC the per-track oracle lambda is bimodal -- 73/217 tracks pin at the grid
floor, 34 want lambda >= 100 -- and the modes are not predictable from tempo. Every deployable
approach tried so far SELECTS one lambda per track, which needs a criterion; every criterion
tried is likelihood-based and the likelihood is corrupted by the integer-frame grid
(rho = -0.23 between ML lambda and oracle lambda). Marginalising over scale needs no criterion:

    p(j|i) = w * exp(-lam_tight |r-1|) + (1-w) * exp(-lam_loose |r-1|)

with ONE global w. The forward pass integrates over scale, so each track leans on whichever
component explains it without anyone deciding which mode it is in.

Question this probe answers: on a stratified SMC subset, does the best GLOBAL mixture beat the
best GLOBAL single lambda -- and does it serve BOTH oracle modes at once?

Everything runs on the R1 chassis (certified event-identical to madmom on 217/217 SMC tracks at
min55, 216/217 at min30), so single-lambda and mixture rows are produced by the SAME engine and
are directly comparable. Absolute values will not match the madmom-produced rows in
FOLLOWUP_FINDINGS; that is expected and irrelevant -- every comparison here is internal.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))

from _harness import FPS, track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob, ev
from rungs.r2_em_dbn import R2GenerativeLambda
from rungs.bar_pointer.readout import state_path_to_events
from rungs.deployment import threshold_crop

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
OUT = Path("/tmp/overnight/scale_mixture_probe.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
JSONL = OUT.with_suffix(".per_track.jsonl")

N_TRACKS = 16
LAM_GRID = [1.0, 5.0, 20.0, 100.0, 500.0]
MIX_PAIRS = [(1.0, 100.0), (2.0, 300.0)]
W_GRID = [0.2, 0.4, 0.6, 0.8]


class ScaleMixture(R2GenerativeLambda):
    """Two exponential components at different scales, mixed with weight w. No dither term."""

    def log_scale_kernel(self, w, lam_tight, lam_loose):
        parts = torch.stack([
            float(np.log(w)) + self.log_kernel(lam_tight),
            float(np.log(1.0 - w)) + self.log_kernel(lam_loose),
        ])
        return torch.logsumexp(parts, dim=0)

    @torch.no_grad()
    def decode_with(self, activations, kernel, deploy_threshold=0.2):
        arr = np.ascontiguousarray(np.asarray(activations, dtype=np.float64))
        arr, first_frame = threshold_crop(arr, deploy_threshold)
        if not arr.size:
            return {"beats": np.array([]), "downbeats": np.array([])}
        snap_acts = arr.astype(np.float64)
        acts = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        densities = self.log_class_densities(acts)
        best = None
        for mi in range(len(self.chassis.state_spaces)):
            dp = self.chassis.dynamic_programs[mi]
            path, score = dp.viterbi(
                self.chassis.log_initial_distributions[mi], kernel, densities,
                state_to_class=self.chassis.state_to_classes[mi], return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path.cpu().numpy(), self.chassis.state_spaces[mi])
        _, path, space = best
        return state_path_to_events(path, space, self.chassis.fps,
                                    snap_to_activations=snap_acts, first_frame=first_frame)


def acts_of(tid):
    return np.stack([bthis_beat_prob(tid), bthis_downbeat_prob(tid)], axis=-1)


def main():
    ids_all = track_ids()
    ids = ids_all[:N_TRACKS]
    refs = {t: load_gt(t) for t in ids}
    acts = {t: acts_of(t) for t in ids}
    print(f"device={DEVICE} n={len(ids)}", flush=True)

    model = ScaleMixture(fps=FPS, device=DEVICE, observation_lambda=6,
                         min_bpm=30.0, max_bpm=215.0)

    fh = open(JSONL, "w")
    single = {}
    for lam in LAM_GRID:
        kern = model.log_kernel(lam)
        per = {}
        for t in ids:
            sc = ev(refs[t], model.decode_with(acts[t], kern)["beats"])
            per[t] = sc
            fh.write(json.dumps({"arm": "single", "lam": lam, "track": t, **sc}) + "\n")
        fh.flush()
        single[lam] = per
        print(f"single lam={lam:<6} F {np.mean([per[t]['F'] for t in ids]):.4f} "
              f"CMLt {np.mean([per[t]['CMLt'] for t in ids]):.4f} "
              f"AMLt {np.mean([per[t]['AMLt'] for t in ids]):.4f}", flush=True)

    # per-track oracle lambda -> the bimodality, computed here rather than assumed
    oracle_lam = {t: max(LAM_GRID, key=lambda L: single[L][t]["F"]) for t in ids}
    tight = [t for t in ids if oracle_lam[t] <= 2.0]
    loose = [t for t in ids if oracle_lam[t] >= 50.0]
    mid = [t for t in ids if t not in tight and t not in loose]
    print(f"\noracle-lambda modes: tight(<=2) {len(tight)}  mid {len(mid)}  loose(>=50) {len(loose)}",
          flush=True)

    mixture = {}
    for lt, ll in MIX_PAIRS:
        for w in W_GRID:
            kern = model.log_scale_kernel(w, lt, ll)
            per = {}
            for t in ids:
                sc = ev(refs[t], model.decode_with(acts[t], kern)["beats"])
                per[t] = sc
                fh.write(json.dumps({"arm": "mix", "lam_tight": lt, "lam_loose": ll,
                                     "w": w, "track": t, **sc}) + "\n")
            fh.flush()
            mixture[f"{lt}/{ll}/w{w}"] = per
            print(f"mix ({lt},{ll}) w={w:<4} F {np.mean([per[t]['F'] for t in ids]):.4f} "
                  f"CMLt {np.mean([per[t]['CMLt'] for t in ids]):.4f} "
                  f"AMLt {np.mean([per[t]['AMLt'] for t in ids]):.4f}", flush=True)
    fh.close()

    def summarise(per):
        out = {}
        for name, grp in (("all", ids), ("tight", tight), ("mid", mid), ("loose", loose)):
            if not grp:
                continue
            out[name] = {m: float(np.mean([per[t][m] for t in grp]))
                         for m in ("F", "CMLt", "AMLt", "CMLc", "AMLc")}
        return out

    best_single = max(single, key=lambda L: np.mean([single[L][t]["F"] for t in ids]))
    best_mix = max(mixture, key=lambda k: np.mean([mixture[k][t]["F"] for t in ids]))
    oracle_F = float(np.mean([max(single[L][t]["F"] for L in LAM_GRID) for t in ids]))

    result = {
        "n": len(ids), "device": DEVICE, "lam_grid": LAM_GRID,
        "mode_counts": {"tight": len(tight), "mid": len(mid), "loose": len(loose)},
        "oracle_lambda_per_track": oracle_lam,
        "per_track_oracle_F": oracle_F,
        "best_single_lambda": best_single,
        "best_single": summarise(single[best_single]),
        "best_mixture": best_mix,
        "best_mixture_scores": summarise(mixture[best_mix]),
        "all_single": {str(L): summarise(single[L]) for L in LAM_GRID},
        "all_mixture": {k: summarise(v) for k, v in mixture.items()},
    }
    OUT.write_text(json.dumps(result, indent=1))
    print(f"\nbest single lambda={best_single}: {result['best_single']['all']}", flush=True)
    print(f"best mixture {best_mix}: {result['best_mixture_scores']['all']}", flush=True)
    print(f"per-track ORACLE lambda ceiling F={oracle_F:.4f}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
