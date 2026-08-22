"""Two-scale (mixture) transition kernel: does separating grid-dither from genuine tempo
change (a) recover a stiff artifact-free lambda, and (b) close the R1-vs-R2 decode gap?

Kernel: p(j|i) = w * D(j|i) + (1-w) * exp-kernel_lambda(j|i)
  D = uniform over {j : |interval_j - interval_i| <= 1}  (the grid-dither component)
  (w, lambda) learned jointly by the SAME exact EM (E-step unchanged; M-step = 2-D argmax).

Stages:
  1. metronome 21.4 (the artifact isolator): expect w ~= dither rate, lambda >> 40
  2. real crops (300, same as train_r2_em): expect lambda_music >> 40 -- first artifact-free
     measurement of musicians' tempo stability
  3. decode comparison on the val fold: R1 lambda=100 vs the learned mixture kernel
     (custom-kernel Viterbi, same DP + readout as R3's decode path)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:2"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.r2_em_dbn import R2GenerativeLambda
from rungs.r1_2016_dbn import DBN2016
from rungs.bar_pointer.readout import state_path_to_events
from crf_baseline import CRFLearnedFactors
from synthetic_grid_probe import synth

DEVICE = "cuda:2"
NUM_CROPS, CROP_FRAMES, EM_ITERS = 300, 700, 8


class MixtureLambda(R2GenerativeLambda):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        intervals = self.chassis.state_spaces[0].interval_frames.astype(np.int64)
        adj = (np.abs(intervals[None, :] - intervals[:, None]) <= 1).astype(np.float32)
        self._log_dither = torch.from_numpy(
            np.log(adj / adj.sum(axis=1, keepdims=True))).to(self.device)   # [V,V], row-normed
        self.mixture_weight = 0.3

    def log_mixture_kernel(self, w, lam) -> torch.Tensor:
        parts = torch.stack([np.log(w) + self._log_dither,
                             np.log(1.0 - w) + self.log_kernel(lam)])
        return torch.logsumexp(parts, dim=0)

    def m_step_2d(self, counts: torch.Tensor):
        ws = torch.linspace(0.01, 0.95, 48, device=self.device)
        lams = torch.logspace(np.log10(2.0), np.log10(600.0), 200, device=self.device)
        scores = -lams[:, None, None] * self._abs_ratio_dev[None]            # [L,V,V]
        log_exp = scores - torch.logsumexp(scores, dim=2, keepdim=True)
        # [W,L,V,V] log mixture
        mix = torch.logsumexp(torch.stack([
            (torch.log(ws)[:, None, None, None]
             + self._log_dither[None, None]).expand(-1, len(lams), -1, -1),
            torch.log(1 - ws)[:, None, None, None] + log_exp[None]]), dim=0)
        objective = (counts[None, None] * mix).sum(dim=(2, 3))               # [W,L]
        flat = int(objective.argmax())
        wi, li = flat // len(lams), flat % len(lams)
        return float(ws[wi]), float(lams[li])

    def em_step_mixture(self, crops):
        counts = self._expected_counts(
            self.marginal_log_likelihood, crops,
            self.log_mixture_kernel(self.mixture_weight, self.transition_lambda))
        self.mixture_weight, self.transition_lambda = self.m_step_2d(counts)
        return self.mixture_weight, self.transition_lambda

    @torch.no_grad()
    def decode(self, activations: torch.Tensor) -> dict:
        densities = self.log_class_densities(activations)
        kernel = self.log_mixture_kernel(self.mixture_weight, self.transition_lambda)
        best = None
        for mi in range(len(self.chassis.state_spaces)):
            dp = self.chassis.dynamic_programs[mi]
            path, score = dp.viterbi(self.chassis.log_initial_distributions[mi], kernel,
                                     densities, state_to_class=self.chassis.state_to_classes[mi],
                                     return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path.cpu().numpy(), self.chassis.state_spaces[mi])
        _, path, space = best
        return state_path_to_events(path, space, self.chassis.fps)


def run_em(crops, label):
    m = MixtureLambda(fps=FPS, device=DEVICE,
                      observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    print(f"\n=== {label} ===", flush=True)
    for it in range(EM_ITERS):
        w, lam = m.em_step_mixture(crops)
        print(f"  EM iter {it}: w = {w:.3f}, lambda = {lam:.2f}", flush=True)
    return m


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)

    # 1. metronome isolator
    metro = [torch.from_numpy(synth(21.4, rng.uniform(0, 21.4), 700)).float().to(DEVICE)
             for _ in range(50)]
    run_em(metro, "metronome 21.4 (expect w ~= dither rate, lambda >> 40)")

    # 2 + 3. real data
    probe = CRFLearnedFactors(fps=FPS, device=DEVICE,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train, val, _ = load_songs(probe)
    model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
    train_acts = final_eval.activations_for(model, train[:NUM_CROPS])
    val_acts = final_eval.activations_for(model, val)
    crops = []
    for e in train[:NUM_CROPS]:
        a = train_acts[e["stem"]]
        if a.shape[0] > CROP_FRAMES + 1:
            s = rng.integers(0, a.shape[0] - CROP_FRAMES)
            a = a[s:s + CROP_FRAMES]
        crops.append(torch.from_numpy(np.ascontiguousarray(a)).float().to(DEVICE))
    m = run_em(crops, "real crops (artifact-free lambda_music)")

    def score_events(get_events):
        fs = []
        for e in val:
            ev = get_events(e)
            est = mir_eval.beat.trim_beats(ev["beats"])
            fs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(e["beat_times"]), est)
                      if len(est) else 0.0)
        return float(np.mean(fs))

    print("\n=== decode comparison (val fold) ===", flush=True)
    r1 = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none",
                 transition_lambda=100.0, **BT_SHIPPED_DECODE)
    print(f"R1 hand-set 100 (shipped decode)      : beatF "
          f"{score_events(lambda e: r1.predict(val_acts[e['stem']])):.4f}", flush=True)
    bare = DBN2016(fps=FPS, device=DEVICE, dtype=torch.float32, bounding="none",
                   transition_lambda=100.0, observation_lambda=6, num_tempi=None,
                   threshold=0.0, correct=False)
    print(f"R1 hand-set 100 (bare viterbi)        : beatF "
          f"{score_events(lambda e: bare.predict(val_acts[e['stem']])):.4f}", flush=True)
    print(f"mixture (w={m.mixture_weight:.3f}, lambda={m.transition_lambda:.1f}) bare : beatF "
          f"{score_events(lambda e: m.decode(torch.from_numpy(val_acts[e['stem']]).float().to(DEVICE))):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
