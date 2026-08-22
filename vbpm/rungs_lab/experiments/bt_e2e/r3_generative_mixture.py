"""R3-on-the-corrected-kernel: per-frame audio-conditioned modulation of the GENUINE-tempo
component of the mixture kernel, trained GENERATIVELY (marginal likelihood, frozen frontend).

Pre-fix, this experiment was meaningless: the marginal was dominated by integer-interval
dithering, so a conditioning net trained on it would learn to predict the artifact. With the
dither component (w) owning the grid flips, the net's modulation of lambda_t targets only
genuine tempo behavior -- the first ON-PROGRAM (generative) test of R3's mechanism.

Kernel per frame t:
  p_t(j|i) = w * Dither(j|i) + (1-w) * softmax_j(-lambda_t * |r_ij - 1|)
  lambda_t = lambda_base * exp(net(x)_t)   (net zero-init: starts at the global optimum)
w, lambda_base fixed at the values learned by mixture_kernel_probe (w=0.370, lambda=93.1);
only the net trains, by Adam on the exact marginal (Fisher-equivalent to generalized EM).

Output: init (=global mixture) vs trained per-frame decode on the val fold, bare Viterbi.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:3"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.r3_conditioned_dbn import R3ConditionedFactors
from rungs.bar_pointer.readout import state_path_to_events
from crf_baseline import CRFLearnedFactors

DEVICE = "cuda:3"
NUM_CROPS, CROP_FRAMES = 300, 700
STEPS, BATCH = 200, 16
W_MIX, LAMBDA_BASE = 0.370, 93.1


class R3GenerativeMixture(R3ConditionedFactors):
    """Per-frame mixture kernel + meter-marginal likelihood on top of the R3 conditioning net."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        intervals = self.chassis.state_spaces[0].interval_frames.astype(np.int64)
        adj = (np.abs(intervals[None, :] - intervals[:, None]) <= 1).astype(np.float32)
        self._log_dither = torch.from_numpy(
            np.log(adj / adj.sum(axis=1, keepdims=True))).to(self.device)
        self.w = W_MIX

    def per_frame_mixture_kernel(self, activations: torch.Tensor) -> torch.Tensor:
        lam_t = self.log_lambda_per_frame(activations).exp()            # [T]
        scores = -lam_t[:, None, None] * self.abs_ratio_dev[None]      # [T,V,V]
        log_exp = scores - torch.logsumexp(scores, dim=2, keepdim=True)
        return torch.logsumexp(torch.stack([
            (np.log(self.w) + self._log_dither)[None].expand(len(lam_t), -1, -1),
            np.log(1.0 - self.w) + log_exp]), dim=0)

    def marginal_log_likelihood(self, activations: torch.Tensor) -> torch.Tensor:
        densities = self.log_class_densities(activations)
        kernel = self.per_frame_mixture_kernel(activations)
        per_meter = [dp.forward_log_likelihood(init, kernel, densities, state_to_class=s2c)
                     for dp, init, s2c in zip(self.chassis.dynamic_programs,
                                              self.chassis.log_initial_distributions,
                                              self.chassis.state_to_classes)]
        return torch.logsumexp(torch.stack(per_meter), dim=0)

    @torch.no_grad()
    def decode_mixture(self, activations: torch.Tensor) -> dict:
        densities = self.log_class_densities(activations)
        kernel = self.per_frame_mixture_kernel(activations)
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


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    r3 = R3GenerativeMixture(fps=FPS, device=DEVICE, lambda_base=LAMBDA_BASE,
                             observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
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
    print(f"{len(crops)} crops | w={W_MIX} lambda_base={LAMBDA_BASE} | net params "
          f"{sum(p.numel() for p in r3.net.parameters())}", flush=True)

    def val_F():
        fs = []
        for e in val:
            ev = r3.decode_mixture(torch.from_numpy(val_acts[e["stem"]]).float().to(DEVICE))
            est = mir_eval.beat.trim_beats(ev["beats"])
            fs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(e["beat_times"]), est)
                      if len(est) else 0.0)
        return float(np.mean(fs))

    print(f"INIT (net=0 == global mixture)     : beatF {val_F():.4f}", flush=True)

    opt = torch.optim.Adam(r3.net.parameters(), lr=1e-3)
    for step in range(STEPS):
        idx = rng.choice(len(crops), size=BATCH, replace=False)
        loss = -torch.stack([r3.marginal_log_likelihood(crops[i]) / crops[i].shape[0]
                             for i in idx]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(r3.net.parameters(), 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            with torch.no_grad():
                spread = torch.cat([r3.log_lambda_per_frame(crops[i]).exp() for i in idx[:4]])
            print(f"step {step+1}: nll/frame {float(loss):.4f} | lambda_t "
                  f"p10 {float(spread.quantile(0.1)):.1f} med {float(spread.median()):.1f} "
                  f"p90 {float(spread.quantile(0.9)):.1f}", flush=True)

    print(f"TRAINED per-frame (generative)     : beatF {val_F():.4f}", flush=True)
    torch.save(r3.state_dict(), Path(__file__).resolve().parent / "r3_generative_mixture.pt")


if __name__ == "__main__":
    main()
