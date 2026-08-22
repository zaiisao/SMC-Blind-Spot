"""Mixture transition kernel (era: experiments/bt_e2e/mixture_kernel_probe.py, verbatim math).
p(j|i) = w * Dither(j|i) + (1-w) * exp-kernel_lambda(j|i);
Dither = uniform over {j: |interval_j - interval_i| <= 1} -- absorbs integer-grid dithering.
(w, lambda) learned jointly by exact EM (E-step unchanged, M-step 2-D argmax)."""
import numpy as np, torch
from rungs.r2_em_dbn import R2GenerativeLambda
from rungs.bar_pointer.readout import state_path_to_events
from rungs.deployment import threshold_crop


class MixtureLambda(R2GenerativeLambda):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        intervals = self.chassis.state_spaces[0].interval_frames.astype(np.int64)
        adj = (np.abs(intervals[None, :] - intervals[:, None]) <= 1).astype(np.float32)
        self._log_dither = torch.from_numpy(
            np.log(adj / adj.sum(axis=1, keepdims=True))).to(self.device)
        self.mixture_weight = 0.3

    def log_mixture_kernel(self, w, lam) -> torch.Tensor:
        parts = torch.stack([np.log(w) + self._log_dither,
                             np.log(1.0 - w) + self.log_kernel(lam)])
        return torch.logsumexp(parts, dim=0)

    def m_step_2d(self, counts: torch.Tensor):
        ws = torch.linspace(0.01, 0.95, 48, device=self.device)
        lams = torch.logspace(np.log10(2.0), np.log10(600.0), 200, device=self.device)
        scores = -lams[:, None, None] * self._abs_ratio_dev[None]
        log_exp = scores - torch.logsumexp(scores, dim=2, keepdim=True)
        mix = torch.logsumexp(torch.stack([
            (torch.log(ws)[:, None, None, None]
             + self._log_dither[None, None]).expand(-1, len(lams), -1, -1),
            torch.log(1 - ws)[:, None, None, None] + log_exp[None]]), dim=0)
        objective = (counts[None, None] * mix).sum(dim=(2, 3))
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
    def decode(self, activations, deploy: bool = False, deploy_threshold: float = 0.2,
               snap: bool = False) -> dict:
        if not torch.is_tensor(activations):
            arr = np.ascontiguousarray(np.asarray(activations, dtype=np.float32))
        else:
            arr = activations.cpu().numpy().astype(np.float32)
        first_frame = 0
        snap_acts = None
        if deploy:
            arr, first_frame = threshold_crop(arr.astype(np.float64), deploy_threshold)
            arr = arr.astype(np.float32)
            if not arr.size:
                return {"beats": np.array([]), "downbeats": np.array([])}
            if snap:
                snap_acts = arr.astype(np.float64)
        acts = torch.from_numpy(arr).to(self.device)
        densities = self.log_class_densities(acts)
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
        return state_path_to_events(path, space, self.chassis.fps, snap_to_activations=snap_acts,
                                    first_frame=first_frame)
