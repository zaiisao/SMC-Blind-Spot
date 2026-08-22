"""R3 mixture model class (see r3_lab.py for training). arm: wt | lamt."""
import numpy as np, torch
from rungs.r3_conditioned_dbn import R3ConditionedFactors
from rungs.bar_pointer.readout import state_path_to_events

class R3Mixture(R3ConditionedFactors):
    def __init__(self, arm='wt', w0=0.37, **kw):
        super().__init__(**kw)
        self.arm = arm
        intervals = self.chassis.state_spaces[0].interval_frames.astype(np.int64)
        adj = (np.abs(intervals[None, :] - intervals[:, None]) <= 1).astype(np.float32)
        self._log_dither = torch.from_numpy(
            np.log(adj / adj.sum(axis=1, keepdims=True))).to(self.device)
        self.w0 = w0

    def modulation(self, activations):                       # [T] net output
        return self.net(activations.t().unsqueeze(0))[0, 0]

    def per_frame_w(self, activations):
        w0_logit = float(np.log(self.w0 / (1 - self.w0)))
        delta = (self.modulation(activations) if self.arm == "wt"
                 else torch.zeros_like(activations[:, 0]))
        return torch.sigmoid(w0_logit + delta)

    def per_frame_lambda(self, activations):
        delta = (self.modulation(activations) if self.arm == "lamt"
                 else torch.zeros_like(activations[:, 0]))
        return self.lambda_base * torch.exp(delta)

    def per_frame_mixture_kernel(self, activations):
        w_t = self.per_frame_w(activations)                                  # [T]
        lam_t = self.per_frame_lambda(activations)                           # [T]
        scores = -lam_t[:, None, None] * self.abs_ratio_dev[None]            # [T,V,V]
        log_exp = scores - torch.logsumexp(scores, dim=2, keepdim=True)
        return torch.logsumexp(torch.stack([
            torch.log(w_t)[:, None, None] + self._log_dither[None],
            torch.log1p(-w_t)[:, None, None] + log_exp]), dim=0)

    def marginal_ll(self, activations):
        densities = self.log_class_densities(activations)
        kernel = self.per_frame_mixture_kernel(activations)
        per_meter = [dp.forward_log_likelihood(init, kernel, densities, state_to_class=s2c)
                     for dp, init, s2c in zip(self.chassis.dynamic_programs,
                                              self.chassis.log_initial_distributions,
                                              self.chassis.state_to_classes)]
        return torch.logsumexp(torch.stack(per_meter), dim=0)

    @torch.no_grad()
    def decode(self, activations, deploy: bool = False, deploy_threshold: float = 0.2,
               snap: bool = False, **_):
        from rungs.deployment import threshold_crop
        arr = activations if isinstance(activations, np.ndarray) else activations.cpu().numpy()
        arr = np.ascontiguousarray(arr.astype(np.float64)); first = 0
        snap_acts = None
        if deploy:
            arr, first = threshold_crop(arr, deploy_threshold)
            snap_acts = arr if snap else None
            if not arr.size: return {"beats": np.array([]), "downbeats": np.array([])}
        acts = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        densities = self.log_class_densities(acts)
        kernel = self.per_frame_mixture_kernel(acts)
        best = None
        for mi in range(len(self.chassis.state_spaces)):
            dp = self.chassis.dynamic_programs[mi]
            path, score = dp.viterbi(self.chassis.log_initial_distributions[mi], kernel, densities,
                                     state_to_class=self.chassis.state_to_classes[mi],
                                     return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path.cpu().numpy(), self.chassis.state_spaces[mi])
        _, path, space = best
        return state_path_to_events(path, space, self.chassis.fps, snap_to_activations=snap_acts,
                                    first_frame=first)

