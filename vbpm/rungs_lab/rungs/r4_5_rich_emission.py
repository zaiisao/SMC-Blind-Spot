"""DEPRECATED: folded into rungs/r4_neural_hmm.py as R4NeuralHMM(emission="gaussian").
Kept until tests/test_r4_5_*.py are ported to GaussianEmission; not registered in train.py.

R4.5 -- LEARNED EMISSION on RICH FEATURES: the observed variable becomes the frontend's
penultimate representation ([T, 256]); the Böck plug-in formula on [T, 2] activations is replaced
by learned class-conditional diagonal Gaussians p(f_t | class), class in {no-beat, beat,
downbeat} (the partition of the bar-pointer state space, unchanged).

FULLY EXACT Baum-Welch, unsupervised:
  E-step: posterior class occupancies gamma_t(c) by autograd on the meter-marginal forward --
          d logZ / d log_dens[t, c] = P(class(z_t) = c | f_{1:T})  (exact; same identity family
          as R2's transition counts).
  M-step: CLOSED FORM -- posterior-weighted Gaussian MLE (weighted mean / variance per class),
          with a variance floor. No optimizer, no learning rate, monotone.

Transition: the artifact-corrected mixture kernel (w, lambda fixed at R2's learned values) --
one axis changes at a time; the transition question is closed at parity.

Symmetry breaking: Gaussians are initialized from the existing pipeline's OWN decode (Viterbi
class labels from the Böck/[T,2] system) -- self-supervised, no human annotations anywhere.
"""
import numpy as np
import torch

from rungs.base import Rung
from rungs.r1_2016_dbn import DBN2016

VARIANCE_FLOOR = 1e-3


class R45RichEmission(Rung):
    INPUT_CHANNELS = None    # feature width is an instance choice (feature_dim)
    TRAIN_MODE = "em"
    ARM_NAME = "r4_5"
    TRAIN_DEFAULTS = dict(em_iterations=10)

    def __init__(self, fps: float, feature_dim: int = 256, beats_per_bar=(3, 4),
                 mixture_w: float = 0.370, transition_lambda: float = 93.1,
                 observation_lambda: int = 6, device: str = "cuda",
                 min_bpm: float = 55.0, max_bpm: float = 215.0,
                 tied_covariance: bool = False):
        # tied_covariance: one shared diagonal covariance for all classes (pooled within-class
        # variance, still exact closed-form M-step). Kills the log-det asymmetry that let rare
        # sharp classes buy constant score subsidies (v0: downbeat +660 nats, 96/256 dims at the
        # variance floor -> emission domination, transition inert). With tied covariance the
        # class comparison is pure Mahalanobis-to-mean under a common metric (LDA-like).
        # chassis supplies the state space + class partition (observation_lambda fixes the
        # partition geometry; the DENSITIES it would compute are never used here)
        self.chassis = DBN2016(fps=fps, min_bpm=min_bpm, max_bpm=max_bpm,
                               beats_per_bar=beats_per_bar, num_tempi=None,
                               threshold=0.0, correct=False,
                               observation_lambda=observation_lambda,
                               dtype=torch.float32, device=device)
        super().__init__(fps=self.chassis.fps, bounding=self.chassis.bounding,
                         eps=self.chassis.eps)
        self.device = device
        self.feature_dim = feature_dim
        intervals = self.chassis.state_spaces[0].interval_frames.astype(np.float32)
        ratio = intervals[None, :] / intervals[:, None]
        dev = torch.from_numpy(np.abs(ratio - 1.0)).to(device)
        scores = -transition_lambda * dev
        log_exp = scores - torch.logsumexp(scores, dim=1, keepdim=True)
        adj = (np.abs(intervals[None, :].astype(np.int64)
                      - intervals[:, None].astype(np.int64)) <= 1).astype(np.float32)
        log_dither = torch.from_numpy(np.log(adj / adj.sum(axis=1, keepdims=True))).to(device)
        self.log_kernel = torch.logsumexp(torch.stack([
            float(np.log(mixture_w)) + log_dither,
            float(np.log(1.0 - mixture_w)) + log_exp]), dim=0)          # [V, V] fixed
        self.tied_covariance = tied_covariance
        # emission parameters: 3 diagonal Gaussians over the feature space
        self.mu = torch.zeros(3, feature_dim, device=device, dtype=torch.float64)
        self.log_var = torch.zeros(3, feature_dim, device=device, dtype=torch.float64)

    def _set_variances(self, sqs, sums, weights):
        """Closed-form variance update from accumulated (sum, sum-of-squares, weight) stats;
        per-class, or pooled within-class when tied_covariance."""
        per_class = (sqs / weights[:, None] - self.mu ** 2)
        if self.tied_covariance:
            within = (sqs - weights[:, None] * self.mu ** 2).sum(0) / weights.sum()
            self.log_var = within.clamp(min=VARIANCE_FLOOR).log()[None].expand(3, -1).clone()
        else:
            self.log_var = per_class.clamp(min=VARIANCE_FLOOR).log()

    # --- emission ---------------------------------------------------------------------------
    def log_class_densities(self, features: torch.Tensor) -> torch.Tensor:
        """[T, D] features -> [T, 3] log N(f_t; mu_c, diag var_c)."""
        f = features.to(torch.float64)
        var = self.log_var.exp()
        diff = f[:, None, :] - self.mu[None]                             # [T, 3, D]
        return (-0.5 * (diff ** 2 / var[None]).sum(-1)
                - 0.5 * self.log_var.sum(-1)[None]
                - 0.5 * self.feature_dim * float(np.log(2 * np.pi)))

    def initialize_from_labels(self, features_list, labels_list):
        """Fit the class Gaussians from hard class labels (self-supervised seed)."""
        sums = torch.zeros(3, self.feature_dim, dtype=torch.float64, device=self.device)
        sqs = torch.zeros_like(sums)
        counts = torch.zeros(3, dtype=torch.float64, device=self.device)
        for f, lab in zip(features_list, labels_list):
            f = f.to(torch.float64)
            for c in range(3):
                m = (lab == c)
                if m.any():
                    sums[c] += f[m].sum(0); sqs[c] += (f[m] ** 2).sum(0)
                    counts[c] += int(m.sum())
        counts = counts.clamp(min=1.0)
        self.mu = sums / counts[:, None]
        self._set_variances(sqs, sums, counts)

    @torch.no_grad()
    def bootstrap_labels(self, activation_crops):
        """Hard class labels from the [T, 2] pipeline's own Viterbi decode -- no annotations.

        Depends only on the activations + chassis, NOT on the feature projection, so the harness can
        call it before fitting a label-dependent projection (LDA).
        """
        labels = []
        for acts in activation_crops:
            densities = self.chassis.log_class_densities(acts).to(torch.float32)
            best = None
            for mi in range(len(self.chassis.state_spaces)):
                dp = self.chassis.dynamic_programs[mi]
                path, score = dp.viterbi(self.chassis.log_initial_distributions[mi], self.log_kernel,
                                         densities, state_to_class=self.chassis.state_to_classes[mi],
                                         return_log_score=True)
                if best is None or score > best[0]:
                    best = (score, path, mi)
            _, path, mi = best
            labels.append(self.chassis.state_to_classes[mi][path].long())
        return labels

    def self_supervised_init(self, feature_crops, activation_crops):
        """Seed the class Gaussians from bootstrap_labels. feature_crops and activation_crops are
        aligned crop-for-crop."""
        labels = self.bootstrap_labels(activation_crops)
        self.initialize_from_labels(feature_crops, labels)
        return labels

    def training_state(self) -> dict:
        """Serializable learned emission; the harness adds the projection stats it owns."""
        return {"mu": self.mu.cpu(), "log_var": self.log_var.cpu()}
    # --- exact inference --------------------------------------------------------------------
    def marginal_log_likelihood(self, features: torch.Tensor,
                                log_densities: torch.Tensor = None) -> torch.Tensor:
        dens = (log_densities if log_densities is not None
                else self.log_class_densities(features)).to(torch.float32)
        per_meter = [dp.forward_log_likelihood(init, self.log_kernel, dens, state_to_class=s2c)
                     for dp, init, s2c in zip(self.chassis.dynamic_programs,
                                              self.chassis.log_initial_distributions,
                                              self.chassis.state_to_classes)]
        return torch.logsumexp(torch.stack(per_meter), dim=0)

    def class_posteriors(self, features: torch.Tensor) -> torch.Tensor:
        """E-step per crop: gamma[t, c] = P(class at t = c | features), exact, via autograd."""
        leaf = self.log_class_densities(features).detach().to(torch.float32).requires_grad_(True)
        self.marginal_log_likelihood(features, log_densities=leaf).backward()
        return leaf.grad.detach().to(torch.float64)

    # --- exact EM ---------------------------------------------------------------------------
    def em_step(self, feature_crops) -> float:
        """One exact EM iteration (E: posteriors; M: closed-form weighted Gaussian MLE).
        Returns total marginal log-likelihood at the OLD parameters (for monotonicity logs)."""
        sums = torch.zeros(3, self.feature_dim, dtype=torch.float64, device=self.device)
        sqs = torch.zeros_like(sums)
        weights = torch.zeros(3, dtype=torch.float64, device=self.device)
        total_ll = 0.0
        for f in feature_crops:
            with torch.no_grad():
                total_ll += float(self.marginal_log_likelihood(f))
            g = self.class_posteriors(f)                                  # [T, 3]
            f64 = f.to(torch.float64)
            sums += g.T @ f64
            sqs += g.T @ (f64 ** 2)
            weights += g.sum(0)
        weights = weights.clamp(min=1e-6)
        self.mu = sums / weights[:, None]
        self._set_variances(sqs, sums, weights)
        return total_ll

    # --- decode -----------------------------------------------------------------------------
    @torch.no_grad()
    def _predict_features(self, features) -> dict:
        """Events only. The harness bypasses this and calls decode() with device tensors directly,
        so [num_frames, D] features are never round-tripped through predict()'s float64 coercion."""
        decoded = self.decode(torch.as_tensor(features, dtype=torch.float32, device=self.device))
        return {"beats": decoded["beats"], "downbeats": decoded["downbeats"]}


    @torch.no_grad()
    def decode(self, features: torch.Tensor, deploy: bool = False) -> dict:
        """Bare Viterbi over the fixed mixture kernel + read-out. `deploy` is accepted for interface
        uniformity; the threshold-crop + snap needs the [T, 2] activations and is done by the
        harness."""
        from rungs.bar_pointer.readout import state_path_to_events
        dens = self.log_class_densities(features).to(torch.float32)
        best = None
        for mi in range(len(self.chassis.state_spaces)):
            dp = self.chassis.dynamic_programs[mi]
            path, score = dp.viterbi(self.chassis.log_initial_distributions[mi], self.log_kernel,
                                     dens, state_to_class=self.chassis.state_to_classes[mi],
                                     return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path.cpu().numpy(), self.chassis.state_spaces[mi])
        _, path, space = best
        return {"path": path, "space": space,
                **state_path_to_events(path, space, self.chassis.fps)}
