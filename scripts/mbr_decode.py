"""Posterior (MBR) decoding for the R1 bar-pointer DBN.

`structured_dp.py` ships a forward recursion (`forward_log_likelihood`) and Viterbi MAP
(`viterbi`) but NO backward pass, so posterior state marginals do not exist. This module adds
them, without modifying anything under vbpm/.

WHY. Viterbi returns argmax_path p(path | y) -- the Bayes rule under 0-1 loss on the WHOLE path.
F-measure, CMLt and AMLt are additive over beats, so MAP is provably not their Bayes rule. The
minimum-Bayes-risk read-out for an additive metric uses per-frame posterior marginals
gamma_t(s) = p(z_t = s | y), which is what the backward pass provides.

TRANSITION STRUCTURE (mirrored exactly from StructuredBarPointerDP._advance_one_frame):
  * a state that is NOT the last state of a beat advances deterministically to state + 1;
  * a last state of beat b-1 at tempo `from` connects to the FIRST state of beat b at every tempo
    `to`, weighted by log_tempo_transition[from, to].
The backward recursion is the transpose of exactly that.

CORRECTNESS GATES (both must pass, or the marginals are meaningless):
  G1  logsumexp_s( alpha_t(s) + beta_t(s) ) == logZ for EVERY frame t, not just the last.
  G2  logZ recovered from the backward recursion == forward_log_likelihood.
"""
import numpy as np
import torch


class PosteriorDecoder:
    """Forward-backward on top of a StructuredBarPointerDP + BarPointerStateSpace."""

    def __init__(self, dp, state_space, device="cuda", dtype=torch.float64):
        self.dp, self.space, self.device, self.dtype = dp, state_space, device, dtype
        self.V = int(state_space.num_tempi)
        self.B = int(state_space.first_states.shape[0])
        self.N = int(state_space.num_states)
        self.first = torch.from_numpy(state_space.first_states).to(device)        # [B, V]
        # last states of the PREVIOUS beat, aligned with first[b] -- same alignment the forward uses
        self.last_prev = torch.from_numpy(
            state_space.last_states[np.arange(self.B) - 1]).to(device)            # [B, V]
        is_last = torch.zeros(self.N, dtype=torch.bool, device=device)
        is_last[torch.from_numpy(state_space.last_states.reshape(-1)).to(device)] = True
        self.is_last = is_last

    def _emit(self, log_emission, frame, state_to_class):
        row = log_emission[frame]
        return row if state_to_class is None else row[state_to_class]

    def forward_alphas(self, log_init, K, log_emission, state_to_class=None):
        T = log_emission.shape[0]
        alphas = torch.empty((T, self.N), dtype=self.dtype, device=self.device)
        a = log_init + self._emit(log_emission, 0, state_to_class)
        alphas[0] = a
        for t in range(1, T):
            a, _ = self.dp._advance_one_frame(
                a, self.dp._kernel_at(K, t),
                lambda s: (torch.logsumexp(s, dim=1), None))
            a = a + self._emit(log_emission, t, state_to_class)
            alphas[t] = a
        return alphas

    def backward_betas(self, K, log_emission, state_to_class=None):
        """beta_t(s) = log p(y_{t+1..T} | z_t = s).  beta_T = 0."""
        T = log_emission.shape[0]
        betas = torch.zeros((T, self.N), dtype=self.dtype, device=self.device)
        for t in range(T - 2, -1, -1):
            msg = self._emit(log_emission, t + 1, state_to_class) + betas[t + 1]   # [N]
            # deterministic advance: successor of s is s+1 (valid wherever s is not a last state)
            b = torch.full((self.N,), -float("inf"), dtype=self.dtype, device=self.device)
            b[:-1] = msg[1:]
            # boundary: last_prev[b, from] -> first[b, to], weight K[from, to]
            Kt = self.dp._kernel_at(K, t + 1)                                      # [V, V]
            tgt = msg[self.first]                                                  # [B, V] over `to`
            mixed = torch.logsumexp(Kt.unsqueeze(0) + tgt.unsqueeze(1), dim=2)     # [B, V] over `from`
            b[self.last_prev.reshape(-1)] = mixed.reshape(-1)
            betas[t] = b
        return betas

    def posterior(self, log_init, K, log_emission, state_to_class=None, atol=1e-6):
        alphas = self.forward_alphas(log_init, K, log_emission, state_to_class)
        betas = self.backward_betas(K, log_emission, state_to_class)
        logZ_fwd = torch.logsumexp(alphas[-1], dim=0)
        per_frame = torch.logsumexp(alphas + betas, dim=1)                         # [T]
        g1 = float((per_frame - logZ_fwd).abs().max())
        logZ_bwd = float(torch.logsumexp(
            log_init + self._emit(log_emission, 0, state_to_class) + betas[0], dim=0))
        g2 = abs(logZ_bwd - float(logZ_fwd))
        gamma = alphas + betas - logZ_fwd                                          # log p(z_t=s | y)
        return gamma, {"G1_max_dev_nats": g1, "G2_logZ_dev_nats": g2,
                       "logZ": float(logZ_fwd), "pass": bool(g1 < 1e-4 and g2 < 1e-4)}

    def beat_posterior(self, gamma):
        """p(a beat ONSET occurs at frame t | y) = posterior mass on first-states of any beat."""
        return torch.logsumexp(gamma[:, self.first.reshape(-1)], dim=1).exp()

    def downbeat_posterior(self, gamma):
        return torch.logsumexp(gamma[:, self.first[0]], dim=1).exp()


def mbr_beats(prob, fps, thresh=0.5):
    """Bayes read-out: ONE beat per contiguous run where p(beat onset) > thresh.

    The earlier version took every local maximum with p > 0.0, which emitted 112-290 beats
    against references of 32-44 -- a read-out bug, not an inference one (the forward-backward
    gates pass at 1e-11). A posterior is nonzero almost everywhere, so "local maximum" is not a
    decision rule; thresholding at 0.5 is the Bayes decision under a symmetric per-frame loss,
    and collapsing each above-threshold RUN to its argmax gives one beat per detected onset.
    """
    p = np.asarray(prob, dtype=float)
    if p.size == 0:
        return np.array([])
    above = p > thresh
    if not above.any():
        return np.array([])
    edges = np.flatnonzero(np.diff(above.astype(np.int8)))
    starts = np.r_[0 if above[0] else [], edges[above[edges + 1]] + 1].astype(int)
    ends = np.r_[edges[~above[edges + 1]] + 1, p.size if above[-1] else []].astype(int)
    if len(ends) and len(starts) and ends[0] <= starts[0]:
        ends = ends[1:]
    n = min(len(starts), len(ends))
    idx = [s + int(np.argmax(p[s:e])) for s, e in zip(starts[:n], ends[:n]) if e > s]
    return np.asarray(sorted(idx), dtype=float) / float(fps)
