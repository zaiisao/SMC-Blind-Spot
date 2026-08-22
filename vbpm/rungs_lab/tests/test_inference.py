"""Certify rungs/bar_pointer/inference against TWO independent implementations.

  hmmlearn     -- an independent forward algorithm + Viterbi (numpy/Cython, sklearn-style)
  torch-struct -- an independent differentiable linear-chain DP (harvardnlp)

If our partition and MAP path match both to ~1e-14, the DP is correct -- the prerequisite for trusting
every learned rung. We use these libraries as ORACLES, not as the engine: torch-struct materializes
the full [num_frames - 1, num_states, num_states] potentials (31.85 GB at our real num_frames=6000,
num_states=1152), so it cannot run the actual model -- but at unit-test scale it is a perfectly good
second opinion.

Also checks the time-varying [num_frames - 1, num_states, num_states] transition that R3+ depend on.
"""
import sys

import numpy as np
import torch
from hmmlearn.hmm import CategoricalHMM
from torch_struct import LinearChainCRF

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab")
from rungs.bar_pointer.inference import forward_log_likelihood, viterbi


def torch_struct_reference(log_initial_distribution, log_transition, log_emission):
    """Same HMM, scored by torch-struct's DP. Folds our factors into its
    [1, num_frames - 1, num_states, num_states] potentials: it scores a chain as
    sum_n potentials[n, z_{n+1}, z_n]."""
    num_frames, num_states = log_emission.shape
    transition_per_frame = (log_transition.unsqueeze(0).expand(num_frames - 1, num_states, num_states)
                            if log_transition.dim() == 2 else log_transition)
    potentials = transition_per_frame.transpose(1, 2) + log_emission[1:].unsqueeze(2)  # [n, next, prev]
    first_potential = (potentials[0]
                       + (log_initial_distribution + log_emission[0]).unsqueeze(0)).unsqueeze(0)
    potentials = torch.cat([first_potential, potentials[1:]], dim=0).unsqueeze(0)
    crf = LinearChainCRF(potentials)
    arcs = crf.argmax[0].reshape(num_frames - 1, -1).argmax(dim=1)
    state_path = torch.cat([(arcs % num_states)[:1], arcs // num_states])
    return crf.partition.reshape(()).item(), state_path.numpy()


def main():
    rng = np.random.default_rng(0)
    num_states, num_symbols, num_frames = 6, 4, 80
    start_probabilities = rng.dirichlet(np.ones(num_states))
    transition_probabilities = rng.dirichlet(np.ones(num_states), size=num_states)
    emission_probabilities = rng.dirichlet(np.ones(num_symbols), size=num_states)

    reference_hmm = CategoricalHMM(n_components=num_states, init_params="", params="")
    reference_hmm.startprob_ = start_probabilities
    reference_hmm.transmat_ = transition_probabilities
    reference_hmm.emissionprob_ = emission_probabilities
    observations, _ = reference_hmm.sample(num_frames, random_state=1)
    observations = observations.flatten()

    log_initial_distribution = torch.log(torch.tensor(start_probabilities, dtype=torch.float64))
    log_transition = torch.log(torch.tensor(transition_probabilities, dtype=torch.float64))
    log_emission = torch.log(torch.tensor(emission_probabilities[:, observations].T,
                                          dtype=torch.float64))

    our_log_likelihood = forward_log_likelihood(log_initial_distribution, log_transition,
                                                log_emission).item()
    our_path = viterbi(log_initial_distribution, log_transition, log_emission).numpy()

    # --- oracle 1: hmmlearn ------------------------------------------------------------------------
    hmmlearn_log_likelihood = reference_hmm.score(observations.reshape(-1, 1))
    hmmlearn_path = reference_hmm.decode(observations.reshape(-1, 1))[1]
    print(f"vs hmmlearn     : LL |gap| {abs(our_log_likelihood - hmmlearn_log_likelihood):.2e} | "
          f"path {float((our_path == hmmlearn_path).mean()) * 100:.1f}% match")
    assert abs(our_log_likelihood - hmmlearn_log_likelihood) < 1e-6
    assert (our_path == hmmlearn_path).all()

    # --- oracle 2: torch-struct --------------------------------------------------------------------
    torch_struct_log_likelihood, torch_struct_path = torch_struct_reference(
        log_initial_distribution, log_transition, log_emission)
    print(f"vs torch-struct : LL |gap| {abs(our_log_likelihood - torch_struct_log_likelihood):.2e} | "
          f"path {float((our_path == torch_struct_path).mean()) * 100:.1f}% match")
    assert abs(our_log_likelihood - torch_struct_log_likelihood) < 1e-6
    assert (our_path == torch_struct_path).all()

    # --- time-varying [num_frames - 1, num_states, num_states] path (what R3+ use) -----------------
    log_transition_time_varying = log_transition.unsqueeze(0).repeat(num_frames - 1, 1, 1)
    time_varying_log_likelihood = forward_log_likelihood(
        log_initial_distribution, log_transition_time_varying, log_emission).item()
    time_varying_path = viterbi(log_initial_distribution, log_transition_time_varying,
                                log_emission).numpy()
    print(f"time-varying    : LL gap vs 2-D "
          f"{abs(time_varying_log_likelihood - our_log_likelihood):.2e} | "
          f"path identical {np.array_equal(time_varying_path, our_path)}")
    assert abs(time_varying_log_likelihood - our_log_likelihood) < 1e-9
    assert np.array_equal(time_varying_path, our_path)

    # --- gradients flow to the factors (the R2+ training objective) -------------------------------
    transition_logits = torch.zeros(num_states, num_states, dtype=torch.float64, requires_grad=True)
    forward_log_likelihood(log_initial_distribution, torch.log_softmax(transition_logits, dim=1),
                           log_emission).backward()
    print(f"grad to transition: {transition_logits.grad.abs().sum().item():.3e}")
    assert transition_logits.grad.abs().sum() > 0

    print("INFERENCE_DP_CERTIFIED (ours; oracles = hmmlearn + torch-struct)")


if __name__ == "__main__":
    main()
