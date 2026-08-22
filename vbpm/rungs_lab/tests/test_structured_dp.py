"""Certify the structured bar-pointer DP against the dense DP on a small instance.

The dense DP (rungs/bar_pointer/inference) is already certified against hmmlearn AND torch-struct. Here we build
the SAME Krebs transition two ways -- structured (gather + [num_tempi, num_tempi] boundary mix) and
dense [num_states, num_states] -- and check the forward LL and the MAP path agree. If they do, the
structured version inherits the whole certificate chain, and we can use it at real scale where dense
is impossible.

Small instance: fps=20 -> intervals 6..22 -> num_states = 4 * 238 = 952, dense = 3.6 MB (affordable).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab")
from rungs.bar_pointer.inference import forward_log_likelihood, viterbi
from rungs.bar_pointer.state_space import BarPointerStateSpace
from rungs.bar_pointer.structured_dp import StructuredBarPointerDP


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_space = BarPointerStateSpace(fps=20.0, min_bpm=55.0, max_bpm=215.0, beats_per_bar=4)
    dynamic_program = StructuredBarPointerDP(state_space, device=device)
    print(state_space)
    print(f"dense here = {state_space.num_states ** 2 * 4 / 1e6:.1f} MB (fine); at real fps it would "
          f"be ~1 GB and 260M ops/frame")

    log_tempo_transition = dynamic_program.build_log_tempo_transition(transition_lambda=100.0)
    num_forbidden = int(torch.isinf(log_tempo_transition).sum())
    print(f"tempo transition: {num_forbidden}/{log_tempo_transition.numel()} entries are -inf "
          f"(madmom thresholds improbable jumps to exactly 0 = FORBIDDEN, not merely expensive)")
    assert num_forbidden > 0, "the thresholded-to-zero jumps must be -inf, not a finite penalty"

    log_initial_distribution = torch.full((state_space.num_states,),
                                          -float(np.log(state_space.num_states)),
                                          dtype=torch.float64, device=device)
    num_frames = 120
    log_emission = torch.log_softmax(
        torch.randn(num_frames, state_space.num_states, dtype=torch.float64, device=device), dim=1)

    # structured
    structured_log_likelihood = dynamic_program.forward_log_likelihood(
        log_initial_distribution, log_tempo_transition, log_emission).item()
    structured_path = dynamic_program.viterbi(log_initial_distribution, log_tempo_transition,
                                              log_emission).cpu().numpy()

    # dense: the SAME transition, written out in full, run through the certified dense DP
    dense_log_transition = dynamic_program.dense_transition(log_tempo_transition)
    dense_log_likelihood = forward_log_likelihood(log_initial_distribution, dense_log_transition,
                                                  log_emission).item()
    dense_path = viterbi(log_initial_distribution, dense_log_transition, log_emission).cpu().numpy()

    print(f"forward LL : structured {structured_log_likelihood:.6f} | dense {dense_log_likelihood:.6f}"
          f" | |gap| {abs(structured_log_likelihood - dense_log_likelihood):.2e}")
    print(f"viterbi    : {float((structured_path == dense_path).mean()) * 100:.1f}% path match")
    assert abs(structured_log_likelihood - dense_log_likelihood) < 1e-3, \
        "structured forward disagrees with dense"
    assert (structured_path == dense_path).all(), "structured viterbi disagrees with dense"

    # compact class-factored emission == pre-gathered per-state emission. The [num_frames, C] table
    # plus a state->class map must give the identical LL and path as the materialized
    # [num_frames, num_states] array it replaces (5,585x smaller at real scale).
    num_classes = 5
    state_to_class = torch.randint(0, num_classes, (state_space.num_states,), device=device)
    log_class_densities = torch.log_softmax(
        torch.randn(num_frames, num_classes, dtype=torch.float64, device=device), dim=1)
    pre_gathered = log_class_densities[:, state_to_class]
    compact_log_likelihood = dynamic_program.forward_log_likelihood(
        log_initial_distribution, log_tempo_transition, log_class_densities,
        state_to_class=state_to_class).item()
    gathered_log_likelihood = dynamic_program.forward_log_likelihood(
        log_initial_distribution, log_tempo_transition, pre_gathered).item()
    compact_path = dynamic_program.viterbi(log_initial_distribution, log_tempo_transition,
                                           log_class_densities,
                                           state_to_class=state_to_class).cpu().numpy()
    gathered_path = dynamic_program.viterbi(log_initial_distribution, log_tempo_transition,
                                            pre_gathered).cpu().numpy()
    print(f"compact emission : LL |gap| {abs(compact_log_likelihood - gathered_log_likelihood):.2e} "
          f"| path {float((compact_path == gathered_path).mean()) * 100:.1f}% match vs pre-gathered")
    assert abs(compact_log_likelihood - gathered_log_likelihood) < 1e-9
    assert (compact_path == gathered_path).all()

    # ...and gradients reach the compact class table (R2+ train through this form)
    class_logits = log_class_densities.clone().requires_grad_(True)
    dynamic_program.forward_log_likelihood(log_initial_distribution, log_tempo_transition,
                                           class_logits, state_to_class=state_to_class).backward()
    assert torch.isfinite(class_logits.grad).all() and class_logits.grad.abs().sum() > 0
    print(f"grad to compact class table: {class_logits.grad.abs().sum().item():.3e}")

    # the advance must be exactly +1 state per frame inside a block (the Krebs property)
    path_positions = state_space.state_positions[structured_path]
    print(f"path bar position spans {path_positions.min():.2f}..{path_positions.max():.2f} "
          f"beat units (should sweep 0..{state_space.beats_per_bar})")

    # gradients flow through the structured forward (the R2+ objective). R2+ parameterize the tempo
    # transition as log_softmax(logits), which never produces an exact zero, so the -inf branch does
    # not arise there.
    tempo_transition_logits = torch.zeros(state_space.num_tempi, state_space.num_tempi,
                                          dtype=torch.float64, device=device, requires_grad=True)
    dynamic_program.forward_log_likelihood(
        log_initial_distribution, torch.log_softmax(tempo_transition_logits, dim=1),
        log_emission).backward()
    print(f"grad to tempo transition: {tempo_transition_logits.grad.abs().sum().item():.3e}")
    assert tempo_transition_logits.grad.abs().sum() > 0
    assert torch.isfinite(tempo_transition_logits.grad).all()

    # ...but R2 may well want to LEARN on top of madmom's thresholded transition, which does carry
    # -inf. Check that -inf entries give finite gradients (they should: logsumexp weights them 0).
    emission_logits = torch.zeros(num_frames, state_space.num_states, dtype=torch.float64,
                                  device=device, requires_grad=True)
    dynamic_program.forward_log_likelihood(
        log_initial_distribution, log_tempo_transition,
        log_emission + emission_logits).backward()
    grad = emission_logits.grad
    print(f"grad with -inf in the transition: finite={bool(torch.isfinite(grad).all())} "
          f"| sum={grad.abs().sum().item():.3e}  (-inf must not poison the backward pass)")
    assert torch.isfinite(grad).all(), "-inf transitions produced non-finite gradients"
    assert grad.abs().sum() > 0

    print("STRUCTURED_DP_CERTIFIED (vs dense, which is certified vs hmmlearn + torch-struct)")


if __name__ == "__main__":
    main()
