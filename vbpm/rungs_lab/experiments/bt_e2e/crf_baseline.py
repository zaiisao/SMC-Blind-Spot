"""CRF BASELINE (formerly "R2") -- the Böck 2016 DBN with factors learned DISCRIMINATIVELY.

NOT A RUNG. The objective below is a supervised CRF conditional likelihood p(z|x) on the
annotated path -- a discriminative structured-prediction objective, off-program for a ladder of
GENERATIVE latent-variable models. The rung that learns the same factor on-program is
rungs/r2_em_dbn.py (unsupervised EM on the exact marginal p(x)). This module is kept as the
discriminative comparison point: generative lambda ~40 vs discriminative lambda ~99 vs hand-set
100 is a finding about the model, and it needs both estimators to state.

The ladder rule made literal: this arm changes ONLY how the factors are produced. Deployment is
DBN2016 itself, constructed with the learned transition_lambda (see make_rung()) -- same state
space, same engine, same read-out, so any R1-vs-R2 difference is attributable to the factors.

What is learned here:
  * transition_lambda -- madmom's tempo-change tolerance, the one hand-set scalar of the
    transition model. Differentiable rebuild per step (exponential kernel, log-row-normalized;
    madmom's hard threshold-to-zero is NOT applied during training -- it is non-differentiable --
    and reappears at deployment through DBN2016's standard constructor).
  * (end-to-end) the emission, implicitly: the observation model keeps Böck's parametric form on
    [T, 2] activations, so training the FRONTEND through this loss is what learns the emission.

The objective is the supervised CRF negative log-likelihood of the ANNOTATED bar-pointer path:

    nll = logZ - score(annotated path),   logZ = exact forward through the structured DP

which is differentiable w.r.t. both transition_lambda and the activations (the DP was built for
exactly this -- see rungs/bar_pointer/structured_dp.py, float32 note included).

The annotated path is constructible EXACTLY in the Krebs state space: between consecutive
annotated beat frames f_i -> f_{i+1}, the pointer occupies the tempo block whose interval equals
the actual frame gap (k = f_{i+1} - f_i), advancing +1 per frame through its k states, then takes
the beat-boundary tempo transition. Segments whose gap falls outside the tempo grid, or whose
bar positions are not consecutive mod beats_per_bar, are unrepresentable -> the crop is skipped
(counted by the caller). Crops must start and end on annotated beat frames so path and logZ cover
the same frames.
"""
import numpy as np
import torch
from torch import nn

from rungs.base import Rung
from rungs.r1_2016_dbn import DBN2016


class CRFLearnedFactors(nn.Module, Rung):
    """Training-side owner of the learned factors. Deployment = make_rung() -> a plain DBN2016.

    Inherits Rung only so it is trained and scored through the same path as the real rungs; the
    module docstring explains why it is not one."""

    INPUT_CHANNELS = 2
    TRAIN_MODE = "gradient"
    ARM_NAME = "crf"
    TRAIN_DEFAULTS = dict(epochs=8, learning_rate=0.05)          # frozen frontend: pure CRF
    E2E_DEFAULTS = dict(epochs=30, learning_rate=1e-3,           # joint: CRF + BCE anchor
                        gradient_clip=0.5)

    def __init__(self, fps: float, beats_per_bar=(3, 4), init_transition_lambda: float = 100.0,
                 device: str = "cuda", min_bpm: float = 55.0, max_bpm: float = 215.0,
                 observation_lambda: int = 16):
        super().__init__()
        # R1's chassis, bare and float32 (training regime); predict() is never called on this.
        # observation_lambda MUST match the deployment decode -- training the CRF against a
        # different beat-region width co-adapts the learned factors to the wrong observation
        # world (measured: lambda learned under 16 was decode-optimal under 16, not under 6).
        self.chassis = DBN2016(fps=fps, min_bpm=min_bpm, max_bpm=max_bpm,
                               beats_per_bar=beats_per_bar, num_tempi=None,
                               threshold=0.0, correct=False,
                               observation_lambda=observation_lambda,
                               dtype=torch.float32, device=device)
        Rung.__init__(self, fps=self.chassis.fps, bounding=self.chassis.bounding,
                      eps=self.chassis.eps)
        self.device = device
        self.log_transition_lambda = nn.Parameter(
            torch.log(torch.tensor(float(init_transition_lambda))))
        self._min_interval = int(self.chassis.state_spaces[0].interval_frames[0])
        self._max_interval = int(self.chassis.state_spaces[0].interval_frames[-1])

    @property
    def transition_lambda(self) -> float:
        return float(self.log_transition_lambda.exp())

    def log_tempo_transition(self) -> torch.Tensor:
        """[V, V] log p(tempo_to | tempo_from), differentiable in transition_lambda.

        madmom's kernel is exp(-lambda * |ratio - 1|); we use |log ratio| in the exponent's
        argument only through ratio itself, so replicate exactly: -lambda * |ratio - 1|, then
        log-row-normalize. (No threshold: training needs gradients through every entry.)
        """
        intervals = torch.from_numpy(
            self.chassis.state_spaces[0].interval_frames.astype(np.float32)).to(self.device)
        ratio = intervals[None, :] / intervals[:, None]
        scores = -self.log_transition_lambda.exp() * (ratio - 1.0).abs()
        return scores - torch.logsumexp(scores, dim=1, keepdim=True)

    def log_class_densities(self, activations: torch.Tensor) -> torch.Tensor:
        return self.chassis.log_class_densities(activations)

    def annotated_state_path(self, beat_frames: np.ndarray, beat_in_bar: np.ndarray,
                             beats_per_bar: int):
        return self.chassis.annotated_state_path(beat_frames, beat_in_bar, beats_per_bar)

    def crf_nll(self, activations: torch.Tensor, state_path: np.ndarray,
                meter_index: int) -> torch.Tensor:
        """-log p(annotated path | activations) = logZ - score(path). Differentiable w.r.t.
        activations (the e2e emission) and transition_lambda."""
        chassis, dp = self.chassis, self.chassis.dynamic_programs[meter_index]
        space = chassis.state_spaces[meter_index]
        densities = self.log_class_densities(activations)
        log_transition = self.log_tempo_transition()
        state_to_class = chassis.state_to_classes[meter_index]
        log_init = chassis.log_initial_distributions[meter_index]

        # score of the annotated path: emission at each frame + the tempo kernel at boundaries
        # (the within-beat +1 advance has probability 1 -> contributes 0)
        path = torch.from_numpy(state_path).to(self.device)
        emission_score = densities[
            torch.arange(len(state_path), device=self.device),
            state_to_class[path].long()].sum()
        intervals = space.state_interval_frames[state_path]
        is_boundary = np.where(np.isin(state_path[1:], space.first_states.reshape(-1)))[0]
        from_index = torch.from_numpy(
            intervals[is_boundary] - self._min_interval).long().to(self.device)
        to_index = torch.from_numpy(
            intervals[is_boundary + 1] - self._min_interval).long().to(self.device)
        transition_score = log_transition[from_index, to_index].sum()
        path_score = log_init[path[0]] + emission_score + transition_score

        log_z = dp.forward_log_likelihood(log_init, log_transition, densities,
                                          state_to_class=state_to_class)
        return log_z - path_score

    def make_rung(self, **kwargs) -> DBN2016:
        """Deployment: a plain DBN2016 whose transition_lambda is the LEARNED value."""
        return DBN2016(fps=self.chassis.fps, transition_lambda=self.transition_lambda,
                       beats_per_bar=tuple(self.chassis.beats_per_bar), **kwargs)

    def trainable_parameters(self):
        return [self.log_transition_lambda]

    def training_step(self, activations: torch.Tensor, path, meter_index) -> torch.Tensor:
        """Length-normalized CRF NLL. In e2e the harness adds the BCE calibration anchor, which
        concerns the frontend rather than this factor."""
        return self.crf_nll(activations, path, meter_index) / len(path)

    @torch.no_grad()
    def decode(self, activations, deploy: bool = True) -> dict:
        """Deploy the learned lambda through a plain DBN2016. deploy=True uses the BT-shipped decode
        (threshold 0.2); deploy=False the bare model."""
        obs = self.chassis.observation_lambda
        decode = (dict(observation_lambda=obs, num_tempi=None, threshold=0.2) if deploy
                  else dict(observation_lambda=obs, num_tempi=None, threshold=0.0, correct=False))
        rung = self.make_rung(device=self.device, dtype=torch.float32, bounding="none", **decode)
        return rung.predict(activations)

    @torch.no_grad()
    def _predict_features(self, activations, deploy: bool = True) -> dict:
        return self.decode(activations, deploy=deploy)
