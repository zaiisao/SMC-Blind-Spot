"""Build/resume/freeze the Beat Transformer, extract [T, 2] activations or [T, 256] penultimate
features from the demix cache, and fit the r4_5 feature projection."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "beat_transformer" / "code"))

MODEL_KWARGS = dict(attn_len=5, instr=5, ntoken=2, dmodel=256, nhead=8,
                    d_hid=1024, nlayers=9, norm_first=True)
BT_SHIPPED_DECODE = dict(observation_lambda=6, num_tempi=None, threshold=0.2)


def build_model(device, init_from=None):
    """A fresh (or resumed) Demixed_DilatedTransformerModel in train mode (e2e)."""
    from DilatedTransformer import Demixed_DilatedTransformerModel
    model = Demixed_DilatedTransformerModel(**MODEL_KWARGS).to(device).train()
    if init_from:
        model.load_state_dict(torch.load(init_from, map_location="cpu")["model"])
    return model


def load_frozen_model(ckpt, device):
    """The frozen frontend: a trained checkpoint in eval mode (parameters never optimized)."""
    from DilatedTransformer import Demixed_DilatedTransformerModel
    model = Demixed_DilatedTransformerModel(**MODEL_KWARGS).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    return model


@torch.no_grad()
def activations_for(model, entries, device):
    """stem -> [T, 2] sigmoid activations (float64 numpy), whole song, one forward each."""
    acts = {}
    for e in entries:
        x = np.load(e["mel_path"])["x"]
        pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
        acts[e["stem"]] = torch.sigmoid(pred[0, :, :2]).double().cpu().numpy()
    return acts


@torch.no_grad()
def features_for(model, entries, device):
    """(feats, acts): penultimate [T, 256] (forward hook on out_linear) + [T, 2] activations."""
    feats, acts, grabbed = {}, {}, {}
    handle = model.out_linear.register_forward_hook(
        lambda module, inp, out: grabbed.__setitem__("f", inp[0].detach()))
    for e in entries:
        x = np.load(e["mel_path"])["x"]
        pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(device))
        feats[e["stem"]] = grabbed["f"][0].float().cpu()
        acts[e["stem"]] = torch.sigmoid(pred[0, :, :2]).double().cpu().numpy()
    handle.remove()
    return feats, acts


def fit_projection(training_features, kind, output_dim, device, bootstrap_labels=None):
    """Fit the affine map conditioning the rich features -> (project, checkpoint_payload, dim):
      standardize : per-dim (x - mean) / std
      pca         : whiten onto the top-`output_dim` PCA axes
      lda         : top-2 LDA directions + PCA of the residual (needs bootstrap_labels)
    The payload is saved so deployment can reproduce `project`."""
    stacked_features = torch.cat(training_features).to(torch.float64)
    feature_mean = stacked_features.mean(0)

    if kind == "standardize":
        feature_std = stacked_features.std(0).clamp(min=1e-3)
        feature_mean32, feature_std32 = feature_mean.to(torch.float32), feature_std.to(torch.float32)
        project = lambda features: ((features - feature_mean32) / feature_std32).to(device)
        return project, {"feature_mean": feature_mean32, "feature_std": feature_std32}, \
            stacked_features.shape[1]

    if kind == "pca":
        covariance = torch.cov((stacked_features - feature_mean).T)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        top = torch.argsort(eigenvalues, descending=True)[:output_dim]
        # whitening: unit-variance along each retained principal axis
        whitening_matrix = (eigenvectors[:, top]
                            / eigenvalues[top].clamp(min=1e-6).sqrt()[None]).to(torch.float32)
        feature_mean32 = feature_mean.to(torch.float32)
        project = lambda features: ((features - feature_mean32) @ whitening_matrix).to(device)
        return project, {"feature_mean": feature_mean32, "projection": whitening_matrix}, output_dim

    if kind == "lda":
        if bootstrap_labels is None:
            raise ValueError("lda projection needs bootstrap labels")
        labels = torch.cat(bootstrap_labels).cpu()
        centered_features = stacked_features - feature_mean
        num_dims = stacked_features.shape[1]
        within_class_scatter = torch.zeros(num_dims, num_dims, dtype=torch.float64)
        between_class_scatter = torch.zeros_like(within_class_scatter)
        for class_index in range(3):
            in_class = (labels == class_index)
            if not in_class.any():
                continue
            class_features = centered_features[in_class]
            class_mean = class_features.mean(0)
            within_class_scatter += (class_features - class_mean).T @ (class_features - class_mean)
            between_class_scatter += int(in_class.sum()) * torch.outer(class_mean, class_mean)
        regularizer = (1e-4 * torch.eye(num_dims, dtype=torch.float64)
                       * within_class_scatter.diagonal().mean())
        eigenvalues, eigenvectors = torch.linalg.eig(
            torch.linalg.solve(within_class_scatter + regularizer, between_class_scatter))
        lda_directions = eigenvectors.real[:, torch.argsort(eigenvalues.real, descending=True)[:2]]
        lda_directions, _ = torch.linalg.qr(lda_directions)              # orthonormal [256, 2]
        residual = centered_features - (centered_features @ lda_directions) @ lda_directions.T
        residual_covariance = torch.cov(residual.T)
        residual_eigenvalues, residual_eigenvectors = torch.linalg.eigh(residual_covariance)
        residual_directions = residual_eigenvectors[
            :, torch.argsort(residual_eigenvalues, descending=True)[:output_dim - 2]]
        directions = torch.cat([lda_directions, residual_directions], dim=1)  # [256, output_dim]
        scale = (stacked_features @ directions).std(0).clamp(min=1e-6)
        projection_matrix = (directions / scale).to(torch.float32)
        feature_mean32 = feature_mean.to(torch.float32)
        project = lambda features: ((features - feature_mean32) @ projection_matrix).to(device)
        return project, {"feature_mean": feature_mean32, "projection": projection_matrix}, output_dim

    raise ValueError(f"unknown projection {kind!r}")
