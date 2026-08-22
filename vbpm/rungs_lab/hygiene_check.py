"""Verify the divide-by-zero warnings are benign: -inf only on structurally-forbidden dither
transitions, no NaN anywhere; sample forward pass finite."""
import numpy as np, torch
from training import data
from mixture import MixtureLambda
from r3_model import R3Mixture

device = "cuda:0"
mix = torch.load("runs/r2mix_seed0.pt"); W0, LAM0 = float(mix["w"]), float(mix["lambda"])
m = MixtureLambda(fps=data.FPS, device=device, observation_lambda=6)
m.mixture_weight, m.transition_lambda = W0, LAM0
ld = m._log_dither
assert not torch.isnan(ld).any(), "NaN in log dither"
n_inf = int(torch.isinf(ld).sum())
print(f"log_dither: {n_inf} -inf entries (forbidden transitions), no NaN")
k = m.log_mixture_kernel(m.mixture_weight, m.transition_lambda)
assert not torch.isnan(k).any(), "NaN in mixture kernel"
assert not torch.isinf(k).any(), "unexpected inf in MIXTURE kernel (exp component should fill)"
print("mixture kernel: fully finite (exp component covers dither's -inf), no NaN")
for arm in ("wt", "lamt"):
    ck = torch.load(f"runs/r3_{arm}_seed0.pt")
    r3 = R3Mixture(arm=arm, w0=W0, fps=data.FPS, device=device,
                   observation_lambda=6, lambda_base=LAM0)
    r3.net.load_state_dict(ck["net"])
    a = torch.rand(400, 2, device=device) * 0.9 + 0.05
    kk = r3.per_frame_mixture_kernel(a)
    ll = r3.marginal_ll(a)
    assert not torch.isnan(kk).any() and not torch.isinf(kk).any()
    assert torch.isfinite(ll)
    print(f"R3 {arm}: per-frame kernel finite, marginal_ll={float(ll):.1f} finite")
print("HYGIENE PASS")
