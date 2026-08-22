"""Is the MMI gradient actually reaching Beat This's weights, and at what magnitude?

WHY. In the fold-0 sweep, MMI/frame followed an IDENTICAL trajectory at alpha=0 (pure MMI, no BCE)
and alpha=10 (BCE-dominated) -- 5.8848/5.8909/5.9004 at epoch 1, and so on, agreeing to within 0.5%.
If MMI were driving the optimisation those would diverge. Meanwhile BCE responded to alpha exactly as
expected. That is the signature of an MMI term that is not contributing, and it is indistinguishable
from a plumbing failure by looking at the metrics -- which is precisely the case this project has
been burned by before.

WHAT IS MEASURED, at the SAME parameters (the untrained fold-0 checkpoint), so the two terms are
directly comparable:
  1. ||d L_MMI/L / d theta||  vs  ||d L_BCE / d theta||   -- relative pull on the actual weights.
  2. cosine(g_MMI, g_BCE) -- if they are near-parallel, MMI is merely a rescaled BCE and cannot
     teach anything new; if near-orthogonal it carries independent information; if ANTI-parallel the
     two terms are fighting, which would explain why more BCE anchoring produced better F.
  3. The same for the EMISSION, to localise a failure: if the emission gradient is healthy but the
     parameter gradient is not, the break is between the emission and the network (the log/sigmoid
     chain); if the emission gradient is already tiny, the break is in the DP or the mask.
  4. Per-layer norms, because a gradient that is fine at the head and vanishes at the frontend is a
     different problem from one that is uniformly small.

NOTE ON SCALE. mmi_loss returns the UNNORMALISED -(log Z_num - log Z_den); the trainer divides by L.
Gate 2 measured max |dL/d emission| = 1 for the unnormalised loss, so after the division the
per-entry emission gradient is bounded by 1/L ~ 6.7e-4 at L=1500. Whether that is small RELATIVE to
BCE is the question -- an absolute number means nothing on its own.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, load_gt
from exp_oracle_tempo_beatonly import make_beatonly_dbn
from mmi import annotation_frames, numerator_mask, mmi_loss

SMC_AUD = ROOT / "SMC_MIREX" / "SMC_MIREX_Audio"
SPLIT = ROOT / "third-party" / "beat_this_annotations" / "smc" / "8-folds.split"


def flat(gs):
    return torch.cat([g.reshape(-1) for g in gs if g is not None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--crop", type=int, default=1500)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cuda:3")
    args = ap.parse_args()
    dev = torch.device(args.device)

    splits = {}
    for line in open(SPLIT):
        p = line.split()
        if len(p) >= 2:
            splits[p[0]] = int(p[-1])
    train = sorted(t for t, f in splits.items() if f not in (args.fold, (args.fold + 7) % 8))

    from beat_this.inference import Audio2Frames
    import beat_this.model.loss as btloss
    a2f = Audio2Frames(checkpoint_path=f"fold{args.fold}", device=args.device)
    model = a2f.model
    bce = btloss.ShiftTolerantBCELoss(pos_weight=1).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]

    m = make_beatonly_dbn(30.0, 16, 100, args.device)
    dp0, s2c = m.dynamic_programs[0], m.state_to_classes[0]
    iv = m.state_spaces[0].interval_frames
    logpi, logA = m.log_initial_distributions[0], m.log_tempo_transition
    edtype = logpi.dtype

    rows = []
    model.eval()                                     # dropout off: compare like with like
    for tid in train[:args.n]:
        y, sr = sf.read(str(SMC_AUD / f"{tid.upper()}.wav"))
        sp = a2f.signal2spect(y, sr)[:args.crop].to(dev)
        L = sp.shape[0]
        gt = np.asarray(load_gt(tid), float)
        fr = annotation_frames(gt, FPS, 0, L)
        if len(fr) < 8:
            continue
        tgt = torch.zeros(L, device=dev)
        tgt[torch.from_numpy(fr).to(dev)] = 1.0
        msk = torch.ones(1, L, dtype=torch.bool, device=dev)

        def terms():
            logits = model(sp.unsqueeze(0))["beat"][0]
            prob = torch.sigmoid(logits).to(edtype)
            em = m.log_class_densities(torch.stack([prob, torch.zeros_like(prob)], 1))
            mk, _, _ = numerator_mask(fr, L, iv, em.device, em.dtype)
            lm, _, _ = mmi_loss(dp0, logpi, logA, em, s2c, mk)
            return lm / L, bce(logits[None], tgt[None], msk), em

        lm, lb, _ = terms()
        gm = flat(torch.autograd.grad(lm, params, retain_graph=False, allow_unused=True))
        lm2, lb2, _ = terms()
        gb = flat(torch.autograd.grad(lb2, params, retain_graph=False, allow_unused=True))

        # emission-level gradients, to localise any break
        logits = model(sp.unsqueeze(0))["beat"][0]
        prob = torch.sigmoid(logits).to(edtype)
        em = m.log_class_densities(torch.stack([prob, torch.zeros_like(prob)], 1))
        em.retain_grad()
        mk, _, _ = numerator_mask(fr, L, iv, em.device, em.dtype)
        (mmi_loss(dp0, logpi, logA, em, s2c, mk)[0] / L).backward()
        gem = em.grad.abs()

        rows.append({
            "tid": tid, "mmi": float(lm), "bce": float(lb),
            "gm": float(gm.norm()), "gb": float(gb.norm()),
            "cos": float(torch.nn.functional.cosine_similarity(gm[None].float(),
                                                               gb[None].float())),
            "gem_max": float(gem.max()), "gem_mean": float(gem.mean()),
            "gem_nonzero": float((gem > 0).float().mean()),
        })
        model.zero_grad(set_to_none=True)
        print(f"  {tid}  ||g_MMI|| {rows[-1]['gm']:.3e}  ||g_BCE|| {rows[-1]['gb']:.3e}  "
              f"ratio {rows[-1]['gm']/max(rows[-1]['gb'],1e-30):.3f}  cos {rows[-1]['cos']:+.3f}",
              flush=True)

    A = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"\nn={len(rows)} tracks, fold-{args.fold} checkpoint, eval mode, crop {args.crop}")
    print(f"  ||d L_MMI/L  / d theta||  : {A('gm'):.4e}")
    print(f"  ||d L_BCE    / d theta||  : {A('gb'):.4e}")
    print(f"  ratio MMI/BCE             : {A('gm')/max(A('gb'),1e-30):.4f}")
    print(f"  cosine(g_MMI, g_BCE)      : {A('cos'):+.4f}")
    print(f"  |dL_MMI/d emission| max   : {A('gem_max'):.4e}   mean {A('gem_mean'):.4e}")
    print(f"  emission entries with nonzero grad : {A('gem_nonzero'):.2%}")
    print("\nreading:")
    print("  ratio << 1        -> MMI is drowned out; it needs its own scale, not a bigger alpha")
    print("  cos ~ +1          -> MMI is a rescaled BCE and can teach nothing new")
    print("  cos ~ -1          -> the two terms fight (would explain F recovering as alpha rises)")
    print("  gem tiny/zero     -> the break is in the DP or mask, not the network chain")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
