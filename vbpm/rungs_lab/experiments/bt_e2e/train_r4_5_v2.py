"""R4.5-v2 (LDA control): projection = 2 LDA discriminant directions (from the init labels'
class statistics -- no supervision beyond what the pipeline already produces) + 14 PCA dims
orthogonal to them, whitened. Controls the v1 confound: unsupervised PCA optimizes variance,
not class separation, and may have discarded downbeat-discriminative directions (v1 signature:
CMLt -0.035 with AMLt fine, dbF -0.023). If v2 recovers dbF, the information was there and the
projection was the bottleneck; if not, the rich-features-don't-help reading strengthens.

Original v1 header follows.
"""
"""R4.5-v1: exact-EM Gaussian emission on a WHITENED 16-dim PCA subspace of the rich features,
with TIED covariance -- the two verified fixes for v0's emission domination (22,600-nat gaps,
uniform-kernel tie, downbeat +660-nat log-det subsidy).

Adds a VALIDITY GATE before any score is interpreted: decode 30 val songs with the real kernel
vs a UNIFORM tempo kernel -- if they still tie, the transition is still inert and the F numbers
say nothing about the bar-pointer. Pre-registered predictions: emission gaps drop ~15x, gate
OPENS (real != uniform), then and only then the rich-vs-Bock emission question is actually posed.

Pipeline:
  1. Extract penultimate features (input of model.out_linear) via forward hook, standardized
     by global train-set mean/std (conditioning only -- an invertible affine map).
  2. Self-supervised init: class labels from the EXISTING pipeline's Viterbi decode (mixture
     kernel on Böck [T,2] densities) -- no annotations touch training at any point.
  3. Exact Baum-Welch (10 iters), logging the total marginal (must be monotone).
  4. Deployed + bare decode on the val fold, six metrics, vs the R1/R2 parity rows.

Pre-registered target (tiebreak table): downbeat CMLt 0.884. Pre-registered risks: emission
domination (256-dim log-densities dwarf transition scores), class collapse, label swap.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import final_eval; final_eval.DEVICE = "cuda:2"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE, load_songs
from rungs.r4_5_rich_emission import R45RichEmission
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events
from mixture_kernel_probe import MixtureLambda
from crf_baseline import CRFLearnedFactors

DEVICE = "cuda:2"
PCA_DIM = 16
NUM_CROPS, CROP_FRAMES, EM_ITERS = 300, 700, 10
W_MIX, LAMBDA_MIX = 0.370, 93.1


def features_for(model, entries):
    """Penultimate [T, 256] via hook on out_linear (plus the [T,2] activations)."""
    feats, acts = {}, {}
    grabbed = {}
    handle = model.out_linear.register_forward_hook(
        lambda module, inp, out: grabbed.__setitem__("f", inp[0].detach()))
    with torch.no_grad():
        for e in entries:
            x = np.load(e["mel_path"])["x"]
            pred, _ = model(torch.from_numpy(x).unsqueeze(0).to(DEVICE))
            feats[e["stem"]] = grabbed["f"][0].float().cpu()
            acts[e["stem"]] = torch.sigmoid(pred[0, :, :2]).double().cpu().numpy()
    handle.remove()
    return feats, acts


def main():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    mix = MixtureLambda(fps=FPS, device=DEVICE,
                        observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    probe = CRFLearnedFactors(fps=FPS, device=DEVICE,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    train, val, _ = load_songs(probe)
    model = final_eval.load_model(Path(__file__).resolve().parent / "vanilla_best_prelim.pt")
    print("extracting rich features (train crops + val)...", flush=True)
    train_feats, train_acts = features_for(model, train[:NUM_CROPS])
    val_feats, val_acts = features_for(model, val)

    # v2 projection: 2 LDA discriminant directions + (PCA_DIM-2) PCA dims orthogonal to them.
    # LDA stats come from init labels computed on the [T,2] activations (label pass moved up).
    print("prepass: init labels for LDA (Viterbi on [T,2] system)...", flush=True)
    mix_kernel_pre = mix.log_mixture_kernel(W_MIX, LAMBDA_MIX)
    lda_feats, lda_labels = [], []
    for e in train[:50]:
        a = train_acts[e["stem"]]
        acts_t = torch.from_numpy(np.ascontiguousarray(a)).float().to(DEVICE)
        dens = mix.log_class_densities(acts_t)
        best = None
        for mi in range(len(mix.chassis.state_spaces)):
            dp = mix.chassis.dynamic_programs[mi]
            path, sc = dp.viterbi(mix.chassis.log_initial_distributions[mi], mix_kernel_pre,
                                  dens, state_to_class=mix.chassis.state_to_classes[mi],
                                  return_log_score=True)
            if best is None or sc > best[0]:
                best = (sc, path, mi)
        _, path, mi = best
        lda_feats.append(train_feats[e["stem"]].to(torch.float64))
        lda_labels.append(mix.chassis.state_to_classes[mi][path].long().cpu())
    F_all = torch.cat(lda_feats); L_all = torch.cat(lda_labels)
    f_mean = F_all.mean(0)
    Fc = F_all - f_mean
    Sw = torch.zeros(256, 256, dtype=torch.float64)
    means = []
    for c in range(3):
        m = (L_all == c)
        mc = Fc[m].mean(0); means.append((mc, float(m.sum())))
        d = Fc[m] - mc
        Sw += d.T @ d
    Sw /= len(Fc)
    Sb = torch.zeros_like(Sw)
    for mc, n in means:
        Sb += (n / len(Fc)) * torch.outer(mc, mc)
    reg = 1e-4 * torch.eye(256, dtype=torch.float64) * Sw.diagonal().mean()
    evals_l, evecs_l = torch.linalg.eigh(torch.linalg.solve(Sw + reg, Sb))
    lda_dirs = evecs_l[:, torch.argsort(evals_l, descending=True)[:2]]      # [256, 2]
    lda_dirs, _ = torch.linalg.qr(lda_dirs)
    # PCA on the residual (orthogonal complement of the LDA plane)
    Fp = Fc - (Fc @ lda_dirs) @ lda_dirs.T
    cov = torch.cov(Fp.T)
    evals, evecs = torch.linalg.eigh(cov)
    top = torch.argsort(evals, descending=True)[:PCA_DIM - 2]
    P = torch.cat([lda_dirs, evecs[:, top]], dim=1)                          # [256, K]
    proj = F_all @ P
    scale = proj.std(0).clamp(min=1e-6)
    W = (P / scale[None]).to(torch.float32)
    f_mean = f_mean.to(torch.float32)
    print(f"LDA+PCA projection: 2 discriminant dims + {PCA_DIM-2} residual PCA dims "
          f"(residual explains {float(evals[top].sum()/evals.sum()):.3f} of residual var)",
          flush=True)
    norm = lambda f: ((f - f_mean) @ W).to(DEVICE)

    # crops + self-supervised init labels from the existing pipeline's own decode
    mix_kernel = mix.log_mixture_kernel(W_MIX, LAMBDA_MIX)
    r45 = R45RichEmission(fps=FPS, device=DEVICE, mixture_w=W_MIX,
                          transition_lambda=LAMBDA_MIX, feature_dim=PCA_DIM,
                          tied_covariance=True,
                          observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    crops, labels = [], []
    print("building self-supervised init labels (Viterbi on [T,2] system)...", flush=True)
    for e in train[:NUM_CROPS]:
        f = train_feats[e["stem"]]; a = train_acts[e["stem"]]
        if f.shape[0] > CROP_FRAMES + 1:
            s = int(rng.integers(0, f.shape[0] - CROP_FRAMES))
            f, a = f[s:s + CROP_FRAMES], a[s:s + CROP_FRAMES]
        acts_t = torch.from_numpy(np.ascontiguousarray(a)).float().to(DEVICE)
        dens = mix.log_class_densities(acts_t)
        best = None
        for mi in range(len(mix.chassis.state_spaces)):
            dp = mix.chassis.dynamic_programs[mi]
            path, score = dp.viterbi(mix.chassis.log_initial_distributions[mi], mix_kernel, dens,
                                     state_to_class=mix.chassis.state_to_classes[mi],
                                     return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path, mi)
        _, path, mi = best
        lab = mix.chassis.state_to_classes[mi][path].long().cpu()
        crops.append(norm(f)); labels.append(lab.to(DEVICE))
    r45.initialize_from_labels(crops, labels)
    frac = torch.stack([(torch.cat(labels) == c).double().mean() for c in range(3)])
    print(f"init label fractions (no-beat/beat/downbeat): {[round(float(x),4) for x in frac]}",
          flush=True)

    def subset_F(n=30):
        """Fast bare-decode F trace on a fixed val subset (streaming signal, not the verdict)."""
        bf, df = [], []
        for e in val[:n]:
            out = r45.decode(norm(val_feats[e["stem"]]))
            est_b = mir_eval.beat.trim_beats(out["beats"])
            bf.append(mir_eval.beat.f_measure(
                mir_eval.beat.trim_beats(e["beat_times"]), est_b) if len(est_b) else 0.0)
            est_d = mir_eval.beat.trim_beats(out["downbeats"])
            df.append(mir_eval.beat.f_measure(
                mir_eval.beat.trim_beats(e["downbeat_times"]), est_d) if len(est_d) else 0.0)
        return float(np.mean(bf)), float(np.mean(df))

    print("exact Baum-Welch:", flush=True)
    for it in range(EM_ITERS):
        ll = r45.em_step(crops)
        sb, sd = subset_F()
        print(f"  EM iter {it}: total marginal ll = {ll:.1f} | subset30 beatF {sb:.4f} "
              f"dbF {sd:.4f}", flush=True)

    # --- VALIDITY GATE: real kernel vs uniform kernel on 30 val songs (bare) ---
    import copy
    r45_uniform = copy.copy(r45)
    V = r45.log_kernel.shape[0]
    r45_uniform.log_kernel = torch.full_like(r45.log_kernel, -float(np.log(V)))
    agree, f_real, f_unif = [], [], []
    for e in val[:30]:
        f = norm(val_feats[e["stem"]])
        o_r, o_u = r45.decode(f), r45_uniform.decode(f)
        agree.append(float(np.mean(o_r["path"] == o_u["path"])))
        for out, acc in ((o_r, f_real), (o_u, f_unif)):
            est = mir_eval.beat.trim_beats(out["beats"])
            acc.append(mir_eval.beat.f_measure(
                mir_eval.beat.trim_beats(e["beat_times"]), est) if len(est) else 0.0)
    gap_stats = []
    for e in val[:10]:
        d = r45.log_class_densities(norm(val_feats[e["stem"]]))
        top2 = torch.topk(d, 2, dim=1).values
        gap_stats.append(float((top2[:, 0] - top2[:, 1]).median()))
    print(f"VALIDITY GATE: median emission top1-top2 gap = {np.median(gap_stats):.1f} nats "
          f"(v0: ~22600)", flush=True)
    print(f"VALIDITY GATE: real-vs-uniform kernel -- beatF {np.mean(f_real):.4f} vs "
          f"{np.mean(f_unif):.4f} (delta {np.mean(f_real)-np.mean(f_unif):+.4f}), "
          f"path agreement {np.mean(agree):.3f}  "
          f"[GATE {'OPEN: transition has a vote' if abs(np.mean(f_real)-np.mean(f_unif)) > 0.005 or np.mean(agree) < 0.8 else 'CLOSED: transition still inert -- do not interpret F'}]",
          flush=True)

    # --- decode: six metrics, deployed (threshold crop + snap on [T,2] acts) and bare ---
    def score(deployed):
        bf, cml, aml, df, dcml, daml = [], [], [], [], [], []
        for e in val:
            f = val_feats[e["stem"]]; a = val_acts[e["stem"]]
            if deployed:
                cropped_a, first = threshold_crop(a, BT_SHIPPED_DECODE["threshold"])
                f_use = f[first:first + cropped_a.shape[0]]
            else:
                cropped_a, first, f_use = a, 0, f
            if not cropped_a.size:
                continue
            out = r45.decode(norm(f_use))
            ev = state_path_to_events(out["path"], out["space"], FPS,
                                      snap_to_activations=cropped_a if deployed else None,
                                      first_frame=first)
            ref = mir_eval.beat.trim_beats(e["beat_times"])
            est = mir_eval.beat.trim_beats(ev["beats"])
            if len(est):
                bf.append(mir_eval.beat.f_measure(ref, est))
                _, c, _, am = mir_eval.beat.continuity(ref, est)
                cml.append(c); aml.append(am)
            else:
                bf.append(0.0); cml.append(0.0); aml.append(0.0)
            ref_d = mir_eval.beat.trim_beats(e["downbeat_times"])
            est_d = mir_eval.beat.trim_beats(ev["downbeats"])
            if len(est_d) and len(ref_d) > 1:
                df.append(mir_eval.beat.f_measure(ref_d, est_d))
                _, c, _, am = mir_eval.beat.continuity(ref_d, est_d)
                dcml.append(c); daml.append(am)
            else:
                df.append(0.0); dcml.append(0.0); daml.append(0.0)
        return tuple(float(np.mean(v)) for v in (bf, cml, aml, df, dcml, daml))

    for deployed, name in ((True, "DEPLOYED"), (False, "BARE    ")):
        b, c, a, d, dc, da = score(deployed)
        print(f"R4.5v2 {name}: beatF {b:.4f}  CMLt {c:.4f}  AMLt {a:.4f}  "
              f"dbF {d:.4f}  dbCMLt {dc:.4f}  dbAMLt {da:.4f}", flush=True)
    print("reference DEPLOYED R1/R2: 0.9504/0.9191/0.9410/0.9057/0.8842/0.9366", flush=True)
    torch.save({"mu": r45.mu.cpu(), "log_var": r45.log_var.cpu(),
                "f_mean": f_mean, "W": W},
               Path(__file__).resolve().parent / "r4_5_v2_gaussians.pt")


if __name__ == "__main__":
    main()
