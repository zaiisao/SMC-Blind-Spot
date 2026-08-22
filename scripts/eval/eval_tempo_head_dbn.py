"""Inject the LEARNED tempo observation q_theta into the beat-only DBN and score SMC beat tracking.

This is the decisive test. Everything measured so far about q_theta is tempo-ESTIMATION accuracy
(46.1% frame accuracy on SMC, up from 12.2%, median log2 bias 0.000, sub-55 BPM band best at 56%).
None of that is beat tracking. The previous head, at 12.2%, made decoding WORSE (-0.069 F at
beta=0.1), so accuracy alone predicts nothing.

WHAT IS BEING ADDED. The bar-pointer DBN has no likelihood term for tempo at all: the Boeck 2016
emission is a function of state POSITION only (Prop 1), so phidot is identified solely through the
transition prior and phase alignment. This supplies the missing factor as an ASR-hybrid scaled
likelihood, so the transition model's tempo prior is not double counted:

    log p(y_t | phi, phidot) = log p(y_t | class(phi))
                             + beta * [ log q_theta(phidot | h) - log p_head(phidot) ] / i

  - p_head is the head's OWN marginal (saved beside the checkpoint), NOT the label prior. Using the
    label prior double-corrects: a head trained with prior^-0.5 weights is already prior-corrected,
    and that bug once injected 5.47 nats per tempo (~11,000 nats/track).
  - the division by i (the state's beat period in frames) makes a tempo contribute ONE BEAT's worth
    of evidence rather than one frame's. This is the same normalisation the oracle ceiling used, and
    it is a heuristic patch for a real problem: h_t comes from a transformer with global attention,
    so adding the term at every frame counts the same evidence T times. See docs/spec/SPECIFICATION.md R5.

REFERENCE POINTS on this exact decoder (beat-only, class-corrected, min_bpm=30, obs_lambda=16,
lambda=100, Beat This activations, n=217):
    tempo-blind null (beta=0)   F 0.6141   CMLt 0.5685   AMLt 0.6988
    oracle tempo    (beta=5)    F 0.6756   CMLt 0.7906   AMLt 0.8169
A learned head that helps should move CMLt off 0.5685 toward 0.7906. CMLt is the primary metric:
the oracle buys +0.2221 CMLt but only +0.0615 F, because F factorises over beats within +/-70 ms
while tempo is a sequence-level property.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
from rungs.bar_pointer.readout import state_path_to_events
from train_tempo_observation import load_net

SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
HEAD = Path(__import__("os").environ.get("HEAD_PT",
                                        "/disk1/jaehoon/tempo_obs_head.pt"))
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
# Low-beta regime. The head's log-ratio range is ~12.7 nats, and the /i normalisation makes each
# BEAT contribute ~beta*12.7 nats, so a 40-beat track carries ~beta*500 nats of tempo evidence
# against the emission's ENTIRE octave preference of ~93 nats/track. At beta=5 the tempo term is
# ~27x stronger than the evidence that could overrule a wrong octave, which is why a wrong head is
# catastrophic (-0.36 CMLt) rather than merely unhelpful. Commensurate weighting is beta ~ 0.1-0.2.
BETAS = [float(x) for x in __import__("os").environ.get("BETAS", "0.0,0.05,0.1,0.2,0.3,0.5").split(",")]
# Octave smearing weights: q_smeared(i) ∝ sum_k w_k * q(i * 2^k), k in {-1,0,+1}.
# WHY. Split by head correctness, the frame-wise injection is not uniformly harmful -- it is two
# opposite effects cancelling (n=217, beat-only, beta=5 vs beta=0):
#     head tempo correct (n=113)   CMLt 0.7594 -> 0.8088   +0.0494  p=0.024
#     head predicts half (n=40)    CMLt 0.4469 -> 0.0848   -0.3621  p=2.1e-05
#     head predicts double (n=14)  CMLt 0.3827 -> 0.0000   -0.3827  p=0.018
# The mechanism works; confidently-wrong OCTAVES destroy it. The head is 75.6% octave-tolerant but
# only 51.6% strict, while the emission demonstrably does discriminate octaves (double-tempo paths
# score ~93 nats worse; truth beats double on 85.7%, n=119, p=5.3e-16). So let the head constrain
# tempo MODULO OCTAVE and let the phase evidence choose the octave.
SMEAR = float(__import__("os").environ.get("SMEAR", "0.0"))   # 0 = off (original behaviour)
# ACF BLEND (step 0 of the head-improvement plan). log q <- (1-W)*log q_head + W*log q_acf.
# Motivation: 37 of the head's 103 failures are tracks where the annotated tempo IS the peak of the
# autocorrelation of the model's own beat activation -- the head reads only embeddings and never sees
# periodicity. Blending recovers 37 and breaks 5 at W=0.8, tau=10 (octave-error 32.3% -> 20.7%).
ACF_W = float(__import__("os").environ.get("ACF_W", "0.0"))
ACF_TAU = float(__import__("os").environ.get("ACF_TAU", "10.0"))
# POOL: collapse the head's per-frame distribution to ONE track-level distribution before
# injecting. Motivation: the head's frame-wise argmax agrees with the oracle's only 13.1% of the
# time, yet a CONSTANT per-track oracle tempo (median IBI, zero time resolution) buys +0.1733 CMLt.
# The form that works is a single tempo per track; every injection so far has been a fluctuating
# per-frame distribution. This tests whether we have been injecting the right information in the
# wrong shape.
POOL = int(__import__("os").environ.get("POOL", "0"))
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"


def main():
    dev = torch.device(DEV)
    m = make_beatonly_dbn(30.0, 16, 100, DEV)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    bf = torch.tensor(space.state_interval_frames, dtype=torch.float64, device=dev)
    s2c = m.state_to_classes[0].to(dev)

    net = load_net(str(HEAD), DEV)
    marg = np.load(str(HEAD) + ".marginal.npy")
    ck = torch.load(str(HEAD), map_location="cpu", weights_only=False)
    grid = np.asarray(ck["grid"], dtype=np.int64)
    assert len(grid) == len(iv) and (grid == iv).all(), \
        f"head grid {len(grid)} != DBN grid {len(iv)} -- the tempo axes must be identical"
    MARG_FIX = __import__("os").environ.get("MARG_FIX", "1") == "1"
    logmarg = torch.tensor(np.log(np.clip(marg, 1e-12, None)), dtype=torch.float64, device=dev)
    logmarg_eff = logmarg
    grid_t = torch.tensor(iv, dtype=torch.float64, device=dev)
    if SMEAR > 0:
        print(f"OCTAVE SMEARING on, weight {SMEAR}", flush=True)
    print(f"grid {len(iv)} tempi, head marginal spread {float(logmarg.max()-logmarg.min()):.3f} nats",
          flush=True)

    per = {b: {} for b in BETAS}
    n_used = 0
    for t in track_ids():
        f = SMC_EMB / f"{t}.npz"
        if not f.exists():
            continue
        got = emissions_beatonly(m, bthis_beat_prob(t))
        if got is None:
            continue
        lcd, dec, ff = got
        T = lcd.shape[0]
        z = np.load(f, allow_pickle=True)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(z["emb"], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            logq = torch.log_softmax(net(x).float(), 1)[0].T.double()          # [T_emb, V]
        # align the head's frames to the DBN's cropped window
        if logq.shape[0] < ff + T:
            pad = ff + T - logq.shape[0]
            logq = torch.cat([logq, logq[-1:].repeat(pad, 1)], 0)
        lq = logq[ff:ff + T]
        if ACF_W > 0:
            a = torch.as_tensor(bthis_beat_prob(t)[ff:ff + T], dtype=torch.float64, device=dev)
            a = a - a.mean()
            Ta = len(a)
            r = torch.stack([(a[:Ta - int(l)] * a[int(l):]).sum() / float(Ta - int(l))
                             if Ta - int(l) >= 8 else torch.tensor(-1e9, dtype=torch.float64,
                                                                   device=dev)
                             for l in iv])
            r = r / ((a * a).sum() / Ta + 1e-12)
            lacf = torch.log_softmax(ACF_TAU * r, dim=0)
            lq = torch.log_softmax((1 - ACF_W) * lq + ACF_W * lacf[None, :], dim=1)
            # BUGFIX (2026-08-22): the prior-correction below subtracts the head's FULL marginal,
            # but after blending the head's log-prior only enters with exponent (1-ACF_W). The
            # residual -ACF_W*log p_head is a spurious per-tempo bias of 0.8*3.87 = 3.10 nats at
            # W=0.8, against a blended log-ratio spread of ~10 nats. Scale the marginal to match.
            if MARG_FIX:
                logmarg_eff = (1 - ACF_W) * logmarg
        if SMEAR > 0:
            # add mass at the octave partners of each grid point, in probability space
            q = lq.exp()
            acc = q.clone()
            for k in (-1, 1):
                tgt = grid_t * (2.0 ** k)
                # nearest grid index for each shifted tempo; out-of-range partners are dropped,
                # which is correct -- a tempo whose octave lies outside 30-215 BPM has no partner
                j = torch.argmin(torch.abs(grid_t[None, :] - tgt[:, None]), dim=1)
                ok = (torch.abs(grid_t[j] - tgt) / tgt) < 0.06
                # index_add_, not indexed assignment: the tempo grid is integer frames, so two
                # source tempi can round to the SAME partner index and plain assignment would
                # silently drop one of them.
                add = torch.zeros_like(q)
                add.index_add_(1, j[ok], q[:, ok])
                acc = acc + SMEAR * add
            lq = torch.log(acc / acc.sum(dim=1, keepdim=True))
        if POOL:
            # collapse to ONE track-level distribution before injecting
            lq = torch.log_softmax(lq.mean(0, keepdim=True), dim=1).expand_as(lq)
        ratio = (lq - logmarg_eff[None, :])
        for beta in BETAS:
            if beta == 0.0:
                em, stc = lcd, s2c
            else:
                em = lcd[:, s2c] + beta * (ratio[:, tmap] / bf.unsqueeze(0))
                stc = None
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], m.log_tempo_transition, em,
                state_to_class=stc, return_log_score=True)
            per[beta][t] = ev(load_gt(t), state_path_to_events(
                path.cpu().numpy(), space, FPS, snap_to_activations=dec, first_frame=ff)["beats"])
        n_used += 1
        if n_used % 50 == 0:
            print(f"  {n_used} tracks", flush=True)

    ids = sorted(set.intersection(*[set(per[b]) for b in BETAS]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    print(f"\nn={len(ids)}")
    print(f"{'beta':>6}" + "".join(f"{k:>9}" for k in METRICS))
    for b in BETAS:
        print(f"{b:>6.1f}" + "".join(f"{A(per[b])[k]:>9.4f}" for k in METRICS))
    print("\nvs tempo-blind (beta=0), paired:")
    base = per[0.0]
    for b in BETAS[1:]:
        out = []
        for k in METRICS:
            d = np.array([per[b][t][k] - base[t][k] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            out.append(f"{k} {d.mean():+.4f}(p={p:.2g})")
        print(f"  beta={b:.1f}  " + "  ".join(out))
    print("\nreference: oracle beta=5 gave F 0.6756 CMLt 0.7906 (null 0.6141 / 0.5685)")
    json.dump({"n": len(ids), "summary": {str(b): A(per[b]) for b in BETAS},
               "per_track": {str(b): per[b] for b in BETAS}},
              open(f"/tmp/overnight/tempo_head_dbn{__import__('os').environ.get('TAG','')}.json",
                   "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
