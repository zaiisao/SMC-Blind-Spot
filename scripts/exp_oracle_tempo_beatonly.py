"""Re-measure the tempo-channel ceiling on the BEAT-ONLY state space.

WHY THIS RE-RUN IS NECESSARY. The tempo-oracle ceiling (+0.0571 F / +0.2398 CMLt, n=217) was
measured with the JOINT beat+downbeat state space (beats_per_bar=(3,4)). We have since shown, on
all 217 tracks, that joint decoding is the wrong decoder for SMC -- reproducing Boeck & Davies
(ISMIR 2020) who report joint decoding "negatively impacts performance for expressive music as
contained in Hainsworth and SMC":

    arm                  F      CMLt      AMLt
    joint_min30     0.5851    0.5102    0.6735
    beatonly_min30  0.6106    0.5672    0.6985      (+0.0255 F p=0.001, +0.0570 CMLt p=0.0014)

Since the tempo channel is what a learned tempo head would feed, its ceiling ON THE DECODER WE WILL
ACTUALLY SHIP is the number that sizes that work -- including the accuracy threshold the head must
cross. The joint-decoder ceiling could over- or under-state it: beat-only removes bar states, so
tempo evidence is no longer competing with meter hypotheses for the same likelihood budget.

SMC is beat-only annotated, so beats_per_bar=1 costs us nothing that is ever evaluated.

Injection is IDENTICAL to the original ceiling test (same Gaussian-in-log-interval likelihood over
the DBN tempo grid, same division by beat length so a tempo contributes one beat's worth of
evidence, same log-likelihood-RATIO form so the transition model's tempo prior is not double
counted). Only the state space changes, so any difference is attributable to the decoder.

ARMS: beta=0 is the tempo-blind null. sigma is log-normal noise on the oracle tempo, used to find
the accuracy at which the gain survives -- on the joint decoder, 94% of the CMLt gain remained at
29.6% accuracy, versus the real head's 12.2%.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob, bthis_downbeat_prob
from rungs.r1_2016_dbn import DBN2016
from rungs.deployment import threshold_crop
from rungs.bar_pointer.readout import state_path_to_events

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
SHARP = 0.15
DEV = "cuda:3"
ARMS = [(0.0, 0.0), (0.0, 1.0), (0.0, 5.0), (0.12, 5.0), (0.25, 5.0)]
# 1 = beat-only, 34 = joint (3,4). Parsed defensively: this module is also IMPORTED (for
# make_beatonly_dbn), and a bare int(sys.argv[1]) crashed on the importer's own flags.
BPB = 1
for _a in sys.argv[1:]:
    if _a.isdigit():
        BPB = int(_a); break


def make_beatonly_dbn(min_bpm, obs_lambda, transition_lambda, device):
    """DBN2016 with beats_per_bar=1, with the class assignment CORRECTED for beat-only decoding.

    BUG THIS FIXES, and it silently crippled the first run of this script. BarPointerStateSpace
    assigns classes as
        position_classes[(state_positions % 1.0) < width] = BEAT
        position_classes[state_positions < width]         = DOWNBEAT      # overwrites BEAT
    With beats_per_bar=1 every state position lies in [0,1), so the second line overwrites ALL of
    them: measured class counts are NO_BEAT/BEAT/DOWNBEAT = 2710/0/211 -- not a single BEAT state.
    Since _log_class_densities feeds the DOWNBEAT class from the DOWNBEAT activation, the decoder
    was tracking SMC using downbeat evidence alone, on a beat-only-annotated corpus. That is why
    the tempo-blind null came out at CMLt 0.3288 when madmom's beat-only decoder scores 0.5672.

    Correction: relabel those states BEAT, and feed a ZERO downbeat channel. Then
        class 0 NO_BEAT  = log((1 - beat) / (obs_lambda - 1))
        class 1 BEAT     = log(beat)                      <- full beat activation, not beat-downbeat
        class 2 DOWNBEAT = log(0), unreachable and unused
    which is the correct two-class beat-only emission.
    """
    from rungs.r1_2016_dbn import DBN2016 as _D
    mm = _D(fps=FPS, min_bpm=min_bpm, max_bpm=215.0, device=device, beats_per_bar=1,
            observation_lambda=obs_lambda, transition_lambda=transition_lambda)
    sp = mm.state_spaces[0]
    pc = sp.position_classes.copy()
    pc[pc == 2] = 1                                        # DOWNBEAT -> BEAT
    sp.position_classes = pc
    mm.state_to_classes = [torch.from_numpy(pc).to(device)]
    return mm


def emissions_beatonly(mm, beat_prob):
    """Beat-only class densities: zero downbeat channel, so BEAT carries the full activation."""
    z = np.zeros_like(beat_prob)
    bb, dd, fl = mm._bound(beat_prob, z)
    dec = mm._decorrelate(bb, dd, fl)
    dec, ff = threshold_crop(dec, mm.threshold)
    if not len(dec):
        return None
    return mm._log_class_densities(beat_prob[ff:ff + len(dec)], z[ff:ff + len(dec)]), dec, ff


def main():
    bpb = (3, 4) if BPB == 34 else BPB
    if BPB == 1:
        m = make_beatonly_dbn(30.0, 16, 100, DEV)
    else:
        m = DBN2016(fps=FPS, min_bpm=30.0, max_bpm=215.0, device=DEV, beats_per_bar=bpb,
                    observation_lambda=16, transition_lambda=100)
    import numpy as _np
    print("  class counts NO_BEAT/BEAT/DOWNBEAT =",
          _np.bincount(m.state_spaces[0].position_classes, minlength=3), flush=True)
    dev = torch.device(DEV)
    nstates = sum(len(s.state_interval_frames) for s in m.state_spaces)
    print(f"beats_per_bar={bpb}  state spaces={len(m.state_spaces)}  total states={nstates:,}",
          flush=True)

    grids = [s.interval_frames.astype(np.float64) for s in m.state_spaces]
    tmaps, bfs = [], []
    for s in m.state_spaces:
        idx = {iv: i for i, iv in enumerate(s.interval_frames)}
        tmaps.append(torch.from_numpy(np.array([idx[v] for v in s.state_interval_frames])).to(dev))
        bfs.append(torch.tensor(s.state_interval_frames, dtype=torch.float64, device=dev))

    per = {a: {} for a in ARMS}
    accs = {a: [] for a in ARMS}
    for t in track_ids():
        bpr = bthis_beat_prob(t)
        if BPB == 1:
            got = emissions_beatonly(m, bpr)
            if got is None:
                continue
            lcd, dec, ff = got
        else:
            a = np.stack([bpr, bthis_downbeat_prob(t)], -1)
            bb, dd, fl = m._bound(a[:, 0], a[:, 1])
            dec = m._decorrelate(bb, dd, fl)
            dec, ff = threshold_crop(dec, m.threshold)
            if not len(dec):
                continue
            aa = a[ff:ff + len(dec)]
            lcd = m._log_class_densities(aa[:, 0], aa[:, 1])
        T = lcd.shape[0]
        bt = np.asarray(load_gt(t), float)
        fr = bt * FPS - ff
        ibi = np.diff(bt) * FPS
        true = np.full(T, np.nan)
        for k in range(len(fr) - 1):
            lo, hi = max(0, int(np.ceil(fr[k]))), min(T, int(np.ceil(fr[k + 1])))
            if hi > lo:
                true[lo:hi] = ibi[k]
        if (~np.isnan(true)).sum() < 10:
            continue
        true = np.where(~np.isnan(true), true, np.nanmedian(true))

        for (sigma, beta) in ARMS:
            rng = np.random.default_rng(0)          # same noise draw across arms, per track
            noisy = true * np.exp(rng.normal(0, sigma, size=T)) if sigma > 0 else true
            accs[(sigma, beta)].append(
                float(np.mean(np.abs(np.log(noisy / true)) < np.log(1.1))))
            best = None
            for space, dp, li, s2c, tm, bf, g in zip(
                    m.state_spaces, m.dynamic_programs, m.log_initial_distributions,
                    m.state_to_classes, tmaps, bfs, grids):
                if beta == 0.0:
                    em, stc = lcd, s2c
                else:
                    d = (np.log(g)[None, :] - np.log(noisy)[:, None]) / SHARP
                    lp = -0.5 * d ** 2
                    lp = lp - torch.logsumexp(torch.tensor(lp, device=dev), dim=1,
                                              keepdim=True).cpu().numpy()
                    ratio = torch.tensor(lp, dtype=torch.float64, device=dev) - np.log(1.0 / len(g))
                    em = lcd[:, s2c] + beta * (ratio[:, tm] / bf.unsqueeze(0))
                    stc = None
                path, sc = dp.viterbi(li, m.log_tempo_transition, em, state_to_class=stc,
                                      return_log_score=True)
                if best is None or sc > best[0]:
                    best = (sc, space, path)
            _, space, path = best
            per[(sigma, beta)][t] = ev(load_gt(t), state_path_to_events(
                path.cpu().numpy(), space, m.fps, snap_to_activations=dec, first_frame=ff)["beats"])

    ids = sorted(set.intersection(*[set(per[a]) for a in ARMS]))
    A = lambda r: {mm: float(np.mean([r[t][mm] for t in ids])) for mm in METRICS}
    print(f"\nn={len(ids)}   beats_per_bar={bpb}")
    print(f"{'sigma':>6}{'beta':>6}{'tempoAcc':>10}" + "".join(f"{mm:>9}" for mm in METRICS))
    for a in ARMS:
        print(f"{a[0]:>6.2f}{a[1]:>6.1f}{np.mean(accs[a]):>10.3f}"
              + "".join(f"{A(per[a])[mm]:>9.4f}" for mm in METRICS))
    base = per[(0.0, 0.0)]
    print("\nvs tempo-blind (beta=0):")
    for a in ARMS[1:]:
        out = []
        for mm in METRICS:
            d = np.array([per[a][t][mm] - base[t][mm] for t in ids])
            p = wilcoxon(d).pvalue if np.any(d) else 1.0
            out.append(f"{mm} {d.mean():+.4f}(p={p:.1e})")
        print(f"  sigma={a[0]:.2f} beta={a[1]:.1f}  " + "  ".join(out))
    json.dump({"beats_per_bar": str(bpb), "n": len(ids),
               "summary": {str(a): A(per[a]) for a in ARMS},
               "tempo_acc": {str(a): float(np.mean(accs[a])) for a in ARMS},
               "per_track": {str(a): per[a] for a in ARMS}},
              open(f"/tmp/overnight/oracle_beatonly_bpb{BPB}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
