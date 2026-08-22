"""Learn the DBN's decoding parameters CONTRASTIVELY, instead of by grid search or by MLE.

THE PROBLEM WITH MLE, MEASURED. Fitting the transition parameter generatively to the SMC
annotations -- i.e. asking "what lambda explains the tempo changes that actually occur?" -- gives
lambda = 11.45 against madmom's shipped 100 (10,267 transitions; the exponential family fits well,
ratios 0.84-0.95). Decoding with lambda=11 costs -0.084 AMLt. The generative optimum is NOT the
decoding optimum, and that is structural rather than unlucky: in the DBN the transition prior is
not doing density estimation, it is doing REGULARISATION. Making it match reality makes decoding
more permissive, the pointer wanders, and continuity degrades. Krebs knew this -- in the ISMIR 2015
paper lambda is set by GRID SEARCH on Cemgil accuracy, never by maximum likelihood.

THE FIX. Keep the likelihood machinery, change the denominator. Conditional (discriminative)
training normalises the true beat sequence not against reality but against COMPETING DECODINGS:

    L = -log  exp(score(z_true)) / sum_k exp(score(z_k))

With the full path set this is exact MMI. Here the denominator is a set of SAMPLED NEGATIVES, i.e.
noise-contrastive estimation of the same quantity -- which removes the cost of a full forward
recursion per step and, more importantly, lets us CHOOSE what the model must beat.

WHY SAMPLED NEGATIVES ARE BETTER THAN A FULL DENOMINATOR HERE. A full denominator spreads gradient
over all paths, nearly all irrelevant. We know the failure modes exactly, so the negatives are
aimed at them:
    octave   double / half tempo   -- the measured equivariance collapse (delta = -0.966 at 34 BPM)
    phase    offset by half a beat -- off-beat tracking
    drift    slow tempo wander     -- precisely what transition_lambda exists to suppress
The beat-only decoder scores AMLt 0.6985 vs CMLt 0.5672 on SMC. That 0.13 gap IS metrical-level
error, because AMLt forgives double/half and CMLt does not. So octave negatives attack exactly the
quantity separating our two headline numbers.

WHAT IS LEARNED. transition_lambda and beta (the tempo-channel weight), by gradient. Both enter the
score as smooth scalars. observation_lambda is NOT learned: it sets the class assignment through a
threshold on state position, which is discrete, so it is swept instead. Learning two scalars by
gradient where 2015 used a grid search is the incremental, conceptually contained upgrade this is
meant to be -- not a new model.

STATE SPACE: beat-only (beats_per_bar=1). SMC is beat-only annotated, and joint decoding measurably
hurts here (+0.0255 F, +0.0570 CMLt for beat-only over joint, n=217, reproducing Boeck & Davies
ISMIR 2020). It is also ~3x smaller, which matters when scoring many paths per step.

HONESTY. Parameters are fitted on a TRAIN split of SMC and reported on a held-out split; the
shipped configuration is evaluated on the same held-out split. A contrastive objective can also be
satisfied by sharpening activations rather than by fixing tempo, so the reported number is always
the DECODED metric, never the loss.
"""
import argparse
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
# beats_per_bar=1 leaves ZERO states in the BEAT class (all 211 on-beat states are relabelled
# DOWNBEAT), so the emission would be driven by the downbeat activation on a beat-only corpus.
# make_beatonly_dbn / emissions_beatonly correct that; see exp_oracle_tempo_beatonly.py.
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


# ----------------------------------------------------------------------------- differentiable bits
def log_tempo_transition(interval_frames, lam, threshold=float(np.spacing(1))):
    """[V, V] log p(tempo_to | tempo_from), differentiable in lam.

    Reproduces madmom's exponential_transition: exp(-lam |ratio - 1|), hard-zeroed below
    np.spacing(1), then row-normalised. The zero mask is taken as a CONSTANT at the current lam
    (detached): it is a discrete set, and the truncation was separately measured to be inert
    (it cuts at a constant 36.04 nats regardless of lam; 4/85 tracks changed at lam=100, 0/85 at
    lam=11). Row-normalisation is done in log space as a logsumexp, which is where the gradient
    w.r.t. lam actually comes from -- normalising after exponentiating would detach it.
    """
    iv = interval_frames.to(torch.float64)
    ratio = iv[None, :] / iv[:, None]
    d = torch.abs(ratio - 1.0)
    with torch.no_grad():
        mask = torch.exp(-lam.detach() * d) <= threshold
    logits = -lam * d
    logits = logits.masked_fill(mask, float("-inf"))
    return logits - torch.logsumexp(logits, dim=1, keepdim=True)


def beats_to_state_path(beat_frames, T, iv_grid, first_state, ivals):
    """Beat times (in frames, within the cropped window) -> a valid state path of length T.

    Between consecutive beats the pointer advances one state per frame at a fixed tempo, which is
    the bar-pointer model's defining property: a tempo of i frames owns exactly i states. The
    inter-beat interval is snapped to the DBN's integer tempo grid. Returns (path, tempo_index
    per frame) or None if the sequence cannot be represented.
    """
    bf = np.asarray(beat_frames, dtype=float)
    bf = bf[(bf > -iv_grid.max()) & (bf < T + iv_grid.max())]
    if len(bf) < 3:
        return None
    path = np.full(T, -1, dtype=np.int64)
    tidx = np.full(T, -1, dtype=np.int64)
    for k in range(len(bf) - 1):
        ib = bf[k + 1] - bf[k]
        if ib < iv_grid.min() or ib > iv_grid.max():
            continue
        v = int(np.argmin(np.abs(iv_grid - ib)))          # snap to the tempo grid
        interval = int(iv_grid[v])
        lo, hi = int(np.floor(bf[k])), int(np.floor(bf[k + 1]))
        for t in range(max(0, lo), min(T, hi)):
            phase = t - bf[k]
            if phase < 0 or phase >= interval:
                continue
            path[t] = first_state[v] + int(phase)
            tidx[t] = v
    if (path < 0).mean() > 0.5:                           # too little coverage to score fairly
        return None
    # fill any gaps by holding the previous state, so the path stays well defined
    last = -1
    for t in range(T):
        if path[t] < 0:
            if last < 0:
                continue
            path[t], tidx[t] = path[last], tidx[last]
        else:
            last = t
    if (path < 0).any():
        first_ok = int(np.argmax(path >= 0))
        path[:first_ok], tidx[:first_ok] = path[first_ok], tidx[first_ok]
    return path, tidx


def path_score(path, tidx, lcd, s2c, log_T, beta, tempo_ratio):
    """Total log score of one state path: emissions + tempo-boundary transitions.

    Within-beat transitions are deterministic (the pointer advances by one state), contributing 0,
    so only BEAT BOUNDARIES -- frames where the tempo index may change -- carry transition mass.
    That mirrors madmom, where the tempo kernel is applied only at boundaries.
    """
    p = torch.as_tensor(path, device=lcd.device)
    ti = torch.as_tensor(tidx, device=lcd.device)
    em = lcd[torch.arange(len(p), device=lcd.device), s2c[p]]
    if beta is not None and tempo_ratio is not None:
        em = em + beta * tempo_ratio[torch.arange(len(p), device=lcd.device), ti]
    sc = em.sum()
    bnd = (ti[1:] != ti[:-1]).nonzero().flatten()
    if len(bnd):
        sc = sc + log_T[ti[bnd], ti[bnd + 1]].sum()
    return sc


# ------------------------------------------------------------------------------------- negatives
def make_negatives(beat_frames, rng, n_drift=2):
    """Hypotheses the true decoding must outscore, aimed at the measured failure modes."""
    bf = np.asarray(beat_frames, dtype=float)
    negs = {}
    if len(bf) >= 3:
        mid = (bf[:-1] + bf[1:]) / 2.0                    # OCTAVE: double tempo
        negs["double"] = np.sort(np.concatenate([bf, mid]))
        negs["half"] = bf[::2]                            # OCTAVE: half tempo
        ibi = np.diff(bf)
        negs["phase"] = bf[:-1] + ibi / 2.0               # PHASE: off-beat
        for j in range(n_drift):                          # DRIFT: slow tempo wander
            w = np.cumsum(rng.normal(0, 0.02, size=len(bf)))
            negs[f"drift{j}"] = bf + w * np.median(ibi)
    return {k: v for k, v in negs.items() if len(v) >= 3}


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-lambda", type=int, default=16)
    ap.add_argument("--init-lambda", type=float, default=100.0)
    ap.add_argument("--init-beta", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--mine-every", type=int, default=25,
                    help="refresh HARD NEGATIVES every N steps (0 = synthetic negatives only).\n                         Synthetic negatives alone drove lambda to 9.02, i.e. straight to the\n                         generative MLE (11.45): the octave corruptions are too easily separated\n                         to give gradient, and beating generic drift is what MLE already does.")
    ap.add_argument("--mine-max-f", type=float, default=0.95,
                    help="only mine a decode as a negative if it scores BELOW this F -- a correct\n                         decode is not an error and must not be pushed down")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-bpm", type=float, default=30.0)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--out", default="/tmp/overnight/contrastive_dbn.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)

    m = make_beatonly_dbn(args.min_bpm, args.obs_lambda, 100, args.device)
    assert int((m.state_spaces[0].position_classes == 1).sum()) > 0, "no BEAT states"
    space = m.state_spaces[0]
    iv_grid = space.interval_frames.astype(np.int64)
    first_state = np.cumsum(np.r_[0, iv_grid[:-1]]).astype(np.int64)
    s2c = m.state_to_classes[0].to(dev)
    iv_t = torch.from_numpy(iv_grid).to(dev)
    # state -> tempo index, so a decoded state path can be scored by the same path_score()
    _idx = {int(v): i for i, v in enumerate(iv_grid)}
    tmap = torch.from_numpy(
        np.array([_idx[int(v)] for v in m.state_spaces[0].state_interval_frames])).to(dev)
    print(f"beat-only space: {len(iv_grid)} tempi, {int(iv_grid.sum()):,} states, "
          f"{60*FPS/iv_grid.max():.0f}-{60*FPS/iv_grid.min():.0f} BPM", flush=True)

    ids = track_ids()
    rng.shuffle(ids)
    split = int(0.6 * len(ids))
    train_ids, test_ids = sorted(ids[:split]), sorted(ids[split:])
    print(f"train {len(train_ids)}  held-out {len(test_ids)}", flush=True)

    # ---- precompute per-track emissions and paths once (the model is fixed; only params train) ---
    def prep(tid):
        got = emissions_beatonly(m, bthis_beat_prob(tid))
        if got is None:
            return None
        lcd, dec, ff = got
        T = lcd.shape[0]
        gt = np.asarray(load_gt(tid), float) * FPS - ff
        pos = beats_to_state_path(gt, T, iv_grid, first_state, None)
        if pos is None:
            return None
        negs = {}
        for k, v in make_negatives(gt, rng).items():
            r = beats_to_state_path(v, T, iv_grid, first_state, None)
            if r is not None:
                negs[k] = r
        if not negs:
            return None
        return {"lcd": lcd, "pos": pos, "negs": negs, "dec": dec, "ff": ff}

    cache = {}
    for tid in train_ids:
        c = prep(tid)
        if c is not None:
            cache[tid] = c
    print(f"prepared {len(cache)} training tracks", flush=True)

    raw_lam = torch.tensor(float(np.log(args.init_lambda)), device=dev, requires_grad=True)
    beta = torch.tensor(float(args.init_beta), device=dev, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw_lam, beta], lr=args.lr)

    keys = list(cache)
    skipped = []

    def mine(log_T_det):
        """Decode each cached track with the CURRENT parameters; keep wrong decodes as negatives.

        This is the lattice-MMI idea: the denominator should contain the hypotheses the decoder
        actually produces, not corruptions we invented. Synthetic negatives sit at a fixed distance
        from the truth, so the gradient stops being informative once they are separated; mined ones
        track the decision boundary as lambda moves. A decode that is already correct is skipped --
        pushing it down would train against the right answer.
        """
        n_mined = 0
        for tid in keys:
            c = cache[tid]
            try:
                path, _ = m.dynamic_programs[0].viterbi(
                    m.log_initial_distributions[0], log_T_det, c["lcd"],
                    state_to_class=s2c, return_log_score=True)
            except Exception:
                continue
            beats = state_path_to_events(path.cpu().numpy(), m.state_spaces[0], FPS,
                                         snap_to_activations=c["dec"], first_frame=c["ff"])["beats"]
            if ev(load_gt(tid), beats)["F"] >= args.mine_max_f:
                c["negs"].pop("mined", None)
                continue
            pa = path.cpu().numpy()
            c["negs"]["mined"] = (pa, tmap[path].cpu().numpy())
            n_mined += 1
        return n_mined
    for step in range(args.steps):
        lam = torch.exp(raw_lam)                       # positivity by construction
        log_T = log_tempo_transition(iv_t, lam)
        if args.mine_every and step % args.mine_every == 0:
            nm = mine(log_T.detach())
            print(f"  [mine] step {step}: {nm}/{len(keys)} tracks decoded wrongly "
                  f"-> hard negatives", flush=True)
        tot, nb = 0.0, 0
        opt.zero_grad()
        for tid in rng.choice(keys, size=min(8, len(keys)), replace=False):
            c = cache[tid]
            sp = path_score(*c["pos"], c["lcd"], s2c, log_T, None, None)
            sn = [path_score(*v, c["lcd"], s2c, log_T, None, None) for v in c["negs"].values()]
            scores = torch.stack([sp] + sn)
            # A single non-finite path score poisons lambda irrecoverably through Adam's state
            # (one nan gradient -> nan moments -> nan forever), which is how the first run ended
            # with lambda=nan and a decode at CMLt 0.0095. Skip rather than propagate.
            if not torch.isfinite(scores).all():
                skipped.append(tid)
                continue
            loss = -(scores[0] - torch.logsumexp(scores, dim=0))
            if not torch.isfinite(loss):
                skipped.append(tid)
                continue
            loss.backward(retain_graph=True)
            tot += float(loss); nb += 1
        if raw_lam.grad is not None and torch.isfinite(raw_lam.grad).all():
            torch.nn.utils.clip_grad_norm_([raw_lam, beta], 5.0)
            prev = raw_lam.detach().clone()
            opt.step()
            if not torch.isfinite(raw_lam):
                with torch.no_grad():
                    raw_lam.copy_(prev)
        else:
            opt.zero_grad()
        if (step + 1) % 25 == 0:
            print(f"  step {step+1}  loss {tot/max(nb,1):.4f}  lambda {float(torch.exp(raw_lam)):.2f}",
                  flush=True)

    if skipped:
        print(f"  skipped {len(skipped)} non-finite scorings "
              f"({len(set(skipped))} distinct tracks)", flush=True)
    lam_learned = float(torch.exp(raw_lam))
    print(f"\nlearned transition_lambda = {lam_learned:.2f}  (shipped 100, generative MLE 11.45)",
          flush=True)

    # ---- held-out decoding: learned vs shipped, same tracks, same everything else ----
    res = {}
    for tag, lv in (("shipped100", 100.0), ("learned", lam_learned), ("mle11", 11.45)):
        mm = make_beatonly_dbn(args.min_bpm, args.obs_lambda, lv, args.device)
        r = {}
        for tid in test_ids:
            got = emissions_beatonly(mm, bthis_beat_prob(tid))
            if got is None:
                continue
            lcd, dec, ff = got
            path, _ = mm.dynamic_programs[0].viterbi(
                mm.log_initial_distributions[0], mm.log_tempo_transition, lcd,
                state_to_class=mm.state_to_classes[0], return_log_score=True)
            r[tid] = ev(load_gt(tid), state_path_to_events(
                path.cpu().numpy(), mm.state_spaces[0], FPS,
                snap_to_activations=dec, first_frame=ff)["beats"])
        res[tag] = r
        print(f"  decoded {tag} (lambda={lv:.2f})", flush=True)

    ok = sorted(set.intersection(*[set(r) for r in res.values()]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ok])) for k in METRICS}
    print(f"\nHELD-OUT n={len(ok)}")
    print(f"{'arm':<14}" + "".join(f"{k:>9}" for k in METRICS))
    for tag in res:
        print(f"{tag:<14}" + "".join(f"{A(res[tag])[k]:>9.4f}" for k in METRICS))
    print("\nlearned minus shipped100, paired:")
    for k in METRICS:
        d = np.array([res["learned"][t][k] - res["shipped100"][t][k] for t in ok])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"  {k} {d.mean():+.4f} (p={p:.3g})")
    json.dump({"lambda_learned": lam_learned, "n_heldout": len(ok),
               "summary": {t: A(res[t]) for t in res},
               "train_ids": train_ids, "test_ids": test_ids},
              open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
