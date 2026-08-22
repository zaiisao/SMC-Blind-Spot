"""A CRF over the bar-pointer state space: multiple evidence sources, LEARNED weights.

WHAT THIS IS. A log-linear (conditional random field) model on the beat-only bar-pointer state
space. The score of a state path is a WEIGHTED SUM of feature functions, and the weights are learned
by conditional maximum likelihood against competing paths:

    score(z) = w_em * SUM_t  log p(y_t | class(phi_t))          Boeck 2016 emission, Beat This
             + w_tempo * SUM_t [log q_theta(phidot_t | h) - log p_head]   learned tempo head
             + w_acf * SUM_t [log q_acf(phidot_t) - log uniform]          activation periodicity
             + w_trans * SUM_boundaries log T_lambda(phidot -> phidot')   tempo transition
             ( + w_bt * SUM_t log p_BT(y_t | class(phi_t))                second frontend, optional )

    L = -log [ exp(score(z_true)) / SUM_k exp(score(z_k)) ]

WHY THIS AND NOT MORE HAND-TUNING. Every combination weight in this project has been set by grid
search: beta for the tempo channel (swept, optimum 0.2), ACF_W (swept, 0.8), the ensemble mixing
weight (swept, 0.6), transition_lambda (swept by Krebs in 2015 on Cemgil accuracy). A CRF makes all
of them LEARNED PARAMETERS of one objective instead of independent sweeps that cannot see each
other. That is the actual content of "combine multiple sources and make one determination".

RELATION TO WHAT WE ALREADY BUILT. The contrastive objective in train_contrastive_dbn_fixed.py IS
CRF conditional ML -- maximising the true path against sampled competitors. It previously had ONE
feature (the transition), which is why it recovered nothing beyond lambda: there was nothing to
weigh. This adds the features.

PRIOR ART -- MUST BE CITED, THE IDEA IS NOT OURS (see docs/related/PRIOR_ART_NEURAL_DBN.md):
  * Durand & Essid, ISMIR 2016 -- four CNNs feeding a CRF for downbeats, fitted separately
    (pycrfsuite + L-BFGS), no gradient into the CNNs. Their closing sentence names "learning the
    deep networks and the conditional random field models jointly" as future work. We are doing what
    they called for, on the bar-pointer state space -- we are NOT proposing the idea.
  * Fuentes et al., ICASSP 2019 -- skip-chain CRF over CRNN outputs, coupling constant hand-chosen.
  * COUNTEREXAMPLE the reviewer will raise: CNN+CRF chord recognition (arXiv:1810.10002) found
    SEPARATE training outperformed joint training. Note it honestly.

HONESTY. Weights are fitted on CORPUS tracks (Beat This activations from /disk1/jaehoon/bt_emb,
which carry beat/downbeat alongside the embeddings) and evaluated on SMC, which is never trained on.
That is the clean answer to the ISMIR oracle-dependence objection. The Beat Transformer feature is
available for SMC only, so it is OFF by default; enabling it would require corpus BT activations and
would make the weights SMC-fitted unless those are extracted first.

EXPECTATION MANAGEMENT. Section 4.12 measured the decoder envelope on Beat This activations at
F=0.6908 with per-track ORACLE hyperparameters; deployed peak-picking is 0.6270. So a CRF over these
features is competing inside a ~0.064 F box and cannot exceed it. Only adding a SECOND FRONTEND
changes that bound (oracle model-selection over the two frontends reaches 0.6978).
"""
import argparse
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
from train_contrastive_dbn_fixed import (log_tempo_transition, build_path, assert_valid,
                                         make_negatives)
from train_tempo_observation import load_net

EMB = Path("/disk1/jaehoon/bt_emb")
SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
HEAD = Path("/disk1/jaehoon/tempo_obs_head.pt")
METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")
FEATS = ("em", "tempo", "acf", "trans")


def acf_logratio(act, iv, tau, dev):
    """Scale-free unbiased ACF over the tempo grid -> log q_acf - log uniform. [V]"""
    a = torch.as_tensor(act, dtype=torch.float64, device=dev)
    a = a - a.mean()
    T = len(a)
    r = torch.stack([(a[:T - int(l)] * a[int(l):]).sum() / float(T - int(l))
                     if T - int(l) >= 8 else torch.tensor(-1e9, dtype=torch.float64, device=dev)
                     for l in iv])
    r = r / ((a * a).sum() / T + 1e-12)
    return torch.log_softmax(tau * r, dim=0) - np.log(1.0 / len(iv))


def crf_score(path_t, ti_t, bnd_t, F, log_T, w):
    """Weighted sum of feature functions for one state path. F holds per-frame feature tables."""
    ar = torch.arange(len(path_t), device=F["em"].device)
    sc = w[0] * F["em"][ar, F["s2c"][path_t]].sum()
    if F.get("tempo") is not None:
        sc = sc + w[1] * F["tempo"][ar, ti_t].sum()
    if F.get("acf") is not None:
        sc = sc + w[2] * F["acf"][ti_t].sum()
    if len(bnd_t):
        sc = sc + w[3] * log_T[ti_t[bnd_t], ti_t[bnd_t + 1]].sum()
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ballroom,hainsworth,beatles,gtzan,asap")
    ap.add_argument("--train-on", choices=["corpus", "smc"], default="corpus",
                    help="corpus: fit on ballroom/hainsworth/beatles/gtzan/asap and evaluate on ALL "
                         "of SMC -- a clean generalisation claim, but lambda and w are fitted on "
                         "steady-tempo material and SMC has 9%% beat-to-beat IBI variation with "
                         "positive lag-1 autocorrelation (SPEC 4.13), so the transfer is BIASED, "
                         "not merely noisy. smc: fit on an SMC train split and evaluate on the "
                         "held-out split -- stronger, but must be labelled SMC-TUNED, which is the "
                         "exact objection the ISMIR reviewers raised. Report BOTH: the gap between "
                         "them is the transfer cost, and it is a measurement rather than a "
                         "confound.")
    ap.add_argument("--smc-train-frac", type=float, default=0.6)
    ap.add_argument("--train-tracks", type=int, default=250)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--wnorm", type=float, default=2.0,
                    help="fixed L2 norm for w (projected after every step). The baseline "
                         "[1,0,0,1] has norm sqrt(2)=1.414; 2.0 leaves room for the extra "
                         "features without letting the vector inflate or collapse.")
    ap.add_argument("--l2", type=float, default=0.0,
                    help="L2 on w. G3 showed the objective is SEPARABLE: once the true path beats\n"
                         "every synthetic negative the loss is ~1e-4, no gradient remains to refine\n"
                         "DIRECTION, and w just inflates in scale (1.006 -> 1.268) while cos(w,w*)\n"
                         "DEGRADES 0.9467 -> 0.9415. L2 removes the scale escape route.")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="hinge margin added to every negative's score, so the objective keeps\n"
                         "pushing after the true path merely wins")
    ap.add_argument("--mine-max-f", type=float, default=0.90,
                    help="a decode at/above this F against the annotation is CORRECT "
                         "and must not be used as a negative")
    ap.add_argument("--mine-every", type=int, default=25,
                    help="refresh HARD negatives by decoding with the CURRENT weights; synthetic\n"
                         "corruptions alone make the problem separable and uninformative")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--obs-lambda", type=int, default=16)
    ap.add_argument("--init-lambda", type=float, default=100.0)
    ap.add_argument("--freeze-lambda", type=int, default=1,
                    help="Hold transition_lambda fixed (default ON). lambda is an unconstrained\n"
                         "FLATTEN-EVERYTHING escape route for this objective: the MMI loss is the\n"
                         "gap between the constrained (annotation-consistent) maximum and the free\n"
                         "maximum, and LOWERING lambda flattens the transition kernel, shrinking\n"
                         "that gap trivially without improving decoding. Measured: lambda slid\n"
                         "100 -> 18.36 while the corpus wrong-decode count went 54 -> 250 of 250.\n"
                         "The unit-norm projection removed exactly this degenerate direction for w;\n"
                         "lambda lives outside the sphere and simply became the next one. Three\n"
                         "independent estimates already agree lambda tuning buys nothing (MLE\n"
                         "11.45, contrastive 9.02, hard-mined 7.55; all about +0.017 F / -0.044\n"
                         "AMLt), so freezing it at Krebs 100 costs no known opportunity and lets\n"
                         "the CRF answer the question it exists for: how to weight the sources.")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/tmp/overnight/crf_decoder.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)

    m = make_beatonly_dbn(30.0, args.obs_lambda, 100, args.device)
    space = m.state_spaces[0]
    iv = space.interval_frames.astype(np.int64)
    first_state = np.cumsum(np.r_[0, iv[:-1]]).astype(np.int64)
    last_state = (np.cumsum(iv) - 1).astype(np.int64)
    s2c = m.state_to_classes[0].to(dev)
    idx = {int(v): i for i, v in enumerate(iv)}
    tmap = torch.from_numpy(np.array([idx[int(v)] for v in space.state_interval_frames])).to(dev)
    iv_t = torch.from_numpy(iv).to(dev)

    net = load_net(str(HEAD), dev)
    marg = np.load(str(HEAD) + ".marginal.npy")
    logmarg = torch.tensor(np.log(np.clip(marg, 1e-12, None)), dtype=torch.float64, device=dev)

    def features(act, emb, dev):
        got = emissions_beatonly(m, act)
        if got is None:
            return None
        lcd, dec, ff = got
        T = lcd.shape[0]
        F = {"em": lcd, "s2c": s2c, "dec": dec, "ff": ff, "T": T}
        if emb is not None:
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(emb, np.float32)).permute(1, 0).unsqueeze(0).to(dev)
                lq = torch.log_softmax(net(x).float(), 1)[0].T.double()
            if lq.shape[0] < ff + T:
                lq = torch.cat([lq, lq[-1:].repeat(ff + T - lq.shape[0], 1)], 0)
            F["tempo"] = (lq[ff:ff + T] - logmarg[None, :])[:, tmap]
        F["acf"] = acf_logratio(act[ff:ff + T], iv, args.tau, dev)[tmap]
        return F

    # ---------- training pool: CORPUS only ----------
    pool = []
    for ds in args.datasets.split(","):
        d = EMB / ds
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npz"))[:args.train_tracks // 5]:
            z = np.load(f, allow_pickle=True)
            ann = STORE / ds / "annotations" / "beats" / f"{f.stem}.beats"
            if not ann.exists():
                continue
            bt = np.array([float(l.split()[0]) for l in open(ann) if l.strip()])
            sf = int(z["start_frame"])
            bf = bt * FPS - sf
            act = 1.0 / (1.0 + np.exp(-np.asarray(z["beat"], np.float64)))
            Fz = features(act, z["emb"], dev)
            if Fz is None:
                continue
            # WINDOWING (two bugs fixed here).
            #  * the filter must be ff <= bf < ff+T. T is the length AFTER the ff crop, so
            #    "bf < T" selected the wrong beats for both the positive path and the mask.
            #  * the mining reference must cover ONLY the decoded span. Previously ref_sec used
            #    EVERY beat in the track while the decode covers a 1500-frame (30 s) window, so on a
            #    3-minute track recall was scored against ~6x too many reference beats: F came out
            #    artificially low and ~70% of ballroom/GTZAN tracks were mined as "wrong", which is
            #    implausible for corpora where Beat This + DBN scores ~0.95.
            bfc = bf[(bf >= Fz["ff"]) & (bf < Fz["ff"] + Fz["T"])] - Fz["ff"]
            if len(bfc) < 8:
                continue
            pos = build_path(bfc, Fz["T"], iv, first_state)
            if pos is None:
                continue
            negs = {}
            for k, v in make_negatives(bfc, rng).items():
                r = build_path(v, Fz["T"], iv, first_state)
                if r is not None:
                    negs[k] = r
            if negs:
                # ref_sec is relative to the WINDOW start (sf): state_path_to_events is called with
                # first_frame=ff and returns (ff+i)/FPS on that same origin.
                # CONSTRAINT MASK for the CRF numerator: at each annotated beat frame (+-tol) the
                # path must occupy a BEAT-class state. The numerator is then the best path
                # CONSISTENT WITH the annotation, recomputed under the current weights -- not a
                # hand-built path.
                # TOLERANCE MUST MATCH THE METRIC. F uses +-70 ms; at 50 fps that is +-3.5
                # frames. The mask previously used +-2 frames (+-40 ms), i.e. it constrained the
                # numerator MORE TIGHTLY than the metric and could exclude paths that would score a
                # perfect F. The numerator must be the best path the metric would ACCEPT.
                tol = int(round(0.070 * FPS))          # 3 frames at 50 fps
                bmask = np.zeros(Fz["T"], bool)
                for fr in np.rint(bfc).astype(int):
                    lo, hi = max(0, fr - tol), min(Fz["T"], fr + tol + 1)
                    bmask[lo:hi] = True
                pool.append({"F": Fz, "pos": pos, "negs": negs,
                             # origin = window start (sf), matching state_path_to_events(
                             # first_frame=ff) which returns (ff+i)/FPS; restricted to the span the
                             # decoder actually covers.
                             "ref_sec": (bfc + Fz["ff"]) / FPS,
                             "bmask": torch.as_tensor(bmask, device=dev)})
    heldout = None
    if args.train_on == "smc":
        pool = []
        sids = sorted(t for t in track_ids() if (SMC_EMB / f"{t}.npz").exists())
        rr = np.random.default_rng(0)
        rr.shuffle(sids)
        cut = int(args.smc_train_frac * len(sids))
        tr_ids, heldout = sorted(sids[:cut]), sorted(sids[cut:])
        for t in tr_ids:
            z = np.load(SMC_EMB / f"{t}.npz", allow_pickle=True)
            Fz = features(np.asarray(bthis_beat_prob(t), np.float64), z["emb"], dev)
            if Fz is None:
                continue
            bf = np.asarray(load_gt(t), float) * FPS          # SMC embeddings have no start_frame
            bfc = bf[(bf >= Fz["ff"]) & (bf < Fz["ff"] + Fz["T"])] - Fz["ff"]
            if len(bfc) < 8:
                continue
            pos = build_path(bfc, Fz["T"], iv, first_state)
            if pos is None:
                continue
            negs = {}
            for k, v in make_negatives(bfc, rng).items():
                r = build_path(v, Fz["T"], iv, first_state)
                if r is not None:
                    negs[k] = r
            if not negs:
                continue
            tol = int(round(0.070 * FPS))
            bmask = np.zeros(Fz["T"], bool)
            for fr in np.rint(bfc).astype(int):
                lo, hi = max(0, fr - tol), min(Fz["T"], fr + tol + 1)
                bmask[lo:hi] = True
            pool.append({"F": Fz, "pos": pos, "negs": negs,
                         "ref_sec": (bfc + Fz["ff"]) / FPS,
                         "bmask": torch.as_tensor(bmask, device=dev)})
        print(f"CRF training pool: {len(pool)} SMC tracks -- SMC-TUNED, evaluated on "
              f"{len(heldout)} held-out SMC tracks", flush=True)
    else:
        print(f"CRF training pool: {len(pool)} CORPUS tracks (SMC never used for fitting)",
              flush=True)
    if not pool:
        print("EMPTY POOL -- aborting"); return

    _w0 = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=dev)
    _w0 = _w0 * (2.0 / _w0.norm())          # start ON the sphere the projection maintains
    w = _w0.clone().requires_grad_(True)
    raw_lam = torch.tensor(float(np.log(args.init_lambda)), dtype=torch.float64,
                           device=dev, requires_grad=True)
    params = [w] if args.freeze_lambda else [w, raw_lam]
    if args.freeze_lambda:
        raw_lam.requires_grad_(False)
    opt = torch.optim.Adam(params, lr=args.lr)

    def to_t(p):
        # build_path already returns (path, tidx, bnd) with bnd = frames t whose t->t+1 crosses a
        # beat boundary, computed from the block construction. Re-deriving boundaries from tempo
        # INDEX CHANGES was the exact bug that biased the earlier contrastive run toward low lambda
        # (37 boundaries/track instead of 89): tempo self-transitions at a boundary are boundaries
        # too. Use theirs.
        path, ti, bnd = p
        return (torch.as_tensor(path, device=dev),
                torch.as_tensor(ti, device=dev),
                torch.as_tensor(bnd, device=dev))

    cache = [{**it, "pos_t": to_t(it["pos"]),
              "negs_t": {k: to_t(v) for k, v in it["negs"].items()}} for it in pool]

    beat_states = (s2c == 1)          # BEAT class in the corrected beat-only space

    def constrained_positive(it, log_T_det, w_det):
        """Numerator = best path that puts a BEAT-class state at every annotated beat.

        WHY THIS REPLACES THE HAND-BUILT PATH. build_path snaps annotation times onto the integer
        tempo grid with a resync-greedy construction, so its beats drift off the activation peaks and
        its emission score is genuinely lower than the model's own decodes. A numerator that loses
        under EVERY weighting gives a margin loss only one descent direction -- shrink all scores --
        which is exactly the observed collapse (w -> [0.118, 0.023, -0.007, 0.051], loss 22.7).
        Constraining the decode instead makes the numerator the best ANNOTATION-CONSISTENT path,
        which is the standard CRF/lattice-MMI numerator.
        """
        F = it["F"]
        em = w_det[0] * F["em"][:, s2c]
        if F.get("tempo") is not None:
            em = em + w_det[1] * F["tempo"]
        if F.get("acf") is not None:
            em = em + w_det[2] * F["acf"][None, :]
        em = em.clone()
        em[it["bmask"][:, None] & (~beat_states)[None, :]] = -1e9   # force a beat where annotated
        try:
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], w_det[3] * log_T_det, em,
                state_to_class=None, return_log_score=True)
        except Exception:
            return None
        pa = path.cpu().numpy()
        ti = tmap[path].cpu().numpy()
        bnd = np.nonzero(ti[1:] != ti[:-1])[0]
        return (torch.as_tensor(pa, device=dev), torch.as_tensor(ti, device=dev),
                torch.as_tensor(bnd, device=dev))

    def mine(log_T_det, w_det):
        """Decode with CURRENT weights; keep a decode as a negative ONLY if it is actually wrong.

        BUG THIS FIXES: the first version compared the decoded STATE PATH to the constructed
        positive for exact equality. The positive is a resync-greedy integer approximation of the
        annotation, so a decode can produce the SAME BEATS yet differ state-by-state -- mining
        duly reported 250/250 corpus tracks "decoded wrongly", on ballroom/GTZAN where beat
        tracking is largely solved. Every CORRECT decode was fed back as a negative, so the
        objective was training the annotation path to beat correct answers: the loss stayed ~10.9
        and w collapsed [1,0,0,1] -> [0.160, 0.030, 0.002, 0.154], which is the only way to shrink
        a margin loss whose positives are losing. Judge by F against the annotation instead.
        """
        n_wrong = 0
        for it in cache:
            F = it["F"]
            em = w_det[0] * F["em"][:, s2c]
            if F.get("tempo") is not None:
                em = em + w_det[1] * F["tempo"]
            if F.get("acf") is not None:
                em = em + w_det[2] * F["acf"][None, :]
            try:
                path, _ = m.dynamic_programs[0].viterbi(
                    m.log_initial_distributions[0], w_det[3] * log_T_det, em,
                    state_to_class=None, return_log_score=True)
            except Exception:
                continue
            pa = path.cpu().numpy()
            beats = state_path_to_events(pa, space, FPS, snap_to_activations=F["dec"],
                                         first_frame=F["ff"])["beats"]
            if ev(it["ref_sec"], beats)["F"] >= args.mine_max_f:
                it["negs_t"].pop("mined", None)          # a correct decode is NOT a negative
                continue
            ti = tmap[path].cpu().numpy()
            bnd = np.nonzero(ti[1:] != ti[:-1])[0]
            it["negs_t"]["mined"] = (torch.as_tensor(pa, device=dev),
                                     torch.as_tensor(ti, device=dev),
                                     torch.as_tensor(bnd, device=dev))
            n_wrong += 1
        return n_wrong

    for step in range(args.steps):
        lam = torch.exp(raw_lam)
        log_T = log_tempo_transition(iv_t, lam, mask_mode="none")
        if args.mine_every and step % args.mine_every == 0:
            nm_ = mine(log_T.detach(), w.detach())
            n_pos = 0
            for it in cache:
                cp = constrained_positive(it, log_T.detach(), w.detach())
                if cp is not None:
                    it["pos_t"] = cp
                    n_pos += 1
            print(f"  [mine] step {step}: {nm_}/{len(cache)} decoded wrongly; "
                  f"{n_pos} constrained numerators refreshed", flush=True)
        opt.zero_grad()
        tot, nb = 0.0, 0
        for it in [cache[i] for i in rng.choice(len(cache), size=min(8, len(cache)), replace=False)]:
            sp = crf_score(*it["pos_t"], it["F"], log_T, w)
            sn = [crf_score(*v, it["F"], log_T, w) for v in it["negs_t"].values()]
            if not sn:
                continue
            # MARGIN: add args.margin to every negative so the loss keeps pressing after the true
            # path merely wins. Without it the problem is separable (G3: loss 1e-4, direction
            # stalls) and the only remaining descent direction is inflating ||w||.
            sc = torch.stack([sp] + [x + args.margin for x in sn])
            if not torch.isfinite(sc).all():
                continue
            loss = -(sc[0] - torch.logsumexp(sc, 0))
            loss.backward(retain_graph=True)
            tot += float(loss); nb += 1
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        # UNIT-NORM PROJECTION, replacing the L2 penalty.
        # L2 did not close the scale escape route -- it REWARDED it. With a margin added to the
        # negatives, driving ||w|| -> 0 sends every score to ~0, so the loss settles on a fixed
        # floor that the L2 term is then happy to sit at. Measured: w collapsed to
        # [+0.102, -0.004, -0.025, -0.023] with w_trans NEGATIVE (i.e. rewarding improbable tempo
        # jumps), loss parked at 1.15, and the corpus wrong-decode count ROSE 175/250 -> 238/250.
        # The decode argmax is invariant to a positive rescaling of the whole weight vector -- which
        # is exactly why G3 has to score recovery by COSINE similarity -- so fixing the norm removes
        # the degenerate direction without changing anything the decoder can see.
        with torch.no_grad():
            n_ = w.norm()
            if float(n_) > 1e-8:
                w.mul_(args.wnorm / n_)
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}  loss {tot/max(nb,1):.4f}  "
                  f"w=[{', '.join(f'{float(x):+.3f}' for x in w)}]  "
                  f"lambda {float(torch.exp(raw_lam)):.2f}", flush=True)

    wl = [float(x) for x in w.detach()]
    lam_l = float(torch.exp(raw_lam))
    print(f"\nLEARNED  w_em {wl[0]:+.3f}  w_tempo {wl[1]:+.3f}  w_acf {wl[2]:+.3f}  "
          f"w_trans {wl[3]:+.3f}   lambda {lam_l:.2f}", flush=True)

    # ---------- evaluate on SMC (never fitted) ----------
    # CONTROL BUG (found by scripts/diagnostics/gate_crf_plumbing.py): the null arm was previously decoded with
    # the LEARNED lambda, so "null" was not the baseline at all -- it read F 0.6215 / CMLt 0.5720
    # instead of the true 0.6141 / 0.5685. Each arm now gets its own kernel.
    logT_learned = log_tempo_transition(iv_t, torch.tensor(lam_l, dtype=torch.float64, device=dev),
                                        mask_mode="none").detach()
    logT_null = log_tempo_transition(iv_t, torch.tensor(100.0, dtype=torch.float64, device=dev),
                                     mask_mode="none").detach()
    wt = torch.tensor(wl, dtype=torch.float64, device=dev)
    base_w = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64, device=dev)
    res = {"null": {}, "crf": {}}
    eval_ids = heldout if heldout is not None else track_ids()
    for t in eval_ids:
        fz = SMC_EMB / f"{t}.npz"
        if not fz.exists():
            continue
        z = np.load(fz, allow_pickle=True)
        Fz = features(np.asarray(bthis_beat_prob(t), np.float64), z["emb"], dev)
        if Fz is None:
            continue
        gt = load_gt(t)
        for nm, ww, log_T in (("null", base_w, logT_null), ("crf", wt, logT_learned)):
            em = ww[0] * Fz["em"][:, s2c]
            if Fz.get("tempo") is not None:
                em = em + ww[1] * Fz["tempo"]
            if Fz.get("acf") is not None:
                em = em + ww[2] * Fz["acf"][None, :]
            path, _ = m.dynamic_programs[0].viterbi(
                m.log_initial_distributions[0], ww[3] * log_T, em,
                state_to_class=None, return_log_score=True)
            res[nm][t] = ev(gt, state_path_to_events(
                path.cpu().numpy(), space, FPS,
                snap_to_activations=Fz["dec"], first_frame=Fz["ff"])["beats"])
    ids = sorted(set(res["null"]) & set(res["crf"]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ids])) for k in METRICS}
    tag = ("HELD-OUT SMC (weights are SMC-TUNED)" if heldout is not None
           else "ALL SMC (never used for fitting)")
    print(f"\n{tag}  n={len(ids)}")
    for nm in ("null", "crf"):
        print(f"  {nm:<6}" + "".join(f"{A(res[nm])[k]:>9.4f}" for k in METRICS))
    print("  paired crf - null:")
    for k in METRICS:
        d = np.array([res["crf"][t][k] - res["null"][t][k] for t in ids])
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"    {k} {d.mean():+.4f} (p={p:.3g})")
    json.dump({"weights": wl, "lambda": lam_l, "n": len(ids),
               "summary": {nm: A(res[nm]) for nm in res}}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
