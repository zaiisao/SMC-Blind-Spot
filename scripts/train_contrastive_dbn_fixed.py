"""Contrastive DBN lambda training -- BUG-FIXED scorer.

Fixes vs scripts/train_contrastive_dbn.py:
 (b) VALID bar-pointer paths. Paths are now generated block-wise (a tempo of i frames owns
     exactly i states, pointer advances +1/frame, wraps only at the block end), the same
     construction as DBN2016.annotated_state_path. Legality is ASSERTED for every path.
     All paths (positive and negatives) cover the SAME frame span [0, T), so their emission
     sums are comparable.
 (c) BOUNDARY TERM at EVERY beat boundary, including tempo self-transitions log_T[v,v]
     (which reward holding tempo and therefore reward HIGH lambda), not only where the tempo
     index changes.
 (e) LAMBDA-INDEPENDENT SUPPORT. madmom's hard threshold mask (np.spacing(1)) forbids more
     tempo pairs as lambda grows, which made 57/129 tracks -inf at lambda=100 and silently
     dropped them. The training objective uses the full (unmasked) row-normalised kernel, so
     the objective is the SAME function at every lambda. --mask keeps the old behaviour.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))
from _harness import FPS, track_ids, load_gt, ev, bthis_beat_prob
from rungs.bar_pointer.readout import state_path_to_events
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly

METRICS = ("F", "CMLt", "AMLt", "CMLc", "AMLc")


def log_tempo_transition(iv, lam, mask_mode="none", threshold=float(np.spacing(1))):
    ivd = iv.to(torch.float64)
    d = torch.abs(ivd[None, :] / ivd[:, None] - 1.0)
    logits = -lam * d
    if mask_mode == "madmom":
        with torch.no_grad():
            mask = torch.exp(-lam.detach() * d) <= threshold
        logits = logits.masked_fill(mask, float("-inf"))
    return logits - torch.logsumexp(logits, dim=1, keepdim=True)


def build_path(beat_frames, T, iv_grid, first_state):
    """Valid bar-pointer path over [0, T). Returns (path, tidx, bnd) or None.

    bnd = frames t such that t -> t+1 crosses a beat boundary (both inside the span).
    """
    min_iv, max_iv = int(iv_grid[0]), int(iv_grid[-1])
    snap = lambda g: int(iv_grid[int(np.argmin(np.abs(iv_grid - g)))])
    bf = np.asarray(beat_frames, float)
    bf = bf[np.isfinite(bf)]
    if len(bf) < 3:
        return None
    bf = np.sort(bf)
    # --- resync-greedy integer beat grid: each gap is an integer in the tempo grid ---
    starts = [int(round(bf[0]))]
    ivs = []
    cur = starts[0]
    for b in bf[1:]:
        gap = int(round(b)) - cur
        if gap <= 0:
            continue
        while gap > max_iv:                     # too long: emit max-tempo blocks
            ivs.append(max_iv); cur += max_iv; starts.append(cur); gap -= max_iv
        gap = snap(max(min_iv, min(max_iv, gap)))
        ivs.append(gap); cur += gap; starts.append(cur)
    if len(ivs) < 2:
        return None
    # --- extend backwards to frame 0 and forwards to T with the edge tempi ---
    while starts[0] > 0:
        starts.insert(0, starts[0] - ivs[0]); ivs.insert(0, ivs[0])
    while starts[-1] < T:
        ivs.append(ivs[-1]); starts.append(starts[-1] + ivs[-1])
    starts = np.array(starts); ivs = np.array(ivs)
    path = np.full(T, -1, np.int64); tidx = np.full(T, -1, np.int64); bnd = []
    for s, iv in zip(starts[:-1], ivs):
        v = int(np.searchsorted(iv_grid, iv))
        lo, hi = max(0, s), min(T, s + iv)
        if hi <= lo:
            continue
        path[lo:hi] = first_state[v] + np.arange(lo - s, hi - s)
        tidx[lo:hi] = v
        if 0 < s + iv <= T - 1 and s + iv - 1 >= 0:
            bnd.append(s + iv - 1)              # frame t with t+1 the next beat
    if (path < 0).any():
        return None
    return path, tidx, np.array(bnd, np.int64)


def assert_valid(path, iv_grid, first_state, last_state, tag=""):
    bad = 0
    for t in range(len(path) - 1):
        a, b = path[t], path[t + 1]
        if a in last_state_set:
            if b not in first_state_set:
                bad += 1
        elif b != a + 1:
            bad += 1
    return bad


def path_score(path_t, ti_t, bnd_t, lcd, s2c, log_T, beta=None, tempo_feat=None):
    ar = torch.arange(len(path_t), device=lcd.device)
    em = lcd[ar, s2c[path_t]]
    if beta is not None and tempo_feat is not None:
        em = em + beta * tempo_feat[ar, ti_t]
    sc = em.sum()
    if len(bnd_t):
        sc = sc + log_T[ti_t[bnd_t], ti_t[bnd_t + 1]].sum()
    return sc


def make_negatives(bf, rng, n_drift=2):
    bf = np.asarray(bf, float); negs = {}
    if len(bf) >= 3:
        mid = (bf[:-1] + bf[1:]) / 2.0
        negs["double"] = np.sort(np.concatenate([bf, mid]))
        negs["half"] = bf[::2]
        ibi = np.diff(bf)
        negs["phase"] = bf[:-1] + ibi / 2.0
        for j in range(n_drift):
            w = np.cumsum(rng.normal(0, 0.02, size=len(bf)))
            negs[f"drift{j}"] = bf + w * np.median(ibi)
    return {k: v for k, v in negs.items() if len(v) >= 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-lambda", type=int, default=16)
    ap.add_argument("--init-lambda", type=float, default=100.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-bpm", type=float, default=30.0)
    ap.add_argument("--mask", default="none", choices=["none", "madmom"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--learn-beta", action="store_true",
                    help="also learn beta, the weight on the q_theta tempo log-ratio")
    ap.add_argument("--head", default="/disk1/jaehoon/tempo_obs_head.pt")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--out", default="/tmp/overnight/contrastive_fixed.json")
    args = ap.parse_args()
    dev = torch.device(args.device); rng = np.random.default_rng(0)

    m = make_beatonly_dbn(args.min_bpm, args.obs_lambda, 100, args.device)
    space = m.state_spaces[0]
    iv_grid = space.interval_frames.astype(np.int64)
    first_state = np.cumsum(np.r_[0, iv_grid[:-1]]).astype(np.int64)
    last_state = (np.cumsum(iv_grid) - 1).astype(np.int64)
    global first_state_set, last_state_set
    first_state_set = set(first_state.tolist()); last_state_set = set(last_state.tolist())
    s2c = m.state_to_classes[0].to(dev)
    iv_t = torch.from_numpy(iv_grid).to(dev)
    print(f"tempo grid: {len(iv_grid)} tempi {iv_grid[0]}..{iv_grid[-1]} frames", flush=True)

    ids = track_ids(); rng.shuffle(ids)
    split = int(0.6 * len(ids))
    train_ids, test_ids = sorted(ids[:split]), sorted(ids[split:])
    print(f"train {len(train_ids)} held-out {len(test_ids)}", flush=True)

    def prep(tid):
        got = emissions_beatonly(m, bthis_beat_prob(tid))
        if got is None:
            return None
        lcd, dec, ff = got
        T = lcd.shape[0]
        gt = np.asarray(load_gt(tid), float) * FPS - ff
        pos = build_path(gt, T, iv_grid, first_state)
        if pos is None:
            return None
        negs = {}
        for k, v in make_negatives(gt, rng).items():
            r = build_path(v, T, iv_grid, first_state)
            if r is not None:
                negs[k] = r
        if not negs:
            return None
        return {"lcd": lcd, "pos": pos, "negs": negs, "dec": dec, "ff": ff, "T": T}

    cache, nbad, nchk = {}, 0, 0
    for tid in train_ids:
        c = prep(tid)
        if c is None:
            continue
        for p in [c["pos"]] + list(c["negs"].values()):
            nbad += assert_valid(p[0], iv_grid, first_state, last_state); nchk += 1
        cache[tid] = c
    print(f"prepared {len(cache)} training tracks; illegal moves over {nchk} paths: {nbad}",
          flush=True)
    net = tfeat = None
    if args.learn_beta:
        import train_tempo_observation as TT
        from decode_tempo_observation import q_for_embeddings
        TT.GRID = TT.tempo_grid(); TT.GRID_BPM = np.sort(60.0 * FPS / TT.GRID)
        net = TT.load_net(args.head, dev)
        marg = np.load(args.head + ".marginal.npy")
        logm = np.log(marg)[None, :]
        ivf = torch.from_numpy(iv_grid.astype(np.float64)).to(dev)
        drop = []
        for tid, c in cache.items():
            f = Path(f"/disk1/jaehoon/smc_bt_embeddings/{tid}.npz")
            if not f.exists():
                drop.append(tid); continue
            emb = np.load(f)["emb"]
            lq = q_for_embeddings(net, emb, dev)[c["ff"]:c["ff"] + c["T"]] - logm
            if lq.shape[0] != c["T"]:
                drop.append(tid); continue
            c["tf"] = torch.tensor(lq, dtype=torch.float64, device=dev) / ivf[None, :]
        for tid in drop:
            cache.pop(tid)
        print(f"tempo features for {len(cache)} tracks ({len(drop)} dropped)", flush=True)

    # move to torch
    for c in cache.values():
        c["pos"] = tuple(torch.as_tensor(x, device=dev) for x in c["pos"])
        c["negs"] = {k: tuple(torch.as_tensor(x, device=dev) for x in v)
                     for k, v in c["negs"].items()}
    keys = list(cache)

    def objective(lam_val, mask_mode):
        lam = torch.tensor(float(lam_val), device=dev, dtype=torch.float64)
        log_T = log_tempo_transition(iv_t, lam, mask_mode)
        L, n, nf = [], 0, 0
        for tid in keys:
            c = cache[tid]
            sp = path_score(*c["pos"], c["lcd"], s2c, log_T)
            sn = [path_score(*v, c["lcd"], s2c, log_T) for v in c["negs"].values()]
            sc = torch.stack([sp] + sn)
            if not torch.isfinite(sc).all():
                nf += 1; continue
            L.append(float(-(sc[0] - torch.logsumexp(sc, 0)))); n += 1
        return float(np.mean(L)), n, nf

    sweep = {}
    if args.sweep:
        for lv in [3, 5, 7.55, 9.02, 11.45, 20, 40, 70, 100, 150, 300, 1000]:
            for mm_ in ["none", "madmom"]:
                l, n, nf = objective(lv, mm_)
                sweep[f"{lv}_{mm_}"] = {"loss": l, "n": n, "nonfinite": nf}
                print(f"  sweep lam={lv:<7} mask={mm_:<7} loss={l:.4f} n={n} nonfinite={nf}",
                      flush=True)

    keys = list(cache)
    raw_lam = torch.tensor(float(np.log(args.init_lambda)), device=dev,
                           dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(0.0, device=dev, dtype=torch.float64, requires_grad=True)
    params = [raw_lam] + ([beta] if args.learn_beta else [])
    opt = torch.optim.Adam(params, lr=args.lr)
    for step in range(args.steps):
        lam = torch.exp(raw_lam)
        log_T = log_tempo_transition(iv_t, lam, args.mask)
        opt.zero_grad(); tot, nb = 0.0, 0
        for tid in rng.choice(keys, size=min(8, len(keys)), replace=False):
            c = cache[tid]
            bt, tf = (beta, c["tf"]) if args.learn_beta else (None, None)
            sc = torch.stack([path_score(*c["pos"], c["lcd"], s2c, log_T, bt, tf)]
                             + [path_score(*v, c["lcd"], s2c, log_T, bt, tf)
                                for v in c["negs"].values()])
            if not torch.isfinite(sc).all():
                continue
            loss = -(sc[0] - torch.logsumexp(sc, 0))
            if not torch.isfinite(loss):
                continue
            loss.backward(retain_graph=True); tot += float(loss); nb += 1
        if raw_lam.grad is not None and torch.isfinite(raw_lam.grad).all():
            torch.nn.utils.clip_grad_norm_(params, 5.0); opt.step()
        else:
            opt.zero_grad()
        if (step + 1) % 25 == 0:
            print(f"  beta {float(beta):+.4f}" if args.learn_beta else "", end="", flush=True)
            print(f"  step {step+1} loss {tot/max(nb,1):.4f} lambda {float(torch.exp(raw_lam)):.3f}",
                  flush=True)
    lam_learned = float(torch.exp(raw_lam))
    print(f"\nlearned transition_lambda = {lam_learned:.3f}", flush=True)

    beta_learned = float(beta)
    if args.learn_beta:
        print(f"learned beta = {beta_learned:.4f}", flush=True)
    res = {}
    arms = [("shipped100", 100.0, 0.0), ("learned", lam_learned, 0.0), ("mle11", 11.45, 0.0)]
    if args.learn_beta:
        arms += [("learned+beta", lam_learned, beta_learned),
                 ("shipped100+beta", 100.0, beta_learned)]
    for tag, lv, bv in arms:
        mm2 = make_beatonly_dbn(args.min_bpm, args.obs_lambda, lv, args.device)
        r = {}
        for tid in test_ids:
            got = emissions_beatonly(mm2, bthis_beat_prob(tid))
            if got is None:
                continue
            lcd, dec, ff = got
            em, stc = lcd, mm2.state_to_classes[0]
            if bv != 0.0:
                f = Path(f"/disk1/jaehoon/smc_bt_embeddings/{tid}.npz")
                if not f.exists():
                    continue
                lq = q_for_embeddings(net, np.load(f)["emb"], dev)[ff:ff + lcd.shape[0]] \
                     - np.log(np.load(args.head + ".marginal.npy"))[None, :]
                if lq.shape[0] != lcd.shape[0]:
                    continue
                tfd = torch.tensor(lq, dtype=torch.float64, device=dev)
                sivf = torch.tensor(mm2.state_spaces[0].state_interval_frames,
                                    dtype=torch.float64, device=dev)
                idx2 = {int(v): i for i, v in enumerate(iv_grid)}
                tm = torch.from_numpy(np.array([idx2[int(v)] for v in
                                    mm2.state_spaces[0].state_interval_frames])).to(dev)
                em = lcd[:, mm2.state_to_classes[0]] + bv * (tfd[:, tm] / sivf.unsqueeze(0))
                stc = None
            path, _ = mm2.dynamic_programs[0].viterbi(
                mm2.log_initial_distributions[0], mm2.log_tempo_transition, em,
                state_to_class=stc, return_log_score=True)
            r[tid] = ev(load_gt(tid), state_path_to_events(
                path.cpu().numpy(), mm2.state_spaces[0], FPS,
                snap_to_activations=dec, first_frame=ff)["beats"])
        res[tag] = r
        print(f"  decoded {tag} (lambda={lv:.3f})", flush=True)
    ok = sorted(set.intersection(*[set(r) for r in res.values()]))
    A = lambda r: {k: float(np.mean([r[t][k] for t in ok])) for k in METRICS}
    print(f"\nHELD-OUT n={len(ok)}")
    print(f"{'arm':<14}" + "".join(f"{k:>9}" for k in METRICS))
    for tag in res:
        print(f"{tag:<14}" + "".join(f"{A(res[tag])[k]:>9.4f}" for k in METRICS))
    stats = {}
    print("\nlearned minus shipped100, paired:")
    for k in METRICS:
        d = np.array([res["learned"][t][k] - res["shipped100"][t][k] for t in ok])
        p = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
        stats[k] = {"delta": float(d.mean()), "p": p}
        print(f"  {k} {d.mean():+.4f} (p={p:.3g})")
    json.dump({"lambda_learned": lam_learned, "beta_learned": beta_learned, "n_heldout": len(ok), "sweep": sweep,
               "summary": {t: A(res[t]) for t in res}, "paired": stats,
               "train_ids": train_ids, "test_ids": test_ids}, open(args.out, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
