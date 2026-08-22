"""q_theta(phidot | h_t): a tempo observation model over the DBN's 60-point tempo grid.

Implements docs/spec/SPECIFICATION.md sec.4 (the missing explicit tempo likelihood factor):

    log p(y_t | z_t) = log p(y_t | class(phi_t)) + beta * [ log q_theta(phidot_t | h_t) - log p(phidot_t) ]

Design decisions forced by the spec:
  R1 (no double counting): the denominator is the HEAD'S OWN MARGINAL, estimated by averaging
     q_theta over held-out corpus frames -- NOT the label prior. A head trained with any implicit
     or explicit prior-correction is already corrected once; dividing by the label prior again
     injected 5.47 nats/tempo last time.
  R4 (soft targets): smoothed interpolated histogram over the local inter-beat interval, widened
     +/-1 and +/-2 BPM at weights 0.5 / 0.25, resampled onto the tempo grid.
  fold honesty / splits: q_theta sees corpus audio ONLY (gtzan/ballroom/hainsworth/beatles/asap,
     plus time-stretched copies of the same corpus). SMC is never trained on and is only decoded.

Subcommands:
  cache  -- build the frame-aligned (embedding, soft-target) memmaps
  train  -- fit q_theta
  eval   -- SMC tempo accuracy (G1), tempo-neutrality (G2), head marginal
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))

FPS = 50.0
STORE = Path("/disk1/jaehoon/dataset_store/beat_this_annotations")
EMB = Path("/disk1/jaehoon/bt_emb")
EMB_STR = Path("/disk1/jaehoon/bt_emb_stretch")
SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
CACHE = Path("/disk1/jaehoon/tempo_obs_cache")
DATASETS = ["gtzan", "ballroom", "hainsworth", "beatles", "asap"]


def tempo_grid():
    from rungs.bar_pointer.state_space import BarPointerStateSpace
    s = BarPointerStateSpace(fps=FPS, min_bpm=30.0, max_bpm=215.0, beats_per_bar=1,
                             observation_lambda=16, num_tempi=60)
    return s.interval_frames.astype(np.float64)


GRID = None  # filled in main
GRID_BPM = None


def load_beats(ds, tid):
    p = STORE / ds / "annotations" / "beats" / f"{tid}.beats"
    if not p.exists():
        return None
    t = []
    for line in open(p):
        pp = line.split()
        if pp:
            t.append(float(pp[0]))
    return np.asarray(t, float)


def soft_target_bpm(bpm0):
    """R4 soft target: interpolated histogram on the tempo GRID for a local tempo of bpm0 BPM.

    Mass 1.0 at bpm0, 0.5 at bpm0 +/- 1 BPM, 0.25 at bpm0 +/- 2 BPM; each delta is split
    linearly between the two neighbouring grid points (GRID_BPM is descending in interval,
    ascending in BPM after flip). Returns a length-60 vector summing to 1, or None if bpm0
    falls outside the grid.
    """
    lo, hi = GRID_BPM[0], GRID_BPM[-1]
    out = np.zeros(len(GRID_BPM))
    tot = 0.0
    for db, w in ((0.0, 1.0), (1.0, 0.5), (-1.0, 0.5), (2.0, 0.25), (-2.0, 0.25)):
        b = bpm0 + db
        if b < lo or b > hi:
            continue
        j = int(np.searchsorted(GRID_BPM, b) - 1)
        j = min(max(j, 0), len(GRID_BPM) - 2)
        f = (b - GRID_BPM[j]) / (GRID_BPM[j + 1] - GRID_BPM[j])
        out[j] += w * (1 - f)
        out[j + 1] += w * f
        tot += w
    if tot < 0.9:          # centre itself out of range
        return None
    return out / out.sum()


def frame_targets(T, beat_frames, start_frame=0):
    """Per-frame soft targets from beat times already converted to FRAMES on this file's grid.

    A frame lying in [beat_k, beat_{k+1}) gets the soft target of the LOCAL interval
    ibi_k = beat_{k+1} - beat_k. Frames outside any labelled interval are masked out (-1).
    Beat frames are beat CENTRES; interval boundaries use ceil(), matching
    scripts/training/build_tempo_labels.py and the oracle injection in exp_oracle_tempo_beatonly.py.
    """
    fr = beat_frames - start_frame
    tgt = np.zeros((T, len(GRID_BPM)), np.float16)
    mask = np.zeros(T, bool)
    cache = {}
    for k in range(len(fr) - 1):
        a, b = int(np.ceil(fr[k])), int(np.ceil(fr[k + 1]))
        a, b = max(0, a), min(T, b)
        if b <= a:
            continue
        ibi = fr[k + 1] - fr[k]
        if ibi <= 0:
            continue
        bpm = 60.0 * FPS / ibi
        key = round(bpm, 2)
        if key not in cache:
            cache[key] = soft_target_bpm(bpm)
        v = cache[key]
        if v is None:
            continue
        tgt[a:b] = v[::-1]      # BPM-ascending -> GRID (interval-ascending) order
        mask[a:b] = True
    return tgt, mask


# --------------------------------------------------------------------------- cache
def build_cache():
    CACHE.mkdir(parents=True, exist_ok=True)
    items = []           # (path, kind, ds, tid, stretch)
    for ds in DATASETS:
        for f in sorted((EMB / ds).glob("*.npz")):
            items.append((f, "plain", ds, f.stem, 1.0))
        for f in sorted((EMB_STR / ds).glob("*__s*.npz")):
            tid = f.stem.split("__s")[0]
            items.append((f, "stretch", ds, tid, float(f.stem.split("__s")[1])))
    print(f"candidate files: {len(items)}  "
          f"(plain {sum(1 for i in items if i[1]=='plain')}, "
          f"stretch {sum(1 for i in items if i[1]=='stretch')})", flush=True)

    embs, tgts, masks, meta = [], [], [], []
    off = 0
    t0 = time.time()
    for n, (f, kind, ds, tid, s) in enumerate(items):
        try:
            z = np.load(f, allow_pickle=True)
            e = z["emb"]
        except Exception as ex:
            print(f"  SKIP unreadable {f.name}: {ex}", flush=True)
            continue
        T = e.shape[0]
        if kind == "plain":
            bt = load_beats(ds, tid)
            if bt is None or len(bt) < 3:
                continue
            bf = bt * FPS
            sf = int(z["start_frame"])
        else:
            bt = np.asarray(z["beat_times"], float)   # ALREADY scaled by s -- do not scale again
            if len(bt) < 3:
                continue
            bf = bt * FPS
            sf = 0
        tg, mk = frame_targets(T, bf, sf)
        if mk.sum() < 50:
            continue
        embs.append(np.ascontiguousarray(e, np.float16))
        tgts.append(tg)
        masks.append(mk)
        meta.append({"ds": ds, "tid": tid, "kind": kind, "stretch": s,
                     "fold": int(z["fold"]), "off": off, "T": T})
        off += T
        if n % 500 == 0:
            print(f"  {n}/{len(items)}  frames={off:,}  {time.time()-t0:.0f}s", flush=True)
    E = np.concatenate(embs); del embs
    Tg = np.concatenate(tgts); del tgts
    Mk = np.concatenate(masks)
    print(f"cached {len(meta)} tracks, {E.shape[0]:,} frames, {Mk.sum():,} labelled", flush=True)
    np.save(CACHE / "emb.npy", E)
    np.save(CACHE / "tgt.npy", Tg)
    np.save(CACHE / "mask.npy", Mk)
    json.dump({"meta": meta, "grid": GRID.tolist()}, open(CACHE / "meta.json", "w"))


# --------------------------------------------------------------------------- model
def make_model(nout):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(s, c, d):
            super().__init__()
            s.c1 = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d)
            s.c2 = nn.Conv1d(c, c, 1)
            s.n1 = nn.GroupNorm(8, c)
            s.n2 = nn.GroupNorm(8, c)
            s.act = nn.GELU()

        def forward(s, x):
            h = s.act(s.n1(s.c1(x)))
            h = s.act(s.n2(s.c2(h)))
            return x + h

    class Net(nn.Module):
        def __init__(s, c=256):
            super().__init__()
            s.inp = nn.Conv1d(512, c, 1)
            s.blocks = nn.ModuleList([Block(c, d) for d in (1, 2, 4, 8, 16, 32, 64, 1)])
            s.out = nn.Conv1d(c, nout, 1)

        def forward(s, x):            # x: [B, 512, T]
            h = s.inp(x)
            for b in s.blocks:
                h = b(h)
            return s.out(h)           # [B, nout, T] logits

    return Net()


def octave_bandmask(grid, band_log2=0.15):
    """[V, V] bool: row t marks every grid bin within +/-band_log2 of t's x2 or /2 partner.

    WHY A BAND AND NOT A BIN -- this is the bug that cost us the octave route. The v1 penalty pushed
    mass off exactly ONE bin. The tempo grid step is 0.0481 log2 (60 bins over 14..100 frames) while
    the octave-error METRIC counts +/-0.15 log2 = +/-3.12 bins around the partner, so probability
    displaced by a single bin STILL SCORED AS AN OCTAVE ERROR: the penalty moved octave-band mass by
    0.4% (0.1469 -> 0.1473). The measured "failure" was a correctly measured null of a loss that did
    nothing. Penalising the band the metric actually scores gives -24% to -38.8% relative octave
    error, 8/8 paired seeds, p=0.008, on two independent corpus splits
    (scripts/diagnostics/sandbox_octave_loss_v2.py).
    """
    lg = np.log2(np.asarray(grid, dtype=np.float64))
    M = np.zeros((len(lg), len(lg)), dtype=bool)
    for i, c in enumerate(lg):
        for k in (-1.0, 1.0):
            M[i] |= np.abs(lg - (c + k)) <= band_log2
    return M


# torch/nn are imported lazily inside the command functions in this file (make_model, train), so
# the module-level classes below need their own imports.
import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber, ported verbatim from PESTO (SonyCSLParis/pesto-full, LGPL-3.0),
    src/losses/equivariance.py."""

    def __init__(self, tau: float):
        super().__init__()
        self.register_buffer("tau", torch.tensor(tau), persistent=False)

    def forward(self, x):
        x = x.abs()
        return torch.where(x.le(self.tau), x ** 2 / 2,
                           self.tau ** 2 / 2 + self.tau * (x - self.tau))


class PestoEquivariance(nn.Module):
    """PESTO's transposition-equivariance objective, adapted from pitch to tempo.

    PESTO (Riou et al., ISMIR 2023 best paper) enforces that shifting the input transposes the
    output by the known amount. Gagnere, Essid & Peeters (ICASSP 2024) adapt exactly this to tempo
    by replacing pitch-shift views with TIME-STRETCH views, and report it beating previous SSL
    tempo approaches.

    Their form (src/losses/equivariance.py), which we follow:
        z_i = q_i . weights                       # plain expectation, NO temperature
        L   = Huber(z2/z1 - r) + Huber(z1/z2 - 1/r), averaged   # RATIO residual, symmetrised

    Two deliberate differences from our earlier attempt, which failed:
      * we used a squared error on a LOG-space soft-argmax with temperature 30; PESTO uses a Huber
        on the ratio of plain expectations. Sharper readouts have worse-conditioned gradients and no
        outlier robustness.
      * PESTO symmetrises (both 1->2 and 2->1). Ours was one-directional.

    WEIGHTS. PESTO's bins are geometric (semitones), so its weights are value**powers. Our tempo
    grid is INTEGER BEAT INTERVALS (14..100 frames), which is not geometric, so the general form of
    the same projection is weights = the physical quantity of each bin, i.e. the interval itself.
    Stretching audio by s multiplies the beat interval by s, so the target ratio is exactly s2/s1.
    """

    def __init__(self, interval_frames, tau: float = 0.25):
        super().__init__()
        self.register_buffer("weights",
                             torch.tensor(np.asarray(interval_frames), dtype=torch.float32),
                             persistent=False)
        self.loss_fn = HuberLoss(tau)

    def project(self, q):                     # q: [..., V] probabilities -> [...] expected interval
        return (q * self.weights).sum(-1)

    def forward(self, q1, q2, ratio):
        z1 = self.project(q1).clamp(min=1e-6)
        z2 = self.project(q2).clamp(min=1e-6)
        r = torch.as_tensor(ratio, dtype=z1.dtype, device=z1.device)
        return 0.5 * (self.loss_fn(z2 / z1 - r).mean() + self.loss_fn(z1 / z2 - 1.0 / r).mean())


def train(args):
    import torch
    dev = torch.device(args.device)
    meta = json.load(open(CACHE / "meta.json"))["meta"]
    E = np.load(CACHE / "emb.npy", mmap_mode="r")
    Tg = np.load(CACHE / "tgt.npy", mmap_mode="r")
    Mk = np.load(CACHE / "mask.npy", mmap_mode="r")
    print(f"cache: {len(meta)} tracks, {E.shape[0]:,} frames", flush=True)

    # split by TRACK ID (not fold): all stretched copies of a track land on the same side, and
    # fold 0 is heavily over-represented (2134/5862 files) so a fold split wastes 36% of the data.
    # SMC is the real test set and is never in this cache at all.
    import hashlib
    isval = lambda m: int(hashlib.md5(m["tid"].encode()).hexdigest(), 16) % 10 == 0
    val = [m for m in meta if isval(m)]
    tr = [m for m in meta if not isval(m)]
    print(f"train {len(tr)} tracks / val(fold0 corpus) {len(val)} tracks", flush=True)
    print("  train by kind: " + str({k: sum(1 for m in tr if m['kind'] == k)
                                     for k in ('plain', 'stretch')}), flush=True)

    W = args.win
    net = make_model(len(GRID)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    rng = np.random.default_rng(0)
    scaler = torch.amp.GradScaler("cuda")

    def batch(pool):
        xs, ys, ms = [], [], []
        for _ in range(args.bs):
            m = pool[rng.integers(len(pool))]
            T = m["T"]
            if T <= W:
                a = 0
                pad = W - T
            else:
                a = int(rng.integers(0, T - W))
                pad = 0
            o = m["off"] + a
            L = min(W, T - a)
            x = np.zeros((W, 512), np.float32); x[:L] = E[o:o + L]
            y = np.zeros((W, len(GRID)), np.float32); y[:L] = Tg[o:o + L]
            k = np.zeros(W, np.float32); k[:L] = Mk[o:o + L]
            xs.append(x); ys.append(y); ms.append(k)
        x = torch.from_numpy(np.stack(xs)).permute(0, 2, 1).to(dev, non_blocking=True)
        y = torch.from_numpy(np.stack(ys)).permute(0, 2, 1).to(dev, non_blocking=True)
        k = torch.from_numpy(np.stack(ms)).to(dev, non_blocking=True)
        return x, y, k

    bandmask_t = torch.from_numpy(octave_bandmask(GRID, args.oct_band)).to(dev)

    # ---- PESTO-style time-stretch equivariance (Gagnere/Essid/Peeters ICASSP 2024 adapt PESTO's
    # transposition-equivariance to tempo by swapping pitch-shift views for TIME-STRETCH views).
    # Our cache already holds the same track at several stretch factors, so the interval ratio is
    # known exactly and needs no annotation.
    equiv = PestoEquivariance(GRID, tau=args.equiv_tau).to(dev) if args.equiv_w > 0 else None
    pair_index = []
    if equiv is not None:
        from collections import defaultdict
        byt = defaultdict(list)
        for mm in tr:                                   # TRAIN split only
            byt[(mm["ds"], mm["tid"])].append(mm)
        for k, v in byt.items():
            if len(v) >= 2:
                pair_index.append(v)
        print(f"  equivariance: {len(pair_index)} train tracks with >=2 stretch factors", flush=True)

    def equiv_batch():
        """Two views of the SAME track at different stretch factors; target ratio = s2/s1.

        Stretching audio by s multiplies the beat INTERVAL by s, and the projection is an expected
        interval, so the ratio the loss enforces is exactly s2/s1. Crops are drawn independently:
        tempo is near-constant within a track, and requiring aligned crops would restrict pairs to
        the overlapping span of two differently-sized arrays for no gain.
        """
        g = pair_index[rng.integers(len(pair_index))]
        i, j = rng.choice(len(g), size=2, replace=False)
        m1, m2 = g[i], g[j]
        if m1["stretch"] == m2["stretch"]:
            return None
        xs = []
        for mm in (m1, m2):
            if mm["T"] <= W:
                return None
            st = int(rng.integers(0, mm["T"] - W))
            o = mm["off"] + st
            xs.append(np.asarray(E[o:o + W], np.float32))
        x = torch.from_numpy(np.stack(xs)).permute(0, 2, 1).to(dev)
        return x, float(m2["stretch"]) / float(m1["stretch"])

    def loss_fn(x, y, k):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = net(x).float()
        lp = torch.log_softmax(lg, 1)
        ce = -(y * lp).sum(1)                       # [B,T]
        loss = (ce * k).sum() / k.sum().clamp(min=1)
        if args.oct_w > 0:
            # mass sitting in the octave BAND around the true tempo, penalised directly
            q = lp.exp()
            bm = bandmask_t[y.argmax(1)].permute(0, 2, 1)          # [B, V, T]
            band = (q * bm).sum(1)                                  # [B, T]
            loss = loss + args.oct_w * (band * k).sum() / k.sum().clamp(min=1)
        return loss

    t0 = time.time()
    for step in range(1, args.steps + 1):
        net.train()
        x, y, k = batch(tr)
        loss = loss_fn(x, y, k)
        if equiv is not None:
            eb = equiv_batch()
            if eb is not None:
                xe, ratio = eb
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lge = net(xe).float()
                qe = torch.softmax(lge, 1).permute(0, 2, 1)        # [2, T, V]
                loss = loss + args.equiv_w * equiv(qe[0], qe[1], ratio)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % args.log_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                vl = np.mean([float(loss_fn(*batch(val))) for _ in range(5)])
            print(f"step {step:6d}  train {float(loss):.4f}  val {vl:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
    torch.save({"sd": net.state_dict(), "grid": GRID.tolist(), "args": vars(args)}, args.out)
    print(f"saved {args.out}", flush=True)

    # head's own marginal, on HELD-OUT corpus frames (R1 denominator)
    marg = head_marginal(net, val, E, dev, rng)
    np.save(str(args.out) + ".marginal.npy", marg)
    print("head marginal (log): min %.3f max %.3f  spread %.3f nats"
          % (np.log(marg).min(), np.log(marg).max(),
             np.log(marg).max() - np.log(marg).min()), flush=True)


def head_marginal(net, pool, E, dev, rng, n=400):
    import torch
    net.eval()
    acc = np.zeros(len(GRID))
    cnt = 0
    with torch.no_grad():
        for m in pool[:n]:
            x = torch.from_numpy(np.asarray(E[m["off"]:m["off"] + m["T"]], np.float32))
            x = x.permute(1, 0).unsqueeze(0).to(dev)
            p = torch.softmax(net(x).float(), 1)[0].mean(1).cpu().numpy()
            acc += p; cnt += 1
    marg = acc / cnt
    return marg / marg.sum()


# --------------------------------------------------------------------------- eval
def load_net(path, dev):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = make_model(len(ck["grid"]))
    net.load_state_dict(ck["sd"])
    return net.to(dev).eval()


def smc_predict(net, dev):
    """q_theta on all 217 SMC tracks. Returns {tid: [T,60] prob}."""
    import torch
    out = {}
    with torch.no_grad():
        for f in sorted(SMC_EMB.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            x = torch.from_numpy(np.asarray(z["emb"], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            out[f.stem] = torch.softmax(net(x).float(), 1)[0].T.cpu().numpy()
    return out


def smc_gt(tid):
    p = STORE / "smc" / "annotations" / "beats" / f"{tid}.beats"
    if not p.exists():
        p = STORE / "smc" / "annotations" / "beats" / f"{tid.upper()}.beats"
    d = np.loadtxt(str(p))
    return d if d.ndim == 1 else d[:, 1]


def evaluate(args):
    import torch
    from scipy.stats import wilcoxon
    dev = torch.device(args.device)
    net = load_net(args.out, dev)
    marg = np.load(str(args.out) + ".marginal.npy")

    Q = smc_predict(net, dev)
    print(f"SMC tracks scored: {len(Q)}", flush=True)

    per_frame_hit, per_track_hit, deltas = [], [], []
    rows = {}
    for tid, q in sorted(Q.items()):
        bt = smc_gt(tid)
        if len(bt) < 3:
            continue
        T = q.shape[0]
        fr = bt * FPS
        ibi = np.diff(fr)
        true = np.full(T, np.nan)
        for k in range(len(fr) - 1):
            a, b = max(0, int(np.ceil(fr[k]))), min(T, int(np.ceil(fr[k + 1])))
            if b > a:
                true[a:b] = ibi[k]
        ok = ~np.isnan(true)
        if ok.sum() < 20:
            continue
        pred = GRID[q.argmax(1)]
        hitf = np.abs(np.log(pred[ok] / true[ok])) < np.log(1.1)
        per_frame_hit.append(hitf.mean())
        # per-track: median of frame-wise argmax vs median IBI
        ptrack = float(np.median(pred[ok]))
        ttrack = float(np.median(true[ok]))
        per_track_hit.append(abs(np.log(ptrack / ttrack)) < np.log(1.1))
        deltas.append(np.log2(ptrack / ttrack))
        rows[tid] = {"pred_bpm": 60 * FPS / ptrack, "true_bpm": 60 * FPS / ttrack,
                     "frame_acc": float(hitf.mean())}

    pf = float(np.mean(per_frame_hit)); pt = float(np.mean(per_track_hit))
    d = np.array(deltas)
    n = len(per_track_hit)
    print("\n=== G1: SMC tempo accuracy within +/-10% ===")
    print(f"  n = {n} tracks")
    print(f"  per-FRAME accuracy (frame argmax vs local IBI):  {pf*100:.1f}%")
    print(f"  per-TRACK accuracy (median argmax vs median IBI): {pt*100:.1f}%")
    print(f"  log2 tempo bias (pred/true): median {np.median(d):+.3f}  mean {d.mean():+.3f}"
          f"  (Wilcoxon vs 0: p={wilcoxon(d).pvalue:.2e})")
    slow = np.array([r["true_bpm"] < 70 for r in rows.values()])
    dd = np.array([np.log2(r["pred_bpm"] / r["true_bpm"]) for r in rows.values()])
    hit = np.array([abs(x) < np.log2(1.1) for x in dd])
    print(f"  slow (<70 BPM) n={slow.sum()}: acc {hit[slow].mean()*100:.1f}%, "
          f"bias {np.median(dd[slow]):+.3f}")
    print(f"  fast (>=70 BPM) n={(~slow).sum()}: acc {hit[~slow].mean()*100:.1f}%, "
          f"bias {np.median(dd[~slow]):+.3f}")
    oct_hit = np.mean([any(abs(x - o) < np.log2(1.1) for o in (-1, 0, 1)) for x in dd])
    print(f"  octave-tolerant accuracy: {oct_hit*100:.1f}%")

    # ---- G2: tempo-neutrality on CONSTANT input -------------------------------
    print("\n=== G2: tempo-neutrality (constant input, 60 s = 3000 frames) ===")
    Tn = 3000
    with torch.no_grad():
        specs = {}
        for name, val in (("zeros", 0.0), ("mean_emb", None)):
            if val is None:
                mm = json.load(open(CACHE / "meta.json"))["meta"]
                E = np.load(CACHE / "emb.npy", mmap_mode="r")
                x0 = np.asarray(E[:200000], np.float32).mean(0)
            else:
                x0 = np.full(512, val, np.float32)
            x = torch.from_numpy(np.tile(x0, (Tn, 1))).permute(1, 0).unsqueeze(0).to(dev)
            lq = torch.log_softmax(net(x).float(), 1)[0].T.cpu().numpy()  # [T,60]
            ratio = lq - np.log(marg)[None, :]
            # accumulated evidence for a path at tempo i: sum_t ratio[t,i]/GRID[i]
            acc = ratio.sum(0) / GRID
            specs[name] = float(acc.max() - acc.min())
            print(f"  {name:9s} spread = {specs[name]:.1f} nats/60 s (beta=1) "
                  f"| argmax tempo {60*FPS/GRID[acc.argmax()]:.0f} BPM")
    print(f"  baseline (shipped emission, obs_lambda=16) = 448 nats/60 s")

    json.dump({"n": n, "frame_acc": pf, "track_acc": pt,
               "bias_log2_median": float(np.median(d)),
               "octave_tolerant_track_acc": float(oct_hit),
               "g2_spread_beta1": specs, "rows": rows},
              open(args.report, "w"), indent=1)
    print(f"\nwrote {args.report}", flush=True)


def check_alignment(args):
    """Label-alignment audit: Beat This activation peaks vs annotated beat CENTRES."""
    from scipy.signal import find_peaks
    from scipy.special import expit
    offs_p, offs_s = [], []
    for ds in DATASETS:
        for f in sorted((EMB / ds).glob("*.npz"))[:40]:
            z = np.load(f, allow_pickle=True)
            a = expit(np.asarray(z["beat"], np.float64))
            pk, _ = find_peaks(a, height=0.5)
            bt = load_beats(ds, f.stem)
            if bt is None or len(pk) < 5:
                continue
            bf = bt * FPS - int(z["start_frame"])
            o = [pk[np.argmin(np.abs(pk - b))] - b for b in bf]
            offs_p.append(np.median(o))
        for f in sorted((EMB_STR / ds).glob("*__s*.npz"))[:40]:
            z = np.load(f, allow_pickle=True)
            a = expit(np.asarray(z["beat"], np.float64))
            pk, _ = find_peaks(a, height=0.5)
            bt = np.asarray(z["beat_times"], float) * FPS
            if len(pk) < 5 or len(bt) < 5:
                continue
            o = [pk[np.argmin(np.abs(pk - b))] - b for b in bt]
            offs_s.append(np.median(o))
    for nm, o in (("plain", offs_p), ("stretch", offs_s)):
        o = np.asarray(o)
        print(f"{nm}: n={len(o)} median offset {np.median(o):+.3f} frames "
              f"({np.median(o)/FPS*1000:+.1f} ms), |med|>1 on {np.mean(np.abs(o)>1)*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "train", "eval", "check"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--equiv-w", type=float, default=0.0,
                    help="weight on the PESTO-style time-stretch equivariance loss. Uses pairs of "
                         "the SAME track at different stretch factors from /disk1/jaehoon/"
                         "bt_emb_stretch, where the interval ratio is known exactly.")
    ap.add_argument("--equiv-tau", type=float, default=0.25, help="Huber tau on the ratio residual")
    ap.add_argument("--oct-w", type=float, default=0.0,
                    help="weight on the octave-band penalty. Sandbox dose-response is monotone and "
                         "only clears the pre-registered bar at w>=20 (split A) / w>=50 (split B); "
                         "w=1 is about 5%% of the objective and does nothing.")
    ap.add_argument("--oct-band", type=float, default=0.15,
                    help="half-width in log2 of the penalised band -- MUST match the metric's "
                         "+/-0.15; using one bin instead is what made v1 a no-op")
    ap.add_argument("--out", default="/disk1/jaehoon/tempo_obs_head.pt")
    ap.add_argument("--report", default="/tmp/overnight/tempo_obs_eval.json")
    args = ap.parse_args()
    GRID = tempo_grid()
    GRID_BPM = np.sort(60.0 * FPS / GRID)          # ascending BPM
    # keep grid order == interval order for the DBN; targets are built in BPM order then flipped
    assert np.allclose(GRID_BPM, (60.0 * FPS / GRID)[::-1])
    if args.cmd == "cache":
        build_cache()
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)
    else:
        check_alignment(args)
