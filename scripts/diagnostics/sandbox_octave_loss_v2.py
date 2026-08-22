"""SANDBOX v2 -- audit + properly-powered re-run of the octave-aware tempo loss variants.

CHANGES vs v1 (scripts/diagnostics/sandbox_octave_loss.py):
  * >=6 seeds, >=3000 steps (v1: 2 seeds, 1200 steps; baseline seed spread 0.4359/0.5441 acc10
    exceeded the 0.024 decision bar -- v1 was pure noise).
  * PAIRED per-seed differences + exact two-sided sign test, instead of comparing seed means.
  * BUGFIX A (penalty/metric support mismatch): v1's `oct` penalty pushed mass off ONE grid bin
    (the exact x2 / /2 partner) while the oct_err METRIC counts a band of +/-0.15 log2 = +/-3.12
    grid bins (grid step = 0.0481 log2). Mass displaced by 1 bin still scores as an octave error.
    New variant `octband` penalises the whole +/-`band` bin window around each partner.
  * BUGFIX B (penalty is linear in q -> vanishing gradient once q is small): `octbandlog`
    penalises -log(1 - mass) which keeps pressure on.
  * Diagnostics added: octave-TOLERANT acc, oct_err conditional on being wrong, and the mean
    probability MASS in the octave band (the quantity the decoder actually consumes, and the
    quantity `fact` was designed to change -- argmax oct_err is the wrong probe for a variant
    whose stated goal is a BIMODAL posterior).
"""
import argparse, json, sys, time
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from sandbox_octave_loss import load_meta, make_net, Factorised, octave_partner_index  # noqa: E402


def sign_test_p(diffs, better="less"):
    """Exact two-sided sign test on nonzero diffs."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0, 0
    k = sum(1 for x in d if x < 0)
    lo = min(k, n - k)
    p = 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, p), k, n


def run(variant, seed, args, meta, grid, E, Tg, Mk, dev):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    V = len(grid)
    tr = [m for m in meta if m["fold"] != 0]
    va = [m for m in meta if m["fold"] == 0]
    if args.stratified:
        # BUGFIX C: meta is ordered by dataset, so v1's `[:900]` / `[:160]` slices selected
        # 900 BALLROOM tracks for training and 160 GTZAN tracks for validation -- a degenerate,
        # single-corpus split. Shuffle with a FIXED split seed (independent of the model seed) so
        # every variant/seed sees the identical, dataset-mixed split.
        srng = np.random.default_rng(12345)
        srng.shuffle(tr); srng.shuffle(va)
    tr = tr[:args.train_tracks]
    va = va[:args.val_tracks]
    up, dn = octave_partner_index(grid)
    up_t = torch.tensor(up, device=dev)
    dn_t = torch.tensor(dn, device=dev)

    # band index table: for each true bin, the set of bins within +/-band of either partner
    B = args.band
    bandmask = np.zeros((V, V), dtype=np.float32)   # [true, grid]
    for i in range(V):
        for p in (up[i], dn[i]):
            if p >= 0:
                bandmask[i, max(0, p - B):min(V, p + B + 1)] = 1.0
    bandmask_t = torch.tensor(bandmask, device=dev)

    net = (Factorised(grid).to(dev) if variant.startswith("fact") else make_net(V).to(dev))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps)
    W = args.win

    def batch(pool):
        xs, ys, ms = [], [], []
        for _ in range(args.bs):
            m = pool[rng.integers(len(pool))]
            if m["T"] <= W:
                continue
            st = int(rng.integers(0, m["T"] - W)); o = m["off"] + st
            xs.append(np.asarray(E[o:o + W], np.float32))
            ys.append(np.asarray(Tg[o:o + W], np.float32))
            ms.append(np.asarray(Mk[o:o + W], bool))
        if not xs:
            return None
        return (torch.from_numpy(np.stack(xs)).permute(0, 2, 1).to(dev),
                torch.from_numpy(np.stack(ys)).permute(0, 2, 1).to(dev),
                torch.from_numpy(np.stack(ms)).to(dev))

    for step in range(args.steps):
        b = batch(tr)
        if b is None:
            continue
        x, y, mk = b
        lq = torch.log_softmax(net(x), 1)
        loss = -(y * lq).sum(1)[mk].mean()
        if variant in ("oct", "octband", "octbandlog", "factband"):
            q = lq.exp()
            tgt = y.argmax(1)                                       # [B,T]
            if variant == "oct":
                pen = 0.0
                for part in (up_t, dn_t):
                    pj = part[tgt]; ok = pj >= 0
                    g = torch.gather(q, 1, pj.clamp(min=0).unsqueeze(1)).squeeze(1)
                    pen = pen + (g * ok.float())[mk].mean()
            else:
                bm = bandmask_t[tgt].permute(0, 2, 1)               # [B,V,T]
                mass = (q * bm).sum(1)                              # [B,T]
                pen = (-torch.log1p(-mass.clamp(max=0.999))
                       if variant.endswith("log") else mass)[mk].mean()
            loss = loss + args.oct_w * pen
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

    net.eval()
    A = dict(acc10=0.0, oct_err=0.0, tol=0.0, mass=0.0)
    n = 0
    with torch.no_grad():
        for m in va:
            o, T = m["off"], m["T"]
            x = torch.from_numpy(np.asarray(E[o:o + T], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            lq = torch.log_softmax(net(x), 1)[0].T                  # [T,V]
            mk = torch.from_numpy(np.asarray(Mk[o:o + T], bool).copy()).to(dev)
            if mk.sum() < 20:
                continue
            y = torch.from_numpy(np.asarray(Tg[o:o + T], np.float32)).to(dev)
            ti = y[mk].argmax(1)
            q = lq[mk].exp()
            mass = float(((q * bandmask_t[ti]).sum(1)).mean())
            pred = grid[lq[mk].argmax(1).cpu().numpy()]
            true = grid[ti.cpu().numpy()]
            lr = np.log2(pred / true)
            A["acc10"] += float(np.mean(np.abs(lr) < np.log2(1.1)))
            A["oct_err"] += float(np.mean(np.abs(np.abs(lr) - 1) < 0.15))
            A["tol"] += float(np.mean(np.abs(lr - np.round(lr)) < np.log2(1.1)))
            A["mass"] += mass
            n += 1
    r = {k: v / max(n, 1) for k, v in A.items()}
    r["oct_frac_of_err"] = r["oct_err"] / max(1 - r["acc10"], 1e-9)
    r["n_val"] = n
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,oct,octband,octbandlog,fact,factband")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--oct-w", type=float, default=1.0)
    ap.add_argument("--band", type=int, default=3)
    ap.add_argument("--train-tracks", type=int, default=900)
    ap.add_argument("--val-tracks", type=int, default=160)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default="/tmp/overnight/sandbox_octave_v2.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    meta, grid = load_meta()
    E = np.load("/disk1/jaehoon/tempo_obs_cache/emb.npy", mmap_mode="r")
    Tg = np.load("/disk1/jaehoon/tempo_obs_cache/tgt.npy", mmap_mode="r")
    Mk = np.load("/disk1/jaehoon/tempo_obs_cache/mask.npy", mmap_mode="r")
    print(f"seeds={args.seeds} steps={args.steps} oct_w={args.oct_w} band={args.band}", flush=True)

    per = {}
    for v in args.variants.split(","):
        per[v] = []
        for s in range(args.seeds):
            t0 = time.time()
            r = run(v, s, args, meta, grid, E, Tg, Mk, dev)
            per[v].append(r)
            print(f"  {v:<11} s{s} acc10 {r['acc10']:.4f} oct {r['oct_err']:.4f} "
                  f"tol {r['tol']:.4f} mass {r['mass']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    base = per["base"]
    print(f"\n{'variant':<11}{'acc10':>8}{'oct_err':>9}{'d_acc':>8}{'octrel':>8}"
          f"{'p_oct':>8}{'k/n':>7}{'VERDICT':>9}")
    summ = {}
    for v, rows in per.items():
        mu = {k: float(np.mean([r[k] for r in rows])) for k in
              ("acc10", "oct_err", "tol", "mass", "oct_frac_of_err")}
        if v == "base":
            summ[v] = mu
            print(f"{v:<11}{mu['acc10']:>8.4f}{mu['oct_err']:>9.4f}{'-':>8}{'-':>8}{'-':>8}{'-':>7}{'(ref)':>9}")
            continue
        do = [rows[i]["oct_err"] - base[i]["oct_err"] for i in range(len(rows))]
        da = [rows[i]["acc10"] - base[i]["acc10"] for i in range(len(rows))]
        rel = [d / max(base[i]["oct_err"], 1e-9) for i, d in enumerate(do)]
        p, k, nn_ = sign_test_p(do)
        ok = (float(np.mean(rel)) <= -0.20) and (float(np.mean(da)) >= -0.01)
        mu.update(d_acc10_paired=float(np.mean(da)), oct_rel_paired=float(np.mean(rel)),
                  p_sign_oct=p, k_better=k, n_seeds=nn_,
                  p_sign_acc=sign_test_p(da)[0], passes=ok)
        summ[v] = mu
        print(f"{v:<11}{mu['acc10']:>8.4f}{mu['oct_err']:>9.4f}{np.mean(da):>+8.4f}"
              f"{np.mean(rel):>+7.1%}{p:>8.3f}{f'{k}/{nn_}':>7}{'PASS' if ok else 'FAIL':>9}")
    print("\ndiagnostics (mean over seeds)")
    print(f"{'variant':<11}{'tol_acc':>9}{'oct|err':>9}{'octmass':>9}")
    for v, mu in summ.items():
        print(f"{v:<11}{mu['tol']:>9.4f}{mu['oct_frac_of_err']:>9.4f}{mu['mass']:>9.4f}")
    json.dump({"summary": summ, "per_seed": per, "args": vars(args)}, open(args.out, "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
