"""SANDBOX: iterate on octave-aware tempo losses WITHOUT touching SMC or the real pipeline.

WHY A SANDBOX. Repeatedly this project has translated a hypothesis straight into the full pipeline
and lost: octave smearing (§8 R-10) and path-score arbitration (§8 R-11) were both proposed,
implemented against the real decoder, and refuted. A loss function is a bigger commitment than
either. So: iterate here, on held-out CORPUS only, until something clears a PRE-REGISTERED bar, and
only then spend a full retrain + SMC evaluation on it.

SMC IS NOT TOUCHED BY THIS SCRIPT. Model selection happens on corpus fold 0. Promoting a winner to
SMC is a separate, deliberate step, so SMC cannot be overfitted by iteration count.

WHAT WE ARE TRYING TO FIX (measured, §4.6/§4.7). The head reaches 46.1% frame accuracy on SMC with
zero bias, and where it has the tempo right, injecting it lifts CMLt 0.7594 -> 0.8088 (n=113,
p=0.024). Where it is octave-wrong it destroys the decode (half: -0.3621 n=40; double: -0.3827
n=14). Re-weighting cannot fix this: the beta sweep peaks at beta=0.2 with +0.0104 CMLt (p=0.14,
n.s.) because at low beta the gain and the catastrophe shrink together. The ONLY route up is moving
tracks out of the octave-wrong groups -- 113/217 correct today.

PRE-REGISTERED SUCCESS CRITERION (fixed before any variant is run; do not move it afterwards):
    a variant PASSES iff, averaged over seeds and vs the baseline at identical budget/seeds,
        (1) octave-error rate falls by >= 20% RELATIVE, and
        (2) acc10 does not fall by more than 0.01 absolute.
Anything else is a FAIL, including "acc10 went up but octave errors did not fall" -- that is the
metric that already looked good while decoding got worse.

VARIANTS
  base   soft-target cross-entropy, exactly what trained the current head
  oct    base + explicit penalty on probability mass at the +/-1 octave partners of the true tempo
  fact   chroma x register factorisation: q(i) = p_c(log2(i) mod 1) * p_r(register(i)).
         Rationale: the head is 75.6% octave-TOLERANT but 51.6% strict, i.e. it already knows tempo
         modulo octave. Factorising lets it stay confident about the chroma while remaining
         calibrated-uncertain about the register, which yields a naturally BIMODAL q exactly when
         the octave is ambiguous. That is octave smearing learned CONDITIONALLY, which is what
         R-10 got wrong by smearing unconditionally.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = Path("/disk1/jaehoon/tempo_obs_cache")


def load_meta():
    d = json.load(open(CACHE / "meta.json"))
    return d["meta"], np.asarray(d["grid"], dtype=np.float64)


def make_net(nout, c=128, blocks=(1, 2, 4, 8, 16, 32)):
    class B(nn.Module):
        def __init__(s, c, d):
            super().__init__()
            s.c1 = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d)
            s.c2 = nn.Conv1d(c, c, 1)
            s.n1, s.n2 = nn.GroupNorm(8, c), nn.GroupNorm(8, c)
            s.a = nn.GELU()

        def forward(s, x):
            h = s.a(s.n1(s.c1(x)))
            return x + s.a(s.n2(s.c2(h)))

    class N(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Conv1d(512, c, 1)
            s.bl = nn.ModuleList([B(c, d) for d in blocks])
            s.out = nn.Conv1d(c, nout, 1)

        def forward(s, x):
            h = s.inp(x)
            for b in s.bl:
                h = b(h)
            return s.out(h)

    return N()


class Factorised(nn.Module):
    """q(i) = p_chroma(chroma(i)) * p_register(register(i)), renormalised over the tempo grid.

    chroma(i) = log2(i) mod 1 discretised into KC bins; register(i) = floor(log2(i) - log2(i_min)).
    Octave partners share a chroma bin and differ only in register, so uncertainty about the octave
    shows up as a flat p_register -- a bimodal q -- while p_chroma stays sharp.
    """

    def __init__(self, grid, kc=24):
        super().__init__()
        lg = np.log2(grid)
        self.kc = kc
        self.nr = int(np.floor(lg.max() - lg.min())) + 1
        cidx = np.floor(((lg - lg.min()) % 1.0) * kc).astype(int) % kc
        ridx = np.clip(np.floor(lg - lg.min()).astype(int), 0, self.nr - 1)
        self.register_buffer("cidx", torch.tensor(cidx, dtype=torch.long))
        self.register_buffer("ridx", torch.tensor(ridx, dtype=torch.long))
        self.net = make_net(kc + self.nr)

    def forward(self, x):
        o = self.net(x)
        lc = torch.log_softmax(o[:, :self.kc], 1)
        lr = torch.log_softmax(o[:, self.kc:], 1)
        return lc[:, self.cidx] + lr[:, self.ridx]        # [B, V, T] unnormalised log q


def octave_partner_index(grid):
    """For each grid index, the indices of its x2 and /2 partners (-1 where out of range)."""
    up, dn = [], []
    for v in grid:
        j = int(np.argmin(np.abs(grid - v * 2)))
        up.append(j if abs(grid[j] - v * 2) / (v * 2) < 0.06 else -1)
        j = int(np.argmin(np.abs(grid - v / 2)))
        dn.append(j if abs(grid[j] - v / 2) / (v / 2) < 0.06 else -1)
    return np.array(up), np.array(dn)


def run(variant, seed, args, meta, grid, E, Tg, Mk, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    V = len(grid)
    tr = [m for m in meta if m["fold"] != 0][:args.train_tracks]
    va = [m for m in meta if m["fold"] == 0][:args.val_tracks]
    up, dn = octave_partner_index(grid)
    up_t = torch.tensor(up, device=dev)
    dn_t = torch.tensor(dn, device=dev)

    net = (Factorised(grid).to(dev) if variant == "fact" else make_net(V).to(dev))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps)
    W = args.win

    def batch(pool):
        xs, ys, ms = [], [], []
        for _ in range(args.bs):
            m = pool[rng.integers(len(pool))]
            if m["T"] <= W:
                continue
            st = int(rng.integers(0, m["T"] - W))
            o = m["off"] + st
            xs.append(np.asarray(E[o:o + W], np.float32))
            ys.append(np.asarray(Tg[o:o + W], np.float32))
            ms.append(np.asarray(Mk[o:o + W], bool))
        if not xs:
            return None
        x = torch.from_numpy(np.stack(xs)).permute(0, 2, 1).to(dev)
        y = torch.from_numpy(np.stack(ys)).permute(0, 2, 1).to(dev)
        mk = torch.from_numpy(np.stack(ms)).to(dev)
        return x, y, mk

    for step in range(args.steps):
        b = batch(tr)
        if b is None:
            continue
        x, y, mk = b
        out = net(x)
        lq = out if variant == "fact" else out
        lq = torch.log_softmax(lq, 1)
        # soft-target cross-entropy over the tempo grid, on labelled frames only
        loss = -(y * lq).sum(1)[mk].mean()
        if variant == "oct":
            # penalise probability mass sitting on the true tempo's octave partners
            q = lq.exp()
            tgt = y.argmax(1)                                     # [B, T]
            pen = torch.zeros_like(loss)
            for part in (up_t, dn_t):
                pj = part[tgt]                                    # [B, T]
                ok = pj >= 0
                if ok.any():
                    g = torch.gather(q, 1, pj.clamp(min=0).unsqueeze(1)).squeeze(1)
                    pen = pen + (g * ok.float())[mk].mean()
            loss = loss + args.oct_w * pen
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()

    # ---- evaluate on held-out CORPUS fold 0 (never SMC) ----
    net.eval()
    accs, octs, n = 0, 0, 0
    with torch.no_grad():
        for m in va:
            o, T = m["off"], m["T"]
            x = torch.from_numpy(np.asarray(E[o:o + T], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            lq = torch.log_softmax(net(x), 1)[0].T                # [T, V]
            mk = torch.from_numpy(np.asarray(Mk[o:o + T], bool)).to(dev)
            if mk.sum() < 20:
                continue
            y = torch.from_numpy(np.asarray(Tg[o:o + T], np.float32)).to(dev)
            pred = grid[lq[mk].argmax(1).cpu().numpy()]
            true = grid[y[mk].argmax(1).cpu().numpy()]
            lr = np.log2(pred / true)
            accs += float(np.mean(np.abs(lr) < np.log2(1.1)))
            octs += float(np.mean(np.abs(np.abs(lr) - 1) < 0.15))
            n += 1
    return {"acc10": accs / max(n, 1), "oct_err": octs / max(n, 1), "n_val": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,oct,fact")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--oct-w", type=float, default=1.0)
    ap.add_argument("--train-tracks", type=int, default=900)
    ap.add_argument("--val-tracks", type=int, default=160)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    dev = torch.device(args.device)

    meta, grid = load_meta()
    E = np.load(CACHE / "emb.npy", mmap_mode="r")
    Tg = np.load(CACHE / "tgt.npy", mmap_mode="r")
    Mk = np.load(CACHE / "mask.npy", mmap_mode="r")
    print(f"cache {E.shape[0]:,} frames, grid {len(grid)}; "
          f"train fold!=0, val fold==0 (CORPUS ONLY, SMC untouched)", flush=True)
    print(f"PRE-REGISTERED PASS: oct_err down >=20% relative AND acc10 down <=0.01 abs\n", flush=True)

    res = {}
    for v in args.variants.split(","):
        rows = []
        for s in range(args.seeds):
            t0 = time.time()
            r = run(v, s, args, meta, grid, E, Tg, Mk, dev)
            rows.append(r)
            print(f"  {v:<5} seed{s}  acc10 {r['acc10']:.4f}  oct_err {r['oct_err']:.4f}  "
                  f"({time.time()-t0:.0f}s, n_val={r['n_val']})", flush=True)
        res[v] = {"acc10": float(np.mean([x["acc10"] for x in rows])),
                  "oct_err": float(np.mean([x["oct_err"] for x in rows]))}

    b = res.get("base")
    print(f"\n{'variant':<8}{'acc10':>9}{'oct_err':>10}{'d_acc10':>10}{'oct rel':>10}{'VERDICT':>10}")
    for v, r in res.items():
        if v == "base":
            print(f"{v:<8}{r['acc10']:>9.4f}{r['oct_err']:>10.4f}{'-':>10}{'-':>10}{'(ref)':>10}")
            continue
        da = r["acc10"] - b["acc10"]
        rel = (r["oct_err"] - b["oct_err"]) / max(b["oct_err"], 1e-9)
        ok = (rel <= -0.20) and (da >= -0.01)
        print(f"{v:<8}{r['acc10']:>9.4f}{r['oct_err']:>10.4f}{da:>+10.4f}{rel:>+9.1%}"
              f"{'PASS' if ok else 'FAIL':>10}")
    json.dump(res, open("/tmp/overnight/sandbox_octave.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
