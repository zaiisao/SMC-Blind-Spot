"""Failure-mode analysis of the learned tempo head q_theta, to decide what is actually fixable.

WHY THIS AND NOT MORE DECODER WORK. The head sits at 24.9% octave error against a measured 24.4%
break-even (§4.9), so a tempo channel is currently worth exactly zero and every downstream
intervention is being evaluated at the root of the curve. The payoff is linear below break-even --
+0.007 CMLt per percentage point -- so the only thing that matters is reducing octave errors.

THE DECISIVE SPLIT. For each track where the head is wrong, ask whether the annotated tempo is even
present in the evidence the head reads:

  A. ACF of the beat activation agrees with the ANNOTATION, head disagrees
     -> the information IS there and the head misreads it. FIXABLE by a better head/loss/context.
  B. ACF agrees with the HEAD (both differ from the annotation)
     -> the activation itself supports the alternative level. Not a head bug; the encoder's beat
        activation is already committed. Fixing this needs different features, not a better head.
  C. ACF agrees with neither -> the track is simply hard.

This distinction has never been measured, and it decides whether to spend the remaining time on the
head (A) or to stop (B). The register probe showed octave information exists in the embeddings at
77% vs a 54% majority floor, so A is plausible; but the head's register confidence had AUROC 0.499,
which is what B would look like.

Also recorded per track, for cross-tabulation: error class (1/2x, 2x, 3x, 1/3x, other), head entropy
and max probability, chroma-vs-register decomposition of the error, the SMC difficulty tags, and the
annotator confidence level. Tags are the dataset authors' own account of why an excerpt was hard.

Everything here is diagnostic on SMC; nothing is trained.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _harness import FPS, track_ids, load_gt, bthis_beat_prob
from tempo_readout import lag_grid, soft_log_period
from train_tempo_observation import load_net

SMC_EMB = Path("/disk1/jaehoon/smc_bt_embeddings")
HEAD = Path("/disk1/jaehoon/tempo_obs_head.pt")
TAGS = ROOT / "SMC_MIREX" / "SMC_MIREX_Tags"
DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
TOL = 0.15                                   # log2 tolerance for "same level"


def read_tags():
    out = {}
    for f in sorted(TAGS.glob("*.tag")):
        tid = "smc_" + re.search(r"SMC_(\d+)", f.name).group(1)
        conf, words = None, []
        for l in open(f, errors="ignore"):
            l = l.strip()
            if not l:
                continue
            m = re.fullmatch(r"([a-zA-Z])\s*([0-9])", l)
            if m:
                conf = int(m.group(2))
            else:
                words.append(l.lower().strip("()"))
        out[tid] = {"conf": conf, "words": words}
    return out


def main():
    dev = torch.device(DEV)
    net = load_net(str(HEAD), DEV)
    ck = torch.load(str(HEAD), map_location="cpu", weights_only=False)
    grid = np.asarray(ck["grid"], dtype=np.float64)
    lags = lag_grid()
    tags = read_tags()

    rows = []
    for t in track_ids():
        f = SMC_EMB / f"{t}.npz"
        if not f.exists():
            continue
        gt = np.asarray(load_gt(t), float)
        if len(gt) < 4:
            continue
        true_iv = float(np.median(np.diff(gt)) * FPS)          # annotated beat period, frames
        z = np.load(f, allow_pickle=True)
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(z["emb"], np.float32)).permute(1, 0).unsqueeze(0).to(dev)
            lq = torch.log_softmax(net(x).float(), 1)[0].T
            q = lq.exp().mean(0)                                # track-level tempo distribution
        head_iv = float(grid[int(q.argmax())])
        ent = float(-(q * torch.log(q.clamp_min(1e-12))).sum())
        # independent evidence: ACF of the model's own beat activation (no head involved)
        bp = torch.from_numpy(bthis_beat_prob(t)).double()
        acf_lp, _ = soft_log_period(bp[None, :], lags, tau=30.0)
        acf_iv = float(2.0 ** float(acf_lp[0]))

        h = np.log2(head_iv / true_iv)
        a = np.log2(acf_iv / true_iv)
        d = np.log2(acf_iv / head_iv)
        def cls(x):
            for v, n in ((0, "correct"), (1, "2x"), (-1, "0.5x"),
                         (np.log2(3), "3x"), (-np.log2(3), "1/3x")):
                if abs(x - v) < TOL:
                    return n
            return "other"
        rows.append({"tid": t, "true_bpm": 60 * FPS / true_iv, "head_bpm": 60 * FPS / head_iv,
                     "acf_bpm": 60 * FPS / acf_iv, "h": h, "a": a, "d": d,
                     "err": cls(h), "acf_err": cls(a), "ent": ent,
                     "maxp": float(q.max()),
                     "tags": tags.get(t, {}).get("words", []),
                     "conf": tags.get(t, {}).get("conf")})

    n = len(rows)
    wrong = [r for r in rows if r["err"] != "correct"]
    print(f"n={n}   head correct {n-len(wrong)}   wrong {len(wrong)}")
    print("\nerror classes:", dict(Counter(r["err"] for r in rows)))

    # ---- THE DECISIVE SPLIT ----
    A = [r for r in wrong if abs(r["a"]) < TOL]                       # ACF right, head wrong
    B = [r for r in wrong if abs(r["d"]) < TOL and abs(r["a"]) >= TOL]  # ACF agrees with head
    C = [r for r in wrong if abs(r["a"]) >= TOL and abs(r["d"]) >= TOL]
    print(f"\nWHERE THE HEAD FAILS (n={len(wrong)}):")
    print(f"  A  activation ACF is RIGHT, head wrong   {len(A):>3}  ({len(A)/max(len(wrong),1):.1%})"
          f"   -> FIXABLE by a better head")
    print(f"  B  activation ACF AGREES with the head   {len(B):>3}  ({len(B)/max(len(wrong),1):.1%})"
          f"   -> activation already committed; needs different features")
    print(f"  C  neither agrees with the annotation    {len(C):>3}  ({len(C)/max(len(wrong),1):.1%})"
          f"   -> hard track")

    print(f"\nhead confidence does NOT separate right from wrong (known, AUROC 0.499):")
    ok = [r for r in rows if r["err"] == "correct"]
    print(f"  entropy  correct {np.mean([r['ent'] for r in ok]):.3f}   "
          f"wrong {np.mean([r['ent'] for r in wrong]):.3f}")
    print(f"  max prob correct {np.mean([r['maxp'] for r in ok]):.3f}   "
          f"wrong {np.mean([r['maxp'] for r in wrong]):.3f}")

    # ---- tags enriched among failures ----
    from scipy.stats import fisher_exact
    allw = sorted({w for r in rows for w in r["tags"]})
    res = []
    for w in allw:
        hw = np.array([w in r["tags"] for r in rows])
        bad = np.array([r["err"] != "correct" for r in rows])
        a_, b_ = int((hw & bad).sum()), int((~hw & bad).sum())
        c_, d_ = int((hw & ~bad).sum()), int((~hw & ~bad).sum())
        if a_ + c_ < 12:
            continue
        orr, p = fisher_exact([[a_, b_], [c_, d_]])
        res.append((p, w, hw[bad].mean(), hw[~bad].mean(), orr))
    print(f"\nSMC difficulty tags vs head failure   ({'tag':<40}{'fail%':>7}{'ok%':>7}{'OR':>6}{'p':>9})")
    for p, w, x, y, orr in sorted(res)[:8]:
        print(f"  {w:<40}{x:>7.1%}{y:>7.1%}{orr:>6.2f}{p:>9.3g}")

    json.dump(rows, open("/tmp/overnight/tempo_head_failures.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
