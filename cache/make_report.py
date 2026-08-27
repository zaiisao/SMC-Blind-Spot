"""Assemble MORNING_REPORT.md from whatever finished overnight."""
import glob, json, os
import numpy as np

C = "/home/sogang/jaehoon/Analyze-SMC/cache"
BANDS = [(0, 55, "<55"), (55, 70, "55-70"), (70, 100, "70-100"), (100, 1e9, ">100")]


def anat(tag):
    p = f"{C}/anat_{tag}.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    b = np.array([x["bpm"] for x in r])
    ne = np.array([x["n_est"] for x in r], float)
    nt = np.array([x["n_true"] for x in r], float)
    rr = ne / nt
    return dict(n=len(r), bpm=b, ratio=rr,
                lo=float(np.mean(rr[b < 55] > 1.5)) if (b < 55).sum() else float("nan"),
                hi=float(np.mean(rr[b >= 100] < 0.7)) if (b >= 100).sum() else float("nan"))


def anat_rows(tag):
    p = f"{C}/anat_{tag}.json"
    if not os.path.exists(p):
        return None
    return {x["stem"]: x for x in json.load(open(p))}


def paired_delta(new_tag, old_tag):
    """doubled/halved deltas over the INTERSECTION of the two arms' track sets.

    error_anatomy can omit a track whose decoder emitted <4 beats -- a real
    failure of that model, which shrinks its own denominator. Differencing raw
    fractions therefore compares different track sets, with the omission
    correlated with failure. Intersect first.
    """
    A, B = anat_rows(new_tag), anat_rows(old_tag)
    if not A or not B:
        return None, None, 0
    common = sorted(set(A) & set(B))
    if not common:
        return None, None, 0

    def frac(d):
        b = np.array([d[s]["bpm"] for s in common])
        r = np.array([d[s]["n_est"] for s in common], float) / np.array(
            [max(d[s]["n_true"], 1) for s in common], float)
        lo = np.mean(r[b < 55] > 1.5) if (b < 55).sum() else np.nan
        hi = np.mean(r[b >= 100] < 0.7) if (b >= 100).sum() else np.nan
        return lo, hi

    (al, ah), (bl, bh) = frac(A), frac(B)
    return al - bl, ah - bh, len(common)


def f_of(tag):
    p = f"{C}/eval_{tag}.json"
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    return float(np.mean([x["F"] for x in r]))


print("# Morning report\n")
print("Everything below ran unattended. Numbers are fold-honest on all 217 SMC")
print("tracks; neither arm saw any SMC track in training.\n")

print("## 1. The overnight headline (already established before you slept)\n")
print("```")
print("Pooled across 1,501 tracks / 12 datasets, fold-honest released model:")
print("  band       n    AMLt-CMLt   doubled   halved")
print("  <55       59      0.2435     0.644     0.000")
print("  55-70    147      0.1919     0.401     0.007")
print("  70-100   358      0.0947     0.162     0.020")
print("  100-140  556      0.0282     0.025     0.022")
print("  >140     381      0.1063     0.013     0.163")
print("```")
print("Slow music is tracked at DOUBLE its rate, fast music at HALF, monotonically")
print("in both directions, minimum where the training corpora's tempo mass sits.\n")

print("## 2. Sign sweep -- does pos_weight cause it?\n")
print("Identical data/seed/epochs; ONLY the global beat pos_weight varies (8/32/128),")
print("bracketing the per-item range (10 at 180bpm .. 68 at 40bpm).\n")
print("```")
print(f"  {'pos_weight':>11}{'doubled(<55)':>14}{'halved(>100)':>14}")
rows = []
for pw in (8, 32, 128):
    a = anat(f"pw{pw}")
    if a:
        rows.append((pw, a["lo"], a["hi"]))
        print(f"  {pw:11d}{a['lo']:14.3f}{a['hi']:14.3f}")
    else:
        print(f"  {pw:11d}{'(pending)':>14}")
print("```")
if len(rows) == 3:
    d_lo = rows[2][1] - rows[0][1]
    d_hi = rows[2][2] - rows[0][2]
    print(f"\nSlope 8 -> 128: doubling {d_lo:+.3f}, halving {d_hi:+.3f}\n")
    if abs(d_lo) < 0.05 and abs(d_hi) < 0.05:
        print("**Doubling barely moves across a 16x weight range.** The prior is NOT in")
        print("the loss normalisation -- it lives in the learned representation, absorbed")
        print("from the training tempo distribution. The per-item fix is aimed at the")
        print("wrong place; the BEAT-COUNT loss is the right tool, because it constrains")
        print("the output directly instead of rebalancing the input.")
    elif d_lo > 0:
        print("**Higher pos_weight => MORE doubling.** The per-item fix is WRONG-SIGNED:")
        print("it raises the weight on exactly the slow tracks that already double.")
        print("Drop it; go to the beat-count loss.")
    else:
        print("**Higher pos_weight => LESS doubling.** The per-item fix is correctly")
        print("aimed, and the two long arms are the real test.")

print("\n## 3. The 2x2 (per-item loss cells)\n")
print("Compared against OUR OWN old-loss arms, not against released checkpoints.")
print("(The released model trained on a different, larger corpus with a different")
print("tempo mass, so its 0.356/0.316 is not the right baseline for these cells.)\n")
print("```")
print(f"  {'cell':<26}{'F(217)':>9}{'doubled':>10}{'halved':>9}")
# SEED-MATCHED. The per-item cells are seed 0 only, so they must be compared
# against the seed-0 old-loss runs -- NOT against the 3-seed mean, which was the
# original hardcode. Seed-matched the per-item shipped delta is +0.0008; against
# the 3-seed mean it looks like +0.0109, i.e. 93% of the apparent gain was two
# worse seeds averaged in on one side only. Within-arm seed spread is 0.019-0.022,
# larger than either effect.
base = {}
for tag, label in (("f0_shipped_s0", "shipped + old loss (s0)"),
                   ("f0_wide_s0", "wide + old loss (s0)"),
                   ("f0_shipped_peritem_s0", "shipped + per-item (s0)"),
                   ("f0_wide_peritem_s0", "wide + per-item (s0)")):
    a = anat(tag)
    f = f_of(tag)
    fs = f"{f:9.4f}" if f is not None else f"{'(pending)':>9}"
    if a:
        base[tag] = a
        print(f"  {label:<26}{fs}{a['lo']:10.3f}{a['hi']:9.3f}")
    else:
        print(f"  {label:<26}{fs}{'(pending)':>10}{'':>9}")
print("```")
print("  (3-seed means for the old-loss arms: shipped 0.5818, wide 0.5734 --")
print("   shown for context only, NOT used in the deltas below)")
for new, old, lab in (("f0_shipped_peritem_s0", "f0_shipped_s0", "shipped"),
                      ("f0_wide_peritem_s0", "f0_wide_s0", "wide")):
    if new in base and old in base:
        # anatomy fractions must be differenced over the SAME tracks:
        # error_anatomy can drop a model's degenerate tracks, shrinking its own
        # denominator, so raw fractions are over different track sets.
        dl, dh, n = paired_delta(new, old)
        if dl is None:
            continue
        improved = dl <= -0.10 and dh <= -0.10
        regressed = dl >= 0.05 or dh >= 0.05
        verdict = ("CONFIRMED" if improved
                   else "REGRESSION" if regressed          # <- was missing
                   else "NO EFFECT" if (abs(dl) < 0.05 and abs(dh) < 0.05)
                   else "MIXED")
        print(f"\n  {lab} (n={n} common tracks): doubled {dl:+.3f}, "
              f"halved {dh:+.3f}  -> **{verdict}**")
print("\nPre-registered gate (cache/phase4_decision.txt, written 00:42 before any")
print("checkpoint existed): CONFIRMED if doubled AND halved both shrink by >=0.10.")
print("KILL if both move <0.05 while per-item weights genuinely vary 7x.\n")

print("## 3b. Beat-count loss -- lambda sweep (25 epochs)\n")
print("Calibration on a trained model: the count term reads 1.943 at <55 bpm and")
print("0.479 at >100 -- it orders exactly like the doubling fractions. Mean 1.09,")
print("so lambda=0.3 would perturb a ~1.2 base loss by 27%; swept gentler.\n")
print("```")
print(f"  {'lambda':>8}{'doubled(<55)':>14}{'halved(>100)':>14}")
cl = {}
for tag, lam in (("cl005", 0.05), ("cl015", 0.15), ("cl050", 0.50)):
    a = anat(tag)
    if a:
        cl[lam] = a
        print(f"  {lam:8.2f}{a['lo']:14.3f}{a['hi']:14.3f}")
    else:
        print(f"  {lam:8.2f}{'(pending)':>14}")
b25 = anat("pw32")   # stock loss at the SAME 25-epoch budget -> the fair baseline
if b25:
    print(f"  {'none':>8}{b25['lo']:14.3f}{b25['hi']:14.3f}   <- stock loss, 25 ep")
print("```")
if cl and b25:
    best = min(cl.items(), key=lambda kv: kv[1]["lo"] + kv[1]["hi"])
    btag = {0.05: "cl005", 0.15: "cl015", 0.50: "cl050"}[best[0]]
    dl, dh, ncom = paired_delta(btag, "pw32")   # intersected, not raw fractions
    if dl is None:
        dl = dh = float("nan"); ncom = 0
    # NOTE: `best` is selected on the same statistic it is then scored on,
    # over 3 candidates with no correction. Treat as exploratory.
    v = ("WORKS" if (dl <= -0.10 and dh <= -0.10) else
         "REGRESSION" if (dl >= 0.05 or dh >= 0.05) else
         "TRADE-OFF" if (dl <= -0.05) != (dh <= -0.05) else
         "NO EFFECT")
    print(f"\n  best lambda={best[0]} (n={ncom} common): doubled {dl:+.3f}, "
          f"halved {dh:+.3f} -> **{v}**")
    print("  lambda selected on the same statistic it is scored on (3 candidates,")
    print("  no correction) -- exploratory, not a test.")
    print("  (compared against pw32 = stock loss at the SAME 25-epoch budget,")
    print("   not against the 100-epoch arms -- different training length)")

print("\n## 4. Also finished overnight\n")
for name, path, what in (
        ("convergence history", f"{C}/../MORNING_REPORT.md", ""),
):
    pass
for f, what in (("conv_shipped", "did the metrical gap EVER improve during training"),
                ("conv_wide", "same, wide-augmentation arm")):
    p = f"{C}/{f}.json"
    if os.path.exists(p):
        r = json.load(open(p))
        g = [x["gap_slow"] for x in r]
        e = [x["epoch"] for x in r]
        print(f"- **{f}** ({what}): gap<70 {g[0]:.4f} at epoch {e[0]} -> "
              f"{g[-1]:.4f} at epoch {e[-1]}, change {g[-1]-g[0]:+.4f}")
    else:
        print(f"- **{f}**: pending ({what})")
p = f"{C}/sdedit_guidance_fidelity.json"
print(f"- **SDEdit guidance sweep**: {'done' if os.path.exists(p) else 'pending'} "
      "(does beat-pathway guidance extend the no-drift noise range)")

print("\n## 5. What I'd do next, depending on section 2\n")
print("- prior NOT in the loss  -> build the beat-count loss |sum(sigmoid) - true_count|;")
print("  it constrains tempo directly and penalises doubling AND halving symmetrically.")
print("- prior IS in the loss    -> the 2x2 finishes the story; add seeds 1-2.")
print("- either way, the cross-dataset table in section 1 stands on its own as a")
print("  characterisation, and does not depend on any fix working.")
