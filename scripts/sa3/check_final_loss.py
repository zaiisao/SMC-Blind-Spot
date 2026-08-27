"""Adversarial invariant tests for FinalPhaseCollarBCELoss.

These are written to BREAK the loss, not to confirm it. Every assertion targets a
specific way the term could be wrong: a collar too narrow to cover a correctly
placed peak (the defect that broke collar14), a gate that leaks gradient, a
penalty that touches the beat itself, a hidden change in total negative pressure
that would confound every tempo reading, degenerate annotations, mixed-tempo
batches, and an ignored mask.

Run:  /disk1/jaehoon/sa3env/bin/python scripts/sa3/check_final_loss.py
"""

import math
import sys
import os

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..",
        "third-party", "beat_this",
    ),
)

import torch

from beat_this.model.loss import (
    ShiftTolerantBCELoss,
    FinalPhaseCollarBCELoss,
)

torch.manual_seed(0)
DT = torch.float64
torch.set_default_dtype(DT)

PASS = 0
FAIL = 0
FAILURES = []
NOTES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        msg = f"FAIL  {name}" + (f"\n      {detail}" if detail else "")
        FAILURES.append(msg)
        print(msg)


def note(s):
    NOTES.append(s)
    print(f"      {s}")


def ibi_frames(bpm, fps=50):
    return int(round(fps * 60.0 / bpm))


def make_targets(bpm, n_beats=16, pad=40, batch=1, fps=50):
    ibi = ibi_frames(bpm, fps)
    T = pad * 2 + ibi * (n_beats - 1) + 1
    y = torch.zeros(batch, T, dtype=DT)
    pos = [pad + i * ibi for i in range(n_beats)]
    y[:, pos] = 1.0
    return y, pos, ibi


# ---------------------------------------------------------------------------
# 1. EXACT reduction to the stock loss when every new term is disabled.
# ---------------------------------------------------------------------------
def test_reduction():
    stock = ShiftTolerantBCELoss(pos_weight=32.0, tolerance=3).to(DT)
    for label, kw in [
        ("gamma=0, collar off", dict(phase_gamma=0.0, relative_collar=False)),
        # the relative collar must ALSO reduce to stock when the floor binds:
        # k=0.0 clips every H up to 2*tolerance, so it cannot differ from stock.
        ("gamma=0, collar on k=0", dict(phase_gamma=0.0, relative_collar=True,
                                        collar_k=0.0)),
    ]:
        final = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3, **kw).to(DT)
        worst = 0.0
        for bpm in (40, 55, 70, 100, 143, 180):
            y, pos, ibi = make_targets(bpm, batch=3)
            z = torch.randn_like(y) * 4.0
            m = torch.ones_like(y)
            m[1, -30:] = 0.0
            m[2] = 0.0
            a = final(z, y, m)
            b = stock(z, y, m)
            worst = max(worst, float((a - b).abs()))
        check(f"reduces EXACTLY to stock ({label}), atol 1e-7",
              worst <= 1e-7, f"max |final - stock| = {worst:.3e}")


# ---------------------------------------------------------------------------
# 2. NEVER pushes down a correctly placed prediction. dL/dz computed directly.
#    This is the bug that broke collar14.
# ---------------------------------------------------------------------------
def test_no_downward_push():
    tempos = (40, 55, 70, 100, 143, 180)
    offsets = range(-3, 4)
    logits = (2.0, 4.0, 8.0, 12.0)
    for label, kw in [
        ("default (collar off)", dict(relative_collar=False)),
        ("relative collar k=0.14, floor 6", dict(relative_collar=True,
                                                 collar_k=0.14)),
    ]:
        loss = FinalPhaseCollarBCELoss(
            pos_weight=32.0, tolerance=3, phase_gamma=1.3, phase_sigma=0.06, **kw
        ).to(DT)
        viol = []
        n = 0
        for bpm in tempos:
            y, pos, ibi = make_targets(bpm)
            for off in offsets:
                for L in logits:
                    z = torch.full_like(y, -8.0)
                    for p in pos:
                        z[0, p + off] = L
                    z.requires_grad_(True)
                    loss(z, y).backward()
                    g = z.grad[0, [p + off for p in pos]]
                    n += 1
                    if float(g.max()) > 1e-12:
                        viol.append((bpm, off, L, float(g.max())))
        check(
            f"never pushes a correct peak DOWN [{label}] "
            f"({n} tempo/offset/logit configs)",
            not viol,
            f"{len(viol)}/{n} violations, first 5: {viol[:5]}",
        )

    # ADVERSARIAL CONTROL: the same test must FAIL for a sub-floor collar, or the
    # test is not measuring anything. min_half < 2*tolerance is rejected by the
    # constructor, so reach past it to prove the floor is what saves us.
    bad = FinalPhaseCollarBCELoss(
        pos_weight=32.0, tolerance=3, phase_gamma=1.3, relative_collar=True,
        collar_k=0.14,
    ).to(DT)
    object.__setattr__(bad, "min_half", 2)
    found = []
    for bpm in tempos:
        y, pos, ibi = make_targets(bpm)
        for off in range(-3, 4):
            z = torch.full_like(y, -8.0)
            for p in pos:
                z[0, p + off] = 8.0
            z.requires_grad_(True)
            bad(z, y).backward()
            if float(z.grad[0, [p + off for p in pos]].max()) > 1e-12:
                found.append((bpm, off))
    check("control: sub-floor collar (min_half=2) DOES push correct peaks down",
          len(found) > 0,
          "no violations found -- the floor test above proves nothing")
    note(f"control violations at min_half=2: {len(found)} of {len(tempos)*7} "
         f"(tempi {sorted(set(b for b, _ in found))})")

    # and the constructor must refuse it
    try:
        FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3, min_half=2)
        ok = False
    except ValueError:
        ok = True
    check("constructor rejects min_half below 2*tolerance", ok)


# ---------------------------------------------------------------------------
# 3. Every gate / evidence term is detached.
# ---------------------------------------------------------------------------
def test_detached():
    loss = FinalPhaseCollarBCELoss(
        pos_weight=32.0, tolerance=3, phase_gamma=1.3, relative_collar=True,
        halving_gamma=0.3,
    ).to(DT)
    y, pos, ibi = make_targets(100, batch=2)
    z = (torch.randn_like(y) * 3.0).requires_grad_(True)

    h = loss._halving_boost(z, y)
    check("halving boost has no grad_fn and does not require grad",
          h.grad_fn is None and not h.requires_grad,
          f"grad_fn={h.grad_fn}, requires_grad={h.requires_grad}")
    c = loss._phase_weight(y)
    check("phase weight has no grad_fn and does not require grad",
          c.grad_fn is None and not c.requires_grad)
    col = loss._collar(y)
    check("collar has no grad_fn and does not require grad",
          col.grad_fn is None and not col.requires_grad)

    # stronger: the gradient must equal the one obtained with the gate frozen as
    # a plain constant. If any gradient leaked through the gate these differ.
    loss(z, y).backward()
    g1 = z.grad.clone()
    z2 = z.detach().clone().requires_grad_(True)
    hconst = h.detach().clone()
    spread = loss.crop(loss.spread(z2))
    ct = loss.crop(y, factor=2)
    la = ct + (1 - loss.crop(loss._collar(y), factor=2))
    posm = ct > 0
    w = torch.where(posm, 1.0 + 0.3 * loss.crop(hconst, factor=2),
                    loss.crop(loss._phase_weight(y), factor=2))
    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        spread, ct, weight=la * w, pos_weight=loss.pos_weight)
    ref.backward()
    d = float((g1 - z2.grad).abs().max())
    check("gradient identical to the gate-frozen reference (no leak)",
          d <= 1e-12, f"max |dg| = {d:.3e}")


# ---------------------------------------------------------------------------
# 4. The penalty is exactly zero at phi = 0 (and outside any interval).
# ---------------------------------------------------------------------------
def test_zero_at_phase_zero():
    loss = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3, phase_gamma=1.3)
    worst_beat = 0.0
    worst_out = 0.0
    gmax = 0.0
    gmin = math.inf
    for bpm in (40, 55, 70, 100, 143, 180):
        y, pos, ibi = make_targets(bpm)
        c = loss._phase_weight(y)
        worst_beat = max(worst_beat, float((c[0, pos] - 1.0).abs().max()))
        outside = torch.ones(y.shape[1], dtype=torch.bool)
        outside[pos[0]: pos[-1] + 1] = False
        worst_out = max(worst_out, float((c[0, outside] - 1.0).abs().max()))
        gmax = max(gmax, float(c.max()))
        gmin = min(gmin, float(c.min()))
    # EXACTLY zero, not merely small: an untruncated Gaussian leaves 8e-16 of
    # penalty on the beat itself, which contradicts the term's whole premise.
    check("c == 1.0 EXACTLY at every annotated beat (phi = 0)",
          worst_beat == 0.0, f"max |c-1| at beats = {worst_beat:.3e}")
    check("c == 1.0 EXACTLY outside [A_0, A_last] (phi undefined)",
          worst_out == 0.0, f"max |c-1| outside = {worst_out:.3e}")
    # ceiling is 1 + gamma, reached exactly when a frame lands on phi = 1/2
    check("1.0 <= c <= 1 + gamma = 2.3 everywhere",
          gmin >= 1.0 and gmax <= 2.3 + 1e-12,
          f"min {gmin:.6f} max {gmax:.6f}")
    note(f"phase weight range over 40-180 bpm: [{gmin:.4f}, {gmax:.4f}] "
         f"(ceiling 1 + gamma = 2.3000)")

    # positive side untouched by the phase term: sweep gamma, positives' total
    # contribution must not move.
    y, pos, ibi = make_targets(100)
    z = torch.randn_like(y) * 3.0
    tot = []
    for g in (0.0, 1.3, 3.0):
        lo = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3, phase_gamma=g)
        sp = lo.crop(lo.spread(z))
        ct = lo.crop(y, factor=2)
        la = ct + (1 - lo.crop(lo._collar(y), factor=2))
        w = torch.where(ct > 0, torch.ones_like(ct),
                        lo.crop(lo._phase_weight(y), factor=2))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            sp, ct, weight=la * w, pos_weight=lo.pos_weight, reduction="none")
        tot.append(float(bce[ct > 0].sum()))
    check("positive-side contribution independent of phase_gamma (0 -> 3)",
          max(tot) - min(tot) == 0.0, f"totals {tot}")


# ---------------------------------------------------------------------------
# 5. Total negative pressure vs stock, quantified per tempo. Reported, not hidden.
#    An unnoticed change in overall negative pressure produces halving on its own
#    and confounds every tempo reading.
# ---------------------------------------------------------------------------
def test_negative_pressure():
    rows = []
    for bpm in (40, 55, 70, 100, 143, 180):
        y, pos, ibi = make_targets(bpm, n_beats=24)
        stock = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3,
                                        phase_gamma=0.0)
        for label, lo in [
            ("phase", FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3,
                                              phase_gamma=1.3)),
            ("phase+collar", FinalPhaseCollarBCELoss(
                pos_weight=32.0, tolerance=3, phase_gamma=1.3,
                relative_collar=True, collar_k=0.14)),
        ]:
            ct = lo.crop(y, factor=2)
            neg = ct <= 0

            def negmass(l):
                la = ct + (1 - l.crop(l._collar(y), factor=2))
                w = torch.where(ct > 0, torch.ones_like(ct),
                                l.crop(l._phase_weight(y), factor=2))
                return float(((la * w) * neg).sum())

            b = negmass(stock)
            a = negmass(lo)
            rows.append((bpm, label, b, a, a / b - 1.0))
    print("\n  NEGATIVE PRESSURE (sum of negative-side weights per excerpt)")
    print("  bpm  variant        stock      final    delta")
    for bpm, label, b, a, r in rows:
        print(f"  {bpm:>3}  {label:<13} {b:8.1f}  {a:8.1f}  {r:+7.2%}")
    for bpm, label, b, a, r in rows:
        note(f"neg pressure {label} @{bpm}bpm: {r:+.2%} vs stock")
    # The phase term is a pure ADDITION on the negative side, so it can only
    # raise the total. Assert only what is structurally true, and report the size.
    ph = [r for bpm, lab, b, a, r in rows if lab == "phase"]
    check("phase term never LOWERS negative pressure (it only adds)",
          min(ph) >= -1e-12, f"min delta {min(ph):+.4%}")
    # The only bound the geometry actually implies is the ceiling gamma: every
    # negative weight lies in [1, 1+gamma], so the total cannot rise by more than
    # gamma = 130%. Any tighter number would be invented, so it is not asserted.
    check("phase-term pressure increase bounded by gamma (structural ceiling)",
          max(ph) <= 1.3 + 1e-12, f"max delta {max(ph):+.4%}")
    # KNOWN CONFOUND, reported not hidden: the increase is NOT tempo-flat.
    slow = [r for bpm, lab, b, a, r in rows if lab == "phase" and bpm <= 55]
    fastp = [r for bpm, lab, b, a, r in rows if lab == "phase" and bpm >= 143]
    print(f"\n  KNOWN CONFOUND: the un-normalised phase term raises total "
          f"negative\n  pressure by {min(slow):+.1%}..{max(slow):+.1%} at 40-55 bpm "
          f"but {min(fastp):+.1%}..{max(fastp):+.1%} at 143-180 bpm.\n"
          f"  Extra negative pressure alone produces halving, and this one is\n"
          f"  tempo-dependent, so ANY tempo-split reading of a phase-term arm is\n"
          f"  confounded unless the arm normalises or the confound is measured out.")
    note(f"tempo skew of the added negative pressure: "
         f"{max(fastp) - min(slow):+.1%} between 40 and 180 bpm")
    check("the tempo skew is real and must be reported (not asserted away)",
          max(fastp) > max(slow),
          "skew absent -- update the confound note above")
    coll = [(bpm, r) for bpm, lab, b, a, r in rows if lab == "phase+collar"]
    fast = [r for bpm, r in coll if bpm >= 70]
    check("relative collar is a NO-OP at >= 65 bpm (floor binds; k=0.14)",
          max(abs(r - p) for (bpm, r), p in zip(
              [(b, r) for b, r in coll if b >= 70],
              [r for bpm, lab, b, a, r in rows if lab == "phase" and bpm >= 70])
          ) <= 1e-12,
          f"fast-tempo deltas differ: {fast}")


# ---------------------------------------------------------------------------
# 6. Degenerate annotations stay finite.
# ---------------------------------------------------------------------------
def test_degenerate():
    T = 400
    cases = {}
    cases["no beats"] = torch.zeros(1, T, dtype=DT)
    y1 = torch.zeros(1, T, dtype=DT); y1[0, 200] = 1.0
    cases["one beat"] = y1
    y2 = torch.zeros(1, T, dtype=DT); y2[0, [100, 300]] = 1.0
    cases["two beats"] = y2
    y3 = torch.zeros(1, T, dtype=DT); y3[0, [100, 101]] = 1.0
    cases["two adjacent beats (IBI=1)"] = y3
    cases["all frames beats"] = torch.ones(1, T, dtype=DT)
    for kwlabel, kw in [("collar off", dict(relative_collar=False)),
                        ("collar on", dict(relative_collar=True, collar_k=0.14))]:
        lo = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3,
                                     phase_gamma=1.3, halving_gamma=0.3,
                                     **kw).to(DT)
        for name, y in cases.items():
            z = (torch.randn_like(y) * 3.0).requires_grad_(True)
            v = lo(z, y)
            v.backward()
            ok = torch.isfinite(v) and bool(torch.isfinite(z.grad).all())
            check(f"finite loss and grad: {name} [{kwlabel}]", ok,
                  f"loss={float(v)}, grad finite="
                  f"{bool(torch.isfinite(z.grad).all())}")


# ---------------------------------------------------------------------------
# 7. Mixed-tempo batch: per-item quantities must differ WITHIN one batch.
# ---------------------------------------------------------------------------
def test_mixed_batch():
    ibis = [ibi_frames(b) for b in (40, 100, 180)]
    T = 40 * 2 + max(ibis) * 15 + 1
    y = torch.zeros(3, T, dtype=DT)
    for i, ibi in enumerate(ibis):
        for k in range(15):
            y[i, 40 + k * ibi] = 1.0
    lo = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3, phase_gamma=1.3,
                                 relative_collar=True, collar_k=0.14).to(DT)
    col = lo._collar(y)
    halves = [int(col[i].sum().item() / max(1, int(y[i].sum().item()))) for i in
              range(3)]
    check("per-item collar width differs within one batch (40/100/180 bpm)",
          halves[0] != halves[1],
          f"per-beat collar frames = {halves}")
    note(f"collar frames per beat at 40/100/180 bpm = {halves} "
         f"(floor 2*tau+1 = 13)")

    # batching must not change any item's loss: compute each item alone.
    z = torch.randn_like(y) * 3.0
    m = torch.ones_like(y)
    per = [float(lo(z[i:i + 1], y[i:i + 1], m[i:i + 1])) for i in range(3)]
    both = float(lo(z, y, m))
    d = abs(both - sum(per) / 3.0)
    check("batched loss == mean of per-item losses (no cross-item leakage)",
          d <= 1e-10, f"batched {both:.12f} vs mean {sum(per)/3.0:.12f}")

    # phase weights must genuinely differ between items
    c = lo._phase_weight(y)
    check("per-item phase weights differ within one batch",
          not torch.allclose(c[0], c[1]))


# ---------------------------------------------------------------------------
# 8. Mask honoured.
# ---------------------------------------------------------------------------
def test_mask():
    y, pos, ibi = make_targets(100, n_beats=20)
    y = y.repeat(2, 1)
    T = y.shape[1]
    z = torch.randn(2, T, dtype=DT) * 3.0
    m = torch.ones_like(y)
    m[0, T // 2:] = 0.0            # padding tail
    m[1] = 0.0                     # whole item unannotated (downbeat case)
    for kw in [dict(relative_collar=False),
               dict(relative_collar=True, collar_k=0.14)]:
        lo = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3,
                                     phase_gamma=1.3, **kw).to(DT)
        base = float(lo(z, y, m))
        z2 = z.clone()
        # perturb well inside the masked regions (> tolerance from any edge)
        z2[0, T // 2 + 10:] += 17.0
        z2[1, 10:-10] += 17.0
        pert = float(lo(z2, y, m))
        check(f"masked frames cannot affect the loss "
              f"[{'relative' if kw.get('relative_collar') else 'fixed'} collar]",
              abs(base - pert) <= 1e-12,
              f"base {base:.12f} vs perturbed {pert:.12f}")
        # and masked frames must receive no gradient
        zz = z.clone().requires_grad_(True)
        lo(zz, y, m).backward()
        g = zz.grad
        check(f"no gradient on fully masked item "
              f"[{'relative' if kw.get('relative_collar') else 'fixed'} collar]",
              float(g[1].abs().max()) == 0.0,
              f"max |grad| on masked item = {float(g[1].abs().max()):.3e}")


# ---------------------------------------------------------------------------
# 9. Parity coupling between phase_gamma and pos_weight (process invariant).
# ---------------------------------------------------------------------------
def test_parity():
    for w, want in [(32.0, 32 / 14 - 1), (20.0, 20 / 14 - 1), (14.0, 0.0)]:
        lo = FinalPhaseCollarBCELoss(pos_weight=w, tolerance=3, phase_gamma=1.3)
        check(f"parity_gamma(pos_weight={w:g}) == w/14 - 1 = {want:.4f}",
              abs(lo.parity_gamma() - want) < 1e-9,
              f"got {lo.parity_gamma():.6f}")
    lo = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3, phase_gamma=1.3)
    check("gamma=1.3 passes parity at pos_weight=32", lo.check_parity())
    lo14 = FinalPhaseCollarBCELoss(pos_weight=14.0, tolerance=3, phase_gamma=1.3)
    try:
        lo14.check_parity()
        raised = False
    except ValueError:
        raised = True
    check("gamma=1.3 at pos_weight=14 is REFUSED (parity gamma is 0 there)",
          raised)
    check("...unless parity_override is set",
          lo14.check_parity(parity_override=True) is False)
    # joint parity with the halving boost
    lj = FinalPhaseCollarBCELoss(pos_weight=32.0, tolerance=3, phase_gamma=1.3,
                                 halving_gamma=0.295)
    want = (32 / 14) * (1 + 0.3385 * 0.295) - 1
    check(f"joint parity with gamma_h=0.295 requires gamma={want:.3f}, "
          f"so 1.3 is refused",
          abs(lj.parity_gamma() - want) < 1e-9 and not lj.check_parity(
              parity_override=True),
          f"parity_gamma={lj.parity_gamma():.4f}")


def main():
    test_reduction()
    test_no_downward_push()
    test_detached()
    test_zero_at_phase_zero()
    test_negative_pressure()
    test_degenerate()
    test_mixed_batch()
    test_mask()
    test_parity()
    print("\n" + "=" * 60)
    print(f"PASSED {PASS}   FAILED {FAIL}")
    if FAILURES:
        print("\nFAILURES VERBATIM:")
        for f in FAILURES:
            print(f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
