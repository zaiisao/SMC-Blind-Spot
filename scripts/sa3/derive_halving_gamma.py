"""Derive gamma_h from cost parity, the same way gamma=1.3 was derived.

The original accounting, per track of N beats under the stock loss:

    double-tempo   all N beats hit, N extra peaks        ~  7N     <- cheapest
    half-tempo     N/2 beats missed                      ~ 16N
    offbeat        all N missed AND N spurious           ~ 39N

7N and 16N are not arbitrary: a spurious peak costs 1 unit of negative BCE, a
missed beat costs pos_weight units. With w+ = 32 and the shift-tolerant collar,
N extra peaks cost about 7N while N/2 missed beats cost about 16N. Doubling is
therefore the cheapest wrong answer available, which is why slow tracks double.

gamma was set by  7N(1 + gamma) = 16N  ->  gamma = 9/7 = 1.286 ~ 1.3, i.e. make
DOUBLING cost what HALVING costs. That is parity, and it is already achieved by
the suppression term alone.

gamma_h = 1.3 was then chosen for SYMMETRY, not derived -- and it breaks parity in
the other direction, because it multiplies the halving cost that was already the
expensive one:

    doubling  7N(1 + gamma)          = 16.0N
    halving   16N(1 + gamma_h)       = 36.8N   at gamma_h = 1.3   <- over-corrected

Measured consequence at epochs 4-19: bidir13 doubles at 37.0% against ph13's
25.9%, exactly the direction over-correcting halving predicts.

This computes gamma_h from the same accounting, and reports what each choice
implies, so the value is derived rather than guessed.
"""
import numpy as np

W_POS = 32.0
TAU = 3


def costs(w_pos=W_POS, tau=TAU):
    """Cost in BCE units of each wrong answer, per N true beats.

    A spurious peak is a negative frame driven high: it contributes on the order
    of 1 unit each, times the (2*tau+1) frames the shift-tolerant window makes
    effective -- but the collar removes 2*tau of them, leaving ~ (2*tau+1)/2 per
    peak in practice. A missed beat is a positive frame left low: w_pos units.
    The published ratio 7 : 16 fixes the constants, so use them directly rather
    than re-deriving with a different convention.
    """
    c_double = 7.0          # per N beats: N extra peaks
    c_half = 16.0           # per N beats: N/2 missed beats at w_pos each
    return c_double, c_half


def main():
    cd, ch = costs()
    print(f"\n  stock costs per N beats:   doubling {cd:.1f}N   halving {ch:.1f}N")
    print(f"  doubling is {ch/cd:.2f}x cheaper -> the model prefers it\n")

    g = ch / cd - 1
    print(f"  suppression term: 7N(1+gamma) = 16N  ->  gamma = {g:.3f}  (shipped 1.3)")
    print(f"    after it:  doubling {cd*(1+g):.1f}N   halving {ch:.1f}N   -> PARITY\n")

    print("  now add the halving boost. It multiplies the halving cost:")
    print(f"  {'gamma_h':>9}{'doubling':>11}{'halving':>10}{'ratio':>9}  verdict")
    for gh in (0.0, 0.2, 0.3, 0.5, 0.8, 1.3):
        d, h = cd * (1 + g), ch * (1 + gh)
        v = ("parity" if abs(h / d - 1) < 0.02 else
             "halving now dearer -> pushes toward DOUBLING" if h > d else
             "doubling dearer")
        print(f"  {gh:9.2f}{d:11.1f}{h:10.1f}{h/d:9.2f}  {v}")

    print("\n  Parity with BOTH terms requires solving  7N(1+gamma) = 16N(1+gamma_h):")
    gh_par = cd * (1 + g) / ch - 1
    print(f"    gamma_h = {gh_par:+.3f}   -- i.e. ZERO. With gamma already at parity,")
    print( "    any positive gamma_h over-corrects. The two terms cannot BOTH be")
    print( "    calibrated to parity; one of them has to give.\n")

    print("  So the parity frame has to be restated. Split the correction:")
    print("  raise doubling and lower halving until they MEET, instead of moving")
    print("  doubling all the way to halving's cost.")
    for split in (0.5,):
        target = np.sqrt(cd * ch)          # geometric mean: symmetric in log-cost
        gd = target / cd - 1
        gh = target / ch - 1
        print(f"    meet at the geometric mean {target:.2f}N:")
        print(f"      gamma   = {gd:.3f}   (doubling {cd:.1f}N -> {cd*(1+gd):.1f}N)")
        print(f"      gamma_h = {gh:.3f}   (halving  {ch:.1f}N -> {ch*(1+gh):.1f}N)")
        print(f"      but gamma_h < 0 is not expressible: the boost can only ADD cost.")

    print("\n  PRACTICAL CHOICE. gamma_h must be > 0 and small enough that halving")
    print("  does not overtake doubling by more than the measurement noise. Taking")
    print("  the largest gamma_h whose halving cost stays within 10% of doubling:")
    gh_max = (cd * (1 + g) * 1.10) / ch - 1
    print(f"    gamma_h_max = {gh_max:.3f}")
    print(f"    at that value: doubling {cd*(1+g):.1f}N   halving {ch*(1+gh_max):.1f}N"
          f"   ratio {ch*(1+gh_max)/(cd*(1+g)):.2f}")
    print(f"\n  RECOMMENDATION: gamma_h = {round(gh_max,2):.2f} (was 1.3, a {1.3/max(gh_max,1e-9):.1f}x"
          f" over-shoot)")


if __name__ == "__main__":
    main()
