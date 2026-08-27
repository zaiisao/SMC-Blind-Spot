"""Does the loss actually implement the three clauses, on realistic inputs?

  1. flanked by beats the model already gets right   (the gate: a_j * a_{j+1})
  2. no annotated beat between those beats and me    (phase is within CONSECUTIVE beats)
  3. I'm near the centre of those two beats          (the bump at phi=0.5)

Earlier checks verified each in isolation. This one prints the ACTUAL weight c(t)
applied at specific frames under specific conditions, so the behaviour can be read
off rather than inferred, and checks whether those frames are counted by the loss
at all -- the shift-tolerance collar zeroes look_at near beats, and if the midpoint
falls inside the collar the penalty is silently inert.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this")
from beat_this.model.loss import GatedPhaseShiftTolerantBCELoss as GL

m = GL(pos_weight=32, tolerance=3, phase_gamma=1.3)


def weights(ibi, beat_logit=4.0, T=1200):
    """Return per-frame (phase, gate, bump, c_final, counted) at frame resolution."""
    t = torch.zeros(1, T)
    beats = list(range(ibi, T - ibi, ibi))
    t[0, beats] = 1.0
    p = torch.full((1, T), -6.0)
    for b in beats:
        p[0, b] = beat_logit
    ph = m._phase(t)
    d = (ph - 0.5).abs(); d = torch.minimum(d, 1 - d)
    bump = torch.nan_to_num(torch.exp(-(d ** 2) / (2 * m.phase_sigma ** 2)), nan=0.0)
    gate = m._gate(p, t)
    c = 1.0 + m.phase_gamma * gate * bump
    cc = m.crop(c, factor=2)
    ct = m.crop(t, factor=2)
    look = ct + (1 - m.spread(t, factor=2))
    counted = (ct <= 0) & (look > 0)
    off = (c.shape[-1] - cc.shape[-1]) // 2
    return ph[0].numpy(), gate[0].numpy(), bump[0].numpy(), cc[0].numpy(), counted[0].numpy(), off, beats


print("\n=== CLAUSE 3: penalty vs distance from the centre  (IBI=40 fr = 75 bpm)")
ph, gate, bump, cc, cnt, off, beats = weights(40)
b0 = beats[3]
print(f"  {'phase':>7}{'frame':>7}{'gate':>8}{'bump':>8}{'weight c':>10}{'counted':>9}")
for frac in (0.0, 0.10, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.90):
    f = int(round(b0 + frac * 40))
    j = f - off
    if not (0 <= j < len(cc)):
        continue
    print(f"  {frac:7.2f}{f:7d}{gate[f]:8.3f}{bump[f]:8.3f}{cc[j]:10.3f}"
          f"{str(bool(cnt[j])):>9}")

print("\n=== CLAUSE 1: same frame, but the model is NOT confident at the beats")
for lg, label in ((4.0, "confident (logit +4)"), (0.0, "unsure (logit 0)"),
                  (-4.0, "absent (logit -4)")):
    ph2, g2, b2, c2, k2, off2, bts = weights(40, beat_logit=lg)
    f = int(round(bts[3] + 0.5 * 40)); j = f - off2
    print(f"  {label:<22} gate={g2[f]:.4f}  weight at phi=0.5 -> {c2[j]:.4f}")

print("\n=== CLAUSE 2: phase is measured within CONSECUTIVE annotated beats")
t = torch.zeros(1, 600)
t[0, [100, 140, 180, 260]] = 1.0        # note the GAP 180 -> 260
ph3 = m._phase(t)[0].numpy()
for f, expect in ((160, "midpoint of 140-180"), (220, "midpoint of 180-260")):
    print(f"  frame {f}: phase={ph3[f]:.3f}   ({expect})")
print(f"  frame 90 (before first beat): phase={ph3[90]}  <- nan, no enclosing interval")
print(f"  frame 300 (after last beat):  phase={ph3[300]}  <- nan")

print("\n=== IS THE MIDPOINT EVEN COUNTED? collar vs tempo")
print("  the loss zeroes look_at within the shift-tolerance collar around each beat;")
print("  if the midpoint falls inside it, the phase penalty is INERT at that tempo.")
print(f"\n  {'BPM':>6}{'IBI(fr)':>9}{'mid dist':>10}{'counted':>9}{'weight at 0.5':>15}")
for bpm in (200, 180, 160, 140, 120, 100, 90, 75, 60, 45, 35):
    ibi = int(round(50 * 60 / bpm))
    if ibi < 6:
        continue
    ph4, g4, b4, c4, k4, off4, bts4 = weights(ibi)
    if len(bts4) < 5:
        continue
    f = int(round(bts4[3] + 0.5 * ibi)); j = f - off4
    if not (0 <= j < len(c4)):
        continue
    print(f"  {bpm:6d}{ibi:9d}{ibi/2:10.1f}{str(bool(k4[j])):>9}{c4[j]:15.3f}")
