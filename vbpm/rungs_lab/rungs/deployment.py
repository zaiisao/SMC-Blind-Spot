"""Model-independent deployment lessons, shared by every rung.

These are madmom's shipped deployment conveniences (DBNDownBeatTrackingProcessor defaults), kept as
OPTIONS because they measurably help deployment F and the lesson should outlive any one rung.
They are ON by default in R1 (matching madmom-as-shipped); rung-to-rung comparisons and the
R1-vs-madmom certificate must opt OUT to the bare model (threshold=0.0, correct=False,
num_tempi=None). Measured contributions (val, 25 songs, {3,4}): threshold +0.005 beat F, peak
snap +0.014 on classical-heavy material (sign flips on tight pop/electronic -- genre-dependent,
not free).
"""
import numpy as np


def threshold_crop(activations: np.ndarray, threshold: float):
    """Crop to the main segment where any activation column reaches `threshold`.

    madmom's threshold_activations, behavior-copied: decoding never sees the quiet intro/outro, so
    the DBN cannot hallucinate beats in dead air (fade-ins/outs -- on our val data this fires almost
    exclusively on Ballroom's 30-second excerpts).

    Returns (cropped_activations, first_frame) -- add first_frame back to any frame index read
    from the cropped array. Empty crop (nothing reaches threshold) returns (empty, 0).
    """
    if not threshold:
        return activations, 0
    above = np.nonzero(np.max(activations, axis=1) >= threshold)[0]
    if not above.size:
        return activations[:0], 0
    first, last = int(above.min()), int(above.max()) + 1
    return activations[first:last], first
