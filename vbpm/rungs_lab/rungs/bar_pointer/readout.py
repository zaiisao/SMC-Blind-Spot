"""MAP state path -> musical events. Shared by R1-R4 (the read-off must be identical across rungs,
or the ladder stops being a controlled comparison).
"""
import numpy as np

from rungs.bar_pointer.state_space import BEAT


def state_path_to_events(state_path, state_space, fps: float, snap_to_activations=None,
                         first_frame: int = 0) -> dict:
    """state_path: [num_frames] state indices from Viterbi. Returns {'beats': seconds,
    'downbeats': seconds}.

    A beat is the ENTRY into any beat region (position class BEAT or DOWNBEAT); a downbeat is the
    entry into the downbeat region. The downbeat is also a beat, so it appears in both lists --
    matching R0/madmom, whose joint DBN likewise emits the downbeat as a beat with position 1.

    Implemented madmom-literally (`np.diff(positions.astype(int)) != 0`, its `correct=False`
    branch): a beat fires only where the integer beat counter CHANGES between frames, so frame 0
    is NEVER a beat even when the path starts inside a beat region. That edge case is real --
    measured on real songs, ~1 in 6 paths starts in-region, and an idealized "entry into region"
    read-out (which counts frame 0) emitted one extra beat at t=0 on exactly those songs. NOTE the
    asymmetry, faithful to madmom: its correct=True branch DOES count a region that starts at
    frame 0 (see below); only correct=False cannot fire there.

    snap_to_activations: pass the [num_frames, 2] activations the model saw to instead report each
    beat at the strongest activation frame WITHIN its beat region (madmom's `correct=True`,
    behavior-copied down to the flat argmax over both columns). A deployment lesson, not the model:
    the Viterbi region entry can sit a frame or two off the perceptual onset, and on soft-onset
    material (classical piano) snapping to the evidence peak is worth ~+0.014 beat F; on tight
    pop/electronic it does nothing or slightly hurts.

    first_frame: offset added to every frame index before conversion to seconds -- pass the crop
    offset when the activations were threshold-cropped (rungs/deployment.py).
    """
    position_classes = state_space.position_classes[np.asarray(state_path)]
    is_in_beat_region = position_classes >= BEAT

    if snap_to_activations is None:
        # madmom's correct=False read-out, op for op: beat where the integer beat counter changes
        # between consecutive frames; downbeat where the counter lands on beat 0 of the bar.
        integer_positions = state_space.state_positions[np.asarray(state_path)].astype(int)
        beat_frames = np.nonzero(np.diff(integer_positions))[0] + 1
        downbeat_frames = beat_frames[integer_positions[beat_frames] == 0]
    else:
        # madmom's correct=True: for each contiguous beat region of the path, report the frame with
        # the strongest activation (flat argmax over both columns, hence the // 2).
        edges = np.nonzero(np.diff(is_in_beat_region.astype(int)))[0] + 1
        if is_in_beat_region[0]:
            edges = np.r_[0, edges]
        if is_in_beat_region[-1]:
            edges = np.r_[edges, len(is_in_beat_region)]
        beat_frames = np.array([
            int(np.argmax(snap_to_activations[left:right]) // 2 + left)
            for left, right in edges.reshape(-1, 2)], dtype=np.int64)
        beat_positions = state_space.state_positions[np.asarray(state_path)[beat_frames]]
        downbeat_frames = beat_frames[beat_positions.astype(int) == 0]

    return {"beats": (beat_frames + first_frame) / fps,
            "downbeats": (downbeat_frames + first_frame) / fps}
