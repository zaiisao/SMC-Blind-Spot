"""The shared bar-pointer chassis for R1-R4: state space, structured DP, dense reference, readout.

R2+ change ONLY how the HMM's factors are produced; everything a bar-pointer rung shares --
the Krebs 2015 state space, the O(states) structured DP (exact Viterbi + forward), the certified
dense reference, and the state-path -> events readout -- lives here so each rung is just its
factor definitions.
"""
from rungs.bar_pointer.state_space import BarPointerStateSpace
from rungs.bar_pointer.structured_dp import StructuredBarPointerDP
from rungs.bar_pointer.readout import state_path_to_events

__all__ = ["BarPointerStateSpace", "StructuredBarPointerDP", "state_path_to_events"]
