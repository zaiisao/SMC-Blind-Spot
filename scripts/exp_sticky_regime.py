"""STICKY TWO-REGIME tempo transition kernel on the R1 DBN chassis -- falsification test.

Structure: duplicate the tempo grid so each physical interval appears twice, once per regime
(0 = steady / lambda_stiff, 1 = yielding / lambda_loose). The [2V, 2V] beat-boundary kernel is

    K_full[(i,a), (j,b)] = log P(s=b | s=a) + log K_{lambda_b}(j | i)

with regime matrix [[1-eps, eps], [eta, 1-eta]]. Read-out is unchanged because duplicated tempo
blocks carry identical interval values.

Gates (run before the sweep):
  A  eps = eta = 0.5 -> the regime is memoryless, so the tempo-marginal process is EXACTLY the
     w=0.5 two-scale mixture. Checked on the exact marginal likelihood (forward), which is the
     quantity that must agree bit-for-bit; Viterbi is max-over-regime rather than sum, so the
     decoded events are reported as a secondary, non-binding comparison.
  B  eps = eta = 1e-9 -> regimes disconnect; decode must equal the better-scoring of the two
     single-lambda decodes.
"""
import json
import sys
import copy
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vbpm" / "rungs_lab"))

from _harness import FPS, track_ids, load_gt, bthis_beat_prob, bthis_downbeat_prob, ev
from rungs.r2_em_dbn import R2GenerativeLambda
from rungs.bar_pointer.state_space import BarPointerStateSpace, NO_BEAT, BEAT, DOWNBEAT
from rungs.bar_pointer.structured_dp import StructuredBarPointerDP
from rungs.bar_pointer.readout import state_path_to_events
from rungs.deployment import threshold_crop

DEVICE = sys.argv[1] if len(sys.argv) > 1 else ("cuda:2" if torch.cuda.is_available() else "cpu")
OUTDIR = Path("/tmp/sticky")
OUTDIR.mkdir(parents=True, exist_ok=True)
JSONL = OUTDIR / "per_track.jsonl"

N_TRACKS = 24
SINGLE_LAMS = [1.0, 5.0, 20.0, 100.0, 200.0, 500.0]
PAIRS = [(100.0, 1.0), (200.0, 5.0)]
EPS_GRID = [0.5, 0.2, 0.1, 0.03, 0.01, 0.003, 1e-9]
W_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
CHASSIS = dict(min_bpm=30.0, max_bpm=215.0, observation_lambda=6)


# --------------------------------------------------------------------------------------------
def duplicated_space(space: BarPointerStateSpace, copies: int = 2) -> BarPointerStateSpace:
    """Re-run BarPointerStateSpace's derived-array construction with the interval grid tiled
    `copies` times. Every array in __init__ is built by concatenation over interval_frames, so
    repeated interval values are structurally fine; this function just makes that explicit."""
    s = copy.copy(space)
    iv = np.tile(space.interval_frames, copies).astype(np.int64)
    s.interval_frames = np.ascontiguousarray(iv)
    s.num_tempi = len(iv)
    bpb = space.beats_per_bar

    num_states_per_beat = int(iv.sum())
    first_state_per_tempo = np.cumsum(np.r_[0, iv[:-1]]).astype(np.int64)
    last_state_per_tempo = (np.cumsum(iv) - 1).astype(np.int64)
    position_within_beat = np.concatenate(
        [np.linspace(0, 1, i, endpoint=False) for i in iv])
    interval_of_each_state = np.concatenate([np.full(i, i, dtype=np.int64) for i in iv])

    s.num_states = bpb * num_states_per_beat
    s.first_states = np.stack([first_state_per_tempo + b * num_states_per_beat
                               for b in range(bpb)])
    s.last_states = np.stack([last_state_per_tempo + b * num_states_per_beat
                              for b in range(bpb)])
    s.state_positions = np.concatenate([position_within_beat + b for b in range(bpb)])
    s.state_interval_frames = np.tile(interval_of_each_state, bpb)

    w = 1.0 / space.observation_lambda
    pc = np.full(s.num_states, NO_BEAT, dtype=np.int64)
    pc[(s.state_positions % 1.0) < w] = BEAT
    pc[s.state_positions < w] = DOWNBEAT
    s.position_classes = pc
    return s


class Sticky(R2GenerativeLambda):
    """R2 chassis (single-lambda / mixture kernels) plus a parallel DUPLICATED chassis carrying
    the 2V tempo grid for the sticky regime model."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.V = self.chassis.state_spaces[0].num_tempi
        self.dup_spaces = [duplicated_space(s) for s in self.chassis.state_spaces]
        self.dup_dps = [StructuredBarPointerDP(s, device=self.device, dtype=torch.float32)
                        for s in self.dup_spaces]
        total = sum(s.num_states for s in self.dup_spaces)
        self.dup_init = [torch.full((s.num_states,), -float(np.log(total)),
                                    dtype=torch.float32, device=self.device)
                         for s in self.dup_spaces]
        self.dup_s2c = [torch.from_numpy(s.position_classes).to(self.device)
                        for s in self.dup_spaces]
        # regime label of every state, from its tempo block index
        self.dup_state_regime = [
            np.concatenate([np.full(int(i), 0 if k < self.V else 1, dtype=np.int64)
                            for k, i in enumerate(s.interval_frames)] * s.beats_per_bar)
            for s in self.dup_spaces]

    # --- kernels -----------------------------------------------------------------------------
    def log_mixture_kernel(self, w, lam_a, lam_b):
        return torch.logsumexp(torch.stack([
            float(np.log(w)) + self.log_kernel(lam_a),
            float(np.log(1.0 - w)) + self.log_kernel(lam_b)]), dim=0)

    def log_sticky_kernel(self, lam_stiff, lam_loose, eps, eta):
        K0, K1 = self.log_kernel(lam_stiff), self.log_kernel(lam_loose)   # [V, V]
        R = np.log(np.array([[1.0 - eps, eps], [eta, 1.0 - eta]]))
        V = self.V
        out = torch.empty((2 * V, 2 * V), dtype=K0.dtype, device=self.device)
        for a in range(2):
            for b in range(2):
                out[a * V:(a + 1) * V, b * V:(b + 1) * V] = float(R[a, b]) + (K0 if b == 0 else K1)
        return out

    # --- decoding ----------------------------------------------------------------------------
    @torch.no_grad()
    def _decode(self, activations, kernel, spaces, dps, inits, s2cs, regimes=None,
                deploy_threshold=0.2):
        arr = np.ascontiguousarray(np.asarray(activations, dtype=np.float64))
        arr, first_frame = threshold_crop(arr, deploy_threshold)
        if not arr.size:
            return {"beats": np.array([]), "downbeats": np.array([])}, float("nan")
        snap = arr.astype(np.float64)
        acts = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        densities = self.log_class_densities(acts)
        best = None
        for mi in range(len(spaces)):
            path, score = dps[mi].viterbi(inits[mi], kernel, densities,
                                          state_to_class=s2cs[mi], return_log_score=True)
            if best is None or score > best[0]:
                best = (score, path.cpu().numpy(), mi)
        score, path, mi = best
        ev_out = state_path_to_events(path, spaces[mi], self.chassis.fps,
                                      snap_to_activations=snap, first_frame=first_frame)
        yield_frac = float("nan")
        if regimes is not None:
            reg = regimes[mi]
            slot = dps[mi].first_state_slot
            dest = path[1:]
            is_boundary = slot[dest] >= 0
            n = int(is_boundary.sum())
            yield_frac = float(reg[dest[is_boundary]].mean()) if n else float("nan")
        return ev_out, score, yield_frac

    def decode_plain(self, acts, kernel):
        return self._decode(acts, kernel, self.chassis.state_spaces, self.chassis.dynamic_programs,
                            self.chassis.log_initial_distributions, self.chassis.state_to_classes)

    def decode_sticky(self, acts, kernel):
        return self._decode(acts, kernel, self.dup_spaces, self.dup_dps, self.dup_init,
                            self.dup_s2c, regimes=self.dup_state_regime)

    # --- exact marginal likelihood (gate A) --------------------------------------------------
    @torch.no_grad()
    def ll_plain(self, acts, kernel):
        d = self.log_class_densities(torch.from_numpy(acts.astype(np.float32)).to(self.device))
        return float(torch.logsumexp(torch.stack([
            dp.forward_log_likelihood(i, kernel, d, state_to_class=c)
            for dp, i, c in zip(self.chassis.dynamic_programs,
                                self.chassis.log_initial_distributions,
                                self.chassis.state_to_classes)]), dim=0))

    @torch.no_grad()
    def ll_sticky(self, acts, kernel):
        d = self.log_class_densities(torch.from_numpy(acts.astype(np.float32)).to(self.device))
        return float(torch.logsumexp(torch.stack([
            dp.forward_log_likelihood(i, kernel, d, state_to_class=c)
            for dp, i, c in zip(self.dup_dps, self.dup_init, self.dup_s2c)]), dim=0))


def acts_of(tid):
    return np.stack([bthis_beat_prob(tid), bthis_downbeat_prob(tid)], axis=-1)


def main():
    ids = track_ids()[:N_TRACKS]
    refs = {t: load_gt(t) for t in ids}
    acts = {t: acts_of(t) for t in ids}
    m = Sticky(fps=FPS, device=DEVICE, **CHASSIS)
    print(f"device={DEVICE} n={len(ids)} V={m.V} "
          f"num_states base={m.chassis.state_spaces[0].num_states} "
          f"dup={m.dup_spaces[0].num_states}", flush=True)

    report = {"n": len(ids), "device": DEVICE, "track_ids": ids,
              "V": m.V, "chassis": CHASSIS}

    # ---------- structural validity of the duplicated state space ----------
    b, d = m.chassis.state_spaces[0], m.dup_spaces[0]
    struct = {
        "num_states_doubles": bool(d.num_states == 2 * b.num_states),
        "intervals_tiled": bool(np.array_equal(d.interval_frames,
                                               np.tile(b.interval_frames, 2))),
        # Only beat 0 can match state-for-state: beat b's offset is b * num_states_per_beat, and
        # num_states_per_beat doubles under duplication. Within beat 0, regime-0 blocks must be
        # exactly the base blocks.
        "first_states_beat0_block0_match": bool(np.array_equal(d.first_states[0, :m.V],
                                                               b.first_states[0])),
        "last_states_beat0_block0_match": bool(np.array_equal(d.last_states[0, :m.V],
                                                              b.last_states[0])),
        "beat_offsets_ok": bool(np.array_equal(
            d.first_states, d.first_states[0][None, :]
            + (np.arange(b.beats_per_bar) * int(d.interval_frames.sum()))[:, None])),
        "block_sizes_ok": bool(np.array_equal(d.last_states - d.first_states + 1,
                                              np.tile(b.interval_frames, 2)[None, :].repeat(
                                                  b.beats_per_bar, 0))),
        "positions_block0_match": bool(np.allclose(
            d.state_positions[:int(b.interval_frames.sum())],
            b.state_positions[:int(b.interval_frames.sum())])),
        "classes_block0_match": bool(np.array_equal(
            d.position_classes[:int(b.interval_frames.sum())],
            b.position_classes[:int(b.interval_frames.sum())])),
    }
    report["structure_checks"] = struct
    print("structure:", struct, flush=True)
    if not all(struct.values()):
        print("STRUCTURE CHECK FAILED -- stopping", flush=True)
        (OUTDIR / "report.json").write_text(json.dumps(report, indent=1))
        return

    fh = open(JSONL, "w")

    def run_arm(name, kernel, sticky, meta):
        per = {}
        yf = []
        for t in ids:
            if sticky:
                e, sc, y = m.decode_sticky(acts[t], kernel)
                yf.append(y)
            else:
                e, sc, y = m.decode_plain(acts[t], kernel)
            s = ev(refs[t], e["beats"])
            per[t] = s
            fh.write(json.dumps({"arm": name, **meta, "track": t, "logscore": sc,
                                 "yield_frac": y, **s}) + "\n")
        fh.flush()
        mean = {k: float(np.mean([per[t][k] for t in ids]))
                for k in ("F", "CMLt", "AMLt", "CMLc", "AMLc")}
        yfm = float(np.nanmean(yf)) if yf else float("nan")
        print(f"{name:<28} F {mean['F']:.4f} CMLt {mean['CMLt']:.4f} AMLt {mean['AMLt']:.4f} "
              f"CMLc {mean['CMLc']:.4f} AMLc {mean['AMLc']:.4f} yield {yfm:.4f}", flush=True)
        return {"per": per, "mean": mean, "yield_frac": yfm}

    # ---------------------------- GATE A ----------------------------
    t0 = time.time()
    gate_ids = ids[:6]
    gateA = []
    for lam_s, lam_l in PAIRS:
        kmix = m.log_mixture_kernel(0.5, lam_s, lam_l)
        kst = m.log_sticky_kernel(lam_s, lam_l, 0.5, 0.5)
        # exact algebraic check: logsumexp over destination regime blocks == mixture kernel
        marg = torch.logsumexp(torch.stack([kst[:m.V, :m.V], kst[:m.V, m.V:]]), dim=0)
        alg = float((marg - kmix).abs().max())
        for t in gate_ids:
            a = acts[t]
            lm, ls = m.ll_plain(a, kmix), m.ll_sticky(a, kst)
            em, _, _ = m.decode_plain(a, kmix)
            es, _, yf = m.decode_sticky(a, kst)
            nb = min(len(em["beats"]), len(es["beats"]))
            dmax = float(np.abs(em["beats"][:nb] - es["beats"][:nb]).max()) if nb else float("nan")
            gateA.append({"pair": [lam_s, lam_l], "track": t, "kernel_max_abs_diff": alg,
                          "ll_mixture": lm, "ll_sticky": ls, "ll_abs_diff": abs(lm - ls),
                          "ll_rel_diff": abs(lm - ls) / abs(lm),
                          "n_beats_mix": len(em["beats"]), "n_beats_sticky": len(es["beats"]),
                          "max_beat_time_diff": dmax, "yield_frac": yf})
            print("GATE A", gateA[-1], flush=True)
    report["gate_A"] = gateA
    kern_ok = max(g["kernel_max_abs_diff"] for g in gateA)
    ll_ok = max(g["ll_rel_diff"] for g in gateA)
    report["gate_A_pass"] = bool(kern_ok < 1e-5 and ll_ok < 1e-5)
    print(f"GATE A: max kernel diff {kern_ok:.3e}, max relative LL diff {ll_ok:.3e}, "
          f"pass={report['gate_A_pass']}", flush=True)

    # ---------------------------- GATE B ----------------------------
    gateB = []
    for lam_s, lam_l in PAIRS:
        kst = m.log_sticky_kernel(lam_s, lam_l, 1e-9, 1e-9)
        for t in gate_ids:
            a = acts[t]
            es, ss, yf = m.decode_sticky(a, kst)
            cand = {}
            for lam in (lam_s, lam_l):
                e, s, _ = m.decode_plain(a, m.log_kernel(lam))
                cand[lam] = (s, e)
            better = max(cand, key=lambda L: cand[L][0])
            eb = cand[better][1]
            same_n = len(eb["beats"]) == len(es["beats"])
            dmax = (float(np.abs(eb["beats"] - es["beats"]).max())
                    if same_n and len(es["beats"]) else float("nan"))
            # The tiled uniform initial spreads mass over TWO regime copies, so the sticky
            # score carries a constant -log 2 bookkeeping term relative to a single-regime
            # decode. Verified empirically: offset = -0.69289..-0.69336 vs -log2 = -0.693147.
            ss_corr = ss + float(np.log(2.0))
            # NOTE ON THE GATE'S PREMISE. eps -> 0 does NOT disconnect the regimes here,
            # because madmom truncates transition probabilities <= np.spacing(1) to EXACTLY
            # zero before normalising. At lam=100 that forbids any |r-1| >= 0.360 outright.
            # A path needing such a transition has score -inf under the stiff kernel alone but
            # is reachable under sticky by paying the ~2*-log(eps) toll, so sticky can and
            # should BEAT the better single. The correct invariant is CONTAINMENT: sticky (after
            # the log-2 correction) is never WORSE than the better single, and decodes
            # identically whenever the winning path uses no forbidden transition.
            contains = bool(ss_corr >= cand[better][0] - 1e-3)
            gateB.append({"pair": [lam_s, lam_l], "track": t,
                          "sticky_score": ss, "sticky_score_log2_corrected": ss_corr,
                          "single_scores": {str(k): v[0] for k, v in cand.items()},
                          "better_single_lambda": better, "same_beat_count": bool(same_n),
                          "max_beat_time_diff": dmax, "yield_frac": yf,
                          "containment_ok": contains,
                          "beats_identical": bool(same_n and (np.isnan(dmax) or dmax < 1e-6))})
            print("GATE B", gateB[-1], flush=True)
    report["gate_B"] = gateB
    n_ident = sum(g["beats_identical"] for g in gateB)
    report["gate_B_beats_identical"] = f"{n_ident}/{len(gateB)}"
    report["gate_B_pass"] = bool(all(g["containment_ok"] for g in gateB))
    print(f"GATE B containment all-pass={report['gate_B_pass']}; "
          f"beats identical on {n_ident}/{len(gateB)} cells", flush=True)
    print(f"GATE B pass={report['gate_B_pass']}", flush=True)
    (OUTDIR / "report.json").write_text(json.dumps(report, indent=1))

    if not (report["gate_A_pass"] and report["gate_B_pass"]):
        print("A GATE FAILED -- not running the sweep", flush=True)
        fh.close()
        return

    # ---------------------------- baselines ----------------------------
    singles = {}
    for lam in SINGLE_LAMS:
        singles[str(lam)] = run_arm(f"single lam={lam}", m.log_kernel(lam), False, {"lam": lam})
        (OUTDIR / "report.json").write_text(json.dumps({**report, "singles": singles}, indent=1))

    mixtures = {}
    for lam_s, lam_l in PAIRS:
        for w in W_GRID:
            key = f"{lam_s}/{lam_l}/w{w}"
            mixtures[key] = run_arm(f"mix {key}", m.log_mixture_kernel(w, lam_s, lam_l), False,
                                    {"lam_stiff": lam_s, "lam_loose": lam_l, "w": w})
            (OUTDIR / "report.json").write_text(
                json.dumps({**report, "singles": singles, "mixtures": mixtures}, indent=1))

    # ---------------------------- sweep ----------------------------
    sticky = {}
    for lam_s, lam_l in PAIRS:
        for e in EPS_GRID:
            key = f"{lam_s}/{lam_l}/eps{e}"
            sticky[key] = run_arm(f"sticky {key}", m.log_sticky_kernel(lam_s, lam_l, e, e), True,
                                  {"lam_stiff": lam_s, "lam_loose": lam_l, "eps": e, "eta": e})
            (OUTDIR / "report.json").write_text(json.dumps(
                {**report, "singles": singles, "mixtures": mixtures, "sticky": sticky}, indent=1))
    fh.close()

    report.update({"singles": singles, "mixtures": mixtures, "sticky": sticky,
                   "elapsed_s": time.time() - t0})
    (OUTDIR / "report.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
