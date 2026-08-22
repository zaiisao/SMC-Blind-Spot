"""R3 per-frame vs global on SMC -- OUT-OF-DOMAIN (SMC is not in our frontend's training), and the
hard tempo-anomaly set the SMC-blind-spot work centered on. Beat-only annotations; decode beats,
score beat-F, stratify by smoothed tempo range (does per-frame help most on real tempo change?)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/sogang/jaehoon/Analyze-SMC/vbpm/rungs_lab")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "bt_e2e"))

import final_eval; final_eval.DEVICE = "cuda:0"
import mir_eval
from train_bt import FPS, BT_SHIPPED_DECODE
from rungs.r3_conditioned_dbn import R3ConditionedFactors

DEMIX = ROOT / "cache" / "bt_demix" / "smc"
BEATS = Path("/home/sogang/jaehoon/Analyze-SMC/beat_this_annotations/smc/annotations/beats")


def smoothed_tempo_range(bt):
    ibi = np.diff(bt)
    if len(ibi) < 8:
        return 1.0
    sm = np.convolve(ibi, np.ones(4) / 4, mode="valid")
    return float(sm.max() / sm.min())


def main():
    r3 = R3ConditionedFactors(fps=FPS, device="cuda:0", lambda_base=100.0,
                              observation_lambda=BT_SHIPPED_DECODE["observation_lambda"])
    songs = []
    for npz in sorted(DEMIX.glob("*.npz")):
        beats_file = BEATS / f"{npz.stem.lower()}.beats"
        if not beats_file.exists():
            continue
        bt = np.array([float(x) for x in open(beats_file) if x.strip()])
        if len(bt) < 8:
            continue
        songs.append(dict(stem=npz.stem, mel_path=npz, beat_times=bt,
                          range=smoothed_tempo_range(bt)))
    print(f"SMC: {len(songs)} songs with demix+beats", flush=True)

    model = final_eval.load_model(ROOT / "experiments/bt_e2e/vanilla_best_prelim.pt")
    acts = {}
    for s in songs:
        x = np.load(s["mel_path"])["x"]
        with torch.no_grad():
            pred, _ = model(torch.from_numpy(x).unsqueeze(0).to("cuda:0"))
        acts[s["stem"]] = torch.sigmoid(pred[0, :, :2]).float()

    trained = torch.load(ROOT / "experiments/bt_e2e/r3_frozen_net.pt", map_location="cuda:0")["net"]
    zero = {k: torch.zeros_like(v) for k, v in r3.net.state_dict().items()}

    def F(grp):
        fs = []
        for s in grp:
            ev = r3.decode(acts[s["stem"]])
            est = mir_eval.beat.trim_beats(ev["beats"])
            fs.append(mir_eval.beat.f_measure(mir_eval.beat.trim_beats(s["beat_times"]), est)
                      if len(est) else 0.0)
        return float(np.mean(fs))

    ranges = np.array([s["range"] for s in songs])
    print(f"smoothed tempo-range: median {np.median(ranges):.3f} p90 {np.percentile(ranges,90):.3f} "
          f"max {ranges.max():.3f}  (SMC is far more tempo-varying than our CV data)", flush=True)
    print(f"{'subset':30s} {'n':>3} {'global':>8} {'perframe':>9} {'delta':>8}", flush=True)
    for label, grp in ([("ALL SMC", songs)] +
                       [(f"range>{t}", [s for s in songs if s["range"] > t])
                        for t in (1.05, 1.10, 1.20, 1.40)]):
        if not grp:
            continue
        r3.net.load_state_dict(zero); fg = F(grp)
        r3.net.load_state_dict(trained); fp = F(grp)
        print(f"{label:30s} {len(grp):>3} {fg:>8.4f} {fp:>9.4f} {fp-fg:>+8.4f}", flush=True)


if __name__ == "__main__":
    main()
