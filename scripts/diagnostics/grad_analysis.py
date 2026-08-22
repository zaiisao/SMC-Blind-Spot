"""Gradient forensics at Beat This's released minimum: why does training converge somewhere
inconsistent with oracle SMC performance?

Three measurements at theta* (final0), using the EXACT training objective
(ShiftTolerantBCELoss, pos_weight=1, 1500-frame crops):

1. GRADIENT MASS: per-tempo-group mean gradients over the training-style data.
   SMC is IN the training mixture, yet ballroom/beatles/hjdb-style corpora have ~zero
   sub-55bpm tracks. How large is the slow-SMC corrective gradient vs the majority gradient?
2. CONFLICT vs STARVATION: pairwise cosines between group gradients (whole-model and
   per-module). cos(g_smc_slow, g_train) << 0 => trade-off minimum (fixing SMC must hurt
   the majority). cos ~ 0 with tiny relative mass => starvation (the optimizer simply
   never received the signal; augmentation/reweighting should work).
3. LOSS FORENSICS: on slow SMC tracks, does the training objective itself prefer the
   octave-doubled solution over the correct one? Synthetic logits: correct / doubled /
   halved / model's actual. If doubled < correct in loss, the OBJECTIVE is complicit;
   if correct < doubled, the objective is fine and optimization/data is the culprit.
"""
import glob, json, os, sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM/external/beat_this")
import numpy as np
import torch, torchaudio

from beat_this.model.beat_tracker import BeatThis
from beat_this.model.loss import ShiftTolerantBCELoss
from beat_this.preprocessing import LogMelSpect

SMC_AUDIO = os.path.expanduser("~/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio")
SMC_ANN = os.path.expanduser("~/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Annotations")
TRAIN_MEL = "/home/sogang/jaehoon/VBPM/cache/mel/real_train_mel50"
OUT_DIR = os.path.expanduser("~/jaehoon/Analyze-SMC/cache/grad")
FPS, CROP, GROUP_CAP, BATCH = 50.0, 1500, 48, 4

device = "cuda"
ckpt = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt"),
                  map_location=device, weights_only=False)
model = BeatThis().to(device)
model.load_state_dict({k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items()
                       if k.startswith("model.")})
model.train()  # gradient analysis at the minimum, but no dropout wanted:
for m in model.modules():
    if isinstance(m, torch.nn.Dropout): m.eval()
mel_module = LogMelSpect(device=device)
beat_loss = ShiftTolerantBCELoss(pos_weight=1).to(device)

params = [p for p in model.parameters() if p.requires_grad]
param_names = [n for n, p in model.named_parameters() if p.requires_grad]
top_prefix = lambda n: n.split(".")[0]
modules = sorted({top_prefix(n) for n in param_names})
print(f"[grad] model modules: {modules}, {sum(p.numel() for p in params)/1e6:.1f}M params", flush=True)


def smc_examples():
    out = []
    for wav in sorted(glob.glob(f"{SMC_AUDIO}/*.wav")):
        tid = os.path.basename(wav)[:-4]
        anns = glob.glob(f"{SMC_ANN}/{tid}_*.txt")
        if not anns: continue
        beats = np.loadtxt(anns[0], usecols=0, ndmin=1)
        if len(beats) < 5: continue
        audio, sr = torchaudio.load(wav)
        audio = audio.mean(0)
        if sr != 22050: audio = torchaudio.functional.resample(audio, sr, 22050)
        spect = mel_module(audio.to(device)).cpu()
        tempo = 60.0 / float(np.median(np.diff(beats)))
        out.append((tid, spect, beats, tempo))
    return out


def train_examples():
    out = []
    for path in sorted(glob.glob(f"{TRAIN_MEL}/*.pt")):
        rec = torch.load(path, map_location="cpu")
        beat_frames = np.where(rec["beat_targets"].numpy() > 0.5)[0]
        if len(beat_frames) < 8: continue
        tempo = 60.0 * FPS / float(np.median(np.diff(beat_frames)))
        out.append((os.path.basename(path), rec["spect"].float(), beat_frames / FPS, tempo))
    return out


def make_targets(beats_sec, n_frames):
    t = torch.zeros(n_frames)
    idx = np.round(np.asarray(beats_sec) * FPS).astype(int)
    t[np.clip(idx[(idx >= 0) & (idx < n_frames)], 0, n_frames - 1)] = 1.0
    return t


def group_gradient(examples, label):
    """Mean gradient of the beat training loss over the group. Returns flat vector + stats."""
    total = torch.zeros(sum(p.numel() for p in params))
    losses, count = [], 0
    for start in range(0, min(len(examples), GROUP_CAP), BATCH):
        chunk = examples[start:start + GROUP_CAP][0:BATCH] if False else examples[start:start + BATCH]
        specs, targs = [], []
        for _, spect, beats_sec, _ in chunk:
            s = spect[:CROP]
            if s.shape[0] < 600: continue
            specs.append(torch.nn.functional.pad(s, (0, 0, 0, CROP - s.shape[0])))
            targs.append(make_targets(beats_sec, CROP))
        if not specs: continue
        spect_b = torch.stack(specs).to(device)
        targ_b = torch.stack(targs).to(device)
        model.zero_grad(set_to_none=True)
        out = model(spect_b)
        loss = beat_loss(out["beat"], targ_b)
        loss.backward()
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().cpu()
                          for p in params])
        total += flat * len(specs)
        losses.append(float(loss) * len(specs)); count += len(specs)
    mean_grad = total / max(count, 1)
    print(f"[grad] {label:14s}: n={count:3d} loss={sum(losses)/max(count,1):.4f} ||g||={mean_grad.norm():.3e}", flush=True)
    return mean_grad, sum(losses) / max(count, 1), count


print("[grad] loading SMC...", flush=True)
smc = smc_examples()
print(f"[grad] {len(smc)} SMC tracks", flush=True)
train = train_examples()
print(f"[grad] {len(train)} train-proxy songs", flush=True)

groups = {
    "smc_slow": [e for e in smc if e[3] < 55],
    "smc_mid": [e for e in smc if 55 <= e[3] < 80],
    "smc_fast": [e for e in smc if e[3] >= 80],
    "train_slow80": [e for e in train if e[3] < 80],
    "train_mid": [e for e in train if 80 <= e[3] < 110],
    "train_fast": [e for e in train if e[3] >= 110],
}
for k, v in groups.items():
    print(f"[grad] group {k}: {len(v)} candidates", flush=True)

grads, stats = {}, {}
for label, examples in groups.items():
    if len(examples) == 0:
        print(f"[grad] group {label} EMPTY (itself a finding)", flush=True); continue
    g, l, n = group_gradient(examples, label)
    grads[label] = g; stats[label] = {"loss": l, "n": n, "norm": float(g.norm())}

# pairwise cosines, whole-model and per-module
sizes = [p.numel() for p in params]
offsets = np.cumsum([0] + sizes)
module_slices = {m: [i for i, n in enumerate(param_names) if top_prefix(n) == m] for m in modules}
def module_cos(a, b, idxs):
    va = torch.cat([a[offsets[i]:offsets[i+1]] for i in idxs])
    vb = torch.cat([b[offsets[i]:offsets[i+1]] for i in idxs])
    return float(torch.nn.functional.cosine_similarity(va, vb, dim=0))

labels = list(grads)
report = {"stats": stats, "cosine": {}, "module_cosine": {}, "module_norm_share": {}}
for i, a in enumerate(labels):
    for b in labels[i+1:]:
        c = float(torch.nn.functional.cosine_similarity(grads[a], grads[b], dim=0))
        report["cosine"][f"{a}|{b}"] = c
        report["module_cosine"][f"{a}|{b}"] = {m: module_cos(grads[a], grads[b], idx)
                                               for m, idx in module_slices.items()}
        print(f"[grad] cos({a}, {b}) = {c:+.3f}", flush=True)
for label in labels:
    shares = {m: float(torch.cat([grads[label][offsets[i]:offsets[i+1]] for i in idx]).norm())
              for m, idx in module_slices.items()}
    report["module_norm_share"][label] = shares

# ---- 3. LOSS FORENSICS on slow SMC: does the objective prefer octave doubling? ----
print("[grad] loss forensics (slow SMC, synthetic solutions):", flush=True)
model.eval()
forensics = []
HI, LO = 8.0, -8.0
with torch.no_grad():
    for tid, spect, beats_sec, tempo in groups["smc_slow"][:40]:
        n = min(spect.shape[0], CROP)
        targ = make_targets(beats_sec, n).unsqueeze(0).to(device)
        def synth(times):
            v = torch.full((1, n), LO, device=device)
            idx = np.round(np.asarray(times) * FPS).astype(int)
            idx = idx[(idx >= 0) & (idx < n)]
            v[0, idx] = HI
            return v
        beats = np.asarray(beats_sec)
        mid = (beats[:-1] + beats[1:]) / 2
        actual = model(spect[:n].unsqueeze(0).to(device))["beat"]
        row = {"tid": tid, "tempo": tempo,
               "correct": float(beat_loss(synth(beats), targ)),
               "doubled": float(beat_loss(synth(np.sort(np.concatenate([beats, mid]))), targ)),
               "halved": float(beat_loss(synth(beats[::2]), targ)),
               "model": float(beat_loss(actual, targ))}
        forensics.append(row)
means = {k: float(np.mean([r[k] for r in forensics])) for k in ("correct", "doubled", "halved", "model")}
print(f"[grad] FORENSICS mean loss over {len(forensics)} slow tracks: "
      f"correct={means['correct']:.4f} doubled={means['doubled']:.4f} "
      f"halved={means['halved']:.4f} model={means['model']:.4f}", flush=True)
report["forensics_mean"] = means; report["forensics"] = forensics

np.savez(f"{OUT_DIR}/group_grads.npz", **{k: v.numpy() for k, v in grads.items()})
json.dump(report, open(f"{OUT_DIR}/grad_report.json", "w"), indent=2)
print("[grad] DONE -> cache/grad/grad_report.json", flush=True)
