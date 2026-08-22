"""The decisive test: is Beat This's minimum a TRADE-OFF or an OPTIMIZATION ARTIFACT?

Split SMC slow (<55bpm) tracks into halves A/B. Run plain SGD from theta* on A's beat loss only.
At checkpoints measure: (a) peak-pick F on HELD-OUT half B (does the correction generalize
within-domain?), (b) loss+F on train-proxy data (does it damage the majority?).

If B improves substantially while train stays intact -> the good minimum EXISTS nearby, is
REACHABLE by gradient descent, generalizes, and costs the majority nothing: the released model's
blind spot is a data-weighting/optimization artifact, not a capacity or objective limit.
"""
import glob, hashlib, json, os, sys
sys.path.insert(0, "/home/sogang/jaehoon/VBPM/external/beat_this")
import numpy as np
import torch, torchaudio
from scipy.signal import find_peaks
from mir_eval.beat import f_measure

from beat_this.model.beat_tracker import BeatThis
from beat_this.model.loss import ShiftTolerantBCELoss
from beat_this.preprocessing import LogMelSpect

SMC_AUDIO = os.path.expanduser("~/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Audio")
SMC_ANN = os.path.expanduser("~/jaehoon/Analyze-SMC/SMC_MIREX/SMC_MIREX_Annotations")
TRAIN_MEL = "/home/sogang/jaehoon/VBPM/cache/mel/real_train_mel50"
FPS, CROP = 50.0, 1500
device = "cuda"

ckpt = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/beat_this-final0.ckpt"),
                  map_location=device, weights_only=False)
state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
mel_module = LogMelSpect(device=device)
beat_loss = ShiftTolerantBCELoss(pos_weight=1).to(device)

def load_smc():
    out = []
    for wav in sorted(glob.glob(f"{SMC_AUDIO}/*.wav")):
        tid = os.path.basename(wav)[:-4]
        anns = glob.glob(f"{SMC_ANN}/{tid}_*.txt")
        if not anns: continue
        beats = np.loadtxt(anns[0], usecols=0, ndmin=1)
        if len(beats) < 5: continue
        tempo = 60.0 / float(np.median(np.diff(beats)))
        if tempo >= 55: continue
        audio, sr = torchaudio.load(wav); audio = audio.mean(0)
        if sr != 22050: audio = torchaudio.functional.resample(audio, sr, 22050)
        out.append((tid, mel_module(audio.to(device)).cpu(), beats))
    return out

slow = load_smc()
half_a = [e for e in slow if int(hashlib.md5(e[0].encode()).hexdigest(), 16) % 2 == 0]
half_b = [e for e in slow if int(hashlib.md5(e[0].encode()).hexdigest(), 16) % 2 == 1]
print(f"[ray] slow SMC: {len(slow)} tracks -> A(train)={len(half_a)} B(heldout)={len(half_b)}", flush=True)

train_sample = []
paths = sorted(glob.glob(f"{TRAIN_MEL}/*.pt"))
for path in paths[:: max(1, len(paths)//48)][:48]:
    rec = torch.load(path, map_location="cpu")
    bf = np.where(rec["beat_targets"].numpy() > 0.5)[0]
    if len(bf) >= 8: train_sample.append((os.path.basename(path), rec["spect"].float(), bf / FPS))

def targets_of(beats_sec, n):
    t = torch.zeros(n)
    idx = np.round(np.asarray(beats_sec) * FPS).astype(int)
    t[np.clip(idx[(idx >= 0) & (idx < n)], 0, n - 1)] = 1.0
    return t

def eval_group(model, examples):
    fs, losses = [], []
    model.eval()
    with torch.no_grad():
        for _, spect, beats_sec in examples:
            n = min(spect.shape[0], 3000)
            logits = model(spect[:n].unsqueeze(0).to(device))["beat"][0]
            losses.append(float(beat_loss(logits.unsqueeze(0), targets_of(beats_sec, n).unsqueeze(0).to(device))))
            prob = torch.sigmoid(logits).cpu().numpy()
            peaks, _ = find_peaks(prob, height=0.2, distance=7)
            ref = np.asarray(beats_sec); ref = ref[ref < n / FPS]
            fs.append(f_measure(ref, peaks / FPS))
    return float(np.mean(fs)), float(np.mean(losses))

for lr in (1e-5, 3e-5):
    model = BeatThis().to(device); model.load_state_dict(state)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    fb, lb = eval_group(model, half_b); ft, lt = eval_group(model, train_sample)
    print(f"[ray] lr={lr:g} step   0: B_heldout F={fb:.3f} loss={lb:.4f} | train F={ft:.3f} loss={lt:.4f}", flush=True)
    rng = np.random.RandomState(0)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.eval()
    for step in range(1, 201):
        batch = [half_a[i] for i in rng.choice(len(half_a), 4)]
        specs = torch.stack([torch.nn.functional.pad(e[1][:CROP], (0, 0, 0, max(0, CROP - e[1].shape[0])))
                             for e in batch]).to(device)
        targs = torch.stack([targets_of(e[2], CROP) for e in batch]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = beat_loss(model(specs)["beat"], targs)
        loss.backward(); optimizer.step()
        if step % 40 == 0:
            fb, lb = eval_group(model, half_b); ft, lt = eval_group(model, train_sample)
            print(f"[ray] lr={lr:g} step {step:3d}: B_heldout F={fb:.3f} loss={lb:.4f} | train F={ft:.3f} loss={lt:.4f}", flush=True)
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout): m.eval()
print("[ray] DONE", flush=True)
