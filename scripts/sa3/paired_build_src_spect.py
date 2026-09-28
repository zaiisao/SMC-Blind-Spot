import os,sys,json,glob
import numpy as np
BT="/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
os.chdir(BT); sys.path.insert(0,BT)
import torch, soxr
torch.set_num_threads(2)
from beat_this.preprocessing import load_audio, LogMelSpect
from beat_this.dataset.mmnpz import MemmappedNpzFile
from pathlib import Path
OUT="/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-Analyze-SMC/e9b382f3-cceb-471e-91d6-43bf1f39be4a/scratchpad/wf/paired/src_spect.npz"
npz=MemmappedNpzFile(Path("data/audio/spectrograms/sdedit.npz"))
stems=sorted({k.split("/")[0] for k in npz.files})
srcof={s:json.load(open(f"/disk1/jaehoon/sdedit_slow/{s}.json"))["source_wav"] for s in stems}
srcs=sorted(set(srcof.values()))
part=int(sys.argv[1]); nparts=int(sys.argv[2])
mine={w for i,w in enumerate(srcs) if i%nparts==part}
stems=sorted([s for s in stems if srcof[s] in mine], key=lambda s:(srcof[s],s))
lm=LogMelSpect()
cache={}
res={}
for s in stems:
    T=np.asarray(npz[f"{s}/track"]).shape[0]
    j=json.load(open(f"/disk1/jaehoon/sdedit_slow/{s}.json"))
    wav=j["source_wav"]; st=j["window_start"]
    if wav not in cache:
        cache.clear()
        y,sr=load_audio(wav,dtype="float32")
        if y.ndim>1: y=y.mean(1)
        if sr!=22050: y=soxr.resample(y,sr,22050)
        cache[wav]=np.asarray(y,dtype=np.float32)
    y=cache[wav]
    i0=int(round(st*22050))
    need=(T+2)*441
    seg=y[i0:i0+need]
    if len(seg)<need: seg=np.pad(seg,(0,need-len(seg)))
    sp=lm(torch.from_numpy(seg)).numpy()
    res[s]=sp[:T].astype(np.float16)
np.savez(OUT.replace(".npz",f"_{part}.npz"),**res)
print("done",part,len(res))
