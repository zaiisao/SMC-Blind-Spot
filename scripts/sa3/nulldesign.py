import os,sys,csv,hashlib,json
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, spearmanr
BT="/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
os.chdir(BT); sys.path.insert(0,BT)
from beat_this.dataset.mmnpz import MemmappedNpzFile
FPS=50; TOL=3; NN=200

def flux(sp):
    d=np.diff(sp,axis=0); f=np.maximum(d,0).sum(axis=1)
    return np.concatenate([[0.0],f])

def zn(f):
    return (f-np.median(f))/(f.std()+1e-9)

def peakmean(f,idx,T):
    w=np.arange(-TOL,TOL+1)
    pos=(idx[:,None]+w[None,:]).clip(0,T-1)
    return f[pos].max(axis=1).mean()

def feats(sp,ref,rng):
    f0=flux(sp); T=len(f0)
    fz=zn(f0); fr=rankdata(f0)/T           # rank-normalised flux
    idx=np.round(np.asarray(ref)*FPS).astype(int)
    idx=idx[(idx>=TOL)&(idx<T-TOL)]
    if len(idx)<3: return None
    span=T-2*TOL
    ibi=np.median(np.diff(idx)) if len(idx)>1 else 25.0
    out={}
    for tag,f in (("z",fz),("r",fr)):
        obs=peakmean(f,idx,T)
        # uniform null
        nu=np.array([peakmean(f,((idx-TOL+rng.integers(1,span))%span)+TOL,T) for _ in range(NN)])
        # on-grid null: shift by fraction of IBI (0.15..0.85 of a beat)
        ng=[]
        for _ in range(NN):
            frac=rng.uniform(0.15,0.85)*ibi
            s=int(round(frac))
            if s==0: s=1
            ng.append(peakmean(f,((idx-TOL+s)%span)+TOL,T))
        ng=np.array(ng)
        out[tag+"_obs"]=obs
        out[tag+"_zu"]=(obs-nu.mean())/(nu.std()+1e-9)
        out[tag+"_zg"]=(obs-ng.mean())/(ng.std()+1e-9)
        out[tag+"_bestu"]=obs-nu.max()
        out[tag+"_bestg"]=obs-ng.max()
        out[tag+"_pctu"]=(obs>nu).mean()
        out[tag+"_pctg"]=(obs>ng).mean()
        out[tag+"_ratg"]=obs/(ng.mean()+1e-9)
    out["loud"]=float(sp.mean()); out["dens"]=float(np.mean(flux(sp)))
    out["dur"]=float(T/FPS); out["nb"]=len(idx); out["ibi"]=float(ibi)
    return out

rng=np.random.default_rng(0)
npz=MemmappedNpzFile(Path("data/audio/spectrograms/sdedit.npz"))
annd=Path("data/annotations/sdedit/annotations/beats")
paired={r["stem"]:r for r in csv.DictReader(open("/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_vs_source.csv"))}
rows=[]
for s in sorted(paired):
    try: sp=np.asarray(npz[f"{s}/track"],dtype=np.float32)
    except KeyError: continue
    ref=np.loadtxt(annd/f"{s}.beats",ndmin=2)[:,0]
    d=feats(sp,ref,rng)
    if d is None: continue
    d["stem"]=s
    d["damage"]=float(paired[s]["F_src"])-float(paired[s]["F_gen"])
    d["diff"]=1-float(paired[s]["F_src"])
    d["bpm"]=float(paired[s]["bpm"])
    rows.append(d)
print("n",len(rows))
json.dump(rows,open("/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-Analyze-SMC/e9b382f3-cceb-471e-91d6-43bf1f39be4a/scratchpad/wf/feats.json","w"))
