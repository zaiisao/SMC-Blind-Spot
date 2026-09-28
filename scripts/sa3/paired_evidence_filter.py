import os,sys,json,csv,hashlib
import numpy as np
from scipy.stats import spearmanr, rankdata
BT="/home/sogang/jaehoon/Analyze-SMC/third-party/beat_this"
os.chdir(BT); sys.path.insert(0,BT)
from beat_this.dataset.mmnpz import MemmappedNpzFile
from pathlib import Path
D="/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-Analyze-SMC/e9b382f3-cceb-471e-91d6-43bf1f39be4a/scratchpad/wf/paired"
FPS=50; TOL=3
src={}
for i in range(8):
    z=np.load(f"{D}/src_spect_{i}.npz")
    for k in z.files: src[k]=z[k].astype(np.float32)
npz=MemmappedNpzFile(Path("data/audio/spectrograms/sdedit.npz"))
annd=Path("data/annotations/sdedit/annotations/beats")

def flux(sp):
    d=np.diff(sp,axis=0); f=np.maximum(d,0).sum(1)
    return np.concatenate([[0.0],f])
def znorm(f):
    return (f-np.median(f))/(f.std()+1e-9)

rng=np.random.default_rng(0)
def zscore(f,idx,T):
    w=np.arange(-TOL,TOL+1)
    def sc(o):
        pos=(o[:,None]+w[None,:]).clip(0,T-1); return f[pos].max(1).mean()
    obs=sc(idx); span=T-2*TOL
    null=np.array([sc(((idx-TOL+rng.integers(1,span))%span)+TOL) for _ in range(200)])
    return (obs-null.mean())/(null.std()+1e-9)

rows={}
for s in sorted(src):
    g=np.asarray(npz[f"{s}/track"],dtype=np.float32); r=src[s]
    T=min(len(g),len(r)); g=g[:T]; r=r[:T]
    ref=np.loadtxt(annd/f"{s}.beats",ndmin=2)[:,0]
    idx=np.round(ref*FPS).astype(int); idx=idx[(idx>=TOL)&(idx<T-TOL)]
    if len(idx)<3: continue
    fg=flux(g); fr=flux(r); zg=znorm(fg); zr=znorm(fr)
    mask=np.zeros(T,bool)
    for i in idx: mask[max(0,i-TOL):i+TOL+1]=True
    d=dict()
    # 1 global flux corr
    d["flux_corr"]=spearmanr(zg,zr).statistic
    # 2 flux corr in beat windows
    d["flux_corr_beat"]=spearmanr(zg[mask],zr[mask]).statistic
    # 2b off-beat corr, and difference
    cb=d["flux_corr_beat"]; co=spearmanr(zg[~mask],zr[~mask]).statistic
    d["flux_corr_beat_minus_off"]=cb-co
    # 3 spectral divergence beat vs off
    dv=np.abs(g-r).mean(1)
    d["neg_div_beat"]=-dv[mask].mean()
    d["neg_div_beat_rel"]=-(dv[mask].mean()-dv[~mask].mean())
    d["neg_div_beat_ratio"]=-(dv[mask].mean()/(dv[~mask].mean()+1e-9))
    # 4 transient sharpness ratio at beats
    w=np.arange(-TOL,TOL+1)
    pos=(idx[:,None]+w[None,:]).clip(0,T-1)
    sg=zg[pos].max(1); sr_=zr[pos].max(1)
    d["sharp_ratio"]=float(np.mean(sg)/(np.mean(sr_)+1e-9))
    d["sharp_diff"]=float(np.mean(sg)-np.mean(sr_))
    d["sharp_ratio_med"]=float(np.median(sg/(np.abs(sr_)+1e-6)))
    # 5 paired z gap
    zzg=zscore(zg,idx,T); zzr=zscore(zr,idx,T)
    d["z_gen"]=zzg; d["z_diff"]=zzg-zzr; d["z_ratio"]=zzg/(abs(zzr)+1e-6)
    # 6 cross-correlation lag of flux curves
    a=zg-zg.mean(); b=zr-zr.mean()
    L=25
    cc=np.array([np.dot(a[max(0,l):T+min(0,l)],b[max(0,-l):T-max(0,l)]) for l in range(-L,L+1)])
    cc/= (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
    lag=np.arange(-L,L+1)[np.argmax(cc)]
    d["neg_abs_lag"]=-abs(float(lag)); d["xcorr_max"]=float(cc.max())
    d["xcorr_at0"]=float(cc[L]); d["xcorr_peakiness"]=float(cc[L]-cc.max())
    # 7 per-beat retention: fraction of beats where gen sharpness >= half src sharpness
    d["retention"]=float(np.mean(sg>=0.5*sr_))
    # 8 beat-window energy corr in mel (spectral shape) 
    d["mel_corr_beat"]=float(np.mean([np.corrcoef(g[i],r[i])[0,1] for i in idx]))
    rows[s]=d
print("scored",len(rows))

paired={r["stem"]:r for r in csv.DictReader(open("/home/sogang/jaehoon/Analyze-SMC/cache/sdedit_vs_source.csv"))}
names=sorted(next(iter(rows.values())).keys())
stems=[s for s in rows if s in paired]
def half(s): return int(hashlib.md5(s.encode()).hexdigest(),16)%2
SEL=[s for s in stems if half(s)==0]; REP=[s for s in stems if half(s)==1]
def partial(x,y,c):
    rx,ry,rc=rankdata(x),rankdata(y),rankdata(c)
    def res(v):
        A=np.vstack([rc,np.ones_like(rc)]).T
        return v-A@np.linalg.lstsq(A,v,rcond=None)[0]
    return spearmanr(res(rx),res(ry)).statistic
out={}
for grp,ss in [("SELECT",SEL),("REPORT",REP)]:
    dam=np.array([float(paired[s]["F_src"])-float(paired[s]["F_gen"]) for s in ss])
    dif=np.array([1-float(paired[s]["F_src"]) for s in ss])
    for n in names:
        e=np.array([rows[s][n] for s in ss])
        ok=np.isfinite(e)
        out.setdefault(n,{})[grp]=(partial(e[ok],dam[ok],dif[ok]),
                                   spearmanr(e[ok],dif[ok]).statistic,
                                   spearmanr(e[ok],dam[ok]).statistic, int(ok.sum()))
print(f"n SELECT={len(SEL)} REPORT={len(REP)}")
print(f"{'measure':28s} {'SEL rp':>8}{'SEL rd':>8}{'REP rp':>8}{'REP rd':>8}{'REP raw':>9}")
for n in sorted(names,key=lambda n:-abs(out[n]["REPORT"][0])):
    a=out[n]["SELECT"]; b=out[n]["REPORT"]
    print(f"{n:28s} {a[0]:8.3f}{a[1]:8.3f}{b[0]:8.3f}{b[1]:8.3f}{b[2]:9.3f}")
json.dump({n:{g:list(v) for g,v in out[n].items()} for n in names},open(f"{D}/results.json","w"),indent=1)
