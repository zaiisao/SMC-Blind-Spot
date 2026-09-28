import json,hashlib,numpy as np
from scipy.stats import rankdata,spearmanr
D="/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-Analyze-SMC/e9b382f3-cceb-471e-91d6-43bf1f39be4a/scratchpad/wf/"
rows=json.load(open(D+"feats.json"))
sel=np.array([int(hashlib.md5(r["stem"].encode()).hexdigest(),16)%2==0 for r in rows])
dam=np.array([r["damage"] for r in rows]); dif=np.array([r["diff"] for r in rows])
def resid(y,X):
    X=np.column_stack([np.ones(len(y))]+[rankdata(c) for c in X])
    b,*_=np.linalg.lstsq(X,rankdata(y),rcond=None); return rankdata(y)-X@b
def ev(e,m):
    e=e[m];d=dam[m];f=dif[m]
    rp=spearmanr(resid(e,[f]),resid(d,[f]))[0]
    rd=spearmanr(e,f)[0]
    return rp,rd
base={k:np.array([r[k] for r in rows]) for k in rows[0] if k not in("stem",)}
cands={}
for k in base:
    if k.split("_")[0] in("z","r") and "_" in k and k not in("z_obs","r_obs"): cands[k]=base[k]
cands["z_obs"]=base["z_obs"]; cands["r_obs"]=base["r_obs"]
# regressed-out variants of a few
ctrl=[base["loud"],base["dens"],base["ibi"]]
for k in list(cands):
    cands[k+"+ctrl"]=resid(cands[k],ctrl)
out=[]
for k,v in cands.items():
    ps,gs=ev(v,sel); pr,gr=ev(v,~sel)
    out.append((k,ps,gs,pr,gr))
out.sort(key=lambda t:t[1])
for k,ps,gs,pr,gr in out:
    print(f"{k:16s} SEL rp={ps:+.3f} rd={gs:+.3f} | REP rp={pr:+.3f} rd={gr:+.3f} {'PASS' if abs(pr)>0.25 and abs(gr)<0.15 else ''}")
