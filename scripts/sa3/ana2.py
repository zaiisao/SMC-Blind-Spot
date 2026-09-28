import json,hashlib,numpy as np
from scipy.stats import rankdata,spearmanr
D="/home/sogang/.tmp/claude-1003/-home-sogang-jaehoon-Analyze-SMC/e9b382f3-cceb-471e-91d6-43bf1f39be4a/scratchpad/wf/"
rows=json.load(open(D+"feats.json"))
sel=np.array([int(hashlib.md5(r["stem"].encode()).hexdigest(),16)%2==0 for r in rows])
dam=np.array([r["damage"] for r in rows]); dif=np.array([r["diff"] for r in rows])
g=lambda k:np.array([r[k] for r in rows])
def design(ks,m=None):
    cols=[rankdata(g(k)) for k in ks]
    return np.column_stack([np.ones(len(dam))]+cols)
def ctrlresid(k):
    X=design(["loud","dens","ibi"]); y=rankdata(g(k))
    b,*_=np.linalg.lstsq(X[sel],y[sel],rcond=None)   # fit on SELECT only
    return y-X@b
def ev(e,m):
    e=e[m];d=dam[m];f=rankdata(dif[m])
    X=np.column_stack([np.ones(m.sum()),f])
    r=lambda y:(rankdata(y)-X@np.linalg.lstsq(X,rankdata(y),rcond=None)[0])
    return spearmanr(r(e),r(d))[0], spearmanr(e,dif[m])[0]
for k in ["r_zg","r_pctg","z_pctg","r_bestg","z_zg","r_zu"]:
    e=ctrlresid(k); ps,gs=ev(e,sel); pr,gr=ev(e,~sel)
    print(f"{k}+ctrl(SELfit)  SEL rp={ps:+.3f} rd={gs:+.3f} | REP rp={pr:+.3f} rd={gr:+.3f}")
