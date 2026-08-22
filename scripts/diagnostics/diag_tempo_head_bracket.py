"""Diagnostic: is the shortfall the head's ACCURACY or the SHAPE/calibration of q_theta?

Arms (all: beat-only PyTorch DBN2016, min_bpm=30, obs_lambda=16, lambda=100, Beat This acts, n=217)
  b0           beta=0 tempo-blind null
  oracle5      Gaussian(SHARP=.15) injection on TRUE local IBI, beta=5   (the P2 ceiling)
  headgauss5   IDENTICAL injection, but the tempo value is the HEAD's per-frame argmax
  headmed5     IDENTICAL injection, head's per-TRACK median argmax (constant tempo)
  oraclemed5   IDENTICAL injection, TRUE per-track median IBI (the sec.8/R-1 replacement baseline)
If headgauss5/headmed5 gain but the log-q injection does not, the defect is q's SHAPE, not its
accuracy.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import wilcoxon
ROOT=Path('/home/sogang/jaehoon/Analyze-SMC')
sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'vbpm'/'rungs_lab'))
from _harness import FPS, track_ids, load_gt, ev
import _harness as H
from rungs.bar_pointer.readout import state_path_to_events
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
import train_tempo_observation as TT
METRICS=("F","CMLt","AMLt","CMLc","AMLc"); SHARP=0.15; DEV='cuda:1'
TT.GRID=TT.tempo_grid(); TT.GRID_BPM=np.sort(60*FPS/TT.GRID)
dev=torch.device(DEV)
net=TT.load_net('/disk1/jaehoon/tempo_obs_head.pt',dev)
m=make_beatonly_dbn(30.,16,100,DEV); space=m.state_spaces[0]
grid=space.interval_frames.astype(np.float64)
idx={iv:i for i,iv in enumerate(space.interval_frames)}
tmap=torch.from_numpy(np.array([idx[v] for v in space.state_interval_frames])).to(dev)
bf=torch.tensor(space.state_interval_frames,dtype=torch.float64,device=dev)
ARMS=['b0','oracle5','headgauss5','headmed5','oraclemed5']
per={a:{} for a in ARMS}; acc={a:[] for a in ARMS}
def gauss_em(lcd,tempo_series):
    d=(np.log(grid)[None,:]-np.log(tempo_series)[:,None])/SHARP
    lp=-0.5*d**2
    lp=lp-torch.logsumexp(torch.tensor(lp,device=dev),dim=1,keepdim=True).cpu().numpy()
    ratio=torch.tensor(lp,dtype=torch.float64,device=dev)-np.log(1.0/len(grid))
    return lcd[:,m.state_to_classes[0]]+5.0*(ratio[:,tmap]/bf.unsqueeze(0))
for t in track_ids():
    got=emissions_beatonly(m,H.bthis_beat_prob(t))
    if got is None: continue
    lcd,dec,ff=got; T=lcd.shape[0]
    emb=np.load(f'/disk1/jaehoon/smc_bt_embeddings/{t}.npz')['emb']
    lq=TT.__dict__ and None
    with torch.no_grad():
        x=torch.from_numpy(np.asarray(emb,np.float32)).permute(1,0).unsqueeze(0).to(dev)
        q=torch.softmax(net(x).float(),1)[0].T.cpu().numpy()
    q=q[ff:ff+T]
    head=grid[q.argmax(1)]
    bt=np.asarray(load_gt(t),float); fr=bt*FPS-ff; ibi=np.diff(bt)*FPS
    true=np.full(T,np.nan)
    for k in range(len(fr)-1):
        a,b=max(0,int(np.ceil(fr[k]))),min(T,int(np.ceil(fr[k+1])))
        if b>a: true[a:b]=ibi[k]
    if (~np.isnan(true)).sum()<10: continue
    true=np.where(~np.isnan(true),true,np.nanmedian(true))
    series={'oracle5':true,'headgauss5':head,
            'headmed5':np.full(T,grid[np.argmin(np.abs(grid-np.median(head)))]),
            'oraclemed5':np.full(T,np.median(true))}
    for a in ARMS:
        if a=='b0': em,stc=lcd,m.state_to_classes[0]
        else: em,stc=gauss_em(lcd,series[a]),None
        if a!='b0': acc[a].append(float(np.mean(np.abs(np.log(series[a]/true))<np.log(1.1))))
        p,_=m.dynamic_programs[0].viterbi(m.log_initial_distributions[0],m.log_tempo_transition,em,state_to_class=stc,return_log_score=True)
        per[a][t]=ev(load_gt(t),state_path_to_events(p.cpu().numpy(),space,m.fps,snap_to_activations=dec,first_frame=ff)['beats'])
ids=sorted(set.intersection(*[set(per[a]) for a in ARMS]))
A=lambda r:{k:float(np.mean([r[t][k] for t in ids])) for k in METRICS}
print(f"n={len(ids)}"); print(f"{'arm':>12}{'tempoAcc':>10}"+''.join(f'{k:>9}' for k in METRICS))
for a in ARMS:
    print(f"{a:>12}{(np.mean(acc[a]) if acc[a] else float('nan')):>10.3f}"+''.join(f"{A(per[a])[k]:>9.4f}" for k in METRICS))
print('\nvs b0:')
for a in ARMS[1:]:
    o=[]
    for k in METRICS:
        d=np.array([per[a][t][k]-per['b0'][t][k] for t in ids]); p=wilcoxon(d).pvalue if np.any(d) else 1.
        o.append(f"{k} {d.mean():+.4f}(p={p:.1e})")
    print(f"  {a:>12} "+'  '.join(o))
json.dump({a:per[a] for a in ARMS},open('/tmp/overnight/diag_head.json','w'))
