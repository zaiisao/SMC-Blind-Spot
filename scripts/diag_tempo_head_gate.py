# Confidence-gated injection (label-free gate): inject beta*ratio only on tracks whose head
# agreement is high, measured WITHOUT labels as the mean max-prob of q over the track.
import sys,json
from pathlib import Path
import numpy as np, torch
from scipy.stats import wilcoxon
ROOT=Path('/home/sogang/jaehoon/Analyze-SMC'); sys.path.insert(0,str(ROOT/'scripts')); sys.path.insert(0,str(ROOT/'vbpm'/'rungs_lab'))
from _harness import FPS, track_ids, load_gt, ev
import _harness as H
from rungs.bar_pointer.readout import state_path_to_events
from exp_oracle_tempo_beatonly import make_beatonly_dbn, emissions_beatonly
import train_tempo_observation as TT
M=("F","CMLt","AMLt"); DEV='cuda:1'
TT.GRID=TT.tempo_grid(); TT.GRID_BPM=np.sort(3000/TT.GRID)
dev=torch.device(DEV); net=TT.load_net('/disk1/jaehoon/tempo_obs_head.pt',dev)
marg=np.load('/disk1/jaehoon/tempo_obs_head.pt.marginal.npy'); logm=torch.tensor(np.log(marg),dtype=torch.float64,device=dev)
m=make_beatonly_dbn(30.,16,100,DEV); space=m.state_spaces[0]; grid=space.interval_frames.astype(float)
idx={iv:i for i,iv in enumerate(space.interval_frames)}
tmap=torch.from_numpy(np.array([idx[v] for v in space.state_interval_frames])).to(dev)
bf=torch.tensor(space.state_interval_frames,dtype=torch.float64,device=dev)
BETAS=[0.0,0.0382,0.1,0.2]
per={b:{} for b in BETAS}; conf={}; hit={}
for t in track_ids():
    got=emissions_beatonly(m,H.bthis_beat_prob(t))
    if got is None: continue
    lcd,dec,ff=got; T=lcd.shape[0]
    emb=np.load(f'/disk1/jaehoon/smc_bt_embeddings/{t}.npz')['emb']
    with torch.no_grad():
        x=torch.from_numpy(np.asarray(emb,np.float32)).permute(1,0).unsqueeze(0).to(dev)
        lg=torch.log_softmax(net(x).float(),1)[0].T.double()
    lg=lg[ff:ff+T]; q=lg.exp().cpu().numpy()
    conf[t]=float(q.max(1).mean())
    bt=np.asarray(load_gt(t),float); pred=grid[q.argmax(1)]
    hit[t]=float(abs(np.log(np.median(pred)/(np.median(np.diff(bt))*FPS)))<np.log(1.1))
    ratio=lg-logm[None,:]
    for b in BETAS:
        if b==0.: em,stc=lcd,m.state_to_classes[0]
        else: em,stc=lcd[:,m.state_to_classes[0]]+b*(ratio[:,tmap]/bf.unsqueeze(0)),None
        p,_=m.dynamic_programs[0].viterbi(m.log_initial_distributions[0],m.log_tempo_transition,em,state_to_class=stc,return_log_score=True)
        per[b][t]=ev(load_gt(t),state_path_to_events(p.cpu().numpy(),space,m.fps,snap_to_activations=dec,first_frame=ff)['beats'])
ids=sorted(per[0.0]); c=np.array([conf[t] for t in ids]); h=np.array([hit[t] for t in ids])
print('conf vs correctness: corr=%.3f  (n=%d); conf quartiles %s'%(np.corrcoef(c,h)[0,1],len(ids),np.round(np.percentile(c,[25,50,75]),3)))
for thr in (0.0,np.percentile(c,50),np.percentile(c,75)):
    sel=[t for t in ids if conf[t]>=thr]
    print(' gate conf>=%.3f  n=%d  headTrackAcc=%.3f'%(thr,len(sel),np.mean([hit[t] for t in sel])))
    for b in BETAS[1:]:
        o=[]
        for k in M:
            d=np.array([per[b][t][k]-per[0.0][t][k] for t in sel]); p=wilcoxon(d).pvalue if np.any(d) else 1.
            o.append(f'{k} {d.mean():+.4f}(p={p:.1e})')
        print('    beta=%.4f '%b+'  '.join(o))
json.dump({'conf':conf,'hit':hit},open('/tmp/overnight/diag_gate.json','w'))
