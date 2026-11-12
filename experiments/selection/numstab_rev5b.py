"""Do on dinh so hoc cua GA-R: nhieu tuong doi 1e-15 tren emb (mo phong khac biet don vi so hoc ARM/x86)."""
import pickle, json, sys, numpy as np, pandas as pd, pa_adapt as pa, rev4_methods as M, rev4_data as rd
D2=pickle.load(open('../../planA_rev4/out/D2.pkl','rb')); y,rec,cal,ev=D2['y'],D2['rec'],D2['cal'],D2['ev']
import json as _j
_S=_j.load(open('../out/ds1_selected_rev5b_20261025.json'))['selected']
CF={'GA-R':('GA-S',_S['GA-R']['hp']),'gradient_anchor':('gradient_anchor',_S['gradient_anchor']['hp']),
    'head_scratch':('head_scratch',_S['head_scratch']['hp']),'head_global':('head_global',_S['head_global']['hp']),
    'GA-R step 0.5 (rev5)':('gate:GA-S',dict(beta=0.1,cap=1.0,trust=None,delta=0.02))}
rows=[]
for name,(m,hp) in CF.items():
    for s in rd.SEEDS:
        S=D2['seeds'][s]; rng=np.random.default_rng(100+s)
        y0,p0,r0=M.run(S['E'],y,rec,cal,ev,S['W0'],S['b0'],(S['A'],rd.ANC_Y),m,fW=S['fw'],**hp); j0=pa.metrics(y0,p0)['jk']
        for k in range(10):
            E=S['E']*(1+1e-15*rng.standard_normal(S['E'].shape)); Aa=S['A']*(1+1e-15*rng.standard_normal(S['A'].shape))
            y_,p_,r_=M.run(E,y,rec,cal,ev,S['W0'],S['b0'],(Aa,rd.ANC_Y),m,fW=S['fw'],**hp)
            d=p_!=p0; rows.append(dict(method=name,seed=s,rep=k,jk0=j0,jk=pa.metrics(y_,p_)['jk'],n_changed=int(d.sum()),recs=' '.join(map(str,sorted(set(r_[d].tolist()))))))
    print(name, flush=True)
df=pd.DataFrame(rows); df.to_csv('../out/e12_numstab_rev5b_20261025.csv',index=False)
for name,g in df.groupby('method',sort=False):
    rngs=g.groupby('seed').jk.agg(['min','max']); recs=sorted(set(' '.join(g.recs).split()))
    print(f"{name:12s} labels changed: max {g.n_changed.max()}, runs with change {(g.n_changed>0).sum()}/{len(g)} | jk range per seed max {(rngs['max']-rngs['min']).max():.4f} | 5-seed mean jk range {g.groupby('rep').jk.mean().min():.4f}-{g.groupby('rep').jk.mean().max():.4f} (ref {g.groupby('seed').jk0.first().mean():.4f}) | recs {recs}")
