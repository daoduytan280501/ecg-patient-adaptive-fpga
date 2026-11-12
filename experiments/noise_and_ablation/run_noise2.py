import numpy as np, pandas as pd, pa_adapt as pa, setup_t4 as S, head_scratch, time
PAIR = np.array([1, 0, 3, 2])            # N<->S, V<->F (nham lan hay gap)
rows=[]
for sd in [1,2,3,4,5]:
    E,W0,b0,anc=S.seed_data(sd); pa.CURRENT_ANCHORS=anc
    for model in ['pair','miss']:
        for p in [0.05,0.10,0.20]:
            for rep in range(3):
                rng=np.random.default_rng(7000*sd+100*rep+int(p*100)); yn=S.y.copy()
                if model=='pair':
                    idx=np.where(S.cal)[0]
                else:                                   # nhip bat thuong bi gan nham la N (bo sot)
                    idx=np.where(S.cal&(S.y!=0))[0]
                k=int(round(p*len(idx))); fl=rng.choice(idx,k,replace=False)
                yn[fl]=PAIR[yn[fl]] if model=='pair' else 0
                for m in ['offset_int','gradient_anchor','head_scratch']:
                    ys,ps=[],[]
                    for r in np.unique(S.rec):
                        mr=S.rec==r; mc,me=mr&S.cal,mr&S.ev
                        W,b=pa.METHODS[m](E[mc],yn[mc],W0,b0,**S.HP[m]); W,b=pa.quantize_head(W,b,7)
                        ps.append((E[me]@W+b).argmax(1)); ys.append(S.y[me])
                    mt=pa.metrics(np.concatenate(ys),np.concatenate(ps))
                    rows.append(dict(seed=sd,noise_model=model,noise=p,rep=rep,method=m,acc=mt['acc'],macroF1=mt['macroF1'],
                                     jk=mt['jk'],Se_S=mt['Se']['S'],PP_S=mt['PP']['S'],Se_F=mt['Se']['F']))
    print('seed',sd,flush=True)
df=pd.DataFrame(rows); df.to_csv('exp5_label_noise_realistic.csv',index=False)
g=df.groupby(['noise_model','method','noise'])[['acc','macroF1','jk','Se_S','PP_S']].agg(lambda v: f'{v.mean():.3f}±{v.std():.3f}')
print(g.to_string()); g.to_csv('exp5_label_noise_realistic_summary.csv')
