import numpy as np, pandas as pd, pa_adapt as pa, setup_t4 as S, head_scratch, time
rows=[]; t0=time.time()
for sd in [1,2,3,4,5]:
    E,W0,b0,anc=S.seed_data(sd); pa.CURRENT_ANCHORS=anc
    for p in [0.0,0.05,0.10,0.20]:
        for rep in range(3 if p>0 else 1):
            rng=np.random.default_rng(1000*sd+rep); yn=S.y.copy()
            idx=np.where(S.cal)[0]; k=int(round(p*len(idx)))
            fl=rng.choice(idx,k,replace=False)
            yn[fl]=(yn[fl]+rng.integers(1,4,k))%4          # doi sang lop khac ngau nhien
            for m in ['offset_int','gradient_anchor','head_scratch']:
                # nhan nhieu chi o doan hieu chinh; danh gia bang nhan that
                ys,ps=[],[]
                for r in np.unique(S.rec):
                    mr=S.rec==r; mc,me=mr&S.cal,mr&S.ev
                    W,b=pa.METHODS[m](E[mc],yn[mc],W0,b0,**S.HP[m]); W,b=pa.quantize_head(W,b,7)
                    ps.append((E[me]@W+b).argmax(1)); ys.append(S.y[me])
                mt=pa.metrics(np.concatenate(ys),np.concatenate(ps))
                rows.append(dict(seed=sd,noise=p,rep=rep,method=m,acc=mt['acc'],macroF1=mt['macroF1'],jk=mt['jk'],
                                 Se_S=mt['Se']['S'],PP_S=mt['PP']['S'],Se_V=mt['Se']['V'],Se_F=mt['Se']['F']))
    print('seed',sd,'xong',round(time.time()-t0),'s',flush=True)
df=pd.DataFrame(rows); df.to_csv('exp5_label_noise.csv',index=False)
g=df.groupby(['method','noise'])[['acc','macroF1','jk','Se_S','PP_S','Se_F']].agg(lambda v: f'{v.mean():.3f} ± {v.std():.3f}')
print(g.to_string()); g.to_csv('exp5_label_noise_summary.csv')
