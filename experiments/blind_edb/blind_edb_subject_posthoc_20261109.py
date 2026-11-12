import pandas as pd, numpy as np, json
from scipy.stats import wilcoxon
T=pd.read_csv('/mnt/user-data/uploads/ECG_FPGA_1DCNN/planA/06_eval/blind_edb_20261026/blind_edb_per_record.csv',index_col=0)
idx=[str(i).split('/')[-1] for i in T.index]; T.index=idx
groups=[['e0118','e0119','e0120','e0121','e0122'],['e0123','e0124','e0125','e0126'],['e0129','e0133'],['e0136','e0139'],['e0147','e0148'],['e0154','e0155'],['e0162','e0163']]
subj={r:r for r in idx}
for g in groups:
    for r in g: subj[r]=g[0]
miss=[r for g in groups for r in g if r not in idx]
meths=['head_scratch','head_scratch_fair','head_global','offset_int','gradient_anchor','GA-R']
D=pd.DataFrame({m:T[m+'|balacc']-T['none|balacc'] for m in meths}); D['subj']=[subj[r] for r in idx]
S=D.groupby('subj').mean()
res={};ps=[]
for m in meths:
    v=S[m].dropna().values; p=wilcoxon(v,alternative='two-sided',zero_method='wilcox').pvalue
    res[m]=dict(n=len(v),up=int((v>0).sum()),down=int((v<0).sum()),tie=int((v==0).sum()),median=float(np.median(v)),p=float(p)); ps.append(p)
o=np.argsort(ps);run=0;adj=[0]*len(ps)
for i,k in enumerate(o): run=max(run,(len(ps)-i)*ps[k]); adj[k]=min(run,1)
for m,a in zip(meths,adj): res[m]['p_holm']=float(a)
print('missing records in csv:',miss,'subjects:',S.shape[0])
for m in meths: print(m,res[m])
json.dump(dict(groups=groups,missing=miss,n_subjects=int(S.shape[0]),results=res),open('/mnt/user-data/outputs/blind_edb_subject_posthoc_20261109.json','w'),indent=1)
