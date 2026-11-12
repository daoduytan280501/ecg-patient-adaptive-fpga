import sys, json, pickle, numpy as np
sys.path.insert(0,'/home/claude/planA_rev5/src')
import rev4_methods as M, rev4_data as rd, pa_adapt as pa
D2=pickle.load(open('/home/claude/planA_rev4/out/D2.pkl','rb')); y,rec,cal,ev=D2['y'],D2['rec'],D2['cal'],D2['ev']
SEL=json.load(open('/home/claude/planA_rev5/out/ds1_selected_rev5b_20261025.json'))['selected']['GA-R']['hp']
RECS=sorted(np.unique(rec)); SEEDS=rd.SEEDS; absent=M.absent_mask(y,rec,cal,ev)
def runm(meth,hp,s,yc=None):
    S=D2['seeds'][s]; return M.run(S['E'],y,rec,cal,ev,S['W0'],S['b0'],(S['A'],rd.ANC_Y),meth,fW=S['fw'],y_cal=yc,**hp)
cfg={'none':('none',{}),'GA-R':('GA-S',SEL),'GA-R no anchors':('GA-S',dict(SEL,beta=0.0))}
print(SEL)
res={}
def recv(meth,hp,yc_fn=None):
    vals=[];hit=n=0;jk=[]
    for s in SEEDS:
        y_,p_,r_=runm(meth,hp,s,None if yc_fn is None else yc_fn(s))
        pr=pa.per_record_scores(y_,p_,r_); vals.append([pr[r]['balacc'] for r in RECS]); jk.append(pa.metrics(y_,p_)['jk'])
        ab=absent&(y_!=0); n+=ab.sum(); hit+=(p_[ab]==y_[ab]).sum()
    return np.nanmean(np.array(vals,float),0), hit/n, float(np.mean(jk))
base,_,_=recv(*cfg['none'])
for name in ['GA-R','GA-R no anchors']:
    meth,hp=cfg[name]; v,ab,jk=recv(meth,hp)
    hn=[]
    for kind in ('miss','pair','uniform'):
        for p in (0.05,0.10,0.20):
            vv=[]
            for rp in range(3):
                vr,_,_=recv(meth,hp,lambda s:M.noisy_labels(y,cal,s,kind,p,rp)); vv.append(vr)
            hn.append(int((base-np.mean(vv,0)>0.05).sum()))
    res[name]=dict(jk=jk,harm_clean=int((base-v>0.05).sum()),absent_Se=float(ab),harm_noisy=hn,harm_noisy_mean=float(np.mean(hn)))
    print(name,res[name],flush=True)
json.dump(res,open('/home/claude/planA_rev5/out/bosung_20261026/abl_noanchor_20261109.json','w'),indent=1)
