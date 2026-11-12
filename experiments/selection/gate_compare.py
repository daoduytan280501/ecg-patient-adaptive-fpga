"""Diem 1: GA-R + cong so voi GA-R tren DS1, theo tung model fold (seed x fold), 3 dieu kien nhan."""
import sys, json, pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import pa_adapt as pa, rev4_methods as M
D1 = pickle.load(open(sys.argv[1], 'rb')); TAG = sys.argv[2]; HP = json.loads(sys.argv[3]) if len(sys.argv) > 3 else dict(beta=0.5, cap=3.0, trust=0.5)
CONDS = [('clean', 0.0, 0), ('uniform', 0.10, 0), ('pair', 0.10, 0)]
rows = []; rev = {'n': 0, 'reverted': 0}
for F in D1['sets']:
    for kind, p, rep in CONDS:
        yc = M.noisy_labels(F['y'], F['cal'], F['seed'], kind, p, rep)
        for name, meth, hp in [('GA-R', 'GA-S', HP), ('GA-R+gate', 'gate:GA-S', dict(HP, delta=0.02))]:
            info = {}
            y_, p_, _ = M.run(F['E'], F['y'], F['rec'], F['cal'], ~F['cal'], F['W0'], F['b0'], F['anchors'], meth, fW=F['fW'], y_cal=yc, info=info, **hp)
            rows.append(dict(seed=F['seed'], fold=F['k'], cond=kind, method=name, mF1=pa.metrics(y_, p_)['macroF1'], n_rec=len(info), n_rev=sum(v['reverted'] for v in info.values())))
df = pd.DataFrame(rows); df.to_csv(f'../out/gate_compare_{TAG}.csv', index=False)
pv = df.pivot_table(index=['seed', 'fold'], columns=['method', 'cond'], values='mF1')
srob = {m: pv[m][['clean', 'uniform', 'pair']].mean(1) for m in ['GA-R', 'GA-R+gate']}
d = srob['GA-R+gate'] - srob['GA-R']
res = dict(per_set_mean_diff=float(d.mean()), per_set_sd=float(d.std(ddof=1)), n_sets=int(len(d)), gate_better=int((d > 0).sum()), gate_worse=int((d < 0).sum()),
           wilcoxon_p=float(wilcoxon(d).pvalue), srob_GAR_sets_sd=float(srob['GA-R'].std(ddof=1)),
           per_cond_diff={c: float((pv['GA-R+gate'][c] - pv['GA-R'][c]).mean()) for c in ['clean', 'uniform', 'pair']},
           revert_rate={c: float(df[(df.method == 'GA-R+gate') & (df.cond == c)].n_rev.sum() / df[(df.method == 'GA-R+gate') & (df.cond == c)].n_rec.sum()) for c in ['clean', 'uniform', 'pair']})
print(json.dumps(res, indent=1)); json.dump(res, open(f'../out/gate_compare_{TAG}.json', 'w'), indent=1)
