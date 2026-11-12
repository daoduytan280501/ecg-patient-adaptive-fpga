"""E14 (protocol_bosung_20261026.md): doi chieu truc tiep GA-R va offset_int theo ban ghi (DS2), sach + 9 dieu kien nhieu.
Cau hinh khoa (ds1_selected_rev5b). Khong chon tham so."""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import pa_adapt as pa, rev4_methods as M, rev4_data as rd
OUT = '../out/bosung_20261026'
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb')); SEL = json.load(open('../out/ds1_selected_rev5b_20261025.json'))['selected']
y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
CONF = {'GA-R': ('GA-S', SEL['GA-R']['hp']), 'offset_int': ('offset_int', SEL['offset_int']['hp']), 'none': ('none', {})}
CONDS = [('clean', 0.0)] + [(k, p) for k in ('miss', 'pair', 'uniform') for p in (0.05, 0.1, 0.2)]
REPS = lambda k: [0] if k == 'clean' else [0, 1, 2]
rows = []
for name, (meth, hp) in CONF.items():
    for s in rd.SEEDS:
        S = D2['seeds'][s]; A = (S['A'], rd.ANC_Y)
        for kind, p in CONDS:
            if name == 'none' and kind != 'clean': continue
            for rep in REPS(kind):
                yc = M.noisy_labels(y, cal, s, kind, p, rep)
                y_, p_, r_ = M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], A, meth, fW=S['fw'], y_cal=yc, **hp)
                for r, v in pa.per_record_scores(y_, p_, r_).items():
                    rows.append(dict(method=name, seed=s, cond=f'{kind}|{p}', rep=rep, rec=int(r), balacc=v['balacc'], acc=v['acc']))
    print(name, 'xong', flush=True)
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/e14_record_scores.csv', index=False)
agg = D.groupby(['method', 'cond', 'rec']).balacc.mean().unstack('rec')
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, k in enumerate(o): run = max(run, (len(p) - i) * p[k]); adj[k] = min(run, 1)
    return adj
res = []
for kind, p in CONDS:
    c = f'{kind}|{p}'; g, o = agg.loc[('GA-R', c)], agg.loc[('offset_int', c)]; d = (g - o).dropna()
    res.append(dict(cond=c, n=len(d), GAR_wins=int((d > 1e-9).sum()), offset_wins=int((d < -1e-9).sum()), ties=int((abs(d) <= 1e-9).sum()),
                    median_diff=float(d.median()), mean_diff=float(d.mean()), p=float(wilcoxon(g[d.index], o[d.index]).pvalue)))
R = pd.DataFrame(res); R['p_holm'] = holm(R.p.values); R.to_csv(f'{OUT}/e14_wilcoxon_GAR_vs_offset.csv', index=False)
print(R.to_string())
