"""E15c tom tat (chi bao cao): tuong quan DS1-DS2, thu hang cau hinh chon, khoang cach oracle, top-10 DS1."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
OUT = '../out/bosung_20261026'
D2 = pd.concat([pd.read_csv(f'{OUT}/ds2_grid_part{i}.csv') for i in (0, 1)])
g1 = pd.read_csv('../out/ds1_selection_rev5b_20261025.csv'); g1 = g1[g1.method == 'GA-R']
g2 = pd.read_csv(f'{OUT}/ds1_e13_head_scratch_fair.csv')
SEL = json.load(open(f'{OUT}/e13_selected.json'))
res, tops = {}, []
for tag, d1, selhp in [('GA-R', g1, SEL['GA_R_S_rob']['hp']), ('head_scratch_fair', g2, SEL['S_rob']['hp'])]:
    m = d1.merge(D2[D2.method == tag], on='hp', suffixes=('_ds1', '_ds2')); assert len(m) == 216, (tag, len(m))
    st = m[m.stable]; sel = st[st.hp.apply(json.loads) == selhp].iloc[0]
    rho = spearmanr(st.S_rob_ds1, st.S_rob_ds2)
    rank = int((st.S_rob_ds2 > sel.S_rob_ds2).sum()) + 1; best = st.loc[st.S_rob_ds2.idxmax()]
    res[tag] = dict(n_stable=len(st), spearman=float(rho.statistic), spearman_p=float(rho.pvalue), sel_S_rob_ds1=float(sel.S_rob_ds1),
                    sel_S_rob_ds2=float(sel.S_rob_ds2), sel_rank_ds2=rank, oracle_S_rob_ds2=float(best.S_rob_ds2), oracle_hp=json.loads(best.hp),
                    oracle_gap=float(best.S_rob_ds2 - sel.S_rob_ds2), sel_jk_clean_ds2=float(sel.jk_clean_ds2),
                    oracle_jk_clean_ds2=float(st.jk_clean_ds2.max()), median_S_rob_ds2=float(st.S_rob_ds2.median()))
    t = st.sort_values('S_rob_ds1', ascending=False).head(10)
    for i, (_, r) in enumerate(t.iterrows()):
        tops.append(dict(method=tag, rank_ds1=i + 1, hp=r.hp, S_rob_ds1=r.S_rob_ds1, S_rob_ds2=r.S_rob_ds2, jk_clean_ds2=r.jk_clean_ds2,
                         jk_uniform_ds2=r.jk_uniform_ds2, jk_pair_ds2=r.jk_pair_ds2))
    if tag == 'GA-R':
        for lr in (0.02, 0.05, 0.1):
            r = m[m.hp == json.dumps(dict(beta=0.1, cap=1.0, trust=0.5, lr=lr))].iloc[0]
            res[tag][f'eta_{lr}'] = dict(S_rob_ds1=float(r.S_rob_ds1), S_rob_ds2=float(r.S_rob_ds2), jk_clean=float(r.jk_clean_ds2),
                                        jk_uniform=float(r.jk_uniform_ds2), jk_pair=float(r.jk_pair_ds2))
    m.to_csv(f'{OUT}/e15c_{tag}_ds1_vs_ds2.csv', index=False)
pd.DataFrame(tops).to_csv(f'{OUT}/e15c_top10.csv', index=False)
json.dump(res, open(f'{OUT}/e15c_summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1)); print(pd.DataFrame(tops).round(4).to_string())
