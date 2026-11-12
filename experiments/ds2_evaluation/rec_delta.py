"""Diem 4: delta balanced accuracy theo ban ghi (GA-R vs none), nhan sach, DS2."""
import pickle, json, numpy as np, pandas as pd
import pa_adapt as pa, rev4_methods as M, rev4_data as rd
D2 = pickle.load(open('../out/D2.pkl', 'rb')); y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
CONF = {'none': ('none', {}), 'GA-R': ('GA-S', dict(beta=0.5, cap=3.0, trust=0.5)), 'offset_int': ('offset_int', dict(tau=20, fbits=8)),
        'gradient_anchor': ('gradient_anchor', dict(beta=0.5)), 'head_scratch': ('head_scratch', dict(lr=0.5, epochs=50))}
def balacc(yy, pp):
    cs = np.unique(yy); return float(np.mean([(pp[yy == c] == c).mean() for c in cs]))
rows = []
for name, (m, hp) in CONF.items():
    for s in rd.SEEDS:
        S = D2['seeds'][s]; A = (S['A'], rd.ANC_Y)
        y_, p_, r_ = M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], A, m, fW=S['fw'], **hp)
        pr10 = pa.per_record_scores(y_, p_, r_, min_beats=10); pr1 = pa.per_record_scores(y_, p_, r_, min_beats=1)
        for r in np.unique(r_):
            rows.append(dict(method=name, seed=s, rec=int(r), ba=pr10[r]['balacc'], ba_all=pr1[r]['balacc']))
df = pd.DataFrame(rows); df.to_csv('../out/e8_record_ba_20261025.csv', index=False)
avg = df.groupby(['method', 'rec']).ba.mean().unstack(0); avg1 = df.groupby(['method', 'rec']).ba_all.mean().unstack(0)
out = {}
for m in ['GA-R', 'offset_int', 'gradient_anchor', 'head_scratch']:
    d = avg[m] - avg['none']
    ps = df[df.method == m].set_index(['seed', 'rec']).ba - df[df.method == 'none'].set_index(['seed', 'rec']).ba
    out[m] = dict(min_delta=float(d.min()), worst_rec=int(d.idxmin()), n_below_005=int((d < -0.05).sum()), n_below_0=int((d < -1e-9).sum()),
                  n_above_0=int((d > 1e-9).sum()), min_delta_allclasses=float((avg1[m] - avg1['none']).min()), n_below_005_allclasses=int(((avg1[m] - avg1['none']) < -0.05).sum()), per_seed_min=float(ps.min()), per_seed_worst=list(map(int, ps.idxmin())), per_seed_n_below_005=int((ps < -0.05).sum()))
    print(m, out[m])
d = (avg['GA-R'] - avg['none']).sort_values()
print(pd.DataFrame(dict(none=avg['none'], GAR=avg['GA-R'], delta=avg['GA-R'] - avg['none'])).loc[d.index].round(3).to_string())
json.dump(out, open('../out/e8_record_delta_20261025.json', 'w'), indent=1)
