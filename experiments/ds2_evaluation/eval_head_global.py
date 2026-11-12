"""Diem 3: head_global (head_scratch giu lop toan cuc) tren DS2 cho 3 so vong; cung giao thuc voi paper_tables_rev4."""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import pa_adapt as pa, rev4_methods as M, rev4_data as rd
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb')); y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
RECS = sorted(np.unique(rec)); SEEDS = rd.SEEDS; absent = M.absent_mask(y, rec, cal, ev)
def runm(meth, hp, s, ycal=None):
    S = D2['seeds'][s]; return M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], (S['A'], rd.ANC_Y), meth, fW=S['fw'], y_cal=ycal, **hp)
def rb(y_, p_, r_): pr = pa.per_record_scores(y_, p_, r_); return {int(r): pr[r]['balacc'] for r in pr}
CONF = {'none': ('none', {}), **{f'head_global_e{e}': ('head_global', dict(lr=0.5, epochs=e)) for e in (50, 150, 300)},
        'head_scratch_e50': ('head_scratch', dict(lr=0.5, epochs=50))}
rows, recbal = [], {}
for name, (m, hp) in CONF.items():
    for s in SEEDS:
        y_, p_, r_ = runm(m, hp, s); mt = pa.metrics(y_, p_); ab = absent & (y_ != 0)
        rows.append(dict(method=name, seed=s, acc=mt['acc'], macroF1=mt['macroF1'], kappa=mt['kappa'], jk=mt['jk'], **{f'Se_{c}': mt['Se'][c] for c in pa.CLASSES},
                         **{f'PP_{c}': mt['PP'][c] for c in pa.CLASSES}, absent_n=int(ab.sum()), absent_hit=int((p_[ab] == y_[ab]).sum())))
        recbal[(name, s)] = rb(y_, p_, r_)
MAIN = pd.DataFrame(rows); MAIN.to_csv('../out/e9_head_global_main.csv', index=False)
base = {r: np.nanmean([recbal[('none', s)][r] for s in SEEDS]) for r in RECS}
CONDS = [(k, p, r) for k in ('miss', 'pair', 'uniform') for p in (0.05, 0.10, 0.20) for r in range(3)]
out = {}; nz = []
for name in [n for n in CONF if n.startswith('head_global')]:
    m, hp = CONF[name]
    v = {r: np.nanmean([recbal[(name, s)][r] for s in SEEDS]) for r in RECS}
    d = np.array([v[r] - base[r] for r in RECS]); ok = np.abs(d) > 1e-12
    w = wilcoxon(d[ok]) if ok.any() else None
    rbn = {}
    for s in SEEDS:
        for kind, p, rep in CONDS:
            y_, p_, r_ = runm(m, hp, s, ycal=M.noisy_labels(y, cal, s, kind, p, rep))
            nz.append(dict(method=name, seed=s, kind=kind, p=p, rep=rep, jk=pa.metrics(y_, p_)['jk'])); rbn[(s, kind, p, rep)] = rb(y_, p_, r_)
    hn = []
    for kind in ('miss', 'pair', 'uniform'):
        for p in (0.05, 0.10, 0.20):
            vv = {r: np.nanmean([rbn[(s, kind, p, rp)][r] for s in SEEDS for rp in range(3)]) for r in RECS}
            hn.append(int(sum(base[r] - vv[r] > 0.05 for r in RECS)))
    g = MAIN[MAIN.method == name]; NZ = pd.DataFrame([x for x in nz if x['method'] == name])
    out[name] = dict(jk=[float(g.jk.mean()), float(g.jk.std())], mF1=[float(g.macroF1.mean()), float(g.macroF1.std())], acc=float(g.acc.mean()),
                     Se_S=float(g.Se_S.mean()), PP_S=float(g.PP_S.mean()), Se_F=float(g.Se_F.mean()), absent_Se=float(g.absent_hit.sum() / g.absent_n.sum()),
                     improved=int((d > 1e-12).sum()), degraded=int((d < -1e-12).sum()), p=float(w.pvalue) if w else 1.0,
                     harm_clean=int((d < -0.05).sum()), harm_noisy_each=hn, harm_noisy_mean=float(np.mean(hn)),
                     noise={f'{k}|{p}': float(NZ[(NZ.kind == k) & (NZ.p == p)].jk.mean()) for k in ('miss', 'pair', 'uniform') for p in (0.05, 0.10, 0.20)},
                     noisy_mean=float(NZ.groupby(['kind', 'p']).jk.mean().mean()))
    print(name, json.dumps(out[name]), flush=True)
pd.DataFrame(nz).to_csv('../out/e9_head_global_noise.csv', index=False)
json.dump(out, open('../out/e9_head_global.json', 'w'), indent=1)
