"""E13 (protocol_bosung_20261026.md): danh gia DS2 cua head_scratch_fair DA CHON tren DS1 (S_rob, va lua chon phu theo mF1 sach),
cung cach tinh voi paper_tables_rev5b.py (bang chinh, nhieu 9 dieu kien x 3 lan, ban ghi bi hai, lop vang mat, Wilcoxon)."""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import pa_adapt as pa, rev4_methods as M, rev4_data as rd, fair_methods  # noqa
OUT = '../out/bosung_20261026'
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb'))
y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']; RECS = sorted(np.unique(rec)); SEEDS = rd.SEEDS
SELJ = json.load(open(f'{OUT}/e13_selected.json'))
SEL5 = json.load(open('../out/ds1_selected_rev5b_20261025.json'))['selected']
METH = {'none': ('none', {}), 'GA-R': ('GA-S', SEL5['GA-R']['hp']), 'head_scratch': ('head_scratch', SEL5['head_scratch']['hp']),
        'head_scratch_fair': ('head_scratch_fair', SELJ['S_rob']['hp']),
        'head_scratch_fair (mF1 sach)': ('head_scratch_fair', SELJ['clean']['hp']),
        'GA-R (mF1 sach)': ('GA-S', SELJ['GA-R_clean']['hp'])}
def runm(meth, hp, s, ycal=None):
    S = D2['seeds'][s]; return M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], (S['A'], rd.ANC_Y), meth, fW=S['fw'], y_cal=ycal, **hp)
def mrow(y_, p_):
    m = pa.metrics(y_, p_)
    return dict(acc=m['acc'], macroF1=m['macroF1'], kappa=m['kappa'], jk=m['jk'], **{f'Se_{c}': m['Se'][c] for c in pa.CLASSES}, **{f'PP_{c}': m['PP'][c] for c in pa.CLASSES})
absent = M.absent_mask(y, rec, cal, ev)
rows, recbal = [], {}
for name, (meth, hp) in METH.items():
    for s in SEEDS:
        y_, p_, r_ = runm(meth, hp, s); row = dict(method=name, seed=s, **mrow(y_, p_))
        ab = absent & (y_ != 0); row['absent_n'], row['absent_hit'] = int(ab.sum()), int((p_[ab] == y_[ab]).sum())
        row['nonF_called_F'] = int(((p_ == 3) & (y_ != 3)).sum()); rows.append(row)
        pr = pa.per_record_scores(y_, p_, r_); recbal[(name, s)] = {int(r): pr[r]['balacc'] for r in pr}
MAIN = pd.DataFrame(rows); MAIN.to_csv(f'{OUT}/e13_main_rows.csv', index=False)
print(MAIN.groupby('method')[['jk', 'macroF1', 'Se_S', 'PP_S', 'Se_F']].mean().round(4), flush=True)
CONDS = [(k, p, r) for k in ('miss', 'pair', 'uniform') for p in (0.05, 0.10, 0.20) for r in range(3)]
NEW = ['head_scratch_fair', 'head_scratch_fair (mF1 sach)', 'GA-R (mF1 sach)']
nz, rbn = [], {}
for name in NEW:
    meth, hp = METH[name]
    for s in SEEDS:
        for kind, p, rep in CONDS:
            y_, p_, r_ = runm(meth, hp, s, M.noisy_labels(y, cal, s, kind, p, rep))
            nz.append(dict(method=name, seed=s, kind=kind, p=p, rep=rep, **mrow(y_, p_)))
            pr = pa.per_record_scores(y_, p_, r_); rbn[(name, s, kind, p, rep)] = {int(r): pr[r]['balacc'] for r in pr}
    print('nhieu', name, flush=True)
NZ = pd.DataFrame(nz); NZ.to_csv(f'{OUT}/e13_noise.csv', index=False)
rv = lambda m: np.array([np.nanmean([recbal[(m, s)][r] for s in SEEDS]) for r in RECS])
base = rv('none'); out = {'config': {k: dict(method=v[0], hp=v[1]) for k, v in METH.items()}, 'harm': {}, 'noise': {}, 'noisy_mean_jk': {}}
for name in NEW:
    hc = int((base - rv(name) > 0.05).sum()); hn = []
    for kind in ('miss', 'pair', 'uniform'):
        for p in (0.05, 0.10, 0.20):
            vv = np.array([np.nanmean([rbn[(name, s, kind, p, rp)][r] for s in SEEDS for rp in range(3)]) for r in RECS])
            hn.append(int((base - vv > 0.05).sum()))
    out['harm'][name] = dict(clean=hc, noisy_each=hn, noisy_mean=float(np.mean(hn)))
    out['noisy_mean_jk'][name] = float(NZ[NZ.method == name].groupby(['kind', 'p']).jk.mean().mean())
out['noise'] = {f'{m}|{k}|{p}': float(g.jk.mean()) for (m, k, p), g in NZ.groupby(['method', 'kind', 'p'])}
ab = MAIN.groupby('method')[['absent_n', 'absent_hit']].sum(); out['absent_Se'] = (ab.absent_hit / ab.absent_n).to_dict()
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, k in enumerate(o): run = max(run, (len(ps) - i) * ps[k]); adj[k] = min(run, 1)
    return adj
W = []
for ref, fam in [('none', NEW), ('GA-R', NEW[:2] + ['head_scratch'])]:
    tmp, ps = [], []
    for m in fam:
        a, b = rv(m), rv(ref); d = a - b; ok = ~np.isnan(d)
        p = float(wilcoxon(a[ok], b[ok]).pvalue) if (abs(d[ok]) > 1e-12).any() else 1.0
        tmp.append(dict(method=m, ref=ref, improved=int((d[ok] > 1e-9).sum()), degraded=int((d[ok] < -1e-9).sum()), p=p, median=float(np.median(d[ok]))))
        ps.append(p)
    for t, ph in zip(tmp, holm(np.array(ps))): t['p_holm'] = float(ph); W.append(t)
pd.DataFrame(W).to_csv(f'{OUT}/e13_wilcoxon.csv', index=False); out['wilcoxon'] = W
out['record_delta'] = {m: dict(min=float(np.nanmin(rv(m) - base)), worst_rec=int(RECS[int(np.nanargmin(rv(m) - base))]), n_below_005=int(((rv(m) - base) < -0.05).sum())) for m in NEW}
json.dump(out, open(f'{OUT}/e13_numbers.json', 'w'), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: out[k] for k in ['harm', 'noisy_mean_jk', 'absent_Se']}, indent=1, ensure_ascii=False, default=float))
print(pd.DataFrame(W).to_string())
