"""rev5 (25/10/2026) - Moi so lieu cho ban thao, cau hinh chon tren DS1 KHONG ro ri (model fold nested, S_rob).
Hai co che: GA-R (gate + GA-S, beta 0.1, cap 1, khong vung tin cay) va offset_int (tau 20).
Dau ra: ../out/paper/*.csv + paper_numbers_rev5b_20261025.json"""
import os, sys, json, pickle, time
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import pa_adapt as pa, rev4_methods as M, rev4_data as rd

OUT = '../out/paper_rev5b'; os.makedirs(OUT, exist_ok=True)
EXP4 = '/mnt/user-data/uploads/ECG_FPGA_1DCNN/planA/06_eval/phaseA_exp4'
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb')); SEL = json.load(open('../out/ds1_selected_rev5b_20261025.json'))
y, rec, cal, ev, t = D2['y'], D2['rec'], D2['cal'], D2['ev'], D2['t']
RECS = sorted(np.unique(rec)); SEEDS = rd.SEEDS
G = SEL['selected']['GA-R']['hp']                        # beta 0.1, cap 1, trust 0.5, lr 0.05
GG = SEL['selected']['gate:GA-R']['hp']
assert SEL['proposed_GA_R'] == 'GA-R' and SEL['selected']['offset_int']['hp']['tau'] == 20
METH = {   # ten trong bai -> (method, hp)
    'none': ('none', {}),
    'head_scratch': ('head_scratch', SEL['selected']['head_scratch']['hp']),
    'head_global': ('head_global', SEL['selected']['head_global']['hp']),
    'offset_int': ('offset_int', SEL['selected']['offset_int']['hp']),
    'gradient_anchor': ('gradient_anchor', SEL['selected']['gradient_anchor']['hp']),
    'GA-R': ('GA-S', G),
}
print('Cấu hình:', json.dumps({k: v[1] for k, v in METH.items()}, ensure_ascii=False), flush=True)
NUM = {'config': {k: dict(method=v[0], hp=v[1]) for k, v in METH.items()}}


def runm(meth, hp, s, calm=None, ycal=None, info=None):
    S = D2['seeds'][s]; A = (S['A'], rd.ANC_Y)
    return M.run(S['E'], y, rec, cal if calm is None else calm, ev, S['W0'], S['b0'], A, meth, fW=S['fw'],
                 y_cal=ycal, info=info, **hp)


def mrow(y_, p_):
    m = pa.metrics(y_, p_)
    return dict(acc=m['acc'], macroF1=m['macroF1'], kappa=m['kappa'], jk=m['jk'],
                **{f'Se_{c}': m['Se'][c] for c in pa.CLASSES}, **{f'PP_{c}': m['PP'][c] for c in pa.CLASSES})


t0 = time.time()
# ---------------------------------------------------------------- 1. bang chinh + theo ban ghi + F + lop vang mat
rows, recbal, PRED = [], {}, {}
absent = M.absent_mask(y, rec, cal, ev)
for name, (meth, hp) in METH.items():
    for s in SEEDS:
        info = {}
        y_, p_, r_ = runm(meth, hp, s, info=info)
        PRED[(name, s)] = (y_, p_, r_)
        row = dict(method=name, seed=s, **mrow(y_, p_))
        ab = absent & (y_ != 0)
        row['absent_n'], row['absent_hit'] = int(ab.sum()), int((p_[ab] == y_[ab]).sum())
        f213 = (r_ == 213) & (y_ == 3); foth = (r_ != 213) & (y_ == 3)
        row.update(SeF_213=float((p_[f213] == 3).mean()), SeF_other=float((p_[foth] == 3).mean()),
                   nonF_called_F=int(((p_ == 3) & (y_ != 3)).sum()))
        row['n_reverted'] = None
        row['n_records'] = len(info)
        rows.append(row)
        pr = pa.per_record_scores(y_, p_, r_); recbal[(name, s)] = {int(r): pr[r]['balacc'] for r in pr}
MAIN = pd.DataFrame(rows); MAIN.to_csv(f'{OUT}/main_rows.csv', index=False)
bl = pd.read_csv(f'{EXP4}/exp4_baselines_DS2.csv')
for mth in ['ft_full', 'ft_full_anchor', 'scratch_patient']:
    for s in SEEDS:
        d = json.load(open(f'{EXP4}/exp4_bl/s{s}_{mth}.json'))
        recbal[(mth, s)] = {int(r): v['balacc'] for r, v in d['rec'].items()}
cols = ['acc', 'macroF1', 'kappa', 'jk', 'Se_S', 'PP_S', 'Se_V', 'PP_V', 'Se_F', 'PP_F']
summ = pd.concat([MAIN.groupby('method')[cols].agg(['mean', 'std']), bl[bl.method.isin(['ft_full', 'ft_full_anchor', 'scratch_patient'])].groupby('method')[cols].agg(['mean', 'std'])])
summ.to_csv(f'{OUT}/main_summary.csv'); NUM['main'] = {m: {c: [float(summ.loc[m, (c, 'mean')]), float(summ.loc[m, (c, 'std')])] for c in cols} for m in summ.index}
NUM['F'] = MAIN.groupby('method')[['Se_F', 'SeF_213', 'SeF_other', 'nonF_called_F']].mean().to_dict('index')
ab = MAIN.groupby('method')[['absent_n', 'absent_hit']].sum(); NUM['absent_Se'] = (ab.absent_hit / ab.absent_n).to_dict()
NUM['absent_n_per_seed'] = int(MAIN[MAIN.method == 'none'].absent_n.iloc[0])
print('1. bảng chính', round(time.time() - t0), 's', flush=True)

# ---------------------------------------------------------------- 2. Wilcoxon + Holm + co hieu ung
def rv(m): return np.array([np.nanmean([recbal[(m, s)][r] for s in SEEDS]) for r in RECS])
def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for i, k in enumerate(o):
        run = max(run, (len(ps) - i) * ps[k]); adj[k] = min(run, 1)
    return adj
def hl(d):
    w = (d[:, None] + d[None, :]) / 2; return float(np.median(w[np.triu_indices(len(d))]))
rng = np.random.default_rng(0); wt = []
fam1 = ['GA-R', 'gradient_anchor', 'offset_int', 'head_scratch', 'head_global', 'ft_full', 'ft_full_anchor', 'scratch_patient']
base = rv('none')
for fam, ref in [(fam1, 'none'), (['GA-R', 'offset_int'], 'ft_full_anchor')]:
    ps, tmp = [], []
    for m in fam:
        a, b = rv(m), rv(ref); d = a - b; ok = ~np.isnan(d); d = d[ok]
        p = wilcoxon(a[ok], b[ok]).pvalue if (np.abs(d) > 1e-12).any() else 1.0
        bm = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
        bh = [hl(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
        tmp.append(dict(method=m, ref=ref, improved=int((d > 1e-9).sum()), degraded=int((d < -1e-9).sum()), p=float(p),
                        median=float(np.median(d)), median_ci=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                        HL=hl(d), HL_ci=[float(np.percentile(bh, 2.5)), float(np.percentile(bh, 97.5))], mean=float(d.mean())))
        ps.append(p)
    for row, pa_ in zip(tmp, holm(np.array(ps))):
        row['p_holm'] = float(pa_); wt.append(row)
pd.DataFrame(wt).to_csv(f'{OUT}/wilcoxon.csv', index=False); NUM['wilcoxon'] = wt
NUM['record_delta'] = {}
for m in ['GA-R', 'offset_int', 'gradient_anchor', 'head_scratch', 'head_global']:
    d = rv(m) - base; ps_ = np.array([recbal[(m, s)][r] - recbal[('none', s)][r] for s in SEEDS for r in RECS]); ps_ = ps_[~np.isnan(ps_)]
    i = int(np.nanargmin(d))
    NUM['record_delta'][m] = dict(min=float(np.nanmin(d)), worst_rec=int(RECS[i]), n_below_005=int((d < -0.05).sum()), n_dec=int((d < -1e-9).sum()),
                                  n_inc=int((d > 1e-9).sum()), pairs_below_005=int((ps_ < -0.05).sum()), pairs_min=float(ps_.min()), n_pairs=int(len(ps_)))
print('2. Wilcoxon', round(time.time() - t0), 's', flush=True)

# ---------------------------------------------------------------- 3. ngan sach nhan
bud = []
for Bm in (1, 2, 3, 4, 5):
    calB = pa.split_first(t, rec, Bm)
    for name in ['offset_int', 'gradient_anchor', 'GA-R']:
        meth, hp = METH[name]
        for s in SEEDS:
            y_, p_, _ = runm(meth, hp, s, calm=calB)
            bud.append(dict(budget=Bm, method=name, seed=s, **mrow(y_, p_)))
BUD = pd.DataFrame(bud); BUD.to_csv(f'{OUT}/budget.csv', index=False)
NUM['budget'] = {f'{m}|{b}': {c: [float(g[c].mean()), float(g[c].std())] for c in ['macroF1', 'jk', 'Se_S', 'PP_S', 'Se_F']}
                 for (m, b), g in BUD.groupby(['method', 'budget'])}
print('3. ngân sách', round(time.time() - t0), 's', flush=True)

# ---------------------------------------------------------------- 4. nhieu nhan + ban ghi bi hai
CONDS = [(k, p, r) for k in ('miss', 'pair', 'uniform') for p in (0.05, 0.10, 0.20) for r in range(3)]
nz, rb_n = [], {}
for name in ['head_scratch', 'head_global', 'offset_int', 'gradient_anchor', 'GA-R']:
    meth, hp = METH[name]
    for s in SEEDS:
        for kind, p, rep in CONDS:
            yn = M.noisy_labels(y, cal, s, kind, p, rep)
            y_, p_, r_ = runm(meth, hp, s, ycal=yn)
            nz.append(dict(method=name, seed=s, kind=kind, p=p, rep=rep, **mrow(y_, p_)))
            pr = pa.per_record_scores(y_, p_, r_); rb_n[(name, s, kind, p, rep)] = {int(r): pr[r]['balacc'] for r in pr}
    print('   nhiễu', name, round(time.time() - t0), 's', flush=True)
NZ = pd.DataFrame(nz); NZ.to_csv(f'{OUT}/noise.csv', index=False)
NUM['noise'] = {f'{m}|{k}|{p}': float(g.jk.mean()) for (m, k, p), g in NZ.groupby(['method', 'kind', 'p'])}
base_r = {r: np.nanmean([recbal[('none', s)][r] for s in SEEDS]) for r in RECS}
harm = {}
for name in ['head_scratch', 'head_global', 'offset_int', 'gradient_anchor', 'GA-R']:
    v = {r: np.nanmean([recbal[(name, s)][r] for s in SEEDS]) for r in RECS}
    hc = sum(base_r[r] - v[r] > 0.05 for r in RECS)
    hn = []
    for kind in ('miss', 'pair', 'uniform'):
        for p in (0.05, 0.10, 0.20):
            vv = {r: np.nanmean([rb_n[(name, s, kind, p, rp)][r] for s in SEEDS for rp in range(3)]) for r in RECS}
            hn.append(sum(base_r[r] - vv[r] > 0.05 for r in RECS))
    harm[name] = dict(clean=int(hc), noisy_mean=float(np.mean(hn)), noisy_each=[int(x) for x in hn])
NUM['harm'] = harm
NUM['noisy_mean_jk'] = {m: float(NZ[NZ.method == m].groupby(['kind', 'p']).jk.mean().mean()) for m in harm}
print('4. nhiễu', round(time.time() - t0), 's', flush=True)

# ---------------------------------------------------------------- 5. ablation quanh GA-R
base_hp = dict(G)
ABL = [('GA-R (all components)', 'GA-S', G),
       ('with self-check gate', 'gate:GA-S', GG),
       ('clipping instead of scale mapping', 'gradient_anchor', G),
       ('without trust region', 'GA-S', dict(G, trust=None)),
       ('cap 3 instead of 1', 'GA-S', dict(G, cap=3.0)),
       ('cap 10 instead of 1', 'GA-S', dict(G, cap=10.0)),
       ('without anchors (beta 0)', 'GA-S', dict(G, beta=0.0)),
       ('without lambda', 'GA-S', dict(G, lam=0.0)),
       ('zero initialisation', 'GA-S', dict(G, init='zero')),
       ('step size 0.5 instead of 0.05', 'GA-S', dict(G, lr=0.5)),
       ('earlier setting (beta 0.1, cap 1, no trust, step 0.5, gate)', 'gate:GA-S', dict(beta=0.1, cap=1.0, trust=None, lr=0.5, delta=0.02)),
       ('rev4 setting (beta 0.5, cap 3, rho 0.5, step 0.5)', 'GA-S', dict(beta=0.5, cap=3.0, trust=0.5)),
       ('gradient_anchor (selected)', 'gradient_anchor', SEL['selected']['gradient_anchor']['hp'])]
abl = []
for nm, meth, hp in ABL:
    for s in SEEDS:
        y_, p_, _ = runm(meth, hp, s); r0 = dict(variant=nm, seed=s, cond='clean', rep=0, **mrow(y_, p_)); abl.append(r0)
        for kind in ('pair', 'uniform'):
            for rep in range(3):
                yn = M.noisy_labels(y, cal, s, kind, 0.10, rep); y_, p_, _ = runm(meth, hp, s, ycal=yn)
                abl.append(dict(variant=nm, seed=s, cond=kind, rep=rep, **mrow(y_, p_)))
AB = pd.DataFrame(abl); AB.to_csv(f'{OUT}/ablation.csv', index=False)
NUM['ablation'] = {}
for nm, *_ in ABL:
    c = AB[(AB.variant == nm) & (AB.cond == 'clean')]
    NUM['ablation'][nm] = dict(jk=[float(c.jk.mean()), float(c.jk.std())], Se_S=float(c.Se_S.mean()), PP_S=float(c.PP_S.mean()),
                               mF1=float(c.macroF1.mean()), Se_F=float(c.Se_F.mean()),
                               pair=float(AB[(AB.variant == nm) & (AB.cond == 'pair')].jk.mean()),
                               uniform=float(AB[(AB.variant == nm) & (AB.cond == 'uniform')].jk.mean()))
print('5. ablation', round(time.time() - t0), 's', flush=True)

# ---------------------------------------------------------------- 6. so nguyen: Mitchell (offset tau) + co gian GA-R
tau = METH['offset_int'][1]['tau']; mit = []
for s in SEEDS:
    S = D2['seeds'][s]; dmax = []
    for r in RECS:
        m = (rec == r) & cal
        if m.sum() == 0: continue
        _, b1 = pa.adapt_offset_int(S['E'][m], y[m], S['W0'], S['b0'], tau=tau, fbits=8)
        _, b2 = pa.adapt_offset(S['E'][m], y[m], S['W0'], S['b0'], tau=float(tau))
        dmax.append(np.abs(b1 - b2))
    y1_, p1_, _ = runm('offset_int', dict(tau=tau, fbits=8), s)
    S_ = D2['seeds'][s]
    y2_, p2_, _ = M.run(S_['E'], y, rec, cal, ev, S_['W0'], S_['b0'], None, 'none', fW=S_['fw'])
    ys, ps = [], []
    for r in RECS:
        mr = rec == r; mc, me = mr & cal, mr & ev
        if me.sum() == 0: continue
        W, b = pa.adapt_offset(S_['E'][mc], y[mc], S_['W0'], S_['b0'], tau=float(tau)); W, b = pa.quantize_head(W, b, S_['fw'])
        ps.append((S_['E'][me] @ W + b).argmax(1)); ys.append(y[me])
    pe = np.concatenate(ps)
    dm = np.concatenate(dmax)
    mit.append(dict(seed=s, max_abs=float(dm.max()), mean_abs=float(dm.mean()), changed=int((pe != p1_).sum()),
                    jk_mitchell=pa.metrics(y1_, p1_)['jk'], jk_exact=pa.metrics(np.concatenate(ys), pe)['jk']))
NUM['mitchell'] = mit
resc = []
for s in SEEDS:
    S = D2['seeds'][s]; A = (S['A'], rd.ANC_Y)
    for r in RECS:
        m = (rec == r) & cal
        W, b = M.ga(S['E'][m], y[m], S['W0'], S['b0'], A, **G)
        mx = float(np.abs(W).max()); resc.append(dict(seed=s, rec=int(r), maxW=mx, alpha=min(1.0, (127 / 128) / mx)))
RS = pd.DataFrame(resc); RS.to_csv(f'{OUT}/rescale.csv', index=False)
NUM['rescale'] = dict(n_rescaled=int((RS.alpha < 1).sum()), n_total=len(RS), alpha_min=float(RS.alpha.min()),
                      alpha_median_rescaled=float(RS[RS.alpha < 1].alpha.median()) if (RS.alpha < 1).any() else 1.0,
                      maxW_max=float(RS.maxW.max()))
print('6. số nguyên', round(time.time() - t0), 's', flush=True)

json.dump(NUM, open(f'{OUT}/paper_numbers_rev5b_20261025.json', 'w'), indent=1, ensure_ascii=False, default=float)
print('XONG', round(time.time() - t0), 's')
