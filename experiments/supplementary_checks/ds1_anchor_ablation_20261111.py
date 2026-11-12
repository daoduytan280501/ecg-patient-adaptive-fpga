"""Ablation neo tren DS1 (cross-validation long nhau, 15 model fold) theo gop y thay 11/11/2026.
So sanh: none, GA-R (cau hinh chon), GA-R beta=0 (trong so mau neo = 0, neo van dung tinh s), head_scratch (fair) (cau hinh chon).
Chi so: S_rob; Se lop vang mat tren nhip validation; so ban ghi DS1 giam balanced accuracy > 0.05 so voi none
(TB 5 seed; nhan sach va TB 9 dieu kien nhieu x 3 lan lap); Wilcoxon theo ban ghi."""
import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/claude/planA_rev5/src')
import pa_adapt as pa, rev4_methods as M, fair_methods  # noqa
from scipy.stats import wilcoxon
SETS = pickle.load(open('/home/claude/planA_rev5/out/D1_nested.pkl', 'rb'))['sets']
SEL = json.load(open('/home/claude/planA_rev5/out/ds1_selected_rev5b_20261025.json'))['selected']
HSF = json.load(open('/home/claude/planA_rev5/out/bosung_20261026/e13_selected.json'))['S_rob']['hp']
CFG = {'none': ('none', {}), 'GA-R': ('GA-S', SEL['GA-R']['hp']), 'GA-R zero anchor weights': ('GA-S', dict(SEL['GA-R']['hp'], beta=0.0)),
       'head_scratch (fair)': ('head_scratch_fair', HSF)}
RECS = sorted({int(r) for F in SETS for r in np.unique(F['rec'][~F['cal']])})

def run_all(meth, hp, kind=None, p=0.0, rep=0):
    ba = {r: [] for r in RECS}; Y = []; P = []; hit = n = 0
    for F in SETS:
        yc = None if kind is None else M.noisy_labels(F['y'], F['cal'], F['seed'], kind, p, rep)
        y_, p_, r_ = M.run(F['E'], F['y'], F['rec'], F['cal'], ~F['cal'], F['W0'], F['b0'], F['anchors'], meth, fW=F['fW'], y_cal=yc, **hp)
        Y.append(y_); P.append(p_)
        pr = pa.per_record_scores(y_, p_, r_)
        for r, v in pr.items(): ba[int(r)].append(v['balacc'])
        ab = M.absent_mask(F['y'], F['rec'], F['cal'], ~F['cal']) & (y_ != 0)
        hit += int((p_[ab] == y_[ab]).sum()); n += int(ab.sum())
    return np.array([np.nanmean(ba[r]) for r in RECS]), pa.metrics(np.concatenate(Y), np.concatenate(P)), hit / max(n, 1), n

res = {}; base, _, _, _ = run_all('none', {})
_, mn, abn, nab = run_all('none', {}); res['none'] = dict(jk=mn['jk'], mF1=mn['macroF1'], absent_Se=abn, n_absent=nab)
NOISE = [(k, p) for k in ('miss', 'pair', 'uniform') for p in (0.05, 0.10, 0.20)]
for name, (m, hp) in CFG.items():
    if name == 'none': continue
    v, mm, ab, nab = run_all(m, hp)
    d = v - base; ok = ~np.isnan(d)
    w = wilcoxon(d[ok])
    harm_noisy = []
    for k, p in NOISE:
        vv = np.mean([run_all(m, hp, k, p, rp)[0] for rp in range(3)], 0)
        harm_noisy.append(int(((vv - base) < -0.05).sum()))
    res[name] = dict(jk=mm['jk'], mF1=mm['macroF1'], absent_Se=ab, n_absent=nab, harm_clean=int((d < -0.05).sum()),
                     up=int((d[ok] > 0).sum()), down=int((d[ok] < 0).sum()), p=float(w.pvalue), harm_noisy=harm_noisy,
                     harm_noisy_mean=float(np.mean(harm_noisy)), n_records=int(ok.sum()))
    print(name, {k: v for k, v in res[name].items() if k != 'harm_noisy'}, flush=True)
json.dump(res, open('/home/claude/planA_rev5/out/bosung_20261026/ds1_anchor_ablation_20261111.json', 'w'), indent=1, default=float)
print(json.dumps(res['none']))
