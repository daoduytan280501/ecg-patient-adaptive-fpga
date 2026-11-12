"""rev5b (25/10/2026) - Chon tren DS1 (model fold nested) them buoc hoc eta va DIEU KIEN ON DINH SO HOC.
On dinh: chay lai voi emb (va neo) nhan nhieu tuong doi 1e-12 (mot lan rut/model fold, nhan sach); cau hinh hop le neu so nhan
thay doi <= 0.05% so nhip danh gia DS1 gop. Chon S_rob cao nhat trong cac cau hinh on dinh."""
import sys, json, pickle, time, itertools
import numpy as np, pandas as pd
import pa_adapt as pa, rev4_methods as M

OUT, D1P, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
SETS = pickle.load(open(D1P, 'rb'))['sets']
CONDS = [('clean', 0.0, 0), ('uniform', 0.10, 0), ('pair', 0.10, 0)]
NOISY = {(F['seed'], F['k'], c[0]): M.noisy_labels(F['y'], F['cal'], F['seed'], *c) for F in SETS for c in CONDS}
PERT = {}
for F in SETS:
    rng = np.random.default_rng(1000 + 10 * F['seed'] + F['k'])
    PERT[(F['seed'], F['k'])] = (F['E'] * (1 + 1e-12 * rng.standard_normal(F['E'].shape)),
                                 F['anchors'][0] * (1 + 1e-12 * rng.standard_normal(F['anchors'][0].shape)))
N_EV = sum(int((~F['cal']).sum()) for F in SETS)
TOL = 0.0005 * N_EV


def score(method, hp, stab=True):
    res = {}; clean_pred = None
    for kind, p, rep in CONDS:
        Y, P = [], []
        for F in SETS:
            y_, p_, _ = M.run(F['E'], F['y'], F['rec'], F['cal'], ~F['cal'], F['W0'], F['b0'], F['anchors'], method, fW=F['fW'],
                              y_cal=NOISY[(F['seed'], F['k'], kind)], **hp)
            Y.append(y_); P.append(p_)
        if kind == 'clean': clean_pred = np.concatenate(P)
        m = pa.metrics(np.concatenate(Y), np.concatenate(P))
        res[f'mF1_{kind}'] = m['macroF1']; res[f'jk_{kind}'] = m['jk']
    res['S_rob'] = float(np.mean([res[f'mF1_{k}'] for k, _, _ in CONDS]))
    if stab and method != 'offset_int' and not method.startswith('gate:offset'):
        P2 = []
        for F in SETS:
            E2, A2 = PERT[(F['seed'], F['k'])]
            _, p_, _ = M.run(E2, F['y'], F['rec'], F['cal'], ~F['cal'], F['W0'], F['b0'], (A2, F['anchors'][1]), method, fW=F['fW'], **hp)
            P2.append(p_)
        res['n_unstable'] = int((np.concatenate(P2) != clean_pred).sum())
    else:
        res['n_unstable'] = 0
    res['stable'] = bool(res['n_unstable'] <= TOL)
    return res


rows = []
def evaluate(method, grid, tag=None):
    best = None
    for hp in grid:
        t0 = time.time(); r = score(method, hp)
        rows.append(dict(method=tag or method, hp=json.dumps(hp), **r, sec=round(time.time() - t0, 1)))
        print(f'{tag or method:18s} {hp} S_rob={r["S_rob"]:.4f} unstable={r["n_unstable"]} {"" if r["stable"] else "LOAI"}', flush=True)
        if r['stable'] and (best is None or r['S_rob'] > best[1]['S_rob'] + 1e-12):
            best = (hp, r)
    return best


ETAS = tuple(float(x) for x in sys.argv[4].split(',')) if len(sys.argv) > 4 else (0.1, 0.2, 0.5)
SEL = {}
SEL['offset_int'] = evaluate('offset_int', [dict(tau=t, fbits=8) for t in (1, 5, 20)])
SEL['head_scratch'] = evaluate('head_scratch', [dict(lr=l, epochs=e) for l in ETAS for e in (50, 150, 300)])
SEL['head_global'] = evaluate('head_global', [dict(lr=l, epochs=e) for l in ETAS for e in (50, 150, 300)])
SEL['gradient_anchor'] = evaluate('gradient_anchor', [dict(beta=b, lr=l) for l in ETAS for b in (0.1, 0.3, 0.5)])
grid_r = [dict(beta=b, cap=c, trust=t, lr=l) for l in ETAS for b, c, t in itertools.product((0.1, 0.3, 0.5), (1.0, 3.0, 10.0), (0.25, 0.5, 1.0, None))]
SEL['GA-R'] = evaluate('GA-S', grid_r, tag='GA-R')
SEL['gate:GA-R'] = evaluate('gate:GA-S', [dict(SEL['GA-R'][0], delta=d) for d in (0.0, 0.02, 0.05)], tag='gate:GA-R')
pd.DataFrame(rows).to_csv(f'{OUT}/ds1_selection_{TAG}.csv', index=False)
out = dict(selected={m: dict(hp=SEL[m][0], **SEL[m][1]) for m in SEL}, n_eval_beats=N_EV, tol=TOL,
           rule='S_rob cao nhat trong cac cau hinh on dinh (nhan doi <= 0.05% khi emb nhieu 1e-12)')
json.dump(out, open(f'{OUT}/ds1_selected_{TAG}.json', 'w'), indent=1, ensure_ascii=False)
print('\nCHON:', {m: SEL[m][0] for m in SEL})
print('GA-R', SEL['GA-R'][1]['S_rob'], 'gate', SEL['gate:GA-R'][1]['S_rob'])
