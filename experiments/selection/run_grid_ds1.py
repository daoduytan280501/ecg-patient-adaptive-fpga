"""Bo sung 26/10/2026 (protocol_bosung_20261026.md): danh gia luoi tren DS1 (model fold nested), cung S_rob va tieu chi on dinh
nhu run_ds1_select_v2.py. argv: GRID PART NPARTS. GRID: e13 (head_scratch_fair 216 cau hinh) | e15b (do nhay T, lambda, eta, so neo GA-R)."""
import sys, json, pickle, time, itertools
import numpy as np, pandas as pd
import pa_adapt as pa, rev4_methods as M, fair_methods  # noqa: F401 (dang ky head_scratch_fair)

GRID, PART, NP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = '../out/bosung_20261026'
SETS = pickle.load(open('../out/D1_nested.pkl', 'rb'))['sets']
CONDS = [('clean', 0.0, 0), ('uniform', 0.10, 0), ('pair', 0.10, 0)]
NOISY = {(F['seed'], F['k'], c[0]): M.noisy_labels(F['y'], F['cal'], F['seed'], *c) for F in SETS for c in CONDS}
PERT = {}
for F in SETS:                                   # giong het run_ds1_select_v2.py (cung hat giong)
    rng = np.random.default_rng(1000 + 10 * F['seed'] + F['k'])
    PERT[(F['seed'], F['k'])] = (F['E'] * (1 + 1e-12 * rng.standard_normal(F['E'].shape)),
                                 F['anchors'][0] * (1 + 1e-12 * rng.standard_normal(F['anchors'][0].shape)))
N_EV = sum(int((~F['cal']).sum()) for F in SETS); TOL = 0.0005 * N_EV


def anchors_m(F, m):
    A, ya = F['anchors']
    if m is None: return A, ya
    idx = np.concatenate([np.where(ya == c)[0][:m] for c in range(4)]); return A[idx], ya[idx]


def score(method, hp, m_anc=None):
    res = {}; clean_pred = None
    for kind, p, rep in CONDS:
        Y, P = [], []
        for F in SETS:
            y_, p_, _ = M.run(F['E'], F['y'], F['rec'], F['cal'], ~F['cal'], F['W0'], F['b0'], anchors_m(F, m_anc), method, fW=F['fW'],
                              y_cal=NOISY[(F['seed'], F['k'], kind)], **hp)
            Y.append(y_); P.append(p_)
        if kind == 'clean': clean_pred = np.concatenate(P)
        mm = pa.metrics(np.concatenate(Y), np.concatenate(P)); res[f'mF1_{kind}'] = mm['macroF1']; res[f'jk_{kind}'] = mm['jk']
    res['S_rob'] = float(np.mean([res[f'mF1_{k}'] for k, _, _ in CONDS]))
    P2 = []
    for F in SETS:
        E2, A2 = PERT[(F['seed'], F['k'])]
        A2m = (A2, F['anchors'][1]) if m_anc is None else anchors_m(dict(anchors=(A2, F['anchors'][1])), m_anc)
        _, p_, _ = M.run(E2, F['y'], F['rec'], F['cal'], ~F['cal'], F['W0'], F['b0'], A2m, method, fW=F['fW'], **hp)
        P2.append(p_)
    res['n_unstable'] = int((np.concatenate(P2) != clean_pred).sum()); res['stable'] = bool(res['n_unstable'] <= TOL)
    return res


ETAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
if GRID == 'e13':
    JOBS = [('head_scratch_fair', dict(lr=l, epochs=T, cap=c, lam=lm, absent=a), None)
            for l, T, c, lm, a in itertools.product(ETAS, (50, 150, 300), (1.0, 3.0, 10.0), (0.0, 0.01), ('fix', 'global'))]
elif GRID == 'e15b':
    JOBS = [('GA-S', dict(beta=0.1, cap=1.0, trust=0.5, lr=l, epochs=T, lam=lm), None)
            for l, T, lm in itertools.product((0.02, 0.05, 0.1), (100, 300, 600), (0.0, 0.01, 0.1))]
    JOBS += [('GA-S', dict(beta=0.1, cap=1.0, trust=0.5, lr=0.05, epochs=300, lam=0.01), m) for m in (5, 10)]
else:
    raise SystemExit(GRID)
JOBS = JOBS[PART::NP]
rows = []
for meth, hp, m in JOBS:
    t0 = time.time(); r = score(meth, hp, m)
    rows.append(dict(method=meth, hp=json.dumps(hp), m_anchor=m if m else 25, **r, sec=round(time.time() - t0, 1)))
    print(meth, hp, m, f"S_rob={r['S_rob']:.4f} unstable={r['n_unstable']}", flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/ds1_{GRID}_part{PART}.csv', index=False)
print('XONG', len(rows))
