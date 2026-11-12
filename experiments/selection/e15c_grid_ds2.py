"""E15c (protocol_bosung_20261026.md): CHI BAO CAO - danh gia toan bo luoi GA-R (216) va head_scratch_fair (216) tren DS2
(5 seed; sach, doi ngau nhien 10 %, nham cap 10 %; rep 0), de do tuong quan DS1-DS2, thu hang va khoang cach oracle.
Khong dung de chon. argv: PART NPARTS"""
import sys, json, pickle, time, numpy as np, pandas as pd
import pa_adapt as pa, rev4_methods as M, rev4_data as rd, fair_methods  # noqa
PART, NP = int(sys.argv[1]), int(sys.argv[2]); OUT = '../out/bosung_20261026'
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb')); y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
CONDS = [('clean', 0.0, 0), ('uniform', 0.10, 0), ('pair', 0.10, 0)]
NOISY = {(s, c[0]): M.noisy_labels(y, cal, s, *c) for s in rd.SEEDS for c in CONDS}
g1 = pd.read_csv('../out/ds1_selection_rev5b_20261025.csv'); g1 = g1[g1.method == 'GA-R']
g2 = pd.concat([pd.read_csv(f'{OUT}/ds1_e13_part{i}.csv') for i in (0, 1)])
JOBS = [('GA-R', 'GA-S', h) for h in g1.hp] + [('head_scratch_fair', 'head_scratch_fair', h) for h in g2.hp]
JOBS = JOBS[PART::NP]; rows = []
for tag, meth, hps in JOBS:
    hp = json.loads(hps); t0 = time.time(); r = {}
    for kind, _, _ in CONDS:
        Y, P = [], []
        for s in rd.SEEDS:
            S = D2['seeds'][s]
            y_, p_, _ = M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], (S['A'], rd.ANC_Y), meth, fW=S['fw'], y_cal=NOISY[(s, kind)], **hp)
            Y.append(y_); P.append(p_)
        m = pa.metrics(np.concatenate(Y), np.concatenate(P)); r[f'mF1_{kind}'] = m['macroF1']; r[f'jk_{kind}'] = m['jk']
    r['S_rob'] = float(np.mean([r[f'mF1_{k}'] for k, _, _ in CONDS]))
    rows.append(dict(method=tag, hp=hps, **r, sec=round(time.time() - t0, 1)))
    pd.DataFrame(rows).to_csv(f'{OUT}/ds2_grid_part{PART}.csv', index=False)
print('XONG', len(rows))
