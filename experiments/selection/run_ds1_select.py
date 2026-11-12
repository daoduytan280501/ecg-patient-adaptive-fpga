"""rev4 - Buoc chon tren DS1 (protocol rev4 muc 2 + v2). Dau ra: ds1_selection_20261022.csv, ds1_selected_20261022.json"""
import sys, json, pickle, time, itertools
import numpy as np, pandas as pd
import pa_adapt as pa, rev4_methods as M

OUT = sys.argv[1]; D1 = pickle.load(open(sys.argv[2], 'rb')); TAG = sys.argv[3]
SETS = D1['sets']
CONDS = [('clean', 0.0, 0), ('uniform', 0.10, 0), ('pair', 0.10, 0)]
NOISY = {(F['seed'], F['k'], c[0]): M.noisy_labels(F['y'], F['cal'], F['seed'], *c) for F in SETS for c in CONDS}


def score(method, hp):
    """Tra dict macroF1/jk cho tung dieu kien (gop 5 seed x 3 fold) + S_rob."""
    res = {}
    for kind, p, rep in CONDS:
        Y, P = [], []
        for F in SETS:
            y_, p_, _ = M.run(F['E'], F['y'], F['rec'], F['cal'], ~F['cal'], F['W0'], F['b0'], F['anchors'],
                              method, fW=F['fW'], y_cal=NOISY[(F['seed'], F['k'], kind)], **hp)
            Y.append(y_); P.append(p_)
        m = pa.metrics(np.concatenate(Y), np.concatenate(P))
        res[f'mF1_{kind}'] = m['macroF1']; res[f'jk_{kind}'] = m['jk']
    res['S_rob'] = float(np.mean([res[f'mF1_{k}'] for k, _, _ in CONDS]))
    return res


rows = []
def evaluate(method, grid, tag=None):
    best = None
    for hp in grid:
        t0 = time.time(); r = score(method, hp)
        rows.append(dict(method=tag or method, hp=json.dumps(hp), **r, sec=round(time.time() - t0, 1)))
        print(f'{tag or method:22s} {hp} S_rob={r["S_rob"]:.4f} (sạch {r["mF1_clean"]:.4f}, uni {r["mF1_uniform"]:.4f}, pair {r["mF1_pair"]:.4f})', flush=True)
        if best is None or r['S_rob'] > best[1]['S_rob'] + 1e-12:
            best = (hp, r)
    return best


SEL = {}
SEL['offset_int'] = evaluate('offset_int', [dict(tau=t, fbits=8) for t in (1, 5, 20)])
SEL['head_scratch'] = evaluate('head_scratch', [dict(lr=0.5, epochs=e) for e in (50, 150, 300)])
SEL['head_global'] = evaluate('head_global', [dict(lr=0.5, epochs=e) for e in (50, 150, 300)])
betas = [dict(beta=b) for b in (0.1, 0.3, 0.5)]
SEL['gradient_anchor'] = evaluate('gradient_anchor', betas)
SEL['GA-P'] = evaluate('GA-P', betas)
SEL['GA-S'] = evaluate('GA-S', betas)
# chon lai gradient_anchor theo tieu chi cu cua A (macro-F1 nhan sach) - de minh bach
ga_rows = [r for r in rows if r['method'] == 'gradient_anchor']
ga_clean = max(ga_rows, key=lambda r: r['mF1_clean'])
base_r = 'GA-S' if SEL['GA-S'][1]['S_rob'] >= SEL['GA-P'][1]['S_rob'] else 'GA-P'
grid_r = [dict(beta=b, cap=c, trust=t) for b, c, t in itertools.product((0.1, 0.3, 0.5), (1.0, 3.0, 10.0), (0.25, 0.5, 1.0, None))]
SEL['GA-R'] = evaluate(base_r, grid_r, tag='GA-R')
deltas = (0.0, 0.02, 0.05)
SEL['gate:offset_int'] = evaluate('gate:offset_int', [dict(SEL['offset_int'][0], delta=d) for d in deltas], tag='gate:offset_int')
SEL['gate:GA-R'] = evaluate(f'gate:{base_r}', [dict(SEL['GA-R'][0], delta=d) for d in deltas], tag='gate:GA-R')

COST = ['offset_int', 'gate:offset_int', 'GA-P', 'GA-S', 'GA-R', 'gate:GA-R']      # tang dan chi phi
cands = sorted(COST, key=lambda m: -SEL[m][1]['S_rob'])
top = SEL[cands[0]][1]['S_rob']
near = [m for m in COST if top - SEL[m][1]['S_rob'] < 0.005]
chosen = near[0]                                                                   # re nhat trong nhom sat dinh
pd.DataFrame(rows).to_csv(f'{OUT}/ds1_selection_{TAG}.csv', index=False)
out = dict(selected={m: dict(hp=SEL[m][0], **SEL[m][1]) for m in SEL}, GA_R_base=base_r,
           gradient_anchor_clean_criterion=dict(hp=json.loads(ga_clean['hp']), mF1_clean=ga_clean['mF1_clean']),
           proposed=chosen, proposed_base=(base_r if 'GA-R' in chosen else None), near_top=near,
           rule='S_rob cao nhat; chenh < 0.005 -> chon re hon (thu tu: ' + ' < '.join(COST) + ')')
json.dump(out, open(f'{OUT}/ds1_selected_{TAG}.json', 'w'), indent=1, ensure_ascii=False)
print('\nĐỀ XUẤT (chọn trên DS1):', chosen, SEL[chosen][0], '| sát đỉnh:', near)
