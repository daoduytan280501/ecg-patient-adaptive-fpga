"""Ban ghi 200 (DS2): vi sao GA-R giam balanced accuracy > 0.05 (cau hinh rev5b)."""
import json, pickle, numpy as np
import pa_adapt as pa, rev4_methods as M, rev4_data as rd
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb')); SEL = json.load(open('../out/ds1_selected_rev5b_20261025.json'))['selected']
y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
out = {}
for name, meth, hp in [('none', 'none', {}), ('offset_int', 'offset_int', SEL['offset_int']['hp']), ('GA-R', 'GA-S', SEL['GA-R']['hp'])]:
    res = []
    for s in rd.SEEDS:
        S = D2['seeds'][s]; A = (S['A'], rd.ANC_Y)
        y_, p_, r_ = M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], A, meth, fW=S['fw'], **hp)
        m = r_ == 200; cm = pa.confusion(y_[m], p_[m]); pr = pa.per_record_scores(y_, p_, r_)[200]
        res.append(dict(seed=s, balacc=pr['balacc'], cm=cm.tolist()))
    out[name] = res
    print(name, [round(r['balacc'], 3) for r in res], np.mean([r['balacc'] for r in res]))
    print(np.sum([np.array(r['cm']) for r in res], 0))
m = rec == 200
print('cal counts', np.bincount(y[m & cal], minlength=4), 'eval counts', np.bincount(y[m & ev], minlength=4))
json.dump(out, open('../out/rec200_rev5b_20261026.json', 'w'))
