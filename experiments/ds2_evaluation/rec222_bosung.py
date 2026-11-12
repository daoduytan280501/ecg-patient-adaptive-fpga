"""Ban ghi 222 (DS2): vi sao head_scratch_fair giam balanced accuracy 0.33 (E13, chi bao cao)."""
import json, pickle, numpy as np
import pa_adapt as pa, rev4_methods as M, rev4_data as rd, fair_methods  # noqa
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb')); y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
SEL = json.load(open('../out/ds1_selected_rev5b_20261025.json'))['selected']; E13 = json.load(open('../out/bosung_20261026/e13_selected.json'))
m = rec == 222; print('cal', np.bincount(y[m & cal], minlength=4), 'eval', np.bincount(y[m & ev], minlength=4))
out = {}
for name, meth, hp in [('none', 'none', {}), ('GA-R', 'GA-S', SEL['GA-R']['hp']), ('head_scratch_fair', 'head_scratch_fair', E13['S_rob']['hp'])]:
    cms, ba = [], []
    for s in rd.SEEDS:
        S = D2['seeds'][s]; y_, p_, r_ = M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], (S['A'], rd.ANC_Y), meth, fW=S['fw'], **hp)
        k = r_ == 222; cms.append(pa.confusion(y_[k], p_[k])); ba.append(pa.per_record_scores(y_, p_, r_)[222]['balacc'])
    out[name] = dict(cm=np.sum(cms, 0).tolist(), balacc=float(np.mean(ba)))
    print(name, round(np.mean(ba), 3)); print(np.sum(cms, 0))
json.dump(out, open('../out/bosung_20261026/rec222.json', 'w'))
