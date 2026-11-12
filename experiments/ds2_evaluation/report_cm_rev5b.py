"""Ma tran nham lan seed 3 (doan danh gia DS2) cho bao cao rev5b."""
import json, pickle, numpy as np
import pa_adapt as pa, rev4_methods as M, rev4_data as rd
D2 = pickle.load(open('../../planA_rev4/out/D2.pkl', 'rb')); SEL = json.load(open('../out/ds1_selected_rev5b_20261025.json'))['selected']
y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
S = D2['seeds'][3]; A = (S['A'], rd.ANC_Y)
out = {}
for name, meth, hp in [('none', 'none', {}), ('offset_int', 'offset_int', SEL['offset_int']['hp']), ('GA-R', 'GA-S', SEL['GA-R']['hp'])]:
    y_, p_, _ = M.run(S['E'], y, rec, cal, ev, S['W0'], S['b0'], A, meth, fW=S['fw'], **hp)
    out[name] = pa.confusion(y_, p_).tolist(); print(name, pa.metrics(y_, p_)['jk'], out[name])
json.dump(out, open('../report/cm_seed3_rev5b.json', 'w'))
