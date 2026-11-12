"""D1 khong ro ri: model fold huan luyen long (nested_train.py), offset + epoch tu inner-val trong T_k."""
import os, io, sys, json, zipfile, pickle, h5py, numpy as np
import pa_adapt as pa, hw_ref as hw, pc_pre_int as pp, rev4_data as rd
MITDB, NF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NAMES = ['conv1d_1', 'conv1d_2', 'conv1d_3', 'rr_dense', 'dense_1', 'dense_out']
def load_w(path):
    f = h5py.File(io.BytesIO(zipfile.ZipFile(path).read('model.weights.h5')), 'r')
    return {n: (f[f'layers/{n}/vars/0'][()], f[f'layers/{n}/vars/1'][()]) for n in NAMES}
parts = [pp.extract_both(r, MITDB) for r in rd.DS1]
cat = lambda k: np.concatenate([p[k] for p in parts])
X1, R1, Xq, y1, g1, T1 = cat('X_f32'), cat('R_f32'), cat('Xq_f'), cat('y'), cat('rec'), cat('T')
folds = rd.make_folds({r: y1[g1 == r] for r in rd.DS1})
sets = []
for s in rd.SEEDS:
    for k, vrecs in enumerate(folds):
        meta = json.load(open(f'{NF}/F2_nfold{k}_s{s}.json')); assert meta['val_recs'] == vrecs and not set(sum(meta['inner_folds'], [])) & set(vrecs)
        vmask = np.isin(g1, vrecs)
        Wd = rd.load_weights(f'{NF}/F2_nfold{k}_s{s}.keras'); assert Wd['rr_dense'][0].shape == (8, 16) and Wd['dense_out'][0].shape == (32, 4)
        qcfg = rd.calibrate(Wd, X1[~vmask], R1[~vmask]); P = hw.quantize_params(Wd, qcfg); fe = qcfg['dense_1']
        E = hw.forward_emb_int(Xq[vmask], pp.rr_float_q(R1[vmask], qcfg['rr']), P, qcfg)
        wo, fw, bq14 = hw.head_global_int(P, meta['offset'])
        ai = pa.sample_anchor_idx(y1, ~vmask, m=25, seed=s); assert not np.isin(g1[ai], vrecs).any()
        A = hw.forward_emb_int(Xq[ai], pp.rr_float_q(R1[ai], qcfg['rr']), P, qcfg)
        sets.append(dict(seed=s, k=k, E=E / 2.0 ** fe, y=y1[vmask], rec=g1[vmask], cal=pa.split_calibration(T1[vmask], g1[vmask], 5),
                         T=T1[vmask], W0=(wo / 2.0 ** fw).T, b0=bq14 / 2.0 ** 14, fW=fw, anchors=(A / 2.0 ** fe, y1[ai]), meta=meta))
        print(f'seed {s} fold {k}: {vmask.sum()} nhip, epoch {meta["epochs"]}, offset {np.round(meta["offset"], 2)}', flush=True)
pickle.dump(dict(y1=y1, g1=g1, T1=T1, folds=folds, sets=sets), open(OUT, 'wb'))
print('xong', OUT)
