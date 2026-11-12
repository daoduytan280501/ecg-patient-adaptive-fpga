"""
rev4_data.py - Nap du lieu cho thi nghiem sua bai A theo gop y cua thay (22/10/2026).
Chi dung NumPy (+h5py de doc trong so .keras). Khong can TensorFlow.

DS2  : emb so nguyen exp4 (exp4_emb_int_s{seed}.npz: E = DS2, A = 100 nhip neo DS1), lop dau ra toan cuc
       doc tu F2_rr8_str_s{seed}.keras + _quant.json (giong phaseA_rev3/base.py), nhan/thoi diem tu f2a_board_data.npz.
DS1  : 15 model fold (3 fold x 5 seed). Emb validation lay tu cache planC exp1 (da kiem: trung pipeline exp4);
       neo cua moi fold (25 nhip/lop tu phan TRAIN cua fold) tinh lai bang hw_ref voi trong so doc tu .keras.
"""
import os, io, json, zipfile
import numpy as np
import h5py
import pa_adapt as pa
import hw_ref as hw
import pc_pre_int as pp

H5_MAP = {'conv1d': 'conv1d_1', 'conv1d_1': 'conv1d_2', 'conv1d_2': 'conv1d_3',
          'dense': 'rr_dense', 'dense_1': 'dense_1', 'dense_2': 'dense_out'}
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
       205, 207, 208, 209, 215, 220, 223, 230]
SEEDS = [1, 2, 3, 4, 5]
ANC_Y = np.repeat(np.arange(4), 25)


def load_weights(path):
    z = zipfile.ZipFile(path)
    f = h5py.File(io.BytesIO(z.read('model.weights.h5')), 'r')
    return {H5_MAP[k]: (f[f'layers/{k}/vars/0'][()], f[f'layers/{k}/vars/1'][()]) for k in H5_MAP}


# ----------------------------- DS2 -----------------------------------------
def load_ds2(V):
    """V: thu muc improve_v4. Tra dict y, rec, t, cal, ev va seeds[seed] = dict(E, A, W0, b0, fw)."""
    D = np.load(os.path.join(V, 'pa_phaseA', 'f2a_board_data.npz'), allow_pickle=True)
    y, rec, t = D['y'], D['rec'], D['t']
    cal = pa.split_calibration(t, rec, 5)
    out = dict(y=y, rec=rec, t=t, cal=cal, ev=~cal, seeds={})
    for s in SEEDS:
        q = json.load(open(os.path.join(V, f'F2_rr8_str_s{s}_quant.json')))
        Wd = load_weights(os.path.join(V, f'F2_rr8_str_s{s}.keras'))
        W, b = Wd['dense_out']
        fw, fb, fe = q['qcfg']['dense_out.kernel'], q['qcfg']['dense_out.bias'], q['qcfg']['dense_1']
        Wq = np.clip(np.round(W * 2 ** fw), -128, 127); bq = np.clip(np.round(b * 2 ** fb), -128, 127)
        b14 = np.round((bq / 2 ** fb + np.array(q['offset'])) * 2 ** 14)
        z = np.load(os.path.join(V, 'pa_phaseA', f'exp4_emb_int_s{s}.npz'))
        out['seeds'][s] = dict(E=z['E'] / 2 ** fe, A=z['A'] / 2 ** fe, W0=Wq / 2 ** fw, b0=b14 / 2 ** 14, fw=fw)
    return out


# ----------------------------- DS1 (CV) ------------------------------------
def frac_bits_for(x, bits=8):
    m = float(np.max(np.abs(x)))
    if m < 1e-12:
        return bits - 1
    return int(max(bits - 1 - max(int(np.ceil(np.log2(m))), 0), 0))


def rr_normalize(R):
    return (R[:, :8] / pp.RR_DIV).clip(0, 3.9)


def _conv_same(x, W, b):
    k = W.shape[0]; pl = (k - 1) // 2; pr = k - 1 - pl
    xp = np.pad(x, ((0, 0), (pl, pr), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=1)
    return np.einsum('nlck,kcf->nlf', win, W, optimize=True) + b


def calibrate(Wd, Xb, R, n=3000, seed=42):
    """Giong het calibrate() cua exp4 (conv 3 khoi, rr_dense 16) nhung nhan dict trong so."""
    idx = np.random.default_rng(seed).choice(len(Xb), min(n, len(Xb)), replace=False)
    q = {'input': 6}
    for name, (W, b) in Wd.items():
        q[f'{name}.kernel'], q[f'{name}.bias'] = frac_bits_for(W), frac_bits_for(b)
    x = Xb[idx, :, None].astype(np.float64)
    for i in (1, 2, 3):
        W, b = Wd[f'conv1d_{i}']
        x = np.maximum(_conv_same(x, W, b), 0); q[f'relu_{i}'] = frac_bits_for(x)
        Lp = x.shape[1] // 2; x = np.maximum(x[:, 0:2 * Lp:2], x[:, 1:2 * Lp:2])
    x = x.reshape(len(x), -1)
    rr = rr_normalize(R[idx]); q['rr'] = frac_bits_for(rr)
    W, b = Wd['rr_dense']; rr = np.maximum(rr @ W + b, 0); q['rr_dense'] = frac_bits_for(rr)
    x = np.concatenate([x, rr], axis=1)
    W, b = Wd['dense_1']; q['dense_1'] = frac_bits_for(np.maximum(x @ W + b, 0))
    return q


def make_folds(recs_y, k=3):
    order = sorted([r for r in recs_y if r != 208],
                   key=lambda r: -(np.sum(recs_y[r] == 1) * 5 + np.sum(recs_y[r] == 2)))
    folds, load = [[] for _ in range(k)], np.zeros((k, 2))
    for r in order:
        j = int(np.argmin(load[:, 0] * 5 + load[:, 1]))
        folds[j].append(r); load[j] += [np.sum(recs_y[r] == 1), np.sum(recs_y[r] == 2)]
    return folds


def load_ds1(mitdb_dir, fold_dir, emb_cache, quant_json_s1, verify=True, verbose=True):
    """Tra dict y1, g1, T1, folds (list ban ghi) va sets: list dict(seed, k, E, y, rec, cal, W0, b0, fW, anchors)."""
    parts = [pp.extract_both(r, mitdb_dir) for r in DS1]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    X1, R1, Xq, y1, g1, T1 = cat('X_f32'), cat('R_f32'), cat('Xq_f'), cat('y'), cat('rec'), cat('T')
    folds = make_folds({r: y1[g1 == r] for r in DS1})
    off = np.array(json.load(open(quant_json_s1))['offset'])
    sets = []
    for s in SEEDS:
        for k, vrecs in enumerate(folds):
            vmask = np.isin(g1, vrecs)
            Wd = load_weights(os.path.join(fold_dir, f'F2_fold{k}_s{s}.keras'))
            qcfg = calibrate(Wd, X1[~vmask], R1[~vmask])
            P = hw.quantize_params(Wd, qcfg)
            z = np.load(os.path.join(emb_cache, f'emb_fold{k}_s{s}.npz'))
            fe, fw, wo, bq14 = int(z['fe']), int(z['fw']), z['wo'], z['bq14']
            assert fe == qcfg['dense_1'], (s, k)
            if verify:                                      # kiem tra chuoi doc trong so + calibrate tren 300 nhip
                v = np.where(vmask)[0][:300]
                Ev = hw.forward_emb_int(Xq[v], pp.rr_float_q(R1[v], qcfg['rr']), P, qcfg)
                assert np.array_equal(Ev, z['E'][:300]), f'emb lech cache exp1 (seed {s} fold {k})'
            ai = pa.sample_anchor_idx(y1, ~vmask, m=25, seed=s)
            A = hw.forward_emb_int(Xq[ai], pp.rr_float_q(R1[ai], qcfg['rr']), P, qcfg)
            sets.append(dict(seed=s, k=k, E=z['E'] / 2.0 ** fe, y=y1[vmask], rec=g1[vmask],
                             cal=pa.split_calibration(T1[vmask], g1[vmask], 5), T=T1[vmask],
                             W0=(wo / 2.0 ** fw).T, b0=bq14 / 2.0 ** 14, fW=fw,
                             anchors=(A / 2.0 ** fe, y1[ai])))
            if verbose:
                print(f'DS1 seed {s} fold {k}: {vmask.sum()} nhịp, neo {len(ai)}', flush=True)
    return dict(y1=y1, g1=g1, T1=T1, folds=folds, sets=sets)
