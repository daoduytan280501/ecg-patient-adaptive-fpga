"""Kiem tra nhan qua cua doan hieu chinh (gop y thay 11/11/2026).
Cau hoi: tien xu ly cua 5 phut dau co dung tin hieu SAU moc hieu chinh khong, va neu cat bo thi ket qua co doi khong?
Cach lam: voi moi ban ghi DS2, cat tin hieu tai thoi diem som nhat ma nhip hieu chinh cuoi cung da du cua so (R+1.1 s)
va da biet dinh R ke tiep (cho RR_{i+1}); loc lai (median + Butterworth hai chieu) CHI tren doan da cat, tinh lai emb so nguyen
cua cac nhip hieu chinh bang hw_ref, chay lai GA-R va offset_int, danh gia tren doan danh gia goc (khong doi)."""
import sys, os, json, pickle, numpy as np
sys.path.insert(0, '/home/claude/planA_rev5/src'); sys.path.insert(0, '/home/claude/planA_rev4/src')
import wfdb, pc_pre_int as pp, hw_ref as hw, rev4_data as rd, rev4_methods as M, pa_adapt as pa
V = '/mnt/user-data/uploads/improve_v4'; DD = '/mnt/user-data/uploads/mitdb_extracted/mit-bih-arrhythmia-database-1.0.0'
B = np.load(f'{V}/pa_phaseA/f2a_board_data.npz', allow_pickle=True)
D2 = pickle.load(open('/home/claude/planA_rev4/out/D2.pkl', 'rb')); y, rec, cal, ev = D2['y'], D2['rec'], D2['cal'], D2['ev']
assert np.array_equal(y, B['y']) and np.array_equal(rec, B['rec'])
SEL = json.load(open('/home/claude/planA_rev5/out/ds1_selected_rev5b_20261025.json'))['selected']

def extract_trunc(r, cut):
    rr = wfdb.rdrecord(os.path.join(DD, str(r)), physical=False)
    ch = rr.sig_name.index('MLII') if 'MLII' in rr.sig_name else 0
    xr = rr.d_signal[:cut, ch].astype(np.int64); sig = (xr - rr.baseline[ch]) / rr.adc_gain[ch]
    ann = wfdb.rdann(os.path.join(DD, str(r)), 'atr'); S = np.asarray(ann.sample); Y = np.asarray(ann.symbol)
    keep = S < cut; S, Y = S[keep], Y[keep]
    yf = pp.clean_float(sig)[::2]; fs = 180.; pre, post = 162, 198
    isb = np.array([s in pp.AAMI for s in Y]); Rs = S[isb]; r_all = Rs / 2; sym = Y[isb]
    X, R, T = [], [], []
    for i in range(1, len(r_all) - 1):
        if pp.AAMI[sym[i]] == 'Q': continue
        c = int(round(r_all[i]))
        if c - pre < 0 or c + post > len(yf): continue
        w = yf[c - pre:c + post]; w = w - np.median(w); m = np.max(np.abs(w))
        if m < 1e-9: continue
        xf = (w / m).astype(np.float32)
        X.append(np.clip(np.round(np.clip(np.round(xf.astype(np.float64) * 64), -128, 127)), -128, 127))
        r8 = pp.rr_float(r_all, i, fs).astype(np.float32)
        rf = np.clip(r8 / pp.RR_DIV, 0, 3.9).astype(np.float64)
        R.append(np.clip(np.round(np.clip(np.round(rf * 32), -128, 127)), -128, 127)); T.append(r_all[i] / fs)
    return np.array(X, np.int8), np.array(R, np.int8), np.array(T)

newX = B['beats_q'].copy(); newR = B['rr_q'].copy(); stats = []
for r in sorted(np.unique(rec)):
    idx = np.where((rec == r) & cal)[0]; t = B['t'][idx]
    ann = wfdb.rdann(os.path.join(DD, str(r)), 'atr'); S = np.asarray(ann.sample)[np.array([a in pp.AAMI for a in ann.symbol])]
    R_last = int(round(t.max() * 360)); nxt = S[S > R_last + 2]
    cut = int(max(R_last + 400, (nxt[0] + 1) if len(nxt) else R_last + 400))
    X, R, T = extract_trunc(r, cut)
    pos = {round(v, 4): k for k, v in enumerate(T)}
    j = np.array([pos[round(v, 4)] for v in t])
    dX = int((X[j] != B['beats_q'][idx]).any(1).sum()); dR = int((R[j] != B['rr_q'][idx]).any(1).sum())
    newX[idx] = X[j]; newR[idx] = R[j]
    stats.append(dict(rec=int(r), n_cal=len(idx), cut_s=cut / 360, win_changed=dX, rr_changed=dR))
    print(stats[-1], flush=True)
res = dict(per_record=stats)
for s in rd.SEEDS:
    q = json.load(open(f'{V}/F2_rr8_str_s{s}_quant.json'))
    P = hw.quantize_params(rd.load_weights(f'{V}/F2_rr8_str_s{s}.keras'), q['qcfg'])
    E_full = hw.forward_emb_int(B['beats_q'][cal], B['rr_q'][cal], P, q['qcfg'])
    Sd = D2['seeds'][s]; fe = q['qcfg']['dense_1']
    assert np.array_equal(E_full / 2 ** fe, Sd['E'][cal]), 'reference emb mismatch'
    E_tr = hw.forward_emb_int(newX[cal], newR[cal], P, q['qcfg'])
    Enew = Sd['E'].copy(); Enew[cal] = E_tr / 2 ** fe
    emb_changed = int((E_tr != E_full).any(1).sum())
    out = dict(emb_changed=emb_changed, n_cal=int(cal.sum()))
    for name, (m, hp) in {'GA-R': ('GA-S', SEL['GA-R']['hp']), 'offset_int': ('offset_int', SEL['offset_int']['hp'])}.items():
        y0, p0, _ = M.run(Sd['E'], y, rec, cal, ev, Sd['W0'], Sd['b0'], (Sd['A'], rd.ANC_Y), m, fW=Sd['fw'], **hp)
        y1, p1, _ = M.run(Enew, y, rec, cal, ev, Sd['W0'], Sd['b0'], (Sd['A'], rd.ANC_Y), m, fW=Sd['fw'], **hp)
        out[name] = dict(jk_full=pa.metrics(y0, p0)['jk'], jk_trunc=pa.metrics(y1, p1)['jk'], labels_changed=int((p0 != p1).sum()))
    res[f'seed{s}'] = out; print(s, out, flush=True)
json.dump(res, open('/home/claude/planA_rev5/out/bosung_20261026/causal_cal_check_20261111.json', 'w'), indent=1, default=float)
