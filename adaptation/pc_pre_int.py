"""
pc_pre_int.py - Phuong an C, buoc 3 (22/10/2026): tien xu ly SO NGUYEN V0' (mo hinh cho PL) va so sanh
voi pipeline so thuc goc (o 4 ecg_pa_exp4_v2: median 73 -> 217, Butterworth 4 bac 40 Hz filtfilt, ::2,
cua so 0.9 s / 1.1 s, chuan hoa trung vi + max|w|, 8 dac trung RR).

V0' (dac ta: planC/00_data/preproc_audit_20261022.md):
  x      = mau ADC nguyen (d_signal, MIT-BIH 11 bit)
  base   = median73 -> median217 tren so nguyen (chinh xac nhu scipy: trung vi giao hoan voi bien doi affine)
  x - base -> FIR pha tuyen tinh 2L+1 tap (= dap ung xung filtfilt cat ngan), he so Q.BH, dau ra giu FY bit le
  ha mau ::2;  cua so 360 mau quanh c = round_half_even(R/2)
  chuan hoa: w2 = 2w - (a+b)  (a, b: 2 phan tu giua -> trung vi x2, chinh xac), m = max|w2|,
             q = round_half_up(64 * w2 / m)   (int8, frac 6)
  RR: so mau nguyen @360 Hz; ti so tinh bang phep chia nguyen co lam tron nua len; clip nhu ban so thuc.
Moi phep lam tron dung 'nua len' (AP_RND) nhu HLS.
"""
import os
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

FS = 360
AAMI = {'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N', 'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
        'V': 'V', 'E': 'V', 'F': 'F', '/': 'Q', 'f': 'Q', 'Q': 'Q'}
CLASSES = ['N', 'S', 'V', 'F']
RR_DIV = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Lam tron nua len cho phep chia nguyen (num, den > 0 hoac num am) -> floor(num/den + 1/2)
# ---------------------------------------------------------------------------
def div_round(num, den):
    num = np.asarray(num, dtype=np.int64); den = np.asarray(den, dtype=np.int64)
    return (2 * num + den) // (2 * den)


def fir_taps(L=30, BH=15, fc=40.0, fs=FS):
    """Dap ung xung tuong duong cua filtfilt(butter(4, fc)) cat ngan 2L+1 tap, luong tu Q.BH, tong = 2^BH."""
    b, a = butter(4, fc / (fs / 2), btype='low')
    N = 4 * L + 401
    d = np.zeros(N); d[N // 2] = 1.0
    h = filtfilt(b, a, d, padlen=0)[N // 2 - L: N // 2 + L + 1]
    h = h / h.sum()
    hq = np.round(h * 2 ** BH).astype(np.int64)
    hq[L] += 2 ** BH - hq.sum()                     # DC gain dung bang 1
    return hq


def clean_int(xr, L=30, BH=15, FY=4):
    """xr: mau nguyen. Tra tin hieu da loc (so nguyen, FY bit le) o 360 Hz."""
    xr = np.asarray(xr, dtype=np.int64)
    base = median_filter(xr, size=73, mode='nearest')
    base = median_filter(base, size=217, mode='nearest')
    x = xr - base
    acc = np.convolve(x, fir_taps(L, BH), mode='same')
    sh = BH - FY
    return (acc + (1 << (sh - 1))) >> sh


def clean_float(sig, fs=FS):
    """Giong het clean_signal cua exp4 (tham chieu)."""
    base = median_filter(sig, size=int(0.2 * fs) | 1, mode='nearest')
    base = median_filter(base, size=int(0.6 * fs) | 1, mode='nearest')
    b, a = butter(4, 40.0 / (fs / 2), btype='low')
    return filtfilt(b, a, sig - base)


def norm_window_int(w):
    """w: 360 so nguyen -> int8 frac 6 (chinh xac den lam tron nua len)."""
    w = np.asarray(w, dtype=np.int64)
    s = np.sort(w)
    n = len(w)
    w2 = 2 * w - (s[n // 2 - 1] + s[n // 2])
    m = np.abs(w2).max()
    if m == 0:
        return np.zeros(n, np.int64), 0
    return np.clip(div_round(64 * w2, m), -128, 127), int(m)


def rr_int(Rs, i, FR=5):
    """8 dac trung RR (int8, FR bit le) cua nhip i tu vi tri dinh R nguyen Rs (mau @360 Hz)."""
    rr = np.diff(np.asarray(Rs, dtype=np.int64))
    pre, post = rr[i - 1], rr[i]
    a10, a30 = max(0, i - 10), max(0, i - 30)
    s10, c10 = rr[a10:i].sum(), i - a10
    s30, c30 = rr[a30:i].sum(), i - a30
    S = 1 << FR
    num = [pre * S, post * S, s10 * S, pre * c10 * S, post * c10 * S, pre * S, pre * c30 * S, post * c30 * S]
    den = [540, 540, 540 * c10, 2 * s10, 2 * s10, 2 * post, 2 * s30, 2 * s30]
    q = np.array([div_round(n_, d_) for n_, d_ in zip(num, den)], dtype=np.int64)
    cl = int(np.round(3.9 * S))
    return np.clip(q, 0, min(cl, 127))


def rr_raw(Rs, i):
    """So nguyen tho cho RR cua nhip i: (pre, post, s10, c10, s30, c30) - doc lap voi so bit le."""
    rr = np.diff(np.asarray(Rs, dtype=np.int64))
    a10, a30 = max(0, i - 10), max(0, i - 30)
    return np.array([rr[i - 1], rr[i], rr[a10:i].sum(), i - a10, rr[a30:i].sum(), i - a30], dtype=np.int64)


def rr_int_from_raw(raw, FR):
    """Vector hoa rr_int: raw (N, 6) -> (N, 8) int8 voi FR bit le (FR lay theo qcfg['rr'] cua tung model)."""
    raw = np.asarray(raw, dtype=np.int64)
    pre, post, s10, c10, s30, c30 = raw.T
    S = 1 << FR
    num = [pre * S, post * S, s10 * S, pre * c10 * S, post * c10 * S, pre * S, pre * c30 * S, post * c30 * S]
    den = [np.full_like(pre, 540), np.full_like(pre, 540), 540 * c10, 2 * s10, 2 * s10, 2 * post, 2 * s30, 2 * s30]
    q = np.stack([div_round(n_, d_) for n_, d_ in zip(num, den)], 1)
    return np.clip(q, 0, min(int(np.round(3.9 * S)), 127)).astype(np.int8)


def rr_float_q(R8, FR):
    """Luong tu RR so thuc nhu exp4 (rr_normalize -> fq -> to_i8) voi FR bit le. R8: (N, 8) float32."""
    rf = np.clip(np.asarray(R8, np.float32) / RR_DIV, 0, 3.9).astype(np.float64)
    return np.clip(np.round(np.clip(np.round(rf * 2 ** FR), -128, 127)), -128, 127).astype(np.int8)


def rr_float(Rs_f, i, fs):
    """Dac trung RR so thuc nhu exp4 (8 dau), Rs_f = r_all (da chia ds), fs = 180."""
    rr = np.diff(Rs_f) / fs
    pre, post = rr[i - 1], rr[i]
    local = np.mean(rr[max(0, i - 10):i]); local30 = np.mean(rr[max(0, i - 30):i])
    return np.array([pre, post, local, pre / local, post / local, pre / post, pre / local30, post / local30])


def extract_both(rec, data_dir, pre_s=0.9, post_s=1.1, ds=2, L=30, BH=15, FY=4, FR=5, fin=6):
    """Mot vong lap duy nhat (cung tap nhip voi exp4): tra dict gom
    Xq_f (int8 tu pipeline so thuc, lam tron nhu exp4), Xq_i (V0'), Rq_f, Rq_i, y, T, rec,
    X_f32 / R_f32 (cua so va 8 dac trung RR so thuc nhu exp4, dung cho calibrate)."""
    import wfdb
    path = os.path.join(data_dir, str(rec))
    r = wfdb.rdrecord(path, physical=False)
    ch = r.sig_name.index('MLII') if 'MLII' in r.sig_name else 0
    xr = r.d_signal[:, ch].astype(np.int64)
    sig = (xr - r.baseline[ch]) / r.adc_gain[ch]          # = p_signal
    ann = wfdb.rdann(path, 'atr')
    samples, symbols = np.asarray(ann.sample), np.asarray(ann.symbol)

    yf = clean_float(sig)[::ds]
    yi = clean_int(xr, L, BH, FY)[::ds]
    fs = FS / ds
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    is_beat = np.array([s in AAMI for s in symbols])
    Rs = samples[is_beat]; r_all = Rs / ds; sym = symbols[is_beat]
    out = {k: [] for k in ['Xq_f', 'Xq_i', 'Rq_f', 'Rq_i', 'y', 'T', 'm_int', 'X_f32', 'R_f32', 'RR_raw']}
    for i in range(1, len(r_all) - 1):
        lab = AAMI[sym[i]]
        if lab == 'Q':
            continue
        c = int(round(r_all[i]))                           # Python round: nua ve chan (giong exp4)
        if c - pre < 0 or c + post > len(yf):
            continue
        w = yf[c - pre:c + post]; w = w - np.median(w); m = np.max(np.abs(w))
        if m < 1e-9:
            continue
        xf = (w / m).astype(np.float32)
        out['X_f32'].append(xf)
        # luong tu nhu exp4: fq (np.round) roi to_i8 (np.round) -> int8 frac 6
        out['Xq_f'].append(np.clip(np.round(np.clip(np.round(xf.astype(np.float64) * 2 ** fin), -128, 127)), -128, 127))
        qi, mi = norm_window_int(yi[c - pre:c + post])
        out['Xq_i'].append(qi); out['m_int'].append(mi)
        r8 = rr_float(r_all, i, fs).astype(np.float32)
        out['R_f32'].append(r8)                            # 8 dac trung chua chuan hoa (nhu R[:, :8] cua exp4)
        rf = np.clip(r8 / RR_DIV, 0, 3.9).astype(np.float64)
        out['Rq_f'].append(np.clip(np.round(np.clip(np.round(rf * 2 ** FR), -128, 127)), -128, 127))
        out['Rq_i'].append(rr_int(Rs, i, FR))
        out['RR_raw'].append(rr_raw(Rs, i))
        out['y'].append(CLASSES.index(lab)); out['T'].append(r_all[i] / fs)
    res = {k: np.array(v) for k, v in out.items()}
    res['Xq_f'] = res['Xq_f'].astype(np.int8); res['Xq_i'] = res['Xq_i'].astype(np.int8)
    res['Rq_f'] = res['Rq_f'].astype(np.int8); res['Rq_i'] = res['Rq_i'].astype(np.int8)
    res['rec'] = np.full(len(res['y']), rec)
    return res


def lsb_stats(a, b):
    d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    return dict(frac_le1=float((d <= 1).mean()), frac_eq=float((d == 0).mean()), max=int(d.max()),
                beats_any_gt1=float((d > 1).any(1).mean()) if d.ndim == 2 else float('nan'))
