"""
pa_adapt.py - Hieu chinh theo benh nhan (patient-adaptive) cho model ECG F2.
Chi dung NumPy -> chay duoc ca tren Colab lan tren ARM Cortex-A53 cua KV260 (PYNQ).

Y tuong: giu nguyen toan bo mang, chi hieu chinh LOP DAU RA (dense_out, 32 -> 4).
Dau vao cua cac ham la vector dac trung E (dau ra dense_1 sau ReLU, 32 chieu)
va trong so toan cuc (W0: 32x4, b0: 4, b0 da cong offset tu cross-validation).

Giao thuc (AAMI EC57): 5 phut dau moi ban ghi (co nhan) de hieu chinh,
phan con lai de danh gia. Moi sieu tham so chon tren DS1 (ham tune_on_folds).
"""
import numpy as np

CLASSES = ['N', 'S', 'V', 'F']
NC = 4

# ----------------------------------------------------------------------------
# 1. Chia doan hieu chinh / danh gia
# ----------------------------------------------------------------------------
def beat_times_from_rr(rr_prev_s, rec):
    """Uoc luong thoi diem moi nhip (giay) tu RR truoc (giay), khi khong luu vi tri dinh R.
    rr_prev_s: RR_i chua chuan hoa (cot 0 cua R). Sai so ~1 nhip do nhip dau/cuoi bi bo."""
    t = np.zeros(len(rec), dtype=np.float64)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        t[idx] = np.cumsum(rr_prev_s[idx])
    return t

def split_calibration(t_s, rec, minutes=5.0):
    """True = thuoc doan hieu chinh (t < minutes phut, tinh tu dau moi ban ghi)."""
    cal = np.zeros(len(rec), dtype=bool)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        cal[idx] = (t_s[idx] - t_s[idx].min()) < minutes * 60.0
    return cal

# ----------------------------------------------------------------------------
# 2. Chi so
# ----------------------------------------------------------------------------
def confusion(y, p, nc=NC):
    cm = np.zeros((nc, nc), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm

def metrics(y, p):
    cm = confusion(y, p)
    n = cm.sum()
    res = {'cm': cm, 'acc': np.trace(cm) / n}
    se, pp, f1 = {}, {}, []
    for c, name in enumerate(CLASSES):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp
        se[name] = tp / (tp + fn) if tp + fn else np.nan
        pp[name] = tp / (tp + fp) if tp + fp else 0.0
        if cm[c].sum():
            s, q = se[name], pp[name]
            f1.append(2 * s * q / (s + q) if s + q else 0.0)
    res['Se'], res['PP'], res['macroF1'] = se, pp, float(np.mean(f1))
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    res['kappa'] = (res['acc'] - pe) / (1 - pe)
    j = np.nansum([se['S'], se['V'], pp['S'], pp['V']])
    res['jk'] = j / 8 + res['kappa'] / 2
    return res

def fmt(m):
    return (f"acc {m['acc']:.4f} | mF1 {m['macroF1']:.4f} | k {m['kappa']:.4f} | jk {m['jk']:.4f} | "
            + ' '.join(f"{c} {m['Se'][c]:.3f}/{m['PP'][c]:.3f}" for c in CLASSES))

# ----------------------------------------------------------------------------
# 3. Luong tu hoa trong so moi theo DUNG dinh dang phan cung hien tai
# ----------------------------------------------------------------------------
def fq(x, f, bits=8):
    s = 2.0 ** f
    return np.clip(np.round(x * s), -2 ** (bits - 1), 2 ** (bits - 1) - 1) / s

def quantize_head(W, b, fW, fb_logit=14, bits_b=32):
    """W: 8-bit voi frac bits fW (giu nhu model toan cuc -> khong doi phan cung).
    b: luu 32-bit Q18.14 nhu bias da gop offset trong IP hien tai."""
    return fq(W, fW, 8), fq(b, fb_logit, bits_b)

# ----------------------------------------------------------------------------
# 4. Cac phuong phap hieu chinh (moi ham tra ve W, b moi cho 1 ban ghi)
# ----------------------------------------------------------------------------
def adapt_none(E, y, W0, b0, **kw):
    return W0.copy(), b0.copy()

def adapt_offset(E, y, W0, b0, tau=1.0, **kw):
    """Chi chinh bias: dich logit theo ti le lop cua benh nhan so voi du doan toan cuc.
    tau: he so lam tron (Dirichlet). Lop khong xuat hien -> khong bi day xuong qua manh."""
    p0 = np.bincount((E @ W0 + b0).argmax(1), minlength=NC) + tau
    pt = np.bincount(y, minlength=NC) + tau
    return W0.copy(), b0 + np.log(pt / pt.sum()) - np.log(p0 / p0.sum())

def adapt_gradient(E, y, W0, b0, lam=0.1, lr=0.5, epochs=300, cap=10.0, **kw):
    """Huan luyen lai lop dau ra bang gradient (softmax CE co trong so lop),
    phat L2 keo ve trong so toan cuc; CHI cap nhat hang cua lop co mat trong doan hieu chinh.
    Dac trung duoc chuan hoa thang do noi bo (s) de lr khong phu thuoc do lon cua E."""
    s = 1.0 / (E.std() + 1e-9)
    Es = E * s
    Wt0 = W0 / s
    Wt, b = Wt0.copy(), b0.copy()
    present = np.bincount(y, minlength=NC) > 0
    cnt = np.bincount(y, minlength=NC).astype(float)
    cw = np.where(cnt > 0, np.minimum(cnt.sum() / (NC * np.maximum(cnt, 1)), cap), 0.0)
    sw = cw[y]; sw = sw / sw.sum()
    Y = np.eye(NC)[y]
    for _ in range(epochs):
        Z = Es @ Wt + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None]
        gW = Es.T @ G + lam * (Wt - Wt0)
        gb = G.sum(0) + lam * (b - b0)
        gW[:, ~present] = 0; gb[~present] = 0
        Wt -= lr * gW; b -= lr * gb
    return Wt * s, b

def adapt_prototype(E, y, W0, b0, alpha=0.5, min_count=5, **kw):
    """Weight imprinting: tron hang trong so cua lop c voi vector trung binh dac trung
    cua lop do (da scale ve cung do lon). Chi can cong don + chia -> cung hoa duoc (phuong an C)."""
    W = W0.copy()
    for c in range(NC):
        m = y == c
        if m.sum() >= min_count:
            mu = E[m].mean(0)
            nm = np.linalg.norm(mu)
            if nm > 1e-9:
                W[:, c] = (1 - alpha) * W0[:, c] + alpha * mu * (np.linalg.norm(W0[:, c]) / nm)
    return W, b0.copy()

METHODS = {'none': adapt_none, 'offset': adapt_offset,
           'gradient': adapt_gradient, 'prototype': adapt_prototype}

def pseudo_labels(E, W0, b0, thr=0.95):
    """Bien the KHONG nhan: lay du doan toan cuc co xac suat >= thr lam nhan gia."""
    Z = E @ W0 + b0
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    keep = P.max(1) >= thr
    return keep, P.argmax(1)

# ----------------------------------------------------------------------------
# 5. Chay toan bo 1 tap (moi ban ghi hieu chinh rieng)
# ----------------------------------------------------------------------------
def run_dataset(E, y, rec, cal, W0, b0, method='none', labelled=True, fW=None,
                thr=0.95, **hp):
    """Tra ve (y_eval, pred_eval, rec_eval). Danh gia CHI tren phan ngoai doan hieu chinh,
    ke ca voi 'none', de moi phuong phap duoc so tren cung tap nhip."""
    fn = METHODS[method]
    ys, ps, rs = [], [], []
    for r in np.unique(rec):
        m = rec == r
        mc, me = m & cal, m & ~cal
        if me.sum() == 0:
            continue
        Ec, yc = E[mc], y[mc]
        if not labelled:
            keep, yhat = pseudo_labels(Ec, W0, b0, thr)
            Ec, yc = Ec[keep], yhat[keep]
        if len(yc) == 0 or method == 'none':
            W, b = W0, b0
        else:
            W, b = fn(Ec, yc, W0, b0, **hp)
        if fW is not None:
            W, b = quantize_head(W, b, fW)
        ps.append((E[me] @ W + b).argmax(1)); ys.append(y[me]); rs.append(rec[me])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(rs)

# ----------------------------------------------------------------------------
# 6. Chon sieu tham so CHI tren DS1 (dung cac model cua tung fold cross-validation)
# ----------------------------------------------------------------------------
def tune_on_folds(folds, method, grid, score='macroF1', labelled=True, **fixed):
    """folds: list dict(E, y, rec, cal, W0, b0, fW) - moi fold la cac ban ghi VALIDATION
    cua fold do, voi E tinh bang model KHONG thay cac ban ghi nay khi train.
    grid: list dict sieu tham so. Tra ve (hp_tot_nhat, bang_ket_qua)."""
    table = []
    for hp in grid:
        Y, P = [], []
        for F in folds:
            global CURRENT_ANCHORS
            CURRENT_ANCHORS = F.get('anchors')
            y_, p_, _ = run_dataset(F['E'], F['y'], F['rec'], F['cal'], F['W0'], F['b0'],
                                    method, labelled, F.get('fW'), **fixed, **hp)
            Y.append(y_); P.append(p_)
        m = metrics(np.concatenate(Y), np.concatenate(P))
        table.append((hp, m[score], m))
    best = max(table, key=lambda t: t[1])
    return best[0], table

DEFAULT_GRIDS = {
    'offset':    [dict(tau=t) for t in (1.0, 5.0, 20.0)],
    'gradient':  [dict(lam=l, lr=0.5, epochs=300) for l in (0.01, 0.1, 1.0)],
    'prototype': [dict(alpha=a, min_count=5) for a in (0.25, 0.5, 0.75)],
    # bien the khong nhan: them nguong tin cay thr cho nhan gia
    'prototype_unl': [dict(alpha=a, min_count=5, thr=t) for a in (0.25, 0.5, 0.75) for t in (0.9, 0.99)],
}


# ============================================================================
# 7. Bo sung (thi nghiem 2): offset mot chieu, offset so nguyen, danh gia tach rieng
# ============================================================================
def _log_ratio(E, y, W0, b0, tau):
    p0 = np.bincount((E @ W0 + b0).argmax(1), minlength=NC) + tau
    pt = np.bincount(y, minlength=NC) + tau
    return np.log(pt / pt.sum()) - np.log(p0 / p0.sum())

def adapt_offset_1s(E, y, W0, b0, tau=5.0, **kw):
    """Offset MOT CHIEU: chi TANG logit cua lop CO MAT trong doan hieu chinh,
    khong bao gio day lop vang mat (hoac lop bi du doan qua nhieu) xuong."""
    present = np.bincount(y, minlength=NC) > 0
    d = np.maximum(_log_ratio(E, y, W0, b0, tau), 0.0)
    d[~present] = 0.0
    return W0.copy(), b0 + d

# --- Phien ban so nguyen cho phan cung (phuong an C) ---
LN2_Q14 = 11357            # round(ln2 * 2^14)
def mitchell_log2_q(n, fbits=8):
    """log2(n) xap xi Mitchell, dang Q.fbits: vi tri bit 1 cao nhat + phan du dich trai.
    Phan cung: bo do bit dan dau + bo dich; khong can bang tra hay phep chia."""
    n = int(n)
    k = n.bit_length() - 1
    frac = ((n - (1 << k)) << fbits) >> k if k > 0 else 0
    return (k << fbits) + frac

def offset_int_q14(n_true, n_pred, tau=5, fbits=8, one_sided=True):
    """Offset Q18.14 cho 1 lop tu hai bo dem so nguyen (n_true: nhan 5 phut dau, n_pred: du doan
    cua model goc). ln(a/b) = (log2 a - log2 b)*ln2. Chi dung cong, dich, 1 phep nhan hang so."""
    d = mitchell_log2_q(n_true + tau, fbits) - mitchell_log2_q(n_pred + tau, fbits)
    if one_sided:
        d = max(d, 0)
    return (d * LN2_Q14) >> fbits          # Q.14

def adapt_offset_1s_int(E, y, W0, b0, tau=5, fbits=8, **kw):
    """Offset mot chieu tinh HOAN TOAN bang so nguyen (dem nhip, Mitchell log2, dich bit).
    Tong so nhip cua hai phan bo bang nhau nen khong can chuan hoa."""
    n_true = np.bincount(y, minlength=NC)
    n_pred = np.bincount((E @ W0 + b0).argmax(1), minlength=NC)
    d = np.array([offset_int_q14(n_true[c], n_pred[c], int(tau), fbits) if n_true[c] > 0 else 0
                  for c in range(NC)], dtype=np.float64) / 2 ** 14
    return W0.copy(), b0 + d

METHODS.update({'offset_1s': adapt_offset_1s, 'offset_1s_int': adapt_offset_1s_int})

def run_dataset_split(E, y, rec, cal, ev, W0, b0, method='none', labelled=True, fW=None,
                      thr=0.95, **hp):
    """Nhu run_dataset nhung tach mat na hieu chinh (cal) va danh gia (ev):
    cho phep thu ngan sach 1/2/5 phut ma VAN danh gia tren cung tap nhip (sau phut thu 5)."""
    fn = METHODS[method]
    ys, ps, rs = [], [], []
    for r in np.unique(rec):
        m = rec == r
        mc, me = m & cal, m & ev
        if me.sum() == 0:
            continue
        Ec, yc = E[mc], y[mc]
        if not labelled:
            keep, yhat = pseudo_labels(Ec, W0, b0, thr)
            Ec, yc = Ec[keep], yhat[keep]
        if len(yc) == 0 or method == 'none':
            W, b = W0, b0
        else:
            W, b = fn(Ec, yc, W0, b0, **hp)
        if fW is not None:
            W, b = quantize_head(W, b, fW)
        ps.append((E[me] @ W + b).argmax(1)); ys.append(y[me]); rs.append(rec[me])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(rs)

def split_first(t_s, rec, minutes):
    """True = thuoc 'minutes' phut dau cua moi ban ghi."""
    return split_calibration(t_s, rec, minutes)

def per_record_scores(y, p, rec, min_beats=10):
    """Chi so theo ban ghi (AAMI 'average statistics'): accuracy, balanced accuracy tren cac lop
    co mat (>= min_beats nhip), va Se cua tung lop (NaN neu lop co < min_beats nhip)."""
    out = {}
    for r in np.unique(rec):
        m = rec == r
        yy, pp = y[m], p[m]
        se = {}
        for c, name in enumerate(CLASSES):
            k = yy == c
            se[name] = (pp[k] == c).mean() if k.sum() >= min_beats else np.nan
        vals = [v for v in se.values() if not np.isnan(v)]
        out[r] = dict(acc=(yy == pp).mean(), balacc=np.mean(vals) if vals else np.nan, **{f'Se_{c}': se[c] for c in CLASSES})
    return out

DEFAULT_GRIDS.update({
    'offset_1s':     [dict(tau=t) for t in (1.0, 5.0, 20.0)],
    'offset_1s_int': [dict(tau=t, fbits=8) for t in (1, 5, 20)],
})


# ============================================================================
# 8. Thi nghiem 3: offset hai chieu so nguyen, gradient co neo, offset + gradient
# ============================================================================
CURRENT_ANCHORS = None      # (A, ya): vector dac trung lay mau tu DS1 cua CHINH model dang dung

def adapt_offset_int(E, y, W0, b0, tau=5, fbits=8, **kw):
    """Offset HAI CHIEU tinh bang so nguyen (dem nhip + log2 Mitchell + dich bit), ap cho ca 4 lop.
    Tuong duong adapt_offset (so thuc) vi tong so nhip cua hai phan bo bang nhau."""
    n_true = np.bincount(y, minlength=NC)
    n_pred = np.bincount((E @ W0 + b0).argmax(1), minlength=NC)
    d = np.array([offset_int_q14(n_true[c], n_pred[c], int(tau), fbits, one_sided=False)
                  for c in range(NC)], dtype=np.float64) / 2 ** 14
    return W0.copy(), b0 + d

def adapt_gradient_anchor(E, y, W0, b0, beta=0.3, lam=0.01, lr=0.5, epochs=300, cap=10.0, **kw):
    """Gradient co NEO (rehearsal): tron them mau dac trung cua ca 4 lop lay tu DS1 (CURRENT_ANCHORS),
    chiem ti trong beta trong ham mat mat -> lop vang mat trong 5 phut dau khong bi lan mat vung quyet dinh.
    Moi hang trong so deu duoc cap nhat (ca 4 lop deu co mat nho neo)."""
    A, ya = CURRENT_ANCHORS
    Eall = np.vstack([E, A]); yall = np.concatenate([y, ya])
    cnt = np.bincount(y, minlength=NC).astype(float)
    cw = np.where(cnt > 0, np.minimum(cnt.sum() / (NC * np.maximum(cnt, 1)), cap), 0.0)
    w_cal = cw[y]; w_cal = (1 - beta) * w_cal / w_cal.sum()
    ca = np.bincount(ya, minlength=NC).astype(float)
    w_anc = beta * (1.0 / (NC * ca[ya]))
    sw = np.concatenate([w_cal, w_anc])
    s = 1.0 / (Eall.std() + 1e-9)
    Es = Eall * s; Wt0 = W0 / s
    Wt, b = Wt0.copy(), b0.copy()
    Y = np.eye(NC)[yall]
    for _ in range(epochs):
        Z = Es @ Wt + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None]
        Wt -= lr * (Es.T @ G + lam * (Wt - Wt0)); b -= lr * (G.sum(0) + lam * (b - b0))
    return Wt * s, b

def adapt_offset_gradient(E, y, W0, b0, tau=5.0, lam=1.0, lr=0.5, epochs=300, **kw):
    """Offset hai chieu truoc, roi gradient (L2 keo ve trong so da dich offset)."""
    W1, b1 = adapt_offset(E, y, W0, b0, tau=tau)
    return adapt_gradient(E, y, W1, b1, lam=lam, lr=lr, epochs=epochs)

METHODS.update({'offset_int': adapt_offset_int, 'gradient_anchor': adapt_gradient_anchor,
                'offset_gradient': adapt_offset_gradient})
DEFAULT_GRIDS.update({
    'offset_int':      [dict(tau=t, fbits=8) for t in (1, 5, 20)],
    'gradient_anchor': [dict(beta=b, lam=0.01, lr=0.5, epochs=300) for b in (0.1, 0.3, 0.5)],
    'offset_gradient': [dict(tau=5.0, lam=l, lr=0.5, epochs=300) for l in (0.1, 1.0, 10.0)],
})

def sample_anchor_idx(y, mask, m=25, seed=0):
    """Chon toi da m chi so moi lop trong mask (bo dem rehearsal ~ 4*m vector 32 chieu)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(NC):
        cand = np.where(mask & (y == c))[0]
        if len(cand):
            idx.append(rng.choice(cand, min(m, len(cand)), replace=False))
    return np.concatenate(idx)


# ============================================================================
# 9. Giao tiep IP phuong an A (ecg_f2a_top): dinh dang thanh ghi + tham chieu so nguyen
# ============================================================================
def head_to_registers(W, b, fW=7, fb=14):
    """W (32x4 so thuc), b (4) -> (w_i8[128] theo chi so o*32+i, b_q14[4] int32) dung dinh dang IP."""
    w = np.clip(np.round(np.asarray(W).T * 2 ** fW), -128, 127).astype(np.int8).reshape(-1)
    bq = np.clip(np.round(np.asarray(b) * 2 ** fb), -2 ** 31, 2 ** 31 - 1).astype(np.int64)
    return w, bq

def pack_head_words(w_i8):
    """128 int8 -> 32 tu 32 bit (4 phan tu/tu, phan tu dau o byte thap) de ghi vao vung head_w AXI-Lite."""
    return np.frombuffer(np.asarray(w_i8, dtype=np.int8).tobytes(), dtype='<u4').copy()

def logits_int(emb_q4, w_i8, b_q14):
    """Tham chieu SO NGUYEN dung nhu IP: logit_q14 = b_q14 + sum(emb_q4 * w_q7) * 2^(14-4-7)."""
    e = np.asarray(emb_q4, dtype=np.int64).reshape(-1, 32)
    w = np.asarray(w_i8, dtype=np.int64).reshape(4, 32)
    return np.asarray(b_q14, dtype=np.int64)[None, :] + (e @ w.T) * 8
