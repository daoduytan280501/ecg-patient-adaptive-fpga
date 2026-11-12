"""
rev4_methods.py - Phuong phap hieu chinh lop dau ra cho rev4 (22/10/2026). Moi ham: (E, y, W0, b0, A, **hp) -> (W, b).
A = (emb neo, nhan neo) hoac None.
"""
import numpy as np
import pa_adapt as pa

NC = 4
W_LO, W_HI = -128 / 128, 127 / 128           # mien int8 voi 7 bit le (dinh dang thanh ghi head_w cua IP)


def head_scratch(E, y, W0, b0, A=None, lr=0.5, epochs=150, cap=10.0, **kw):
    """Nhu exp5/src/head_scratch.py (bai A)."""
    cnt = np.bincount(y, minlength=NC).astype(float)
    cw = np.where(cnt > 0, np.minimum(cnt.sum() / (NC * np.maximum(cnt, 1)), cap), 0.0)
    sw = cw[y] / cw[y].sum()
    s = 1.0 / (E.std() + 1e-9); Es = E * s
    W = np.zeros_like(W0); b = np.zeros_like(b0)
    b[cnt == 0] = -1e3
    Y = np.eye(NC)[y]
    for _ in range(epochs):
        Z = Es @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None]
        W -= lr * Es.T @ G; b -= lr * G.sum(0) * (cnt > 0)
    return W * s, b


def ga(E, y, W0, b0, A, beta=0.1, lam=0.01, lr=0.5, epochs=300, cap=10.0, proj=False, trust=None,
       return_info=False, init='global', **kw):
    """gradient_anchor cua bai A (proj=False, trust=None, cap=10 -> trung pa.adapt_gradient_anchor),
    them: proj = chieu W ve mien int8 sau moi buoc; trust = ban kinh vung tin cay tuong doi ||W-W0|| <= trust*||W0||."""
    Aa, ya = A
    Eall = np.vstack([E, Aa]); yall = np.concatenate([y, ya])
    cnt = np.bincount(y, minlength=NC).astype(float)
    cw = np.where(cnt > 0, np.minimum(cnt.sum() / (NC * np.maximum(cnt, 1)), cap), 0.0)
    w_cal = cw[y]; w_cal = (1 - beta) * w_cal / w_cal.sum()
    ca = np.bincount(ya, minlength=NC).astype(float)
    sw = np.concatenate([w_cal, beta * (1.0 / (NC * ca[ya]))])
    s = 1.0 / (Eall.std() + 1e-9)
    Es = Eall * s; Wt0 = W0 / s
    Wt, b = (Wt0.copy(), b0.copy()) if init == 'global' else (np.zeros_like(Wt0), np.zeros_like(b0))
    Y = np.eye(NC)[yall]
    r_lim = None if trust is None else trust * np.linalg.norm(Wt0)
    for _ in range(epochs):
        Z = Es @ Wt + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None]
        Wt -= lr * (Es.T @ G + lam * (Wt - Wt0)); b -= lr * (G.sum(0) + lam * (b - b0))
        if r_lim is not None:
            D = Wt - Wt0; n = np.linalg.norm(D)
            if n > r_lim:
                Wt = Wt0 + D * (r_lim / n)
        if proj:
            Wt = np.clip(Wt * s, W_LO, W_HI) / s
    W = Wt * s
    if return_info:
        return W, b, dict(n_sat=int((np.abs(np.round(W * 128)) > 127).sum() if not proj else 0),
                          n_sat_raw=int(((W * 128 > 127.5) | (W * 128 < -128.5)).sum()),
                          dW=float(np.linalg.norm(W - W0) / (np.linalg.norm(W0) + 1e-12)))
    return W, b


def offset_int(E, y, W0, b0, A=None, tau=5, fbits=8, **kw):
    return pa.adapt_offset_int(E, y, W0, b0, tau=tau, fbits=fbits)


def rescale_int8(W, b):
    """GA-S: argmax(EW+b) bat bien voi (aW, ab), a>0 -> co gian de max|W| vua mien int8, khong doi quyet dinh."""
    m = np.abs(W).max()
    if m > W_HI:
        a = W_HI / m
        return W * a, b * a
    return W, b


def ga_s(*a, trust=None, **k):
    W, b = ga(*a, proj=False, trust=trust, **k)
    return rescale_int8(W, b)


def head_global(E, y, W0, b0, A=None, lr=0.5, epochs=150, cap=10.0, **kw):
    """head_scratch giu lop dau ra toan cuc (gop y thay 25/10, diem 3): giong head_scratch (chi nhip hieu chinh,
    cung chuan hoa, cung trong so lop cap 10, khong neo, khong lambda) nhung KHOI TAO tu (W0, b0) va
    KHONG co dinh bias lop vang mat o -1e3 (lop vang mat giu trong so toan cuc, chi bi gradient day xuong)."""
    cnt = np.bincount(y, minlength=NC).astype(float)
    cw = np.where(cnt > 0, np.minimum(cnt.sum() / (NC * np.maximum(cnt, 1)), cap), 0.0)
    sw = cw[y] / cw[y].sum()
    s = 1.0 / (E.std() + 1e-9); Es = E * s
    W = W0 / s; b = b0.copy()
    Y = np.eye(NC)[y]
    for _ in range(epochs):
        Z = Es @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None]
        W -= lr * Es.T @ G; b -= lr * G.sum(0)
    return rescale_int8(W * s, b)


BASE = {'offset_int': offset_int, 'GA-S': ga_s, 'gradient_anchor': lambda *a, **k: ga(*a, **k),
        'GA-P': lambda *a, **k: ga(*a, proj=True, **k), 'GA-R': lambda *a, **k: ga(*a, proj=True, **k),
        'head_scratch': head_scratch, 'head_global': head_global}


def macro_recall(y, p):
    cs = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in cs])) if len(cs) else 0.0


def gated(base, E, y, W0, b0, A, delta=0.02, frac=0.8, fW=7, **hp):
    """Cong tu kiem tra: hieu chinh tren 80 % dau, kiem tren 20 % sau (nhan cua doan hieu chinh), so voi lop toan cuc."""
    n = len(y); na = int(round(frac * n))
    if na < 10 or n - na < 5:
        return BASE[base](E, y, W0, b0, A, **hp)
    W1, b1 = BASE[base](E[:na], y[:na], W0, b0, A, **hp)
    W1, b1 = pa.quantize_head(W1, b1, fW)
    Ec, yc = E[na:], y[na:]
    s_ad = macro_recall(yc, (Ec @ W1 + b1).argmax(1))
    s_gl = macro_recall(yc, (Ec @ W0 + b0).argmax(1))
    if s_ad < s_gl - delta:
        return W0.copy(), b0.copy()
    return BASE[base](E, y, W0, b0, A, **hp)


def adapt(method, E, y, W0, b0, A, fW=7, **hp):
    """method: ten trong BASE hoac 'gate:<base>'."""
    if method == 'none':
        return W0.copy(), b0.copy()
    if method.startswith('gate:'):
        return gated(method[5:], E, y, W0, b0, A, fW=fW, **hp)
    return BASE[method](E, y, W0, b0, A, **hp)


def run(E, y, rec, cal, ev, W0, b0, A, method, fW=7, y_cal=None, quant=True, info=None, **hp):
    """Moi ban ghi hieu chinh rieng. y_cal: nhan (co the nhieu) dung cho doan hieu chinh; danh gia bang y that."""
    yc_all = y if y_cal is None else y_cal
    ys, ps, rs = [], [], []
    for r in np.unique(rec):
        m = rec == r
        mc, me = m & cal, m & ev
        if me.sum() == 0:
            continue
        if mc.sum() == 0 or method == 'none':
            W, b = W0, b0
        else:
            W, b = adapt(method, E[mc], yc_all[mc], W0, b0, A, fW=fW, **hp)
        if info is not None:
            info[r] = dict(n_sat=int(((W * 2 ** fW > 127.5) | (W * 2 ** fW < -128.5)).sum()),
                           dW=float(np.linalg.norm(W - W0) / (np.linalg.norm(W0) + 1e-12)),
                           reverted=bool(np.array_equal(W, W0) and np.array_equal(b, b0)))
        if quant:
            W, b = pa.quantize_head(W, b, fW)
        ps.append((E[me] @ W + b).argmax(1)); ys.append(y[me]); rs.append(rec[me])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(rs)


# ---------------------------- nhieu nhan (giong exp5) -------------------------
PAIR = np.array([1, 0, 3, 2])


def noisy_labels(y, cal, seed, kind, p, rep):
    yn = y.copy()
    if kind == 'clean' or p == 0:
        return yn
    if kind == 'uniform':
        rng = np.random.default_rng(1000 * seed + rep); idx = np.where(cal)[0]; k = int(round(p * len(idx)))
        fl = rng.choice(idx, k, replace=False); yn[fl] = (yn[fl] + rng.integers(1, 4, k)) % 4
    elif kind == 'pair':
        rng = np.random.default_rng(7000 * seed + 100 * rep + int(p * 100)); idx = np.where(cal)[0]
        k = int(round(p * len(idx))); fl = rng.choice(idx, k, replace=False); yn[fl] = PAIR[yn[fl]]
    elif kind == 'miss':
        rng = np.random.default_rng(7000 * seed + 100 * rep + int(p * 100)); idx = np.where(cal & (y != 0))[0]
        k = int(round(p * len(idx))); fl = rng.choice(idx, k, replace=False); yn[fl] = 0
    return yn


def absent_mask(y, rec, cal, ev):
    """Mat na (theo dung thu tu dau ra cua run) cho nhip danh gia thuoc lop KHONG co trong doan hieu chinh cua ban ghi."""
    out = []
    for r in np.unique(rec):
        m = rec == r
        me = m & ev
        if me.sum() == 0:
            continue
        present = np.bincount(y[m & cal], minlength=NC) > 0
        out.append(~present[y[me]])
    return np.concatenate(out)
