"""hw_ref.py - Tham chieu SO NGUYEN bit-exact voi IP ecg_f2(a)_top (Vitis HLS).
Khac voi fake-quant cu (np.round = lam tron chan): AP_RND cua HLS lam tron NUA LEN (floor(x+0.5)),
AP_SAT bao hoa. Moi phep tinh deu tren so nguyen int64 -> khop tung bit voi C-sim/co-sim/board.
Dau vao: beats_q (N,360) int8 frac 6; rr_q (N,8) int8 frac 5. Trong so: so nguyen da luong tu (Wq) + so bit thap phan.
"""
import numpy as np

def _act(acc, f_acc, f_out, bits=8):
    """acc: so nguyen voi f_acc bit thap phan (>0 sau ReLU). ReLU -> AP_RND (nua len) -> AP_SAT."""
    acc = np.maximum(acc, 0)
    sh = f_acc - f_out
    q = (acc + (1 << (sh - 1))) >> sh if sh > 0 else acc << (-sh)
    return np.minimum(q, (1 << (bits - 1)) - 1)

def _conv(x, W, b, fx, fw, fb):
    """x (n,L,C) int; W (k,C,F) int; b (F,) int. Tra (acc, f_acc). 'same' padding nhu HLS (pad trai (k-1)//2)."""
    k = W.shape[0]; pl = (k - 1) // 2; pr = k - 1 - pl
    xp = np.pad(x, ((0, 0), (pl, pr), (0, 0)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=1)        # (n,L,C,k)
    f = fx + fw
    acc = np.einsum('nlck,kcf->nlf', win, W, optimize=True) + (b.astype(np.int64) << (f - fb))
    return acc, f

def _pool(x):
    L = x.shape[1] // 2
    return np.maximum(x[:, 0:2 * L:2], x[:, 1:2 * L:2])

def quantize_params(Wd, qcfg):
    """Wd: dict ten lop -> (W, b) so thuc (Keras). Tra trong so so nguyen giong het weights.h
    (weights.h duoc sinh bang fq = np.round, nen o day cung dung np.round cho TRONG SO)."""
    P = {}
    for name in ['conv1d_1', 'conv1d_2', 'conv1d_3', 'rr_dense', 'dense_1', 'dense_out']:
        W, b = Wd[name]
        fw, fb = qcfg[f'{name}.kernel'], qcfg[f'{name}.bias']
        P[name] = (np.clip(np.round(W * 2 ** fw), -128, 127).astype(np.int64), fw,
                   np.clip(np.round(b * 2 ** fb), -128, 127).astype(np.int64), fb)
    return P

def forward_emb_int(beats_q, rr_q, P, qcfg, batch=1024):
    """-> emb (N,32) int, so bit thap phan = qcfg['dense_1'] (=4). Khop bit voi emb_out cua IP."""
    fin, frr = qcfg['input'], qcfg['rr']
    outs = []
    for s in range(0, len(beats_q), batch):
        x = beats_q[s:s + batch, :, None].astype(np.int64); fx = fin
        for i in (1, 2, 3):
            W, fw, b, fb = P[f'conv1d_{i}']
            acc, fa = _conv(x, W, b, fx, fw, fb)
            fo = qcfg[f'relu_{i}']
            x = _pool(_act(acc, fa, fo)); fx = fo
        x = x.reshape(len(x), -1)                       # thu tu (p, c) giong D1_W[o][p*32+c]
        W, fw, b, fb = P['rr_dense']
        r = rr_q[s:s + batch].astype(np.int64)
        fa = frr + fw
        r = _act(r @ W + (b << (fa - fb)), fa, qcfg['rr_dense'])
        W, fw, b, fb = P['dense_1']
        # hai phan dau vao co so bit thap phan khac nhau (pool3 vs rr_dense) -> dua ve cung thang
        f1, f2 = qcfg['relu_3'], qcfg['rr_dense']
        fa = max(f1, f2) + fw
        xx = np.concatenate([x << (max(f1, f2) - f1), r << (max(f1, f2) - f2)], axis=1)
        acc = xx @ W + (b << (fa - fb))
        outs.append(_act(acc, fa, qcfg['dense_1']))
    return np.concatenate(outs)

def head_global_int(P, off, fb_logit=14):
    """Lop dau ra toan cuc nhu IP: w int8 (frac 7), bias Q18.14 = bias 8-bit + offset CV (np.round o Q14)."""
    W, fw, b, fb = P['dense_out']
    bq = np.round((b / 2 ** fb + np.asarray(off)) * 2 ** fb_logit).astype(np.int64)
    return W.T.astype(np.int64), fw, bq                   # W.T: (4,32)

def logits_int(emb, w_oi, fw, b_q14, f_emb=4, fb_logit=14):
    """logit Q18.14 = b + sum(emb * w) << (14 - f_emb - fw)."""
    return b_q14[None, :] + (emb.astype(np.int64) @ w_oi.T) * (1 << (fb_logit - f_emb - fw))
