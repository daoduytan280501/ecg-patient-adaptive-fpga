import numpy as np, pa_adapt as pa
def adapt_head_scratch(E, y, W0, b0, lr=0.5, epochs=300, cap=10.0, **kw):
    cnt = np.bincount(y, minlength=pa.NC).astype(float)
    cw = np.where(cnt > 0, np.minimum(cnt.sum() / (pa.NC * np.maximum(cnt, 1)), cap), 0.0)
    sw = cw[y] / cw[y].sum()
    s = 1.0 / (E.std() + 1e-9); Es = E * s
    W = np.zeros_like(W0); b = np.zeros_like(b0)
    # lớp vắng mặt trong 5 phút đầu: không thể học -> logit rất âm (đúng bản chất "học từ đầu")
    b[cnt == 0] = -1e3
    Y = np.eye(pa.NC)[y]
    for _ in range(epochs):
        Z = Es @ W + b; Z -= Z.max(1, keepdims=True)
        Pp = np.exp(Z); Pp /= Pp.sum(1, keepdims=True)
        G = (Pp - Y) * sw[:, None]
        W -= lr * Es.T @ G; b -= lr * G.sum(0) * (cnt > 0)
    return W * s, b
pa.METHODS['head_scratch'] = adapt_head_scratch
pa.DEFAULT_GRIDS['head_scratch'] = [dict(lr=0.5, epochs=e) for e in (50, 150, 300)]