"""fair_methods.py (26/10/2026) - baseline head_scratch "cong bang" theo gop y cua thay (diem 1).
head_scratch_fair co MOI cong cu cua GA-R tru vector neo va khoi tao tu lop toan cuc:
  - trong so lop co cap (cap), chuan hoa s = 1/std(E), suy giam trong so lambda (ve 0, vi khoi tao 0),
  - cung luoi buoc hoc eta va so vong T nhu GA-R, cung tieu chi on dinh,
  - anh xa bat bien thang alpha vao thanh ghi int8 (GA-S), khong cat bao hoa,
  - hai cach xu ly lop vang mat (khong co nhip nao trong doan hieu chinh):
      absent='fix'    : bias lop vang mat co dinh -1e3 (nhu head_scratch goc),
      absent='global' : cot trong so va bias cua lop vang mat giu nguyen nhu lop toan cuc (W0, b0), khong hoc.
Dang ky vao rev4_methods.BASE khi import."""
import numpy as np
import rev4_methods as M

NC = 4


def head_scratch_fair(E, y, W0, b0, A=None, lr=0.05, epochs=300, cap=1.0, lam=0.0, absent='fix', **kw):
    cnt = np.bincount(y, minlength=NC).astype(float)
    pres = cnt > 0
    cw = np.where(pres, np.minimum(cnt.sum() / (NC * np.maximum(cnt, 1)), cap), 0.0)
    sw = cw[y] / cw[y].sum()
    s = 1.0 / (E.std() + 1e-9); Es = E * s
    W = np.zeros_like(W0, dtype=float); b = np.zeros_like(b0, dtype=float)
    if absent == 'fix':
        b[~pres] = -1e3
        mW = np.ones(NC); mb = pres.astype(float)            # W cua moi lop van hoc (nhu goc), bias lop vang mat co dinh
    elif absent == 'global':
        W[:, ~pres] = W0[:, ~pres] / s; b[~pres] = b0[~pres]
        mW = pres.astype(float); mb = pres.astype(float)     # lop vang mat: giu nguyen lop toan cuc
    else:
        raise ValueError(absent)
    Y = np.eye(NC)[y]
    for _ in range(epochs):
        Z = Es @ W + b; Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) * sw[:, None]
        W -= lr * (Es.T @ G + lam * W) * mW[None, :]
        b -= lr * (G.sum(0) + lam * b) * mb
    return M.rescale_int8(W * s, b)


M.BASE['head_scratch_fair'] = head_scratch_fair
