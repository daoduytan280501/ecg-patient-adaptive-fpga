"""Tao du lieu kiem thu T3 cho IP ecg_f2a_top theo DUNG luong ARM se chay:
   emb_out (tu IP) -> hieu chinh gradient co neo (pa_adapt) -> luong tu hoa -> thanh ghi -> logit so nguyen.
   Chay: python make_t3_golden.py  (can tb_emb_out.txt do C-sim T1 xuat, weights DO_W/DO_BTOT tu f2_hw_config/weights)"""
import numpy as np, re, pa_adapt as pa
E = np.loadtxt('tb_emb_out.txt', dtype=np.int64)           # don vi 2^-4
y = np.loadtxt('test_labels_true.txt', dtype=int)[:len(E)]
txt = open('weights_f2a.h').read()
def arr(name):
    body = re.search(name + r'\[[^=]*=\s*\{(.*?)\};', txt, re.S).group(1)
    return np.array([float(v) for v in re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', body)])
W0 = arr('DO_W').reshape(4, 32).T                           # (32 x 4)
b0 = arr('DO_BTOT')
Ef = E / 16.0
cal = np.arange(len(E)) < len(E) // 2                         # nua dau lam "5 phut dau"
ai = pa.sample_anchor_idx(y, ~cal, m=10); pa.CURRENT_ANCHORS = (Ef[ai], y[ai])
W, b = pa.adapt_gradient_anchor(Ef[cal], y[cal], W0, b0, beta=0.1, lam=0.01, lr=0.5, epochs=300)
w_i8, b_q14 = pa.head_to_registers(W, b)
lg = pa.logits_int(E, w_i8, b_q14)
np.savetxt('test_head_adapted.txt', np.concatenate([w_i8.astype(np.int64), b_q14])[None, :], fmt='%d')
np.savetxt('test_logits_adapted.txt', lg, fmt='%d')
ok0 = ((E / 16) @ W0 + b0).argmax(1); ok1 = lg.argmax(1)
print('W thay doi:', int((w_i8 != pa.head_to_registers(W0, b0)[0]).sum()), '/128 | acc truoc/sau (nua sau):',
      round((ok0[~cal] == y[~cal]).mean(), 4), round((ok1[~cal] == y[~cal]).mean(), 4))
print('Tu 32 bit dau tien cua head_w:', [hex(v) for v in pa.pack_head_words(w_i8)[:3]])
