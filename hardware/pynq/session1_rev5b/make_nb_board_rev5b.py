"""Sinh notebook board kv260_f2a_rev5b_20261025.ipynb (goi y 5 cua thay, phuong an 2)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n')))
def code(s): cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s.strip('\n')))

md(r"""
# KV260 — rev5b: cấu hình chọn trên DS1 không rò rỉ + bước học ổn định (25/10/2026): GA-R, offset_int, gradient_anchor, head_scratch, head_global
Chạy trong thư mục `F2A_BOARD_RUN` trên board (đã có: `ecg_f2a.bit`, `ecg_f2a.hwh`, `xecg_f2a_top_hw.h`, `lib_ecg_f2a.py`, `lib_ecg_f2.py`, `weights_f2a.h`, `f2a_board_data.npz`).
**Chép thêm** (từ `planA/05_board/ecg_f2a_rev5b_20261025/`): `pa_adapt.py`, `rev4_methods.py` (ghi đè bản rev4 trên board — bản rev5 chỉ thêm hàm `head_global`), `rev5b_board_ref_seed3_20261025.npz`, `rev5b_board_methods_20261025.json`.

**Trước khi đo:** `sudo systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer`; đóng mọi notebook khác (hai tiến trình dùng PL làm treo board).

1. Mỗi phương pháp: hiệu chỉnh trên ARM theo từng bản ghi (5 phút đầu) → ghi lớp đầu ra → phân loại phần còn lại trên IP → so **từng nhãn** với dự đoán tham chiếu số nguyên (phải lệch 0 nhịp).
2. Năng lượng / thời gian / bộ nhớ của bước hiệu chỉnh (INA260, giống `kv260_f2a_adapt_energy.ipynb`).
Kết quả: `f2a_rev5b_board_20261025.json`, `f2a_rev5b_board_pred_20261025.npz`.
""")
code(r"""
import time, json, gc, tracemalloc, numpy as np
from pynq import Overlay, Clocks
import lib_ecg_f2a as L, pa_adapt as pa, lib_ecg_f2 as LF, rev4_methods as M
ol = Overlay('ecg_f2a.bit'); Clocks.fclk0_mhz = 100; print('fclk0:', Clocks.fclk0_mhz, 'MHz')
regs = L.parse_hw_header('xecg_f2a_top_hw.h'); ip = L.ECGF2A(ol.ecg_f2a_top_0.mmio, regs)
W_G, B_G = L.parse_global_head('weights_f2a.h'); ip.write_head(W_G, B_G)
D = np.load('f2a_board_data.npz')
beats, rrs, y, rec, t = D['beats_q'], D['rr_q'], D['y'], D['rec'], D['t']
W0 = W_G.reshape(4, 32).T / 128.0; b0 = B_G / 16384.0
t0 = time.perf_counter()
_, E_all, P_base = ip.infer_batch(beats, rrs)               # emb + nhãn gốc lấy từ IP (bit-exact với phần cứng)
_, EA, _ = ip.infer_batch(D['anc_beats_q'], D['anc_rr_q'])
print(f'Lượt gốc: {time.perf_counter() - t0:.1f} s cho {len(y)} nhịp')
A = (EA / 16.0, D['anc_y'])                                  # seed 3: emb có 4 bit lẻ
cal = pa.split_first(t, rec, 5); ev = ~cal; RECS = np.unique(rec)
REF = np.load('rev5b_board_ref_seed3_20261025.npz'); METH = json.load(open('rev5b_board_methods_20261025.json'))
CAL = {r: (E_all[(rec == r) & cal] / 16.0, y[(rec == r) & cal]) for r in RECS}
print('Nhịp hiệu chỉnh:', cal.sum(), '| đánh giá:', ev.sum(), '| phương pháp:', list(METH))
""")
md("## 1. Hiệu chỉnh trên ARM → phân loại trên IP → đối chiếu từng nhãn với tham chiếu")
code(r"""
def adapt_one(name, r):
    Ec, yc = CAL[r]; m = METH[name]
    W, b = M.adapt(m['method'], Ec, yc, W0, b0, A, fW=7, **m['hp'])
    return pa.head_to_registers(W, b)

PRED, TAD, RES = {}, {}, {}
for name in METH:
    PRED[name] = -np.ones(len(y), np.int64); TAD[name] = []
    for r in RECS:
        ta = time.perf_counter(); w_i8, b_q14 = adapt_one(name, r); TAD[name].append(time.perf_counter() - ta)
        ip.write_head(w_i8, b_q14)
        idx = np.where((rec == r) & ev)[0]
        _, _, PRED[name][idx] = ip.infer_batch(beats[idx], rrs[idx])
    ip.write_head(W_G, B_G)
    mis = int((PRED[name][ev] != REF[f'pred_{name}'][ev]).sum())
    mt = pa.metrics(y[ev], PRED[name][ev])
    RES[name] = dict(mismatch_vs_ref=mis, jk=mt['jk'], acc=mt['acc'], macroF1=mt['macroF1'], jk_ref=METH[name]['jk'],
                     t_adapt_ms_mean=float(np.mean(TAD[name]) * 1e3), t_adapt_ms_max=float(np.max(TAD[name]) * 1e3))
    print(f"{name:22s} lệch tham chiếu: {mis:3d} nhịp | jk board {mt['jk']:.4f} (tham chiếu {METH[name]['jk']:.4f}) | "
          f"hiệu chỉnh {RES[name]['t_adapt_ms_mean']:.1f} ms/bản ghi (max {RES[name]['t_adapt_ms_max']:.1f})")
np.savez_compressed('f2a_rev5b_board_pred_20261025.npz', **{f'pred_{k}': v for k, v in PRED.items()})
print('TẤT CẢ KHỚP:', all(v['mismatch_vs_ref'] == 0 for v in RES.values()))
""")
md("## 2. Năng lượng + thời gian của bước hiệu chỉnh (lặp cả 22 bản ghi tới ≥ 15 s, mẫu công suất mỗi 50 ms)")
code(r"""
pm = LF.PowerMonitor(); idle0 = pm.baseline(seconds=10)
EN = {}
for name in METH:
    gc.collect(); time.sleep(2)
    _, st = pm.measure(lambda: [adapt_one(name, r) for r in RECS], period=0.05, min_seconds=15)
    n = st['reps'] * len(RECS)
    EN[name] = dict(seconds=st['seconds'], reps=st['reps'], n_records=n, p_mean_w=st['p_mean_w'], t_per_record_ms=st['seconds'] / n * 1e3)
idle1 = pm.baseline(seconds=10); idle = (idle0 + idle1) / 2
for name, r in EN.items():
    r['p_active_w'] = r['p_mean_w'] - idle; r['energy_mJ_per_record'] = r['p_active_w'] * r['seconds'] / r['n_records'] * 1e3
    print(f"{name:22s} {r['t_per_record_ms']:9.2f} ms/bản ghi | +{r['p_active_w']:.3f} W | {r['energy_mJ_per_record']:8.3f} mJ/bản ghi")
print('idle trước/sau:', round(idle0, 3), round(idle1, 3), 'W')
""")
md("## 3. Bộ nhớ đỉnh (tracemalloc) khi hiệu chỉnh 1 bản ghi + lưu kết quả")
code(r"""
MEM = {}
for name in METH:
    peaks = []
    for r in RECS:
        tracemalloc.start(); adapt_one(name, r); peaks.append(tracemalloc.get_traced_memory()[1]); tracemalloc.stop()
    MEM[name] = dict(peak_kB_max=max(peaks) / 1024, peak_kB_mean=float(np.mean(peaks)) / 1024)
print(MEM)
json.dump(dict(fclk_mhz=Clocks.fclk0_mhz, idle_w=[idle0, idle1], bitexact=RES, energy=EN, memory=MEM, methods=METH),
          open('f2a_rev5b_board_20261025.json', 'w'), indent=1, default=float, ensure_ascii=False)
print('Đã lưu f2a_rev5b_board_20261025.json')
""")
nb = dict(cells=cells, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')), nbformat=4, nbformat_minor=5)
for c in nb['cells']:
    c['source'] = [l + '\n' for l in c['source'].split('\n')]; c['source'][-1] = c['source'][-1].rstrip('\n')
json.dump(nb, open(os.path.join(HERE, 'kv260_f2a_rev5b_20261025.ipynb'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok')
