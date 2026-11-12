KV260 - PHUONG AN A (ecg_f2a_top): bo file cho board
======================================================
A. Chuan bi tren may tinh / Colab
  1. Vivado: lay design_1_wrapper.bit -> doi ten ecg_f2a.bit ; design_1.hwh -> ecg_f2a.hwh
  2. Vitis HLS: chep file thanh ghi do HLS sinh ra:
       ECG_F2A_HLS/ECG_F2A/solution1/impl/misc/drivers/ecg_f2a_top_v1_0/src/xecg_f2a_top_hw.h
     (driver doc offset head_w/head_b/emb_out... TU file nay, khong ghi cung)
  3. Colab: dan cell_export_board_f2a.py vao cuoi notebook ecg_pa_exp3 (sau muc 7), chay
     -> pa_phaseA/f2a_board_data.npz (toan bo DS2 luong tu int8 + thoi diem nhip + 100 nhip neo DS1)
     Dong "Khop vector C-sim: beats True | rr True" xac nhan luong tu hoa giong vector C-sim.

B. Chep vao CUNG mot thu muc tren board (Jupyter PYNQ)
  ecg_f2a.bit  ecg_f2a.hwh  xecg_f2a_top_hw.h  f2a_board_data.npz
  va toan bo file trong goi nay: lib_ecg_f2a.py  pa_adapt.py  weights_f2a.h  kv260_ecg_f2a.ipynb
  test_beats_q.txt  test_rr_q.txt  test_logits_sim.txt  test_emb_sim.txt  test_head_adapted.txt  test_logits_adapted.txt

C. Chay kv260_ecg_f2a.ipynb - ket qua mong doi
  - "Doc lai head_w/head_b khop: True"
  - [T1] logit khop C-sim: True | emb khop C-sim: True     (muc 5 cua luong kiem chung)
  - [T3] logit (trong so hieu chinh) khop C-sim: True
  - Latency ~0.96 ms/nhip (xap xi ECG_F2)
  - DS2: none / offset_int / gradient_anchor tren phan sau phut 5 (seed 3), + thoi gian hieu chinh ARM/ban ghi
  Gui lai: output cac cell va f2a_board_results.npz

Ghi chu: sim_mmio.py gia lap thanh ghi IP de kiem thu driver khong can board (da dung de kiem thu goi nay).
