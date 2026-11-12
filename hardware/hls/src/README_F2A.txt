ECG_F2A_HLS - IP phuong an A (hieu chinh theo benh nhan), dua tren ECG_F2 seed 3
==================================================================================
Thay doi so voi ECG_F2 (moi lop Conv/Pool/RR/Dense1 GIU NGUYEN, cung weights seed 3):
  + head_w[128] (int8, frac 7)  : trong so lop dau ra, ARM GHI qua AXI-Lite (chi so o*32 + i)
  + head_b[4]   (Q18.14, 32 bit): bias lop dau ra (da gop offset), ARM GHI
  + emb_out[32] (int8, frac 4)  : vector dac trung dense1 cua nhip vua xu ly, ARM DOC
  Top function: ecg_f2a_top (ten moi, tranh nham voi ecg_f2_top)

VITIS HLS (project moi, vi du ECG_F2A)
  Source    : ecg_f2a.cpp, ecg_f2a.h, weights_f2a.h
  Test Bench: ecg_f2a_tb.cpp, test_beats_q.txt, test_rr_q.txt, test_labels_true.txt,
              test_labels_sim.txt, test_logits_sim.txt, test_head_adapted.txt, test_logits_adapted.txt
  Top       : ecg_f2a_top | Part xck26-sfvc784-2LV-c | Clock 10 ns
  C Sim (de trong tham so) - ket qua mong doi (da chay thu bang g++ + thu vien ap_fixed cua Xilinx):
    [T1] Hoi quy (trong so toan cuc): logit lech 0 | nhan lech 0 | accuracy 0.5440  -> PASS
    [T2] Ghi lop dau ra moi nhip: logit/nhan sai 0 | emb thay doi 0 | (so nhip doi nhan ...: 770)  -> PASS
    [T3] Trong so hieu chinh tu Python: logit lech 0  -> PASS
    PASS: tat ca kiem thu dat
  Co-sim: tham so 10
  Ghi lai: latency (du kien chi tang vai chuc chu ky so voi 82 506 cua ECG_F2), tai nguyen tung khoi, dung luong control_s_axi

Y nghia 3 kiem thu
  T1: nap trong so goc -> phai trung bit-exact voi ECG_F2 (khong lam hong gi)
  T2: moi nhip mot bo trong so khac (ngau nhien, bien +-1.0, bias lon) -> kiem tra ghi/doc thanh ghi, dau so,
      va emb_out khong phu thuoc lop dau ra
  T3: dung dung luong ARM se chay: emb_out cua IP -> gradient co neo (pa_adapt.py) -> luong tu hoa ->
      thanh ghi -> so sanh logit IP voi tham chieu so nguyen Python (pa_adapt.logits_int)
      (make_t3_golden.py tao lai test_head_adapted.txt / test_logits_adapted.txt tu tb_emb_out.txt do C-sim ghi ra)
      Luu y: do chinh xac "truoc/sau" in ra boi make_t3_golden.py chi de kiem tra chuc nang
      (tap test can bang 250 nhip/lop), KHONG phai ket qua bao cao.

Dinh dang thanh ghi cho driver ARM (pa_adapt.py, muc 9)
  head_to_registers(W, b) -> w_i8[128], b_q14[4]
  pack_head_words(w_i8)   -> 32 tu 32 bit (phan tu dau o byte thap), ghi vao vung head_w
  logits_int(emb_q4, w_i8, b_q14) -> logit so nguyen dung nhu IP
