// ============================================================
// ecg_f2a.h - Phuong an A (hieu chinh theo benh nhan): lop dau ra GHI DUOC qua AXI-Lite,
//             xuat vector dac trung dense1 (32 x int8) cho ARM. Dua tren ecg_f2.h (seed 3).
// Input: 360 mau ECG @180 Hz + 8 dac trung RR
// Conv(16,k7)-Pool -> Conv(32,k5)-Pool -> Conv(32,k5)-Pool
// -> Flatten(1440) ++ RR Dense(16) -> Dense(32) -> Dense(4) -> argmax
// ============================================================
#ifndef ECG_F2A_H
#define ECG_F2A_H
#include "ap_fixed.h"
#include "ap_int.h"

#define IN_LEN   360
#define N_RR     8
#define C1_F 16
#define C1_K 7
#define C2_F 32
#define C2_K 5
#define C3_F 32
#define C3_K 5
#define L1 IN_LEN
#define L2 (IN_LEN / 2)
#define L3 (IN_LEN / 4)
#define L4 (IN_LEN / 8)
#define N_FLAT   1440
#define RR_H     16
#define D1_IN    (N_FLAT + RR_H)
#define D1_OUT   32
#define NUM_CLASSES 4

typedef ap_fixed<8, 2> in_t;   // frac 6
typedef ap_fixed<8, 3> rr_t;   // frac 5
typedef ap_fixed<8, 1> w1_t;   // frac 7
typedef ap_fixed<8, 1> b1_t;   // frac 7
typedef ap_fixed<8, 2, AP_RND, AP_SAT> a1_t;   // frac 6
typedef ap_fixed<8, 1> w2_t;   // frac 7
typedef ap_fixed<8, 1> b2_t;   // frac 7
typedef ap_fixed<8, 3, AP_RND, AP_SAT> a2_t;   // frac 5
typedef ap_fixed<8, 1> w3_t;   // frac 7
typedef ap_fixed<8, 1> b3_t;   // frac 7
typedef ap_fixed<8, 4, AP_RND, AP_SAT> a3_t;   // frac 4
typedef ap_fixed<8, 1> wr_t;   // frac 7
typedef ap_fixed<8, 1> br_t;   // frac 7
typedef ap_fixed<8, 3, AP_RND, AP_SAT> ar_t;   // frac 5
typedef ap_fixed<8, 1> wd1_t;   // frac 7
typedef ap_fixed<8, 1> bd1_t;   // frac 7
typedef ap_fixed<8, 4, AP_RND, AP_SAT> ad1_t;   // frac 4
typedef ap_fixed<8, 1> wo_t;   // frac 7
typedef ap_fixed<48, 30>  acc_t;        // bo cong don (18 bit thap phan, du rong cho 1456 phep cong)
typedef ap_fixed<32, 18>  logit_t;      // logit, 14 bit thap phan
typedef ap_uint<2>        label_t;      // 0=N 1=S 2=V 3=F

#define N_HEAD_W (NUM_CLASSES * D1_OUT)   // 128 trong so int8, chi so o*D1_OUT + i

// head_w, head_b: do ARM ghi (mac dinh = trong so toan cuc DO_W, DO_BTOT; sau hieu chinh = trong so moi)
// emb_out     : vector dac trung dense1 (sau ReLU, 8 bit) cua nhip vua xu ly, de ARM thu thap 5 phut dau
void ecg_f2a_top(in_t ecg_in[IN_LEN], rr_t rr_in[N_RR],
                 wo_t head_w[N_HEAD_W], logit_t head_b[NUM_CLASSES],
                 logit_t logits[NUM_CLASSES], ad1_t emb_out[D1_OUT], label_t &label);
#endif
