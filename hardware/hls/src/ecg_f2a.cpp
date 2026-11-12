// ============================================================
// ecg_f2a.cpp - Phuong an A: IP F2 voi lop dau ra ghi duoc + xuat vector dac trung
// Giao tiep AXI-Lite: ecg_in[360], rr_in[8], head_w[128], head_b[4] (vao)
//                     logits[4], emb_out[32], label (ra)
// Toan bo cac lop Conv/Pool/RR/Dense1 GIU NGUYEN tu ecg_f2.cpp (seed 3).
// ============================================================
#include "ecg_f2a.h"
#include "weights_f2a.h"

// ---------- Conv1D(16, k7, same) + ReLU: cua so truot ----------
static void conv1_relu(in_t in[360][1], a1_t out[360][16]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=C1_W type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=C1_W type=complete dim=3
    in_t win[7][1];
#pragma HLS ARRAY_PARTITION variable=win type=complete dim=0
    for (int k = 0; k < 7; k++)
        for (int c = 0; c < 1; c++)
            win[k][c] = 0;
    C1_STEP:
    for (int i = 0; i < 360 + 7 - 1; i++) {
        int new_pos = i - 3;
        int pos     = i - 6;
        C1_SHIFT:
        for (int c = 0; c < 1; c++) {
#pragma HLS PIPELINE II=1
            for (int k = 0; k < 6; k++) win[k][c] = win[k + 1][c];
            win[6][c] = (new_pos >= 0 && new_pos < 360) ? in[new_pos][c] : (in_t)0;
        }
        if (pos >= 0 && pos < 360) {
            C1_OC:
            for (int o = 0; o < 16; o++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int k = 0; k < 7; k++)
                    for (int c = 0; c < 1; c++)
                        acc += win[k][c] * C1_W[o][k][c];
                acc += C1_B[o];
                out[pos][o] = (acc > 0) ? (a1_t)acc : (a1_t)0;
            }
        }
    }
}

// ---------- MaxPool(2) ----------
static void pool1(a1_t in[360][16], a1_t out[180][16]) {
#pragma HLS INLINE off
    P1_POS:
    for (int p = 0; p < 180; p++) {
        P1_CH:
        for (int c = 0; c < 16; c++) {
#pragma HLS PIPELINE II=1
            a1_t a = in[2 * p][c], b = in[2 * p + 1][c];
            out[p][c] = (a > b) ? a : b;
        }
    }
}

// ---------- Conv1D(32, k5, same) + ReLU: cua so truot ----------
static void conv2_relu(a1_t in[180][16], a2_t out[180][32]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=C2_W type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=C2_W type=complete dim=3
    a1_t win[5][16];
#pragma HLS ARRAY_PARTITION variable=win type=complete dim=0
    for (int k = 0; k < 5; k++)
        for (int c = 0; c < 16; c++)
            win[k][c] = 0;
    C2_STEP:
    for (int i = 0; i < 180 + 5 - 1; i++) {
        int new_pos = i - 2;
        int pos     = i - 4;
        C2_SHIFT:
        for (int c = 0; c < 16; c++) {
#pragma HLS PIPELINE II=1
            for (int k = 0; k < 4; k++) win[k][c] = win[k + 1][c];
            win[4][c] = (new_pos >= 0 && new_pos < 180) ? in[new_pos][c] : (a1_t)0;
        }
        if (pos >= 0 && pos < 180) {
            C2_OC:
            for (int o = 0; o < 32; o++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int k = 0; k < 5; k++)
                    for (int c = 0; c < 16; c++)
                        acc += win[k][c] * C2_W[o][k][c];
                acc += C2_B[o];
                out[pos][o] = (acc > 0) ? (a2_t)acc : (a2_t)0;
            }
        }
    }
}

// ---------- MaxPool(2) ----------
static void pool2(a2_t in[180][32], a2_t out[90][32]) {
#pragma HLS INLINE off
    P2_POS:
    for (int p = 0; p < 90; p++) {
        P2_CH:
        for (int c = 0; c < 32; c++) {
#pragma HLS PIPELINE II=1
            a2_t a = in[2 * p][c], b = in[2 * p + 1][c];
            out[p][c] = (a > b) ? a : b;
        }
    }
}

// ---------- Conv1D(32, k5, same) + ReLU: cua so truot ----------
static void conv3_relu(a2_t in[90][32], a3_t out[90][32]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=C3_W type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=C3_W type=complete dim=3
    a2_t win[5][32];
#pragma HLS ARRAY_PARTITION variable=win type=complete dim=0
    for (int k = 0; k < 5; k++)
        for (int c = 0; c < 32; c++)
            win[k][c] = 0;
    C3_STEP:
    for (int i = 0; i < 90 + 5 - 1; i++) {
        int new_pos = i - 2;
        int pos     = i - 4;
        C3_SHIFT:
        for (int c = 0; c < 32; c++) {
#pragma HLS PIPELINE II=1
            for (int k = 0; k < 4; k++) win[k][c] = win[k + 1][c];
            win[4][c] = (new_pos >= 0 && new_pos < 90) ? in[new_pos][c] : (a2_t)0;
        }
        if (pos >= 0 && pos < 90) {
            C3_OC:
            for (int o = 0; o < 32; o++) {
#pragma HLS PIPELINE II=1
                acc_t acc = 0;
                for (int k = 0; k < 5; k++)
                    for (int c = 0; c < 32; c++)
                        acc += win[k][c] * C3_W[o][k][c];
                acc += C3_B[o];
                out[pos][o] = (acc > 0) ? (a3_t)acc : (a3_t)0;
            }
        }
    }
}

// ---------- MaxPool(2) ----------
static void pool3(a3_t in[90][32], a3_t out[45][32]) {
#pragma HLS INLINE off
    P3_POS:
    for (int p = 0; p < 45; p++) {
        P3_CH:
        for (int c = 0; c < 32; c++) {
#pragma HLS PIPELINE II=1
            a3_t a = in[2 * p][c], b = in[2 * p + 1][c];
            out[p][c] = (a > b) ? a : b;
        }
    }
}

// ---------- Nhanh RR: Dense(8 -> 16) + ReLU ----------
static void rr_dense(rr_t in[N_RR], ar_t out[RR_H]) {
#pragma HLS INLINE off
    RR_OUT:
    for (int o = 0; o < RR_H; o++) {
        acc_t acc = RR_B[o];
        RR_IN:
        for (int i = 0; i < N_RR; i++) {
#pragma HLS PIPELINE II=1
            acc += in[i] * RR_W[o][i];
        }
        out[o] = (acc > 0) ? (ar_t)acc : (ar_t)0;
    }
}

// ---------- Flatten ++ RR -> Dense(32) + ReLU (tuan tu, 1 bo nhan) ----------
static void dense1_relu(a3_t in[45][32], ar_t rr[RR_H], ad1_t out[D1_OUT]) {
#pragma HLS INLINE off
#pragma HLS BIND_STORAGE variable=D1_W type=rom_1p impl=bram
    D1_OUT_LOOP:
    for (int o = 0; o < D1_OUT; o++) {
        acc_t acc = D1_B[o];
        D1_FLAT:
        for (int p = 0; p < 45; p++) {
            D1_CH:
            for (int c = 0; c < 32; c++) {
#pragma HLS PIPELINE II=1
                acc += in[p][c] * D1_W[o][p * 32 + c];
            }
        }
        D1_RR:
        for (int r = 0; r < RR_H; r++) {
#pragma HLS PIPELINE II=1
            acc += rr[r] * D1_W[o][N_FLAT + r];
        }
        out[o] = (acc > 0) ? (ad1_t)acc : (ad1_t)0;
    }
}

// ---------- Dense(4) + argmax: trong so/bias lay tu thanh ghi (ghi duoc) ----------
static void dense_out_argmax(ad1_t in[D1_OUT], const wo_t w[N_HEAD_W], const logit_t b[NUM_CLASSES],
                             logit_t logits[NUM_CLASSES], label_t &label) {
#pragma HLS INLINE off
    logit_t best = 0;
    label_t best_idx = 0;
    DO_OUT:
    for (int o = 0; o < NUM_CLASSES; o++) {
        logit_t acc = b[o];
        DO_IN:
        for (int i = 0; i < D1_OUT; i++) {
#pragma HLS PIPELINE II=1
            acc += in[i] * w[o * D1_OUT + i];
        }
        logits[o] = acc;
        if (o == 0 || acc > best) { best = acc; best_idx = o; }
    }
    label = best_idx;
}

// ---------- TOP ----------
void ecg_f2a_top(in_t ecg_in[IN_LEN], rr_t rr_in[N_RR],
                 wo_t head_w[N_HEAD_W], logit_t head_b[NUM_CLASSES],
                 logit_t logits[NUM_CLASSES], ad1_t emb_out[D1_OUT], label_t &label) {
#pragma HLS INTERFACE mode=s_axilite port=ecg_in
#pragma HLS INTERFACE mode=s_axilite port=rr_in
#pragma HLS INTERFACE mode=s_axilite port=head_w
#pragma HLS INTERFACE mode=s_axilite port=head_b
#pragma HLS INTERFACE mode=s_axilite port=logits
#pragma HLS INTERFACE mode=s_axilite port=emb_out
#pragma HLS INTERFACE mode=s_axilite port=label
#pragma HLS INTERFACE mode=s_axilite port=return
    in_t  x0[360][1];
    a1_t  r1[360][16];  a1_t p1[180][16];
    a2_t  r2[180][32];  a2_t p2[90][32];
    a3_t  r3[90][32];  a3_t p3[45][32];
    ar_t  rh[RR_H];
    ad1_t h[D1_OUT];
    for (int t = 0; t < IN_LEN; t++) {
#pragma HLS PIPELINE II=1
        x0[t][0] = ecg_in[t];
    }
    conv1_relu(x0, r1); pool1(r1, p1);
    conv2_relu(p1, r2); pool2(r2, p2);
    conv3_relu(p2, r3); pool3(r3, p3);
    rr_dense(rr_in, rh);
    dense1_relu(p3, rh, h);
    EMB_OUT:
    for (int i = 0; i < D1_OUT; i++) {
#pragma HLS PIPELINE II=1
        emb_out[i] = h[i];
    }
    dense_out_argmax(h, head_w, head_b, logits, label);
}
