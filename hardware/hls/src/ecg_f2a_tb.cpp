// ============================================================
// ecg_f2a_tb.cpp - Testbench IP ecg_f2a_top (phuong an A)
// File Test Bench: weights_f2a.h + test_beats_q.txt, test_rr_q.txt, test_labels_true.txt,
//                  test_labels_sim.txt, test_logits_sim.txt  (giong ECG_F2)
//   Tuy chon (neu co):  test_emb_sim.txt      : 32 so nguyen/dong, don vi 2^-4 (vector dac trung Python)
//                       test_head_adapted.txt : 128 so nguyen (W, don vi 2^-7) + 4 so nguyen (b, don vi 2^-14)
//                       test_logits_adapted.txt: logit Python voi lop dau ra da hieu chinh (don vi 2^-14)
// Input Arguments: (trong) = toan bo | 10 = 10 mau dau (Co-sim)
//
// T1  Hoi quy : nap trong so toan cuc (DO_W, DO_BTOT) -> logit + nhan khop bit-exact voi Python
// T2  Ghi lop dau ra: moi nhip dung mot bo trong so KHAC (ngau nhien, co bien +-1.0, bias lon am/duong);
//     logit phai = b + sum(emb_out*w) tinh lai trong testbench, emb_out phai giong T1 (khong phu thuoc head)
// T3  (tuy chon) trong so hieu chinh tu Python -> khop bit-exact test_logits_adapted.txt
// ============================================================
#include "ecg_f2a.h"
#include "weights_f2a.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>

#define N_MAX 2000
static in_t ecg[N_MAX][IN_LEN];
static rr_t rr[N_MAX][N_RR];
static int  y_true[N_MAX], y_sim[N_MAX];
static long lg_sim[N_MAX][NUM_CLASSES];
static int  emb_ref[N_MAX][D1_OUT];       // emb_out cua T1 (so nguyen don vi 2^-4)

template <typename T> T raw8(int v) { ap_int<8> r = v; T x; x.range(7, 0) = r.range(7, 0); return x; }
static long q14(logit_t v) { return (long)(v * 16384).to_int64(); }
static logit_t from_q14(long v) { logit_t x; x.range(31, 0) = ap_int<32>(v).range(31, 0); return x; }
static int  q4(ad1_t v) { return (int)(v * 16).to_int(); }

static int load() {
    std::ifstream fb("test_beats_q.txt"), fr("test_rr_q.txt"), ft("test_labels_true.txt"),
                  fs("test_labels_sim.txt"), fl("test_logits_sim.txt");
    if (!fb || !fr || !ft || !fs || !fl) { std::cerr << "Thieu file test_*.txt\n"; return -1; }
    int n = 0, v;
    while (n < N_MAX && (fb >> v)) {
        ecg[n][0] = raw8<in_t>(v);
        for (int t = 1; t < IN_LEN; t++) { fb >> v; ecg[n][t] = raw8<in_t>(v); }
        for (int i = 0; i < N_RR; i++) { fr >> v; rr[n][i] = raw8<rr_t>(v); }
        ft >> y_true[n]; fs >> y_sim[n];
        for (int o = 0; o < NUM_CLASSES; o++) fl >> lg_sim[n][o];
        n++;
    }
    return n;
}

static unsigned rng_state = 12345u;
static int rnd(int lo, int hi) { rng_state = rng_state * 1103515245u + 12345u; return lo + (int)((rng_state >> 8) % (unsigned)(hi - lo + 1)); }

int main(int argc, char *argv[]) {
    int n = load();
    if (n < 0) return 1;
    if (argc > 1 && atoi(argv[1]) > 0 && atoi(argv[1]) < n) n = atoi(argv[1]);
    std::cout << "So mau: " << n << "\n";
    int fail = 0;

    // ---------------- T1: hoi quy voi trong so toan cuc ----------------
    wo_t hw[N_HEAD_W]; logit_t hb[NUM_CLASSES];
    for (int o = 0; o < NUM_CLASSES; o++) { hb[o] = DO_BTOT[o]; for (int i = 0; i < D1_OUT; i++) hw[o * D1_OUT + i] = DO_W[o][i]; }
    int lerr = 0, laberr = 0, ok = 0;
    std::ifstream fe("test_emb_sim.txt"); bool has_emb = (bool)fe; int emb_err = 0;
    std::ofstream fdump("tb_emb_out.txt");            // vector dac trung do IP xuat (don vi 2^-4), dung tao du lieu T3
    for (int s = 0; s < n; s++) {
        logit_t lg[NUM_CLASSES]; ad1_t emb[D1_OUT]; label_t lab;
        ecg_f2a_top(ecg[s], rr[s], hw, hb, lg, emb, lab);
        for (int o = 0; o < NUM_CLASSES; o++) if (q14(lg[o]) != lg_sim[s][o]) { lerr++; break; }
        if ((int)lab != y_sim[s]) laberr++;
        if ((int)lab == y_true[s]) ok++;
        for (int i = 0; i < D1_OUT; i++) { emb_ref[s][i] = q4(emb[i]); fdump << emb_ref[s][i] << (i + 1 < D1_OUT ? " " : "\n"); }
        if (has_emb) { bool e = false; for (int i = 0; i < D1_OUT; i++) { int v; fe >> v; if (v != emb_ref[s][i]) e = true; } if (e) emb_err++; }
    }
    std::cout << "[T1] Hoi quy (trong so toan cuc): logit lech " << lerr << " | nhan lech " << laberr
              << " | accuracy " << std::fixed << std::setprecision(4) << (double)ok / n;
    if (has_emb) std::cout << " | emb lech Python " << emb_err;
    std::cout << (lerr == 0 && laberr == 0 && emb_err == 0 ? "  -> PASS\n" : "  -> FAIL\n");
    if (lerr || laberr || emb_err) fail++;

    // ---------------- T2: ghi lop dau ra moi o MOI nhip ----------------
    int t2_err = 0, t2_emb = 0, t2_changed = 0;
    for (int s = 0; s < n; s++) {
        int mode = s % 4;
        for (int k = 0; k < N_HEAD_W; k++) {
            int v = (mode == 0) ? rnd(-128, 127) : (mode == 1) ? ((k & 1) ? 127 : -128)
                  : (mode == 2) ? (int)(DO_W[k / D1_OUT][k % D1_OUT] * 128).to_int() + rnd(-8, 8) : 0;
            if (v > 127) v = 127; if (v < -128) v = -128;
            hw[k] = raw8<wo_t>(v);
        }
        for (int o = 0; o < NUM_CLASSES; o++) {
            long b = (mode == 1) ? ((o & 1) ? (1L << 31) - 1 : -(1L << 31)) / 4 : (long)rnd(-200000, 200000);
            hb[o] = from_q14(b);
        }
        logit_t lg[NUM_CLASSES]; ad1_t emb[D1_OUT]; label_t lab;
        ecg_f2a_top(ecg[s], rr[s], hw, hb, lg, emb, lab);
        // tham chieu trong testbench, cung kieu du lieu
        logit_t best = 0; int bi = 0; bool bad = false;
        for (int o = 0; o < NUM_CLASSES; o++) {
            logit_t acc = hb[o];
            for (int i = 0; i < D1_OUT; i++) acc += emb[i] * hw[o * D1_OUT + i];
            if (q14(acc) != q14(lg[o])) bad = true;
            if (o == 0 || acc > best) { best = acc; bi = o; }
        }
        if ((int)lab != bi) bad = true;
        if (bad) t2_err++;
        for (int i = 0; i < D1_OUT; i++) if (q4(emb[i]) != emb_ref[s][i]) { t2_emb++; break; }
        if (bi != y_sim[s]) t2_changed++;
    }
    std::cout << "[T2] Ghi lop dau ra moi nhip: logit/nhan sai " << t2_err << " | emb thay doi " << t2_emb
              << " | (so nhip doi nhan do trong so moi: " << t2_changed << ")"
              << (t2_err == 0 && t2_emb == 0 && t2_changed > 0 ? "  -> PASS\n" : "  -> FAIL\n");
    if (t2_err || t2_emb || t2_changed == 0) fail++;

    // ---------------- T3 (tuy chon): trong so hieu chinh tu Python ----------------
    std::ifstream fh("test_head_adapted.txt"), fla("test_logits_adapted.txt");
    if (fh && fla) {
        int v; long b;
        for (int k = 0; k < N_HEAD_W; k++) { fh >> v; hw[k] = raw8<wo_t>(v); }
        for (int o = 0; o < NUM_CLASSES; o++) { fh >> b; hb[o] = from_q14(b); }
        int e3 = 0;
        for (int s = 0; s < n; s++) {
            logit_t lg[NUM_CLASSES]; ad1_t emb[D1_OUT]; label_t lab;
            ecg_f2a_top(ecg[s], rr[s], hw, hb, lg, emb, lab);
            bool bad = false;
            for (int o = 0; o < NUM_CLASSES; o++) { fla >> b; if (q14(lg[o]) != b) bad = true; }
            if (bad) e3++;
        }
        std::cout << "[T3] Trong so hieu chinh tu Python: logit lech " << e3 << (e3 == 0 ? "  -> PASS\n" : "  -> FAIL\n");
        if (e3) fail++;
    } else std::cout << "[T3] Bo qua (chua co test_head_adapted.txt / test_logits_adapted.txt)\n";

    std::cout << (fail == 0 ? "PASS: tat ca kiem thu dat\n" : "FAIL\n");
    return fail ? 1 : 0;
}
