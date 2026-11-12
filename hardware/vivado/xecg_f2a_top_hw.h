// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x000 : Control signals
//         bit 0  - ap_start (Read/Write/COH)
//         bit 1  - ap_done (Read/COR)
//         bit 2  - ap_idle (Read)
//         bit 3  - ap_ready (Read/COR)
//         bit 7  - auto_restart (Read/Write)
//         bit 9  - interrupt (Read)
//         others - reserved
// 0x004 : Global Interrupt Enable Register
//         bit 0  - Global Interrupt Enable (Read/Write)
//         others - reserved
// 0x008 : IP Interrupt Enable Register (Read/Write)
//         bit 0 - enable ap_done interrupt (Read/Write)
//         bit 1 - enable ap_ready interrupt (Read/Write)
//         others - reserved
// 0x00c : IP Interrupt Status Register (Read/TOW)
//         bit 0 - ap_done (Read/TOW)
//         bit 1 - ap_ready (Read/TOW)
//         others - reserved
// 0x060 : Data signal of label_r
//         bit 1~0 - label_r[1:0] (Read)
//         others  - reserved
// 0x064 : Control signal of label_r
//         bit 0  - label_r_ap_vld (Read/COR)
//         others - reserved
// 0x010 ~
// 0x017 : Memory 'rr_in' (8 * 8b)
//         Word n : bit [ 7: 0] - rr_in[4n]
//                  bit [15: 8] - rr_in[4n+1]
//                  bit [23:16] - rr_in[4n+2]
//                  bit [31:24] - rr_in[4n+3]
// 0x020 ~
// 0x02f : Memory 'head_b' (4 * 32b)
//         Word n : bit [31:0] - head_b[n]
// 0x030 ~
// 0x03f : Memory 'logits' (4 * 32b)
//         Word n : bit [31:0] - logits[n]
// 0x040 ~
// 0x05f : Memory 'emb_out' (32 * 8b)
//         Word n : bit [ 7: 0] - emb_out[4n]
//                  bit [15: 8] - emb_out[4n+1]
//                  bit [23:16] - emb_out[4n+2]
//                  bit [31:24] - emb_out[4n+3]
// 0x080 ~
// 0x0ff : Memory 'head_w' (128 * 8b)
//         Word n : bit [ 7: 0] - head_w[4n]
//                  bit [15: 8] - head_w[4n+1]
//                  bit [23:16] - head_w[4n+2]
//                  bit [31:24] - head_w[4n+3]
// 0x200 ~
// 0x3ff : Memory 'ecg_in' (360 * 8b)
//         Word n : bit [ 7: 0] - ecg_in[4n]
//                  bit [15: 8] - ecg_in[4n+1]
//                  bit [23:16] - ecg_in[4n+2]
//                  bit [31:24] - ecg_in[4n+3]
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XECG_F2A_TOP_CONTROL_ADDR_AP_CTRL      0x000
#define XECG_F2A_TOP_CONTROL_ADDR_GIE          0x004
#define XECG_F2A_TOP_CONTROL_ADDR_IER          0x008
#define XECG_F2A_TOP_CONTROL_ADDR_ISR          0x00c
#define XECG_F2A_TOP_CONTROL_ADDR_LABEL_R_DATA 0x060
#define XECG_F2A_TOP_CONTROL_BITS_LABEL_R_DATA 2
#define XECG_F2A_TOP_CONTROL_ADDR_LABEL_R_CTRL 0x064
#define XECG_F2A_TOP_CONTROL_ADDR_RR_IN_BASE   0x010
#define XECG_F2A_TOP_CONTROL_ADDR_RR_IN_HIGH   0x017
#define XECG_F2A_TOP_CONTROL_WIDTH_RR_IN       8
#define XECG_F2A_TOP_CONTROL_DEPTH_RR_IN       8
#define XECG_F2A_TOP_CONTROL_ADDR_HEAD_B_BASE  0x020
#define XECG_F2A_TOP_CONTROL_ADDR_HEAD_B_HIGH  0x02f
#define XECG_F2A_TOP_CONTROL_WIDTH_HEAD_B      32
#define XECG_F2A_TOP_CONTROL_DEPTH_HEAD_B      4
#define XECG_F2A_TOP_CONTROL_ADDR_LOGITS_BASE  0x030
#define XECG_F2A_TOP_CONTROL_ADDR_LOGITS_HIGH  0x03f
#define XECG_F2A_TOP_CONTROL_WIDTH_LOGITS      32
#define XECG_F2A_TOP_CONTROL_DEPTH_LOGITS      4
#define XECG_F2A_TOP_CONTROL_ADDR_EMB_OUT_BASE 0x040
#define XECG_F2A_TOP_CONTROL_ADDR_EMB_OUT_HIGH 0x05f
#define XECG_F2A_TOP_CONTROL_WIDTH_EMB_OUT     8
#define XECG_F2A_TOP_CONTROL_DEPTH_EMB_OUT     32
#define XECG_F2A_TOP_CONTROL_ADDR_HEAD_W_BASE  0x080
#define XECG_F2A_TOP_CONTROL_ADDR_HEAD_W_HIGH  0x0ff
#define XECG_F2A_TOP_CONTROL_WIDTH_HEAD_W      8
#define XECG_F2A_TOP_CONTROL_DEPTH_HEAD_W      128
#define XECG_F2A_TOP_CONTROL_ADDR_ECG_IN_BASE  0x200
#define XECG_F2A_TOP_CONTROL_ADDR_ECG_IN_HIGH  0x3ff
#define XECG_F2A_TOP_CONTROL_WIDTH_ECG_IN      8
#define XECG_F2A_TOP_CONTROL_DEPTH_ECG_IN      360

