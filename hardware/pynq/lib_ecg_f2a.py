"""lib_ecg_f2a.py - Driver PYNQ cho IP ecg_f2a_top (phuong an A: lop dau ra ghi duoc + xuat vector dac trung).
Offset thanh ghi doc TU DONG tu file xecg_f2a_top_hw.h do Vitis HLS sinh ra (khong ghi cung trong code):
  ECG_F2A_HLS/ECG_F2A/solution1/impl/misc/drivers/ecg_f2a_top_v1_0/src/xecg_f2a_top_hw.h
"""
import re, time
import numpy as np

REQUIRED = ['AP_CTRL', 'ECG_IN_BASE', 'RR_IN_BASE', 'HEAD_W_BASE', 'HEAD_B_BASE',
            'LOGITS_BASE', 'EMB_OUT_BASE', 'LABEL_DATA']

def parse_hw_header(path):
    """Doc cac #define XECG_F2A_TOP_CONTROL_ADDR_<TEN> 0x.. -> dict {TEN: offset}."""
    txt = open(path).read()
    regs = {m.group(1): int(m.group(2), 16)
            for m in re.finditer(r'#define\s+XECG_F2A_TOP_CONTROL_ADDR_(\w+)\s+0x([0-9a-fA-F]+)', txt)}
    # HLS doi ten cong 'label' thanh 'label_r' (tranh trung tu khoa) -> chap nhan ca hai ten
    if 'LABEL_DATA' not in regs and 'LABEL_R_DATA' in regs:
        regs['LABEL_DATA'] = regs['LABEL_R_DATA']
    miss = [k for k in REQUIRED if k not in regs]
    if miss:
        raise ValueError(f'Khong thay thanh ghi {miss} trong {path}')
    return regs

def parse_global_head(weights_path):
    """Lay trong so lop dau ra toan cuc (DO_W, DO_BTOT) tu weights_f2a.h -> (w_i8[128], b_q14[4])."""
    txt = open(weights_path).read()
    def arr(name):
        body = re.search(name + r'\[[^=]*=\s*\{(.*?)\};', txt, re.S).group(1)
        return np.array([float(v) for v in re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', body)])
    w = np.round(arr('DO_W') * 128).astype(np.int8)            # thu tu o*32+i, giong IP
    b = np.round(arr('DO_BTOT') * 16384).astype(np.int64)
    return w, b

def _pack_i8(v):
    v = np.asarray(v, dtype=np.int8)
    pad = (-len(v)) % 4
    if pad:
        v = np.concatenate([v, np.zeros(pad, np.int8)])
    return np.frombuffer(v.tobytes(), dtype='<u4')

class ECGF2A:
    def __init__(self, mmio, regs):
        """mmio: pynq.MMIO (hoac doi tuong gia lap co .read/.write/.array); regs: tu parse_hw_header."""
        self.mmio, self.r = mmio, regs
        self.w_head = None; self.b_head = None

    # ---------- ghi/doc vung nho ----------
    def _write_words(self, off, words):
        a = self.mmio.array
        a[off // 4: off // 4 + len(words)] = np.asarray(words, dtype=np.uint32)

    def _read_words(self, off, n):
        return np.array(self.mmio.array[off // 4: off // 4 + n], dtype=np.uint32)

    # ---------- lop dau ra ----------
    def write_head(self, w_i8, b_q14):
        """w_i8: 128 int8 (chi so o*32+i, don vi 2^-7); b_q14: 4 so nguyen (don vi 2^-14)."""
        w_i8 = np.asarray(w_i8, dtype=np.int8); b_q14 = np.asarray(b_q14, dtype=np.int64)
        assert w_i8.size == 128 and b_q14.size == 4
        self._write_words(self.r['HEAD_W_BASE'], _pack_i8(w_i8))
        self._write_words(self.r['HEAD_B_BASE'], (b_q14 & 0xFFFFFFFF).astype(np.uint32))
        self.w_head, self.b_head = w_i8.copy(), b_q14.copy()

    def read_head(self):
        w = self._read_words(self.r['HEAD_W_BASE'], 32).view(np.int8).copy()
        b = self._read_words(self.r['HEAD_B_BASE'], 4).view(np.int32).astype(np.int64)
        return w, b

    # ---------- suy luan 1 nhip ----------
    def infer(self, beat_q, rr_q, timeout_s=0.1):
        """beat_q: 360 int8 (frac 6), rr_q: 8 int8 (frac 5) -> (logits_q14[4], emb_q4[32], label)."""
        self._write_words(self.r['ECG_IN_BASE'], _pack_i8(beat_q))
        self._write_words(self.r['RR_IN_BASE'], _pack_i8(rr_q))
        self.mmio.write(self.r['AP_CTRL'], 1)
        t0 = time.perf_counter()
        while not (self.mmio.read(self.r['AP_CTRL']) & 0x6):         # ap_done (bit1) hoac ap_idle (bit2)
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError('IP khong xong (ap_done)')
        lg = self._read_words(self.r['LOGITS_BASE'], 4).view(np.int32).astype(np.int64)
        emb = self._read_words(self.r['EMB_OUT_BASE'], 8).view(np.int8).astype(np.int64)
        lab = int(self.mmio.read(self.r['LABEL_DATA']) & 0x3)
        return lg, emb, lab

    def infer_batch(self, beats_q, rrs_q):
        n = len(beats_q)
        L = np.zeros((n, 4), np.int64); Em = np.zeros((n, 32), np.int64); Y = np.zeros(n, int)
        for k in range(n):
            L[k], Em[k], Y[k] = self.infer(beats_q[k], rrs_q[k])
        return L, Em, Y
