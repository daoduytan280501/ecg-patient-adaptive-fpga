"""Gia lap MMIO cua IP ecg_f2a_top de kiem thu driver khi CHUA co board.
IP gia: tra ve emb tu bang tra (theo nhip dau vao) va tinh logit bang tham chieu so nguyen pa_adapt.logits_int
tu THANH GHI head_w/head_b hien tai -> kiem tra dong goi, offset, dau so cua driver."""
import numpy as np, pa_adapt as pa

class SimMMIO:
    def __init__(self, regs, beats_q, emb_q4):
        self.r = regs; self.array = np.zeros(16384, dtype=np.uint32)
        self.lookup = {bytes(np.asarray(b, np.int8)): e for b, e in zip(beats_q, emb_q4)}
    def write(self, off, v):
        self.array[off // 4] = np.uint32(v & 0xFFFFFFFF)
        if off == self.r['AP_CTRL'] and v & 1:
            self._run()
    def read(self, off):
        return int(self.array[off // 4])
    def _run(self):
        a, r = self.array, self.r
        beat = a[r['ECG_IN_BASE'] // 4: r['ECG_IN_BASE'] // 4 + 90].view(np.int8)
        emb = np.asarray(self.lookup[bytes(beat)], dtype=np.int64)
        w = a[r['HEAD_W_BASE'] // 4: r['HEAD_W_BASE'] // 4 + 32].view(np.int8)
        b = a[r['HEAD_B_BASE'] // 4: r['HEAD_B_BASE'] // 4 + 4].view(np.int32).astype(np.int64)
        lg = pa.logits_int(emb, w, b)[0]
        a[r['LOGITS_BASE'] // 4: r['LOGITS_BASE'] // 4 + 4] = (lg & 0xFFFFFFFF).astype(np.uint32)
        a[r['EMB_OUT_BASE'] // 4: r['EMB_OUT_BASE'] // 4 + 8] = np.frombuffer(emb.astype(np.int8).tobytes(), '<u4')
        a[r['LABEL_DATA'] // 4] = int(np.argmax(lg))
        a[r['AP_CTRL'] // 4] = 0x6
