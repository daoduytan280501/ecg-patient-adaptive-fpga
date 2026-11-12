"""
lib_ecg_f2.py - Thu vien chay IP ecg_f2_top tren Kria KV260 (PYNQ)
Cung cap:
  EcgF2(bitfile)          : nap overlay, tim ban do thanh ghi
  .classify(beat, rr)     : phan loai 1 nhip, tra (logits, label, latency_us)
  .classify_many(...)     : chay nhieu nhip, do latency/throughput
  NumpyF2(params_npz)     : chay CUNG model tren CPU ARM (int8 bit-exact va float)
  PowerMonitor()          : do cong suat board qua cam bien (PYNQ rails hoac hwmon)
"""
import json, time, os, glob
import numpy as np

LF = 14                      # logit co 14 bit thap phan
CLASS_NAMES = ['N', 'S', 'V', 'F']
CLASS_FULL = {'N': 'Nhip binh thuong', 'S': 'Ngoai tam thu tren that',
              'V': 'Ngoai tam thu that', 'F': 'Nhip hoa hop'}


# ======================= PHAN CUNG =======================
class EcgF2:
    def __init__(self, bitfile='ecg_f2.bit', header='xecg_f2_top_hw.h', fclk_mhz=100):
        from pynq import Overlay, Clocks
        self.ol = Overlay(bitfile)
        Clocks.fclk0_mhz = fclk_mhz                      # thiet ke dong timing o 100 MHz (PYNQ mac dinh 150)
        self.fclk_mhz = float(Clocks.fclk0_mhz)
        print(f'fclk0 = {self.fclk_mhz:.3f} MHz')
        self.ip = None
        for name, desc in self.ol.ip_dict.items():
            if 'ecg_f2' in name:
                self.ip = getattr(self.ol, name)
                break
        if self.ip is None:
            raise RuntimeError('Khong tim thay IP ecg_f2_top trong overlay: '
                               + ', '.join(self.ol.ip_dict))
        self.addr = self._addrs(header)
        print('Dia chi thanh ghi:', {k: hex(v) for k, v in self.addr.items()})

    def _addrs(self, header):
        """Doc dia chi tu xecg_f2_top_hw.h; neu thieu file thi dung mac dinh."""
        want = {'AP_CTRL': ['ADDR_AP_CTRL'], 'LABEL': ['ADDR_LABEL_R_DATA', 'ADDR_LABEL_DATA'],
                'LOGITS': ['ADDR_LOGITS_BASE'], 'RR': ['ADDR_RR_IN_BASE'], 'ECG': ['ADDR_ECG_IN_BASE']}
        found = {}
        if os.path.exists(header):
            for line in open(header):
                p = line.split()
                if len(p) >= 3 and p[0] == '#define':
                    for key, names in want.items():
                        if any(p[1].endswith(n) for n in names):
                            found[key] = int(p[2], 0)
        # dia chi THAT cua IP ecg_f2_top (doc tu xecg_f2_top_hw.h o lan chay truoc, bit-exact 50/50)
        default = {'AP_CTRL': 0x00, 'LABEL': 0x30, 'LOGITS': 0x20, 'RR': 0x10, 'ECG': 0x200}
        if not os.path.exists(header):
            print(f'Khong thay {header}: dung dia chi mac dinh da kiem chung cua ecg_f2_top')
        for k, v in default.items():
            found.setdefault(k, v)
        return found

    # ---- truy cap thanh ghi ----
    def _prep(self, data):
        b = np.asarray(data, np.int8).tobytes()
        if len(b) % 4:
            b += b'\x00' * (4 - len(b) % 4)
        return np.frombuffer(b, dtype='<u4')

    def _write_bytes(self, base, data):
        """Ghi mang int8 vao vung nho AXI. Uu tien ghi ca khoi qua mmio.array (nhanh hon ~20 lan)."""
        words = self._prep(data)
        arr = getattr(self.ip.mmio, 'array', None)
        if arr is not None:
            i = base // 4
            arr[i:i + len(words)] = words
        else:
            for k, w in enumerate(words):
                self.ip.write(base + 4 * k, int(w))

    def classify(self, beat_q, rr_q, prepped=False):
        """beat_q: 360 so nguyen int8 (frac 6); rr_q: 8 so nguyen int8.
        prepped=True: beat_q/rr_q da la mang uint32 (dung trong do throughput)."""
        if prepped:
            arr = self.ip.mmio.array
            arr[self.addr['ECG'] // 4: self.addr['ECG'] // 4 + len(beat_q)] = beat_q
            arr[self.addr['RR'] // 4: self.addr['RR'] // 4 + len(rr_q)] = rr_q
        else:
            self._write_bytes(self.addr['ECG'], beat_q)
            self._write_bytes(self.addr['RR'], rr_q)
        t0 = time.perf_counter()
        self.ip.write(self.addr['AP_CTRL'], 1)          # ap_start
        while not (self.ip.read(self.addr['AP_CTRL']) & 0x2):
            pass                                         # cho ap_done
        t1 = time.perf_counter()
        logits = np.array([np.int32(np.uint32(self.ip.read(self.addr['LOGITS'] + 4 * o)))
                           for o in range(4)], dtype=np.int64)
        label = self.ip.read(self.addr['LABEL']) & 0x3
        return logits, int(label), (t1 - t0) * 1e6      # us: chi thoi gian tinh toan

    def classify_many(self, beats_q, rr_q, verbose_every=0):
        """Tra dict: nhan, logit, thoi gian tinh toan va tong thoi gian (gom ghi/doc AXI)."""
        n = len(beats_q)
        preds = np.zeros(n, np.int8); lgs = np.zeros((n, 4), np.int64)
        t_core = np.zeros(n); t_all = np.zeros(n)
        pb = [self._prep(b) for b in beats_q]
        pr = [self._prep(r) for r in rr_q]
        for i in range(n):
            s = time.perf_counter()
            lg, lab, tc = self.classify(pb[i], pr[i], prepped=True)
            t_all[i] = (time.perf_counter() - s) * 1e6
            preds[i], lgs[i], t_core[i] = lab, lg, tc
            if verbose_every and (i + 1) % verbose_every == 0:
                print(f'  {i + 1}/{n} nhip...')
        return dict(pred=preds, logits=lgs, t_core_us=t_core, t_total_us=t_all)


# ======================= CPU ARM / PC =======================
class NumpyF2:
    """Chay cung model bang NumPy: int8 (bit-exact nhu FPGA) hoac float32."""
    def __init__(self, npz='f2_params.npz'):
        z = np.load(npz, allow_pickle=True)
        self.P = z['P'].item()
        self.F = z['F'].item()          # trong so float + offset
        self.qcfg = z['qcfg'].item()

    # ---- int8, giong het phan cung ----
    @staticmethod
    def _requant_relu(acc, units, fa):
        s = units - fa
        v = (acc + (1 << (s - 1))) >> s if s > 0 else acc << (-s)
        return np.where(acc > 0, np.clip(v, -128, 127), 0)

    def int8(self, xq, rq, batch=256):
        P = self.P; out = []
        for s in range(0, len(xq), batch):
            x = np.asarray(xq[s:s + batch], np.int64)[:, :, None]
            for c in P['conv']:
                K = c['K']; pl = (K - 1) // 2
                xp = np.pad(x, ((0, 0), (pl, K - 1 - pl), (0, 0)))
                win = np.lib.stride_tricks.sliding_window_view(xp, K, axis=1)
                U = max(c['fin'] + c['fw'], c['fb'])
                acc = np.einsum('nlck,fkc->nlf', win, c['W'], optimize=True) * (1 << (U - c['fin'] - c['fw'])) \
                      + c['B'] * (1 << (U - c['fb']))
                a = self._requant_relu(acc, U, c['fa'])
                Lp = a.shape[1] // 2
                x = np.maximum(a[:, 0:2 * Lp:2], a[:, 1:2 * Lp:2])
            flat = x.reshape(len(x), -1)
            r = np.asarray(rq[s:s + batch], np.int64)
            d = P['rrd']; U = max(d['fin'] + d['fw'], d['fb'])
            acc = (r @ d['W'].T) * (1 << (U - d['fin'] - d['fw'])) + d['B'] * (1 << (U - d['fb']))
            r16 = self._requant_relu(acc, U, d['fa'])
            d = P['d1']; nf = flat.shape[1]
            U = max(d['fin_cnn'] + d['fw'], d['fin_rr'] + d['fw'], d['fb'])
            acc = (flat @ d['W'][:, :nf].T) * (1 << (U - d['fin_cnn'] - d['fw'])) \
                + (r16 @ d['W'][:, nf:].T) * (1 << (U - d['fin_rr'] - d['fw'])) + d['B'] * (1 << (U - d['fb']))
            h = self._requant_relu(acc, U, d['fa'])
            o = P['do']
            out.append((h @ o['W'].T) * (1 << (LF - o['fin'] - o['fw'])) + o['Btot'])
        return np.concatenate(out)

    # ---- float32 ----
    def float32(self, X, Rn, batch=256):
        F = self.F; out = []
        for s in range(0, len(X), batch):
            x = np.asarray(X[s:s + batch], np.float32)[:, :, None]
            for W, b, K in F['conv']:
                pl = (K - 1) // 2
                xp = np.pad(x, ((0, 0), (pl, K - 1 - pl), (0, 0)))
                win = np.lib.stride_tricks.sliding_window_view(xp, K, axis=1)
                a = np.maximum(np.einsum('nlck,kcf->nlf', win, W, optimize=True) + b, 0)
                Lp = a.shape[1] // 2
                x = np.maximum(a[:, 0:2 * Lp:2], a[:, 1:2 * Lp:2])
            flat = x.reshape(len(x), -1)
            r = np.maximum(np.asarray(Rn[s:s + batch], np.float32) @ F['rrd'][0] + F['rrd'][1], 0)
            h = np.maximum(np.concatenate([flat, r], 1) @ F['d1'][0] + F['d1'][1], 0)
            out.append(h @ F['do'][0] + F['do'][1] + F['offset'])
        return np.concatenate(out)


# ======================= CONG SUAT =======================
class PowerMonitor:
    """Do cong suat board KV260.
    Uu tien PYNQ rails; neu khong co thi doc truc tiep cam bien INA qua sysfs hwmon.
    Moi lan doc deu bat loi rieng tung file -> mot cam bien hong khong lam chet luong do.
    Moi cam bien chi dem MOT lan: uu tien thanh ghi power*_input (INA260), khong cong dong x ap.
    """
    def __init__(self, verbose=True):
        self.mode, self.rails, self.files = None, None, []
        try:
            from pynq import get_rails
            r = get_rails()
            if r:
                # chi giu rail thuc su doc duoc
                ok = {}
                for k, v in r.items():
                    try:
                        float(v.power.value); ok[k] = v
                    except Exception:
                        pass
                if ok:
                    self.rails, self.mode = ok, 'pynq'
        except Exception as e:
            if verbose:
                print('PYNQ rails khong dung duoc:', type(e).__name__)
        if self.mode is None:
            self.files = self._scan_hwmon()
            if self.files:
                self.mode = 'hwmon'
        if verbose:
            print('Nguon do cong suat:', self.mode or 'KHONG CO (dung dong ho do o cam)')
            if self.mode == 'hwmon':
                print(f'  {len(self.files)} kenh: ' +
                      ', '.join(n for n, _, _ in self.files[:8]) + (' ...' if len(self.files) > 8 else ''))
            print('  Cong suat doc thu:', self.read_w(), 'W')

    @staticmethod
    def _read_int(path):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _scan_hwmon(self):
        """Tra ve danh sach (ten, kieu, duong dan) cac kenh doc duoc."""
        out = []
        for d in sorted(glob.glob('/sys/class/hwmon/hwmon*')):
            name = 'hwmon'
            try:
                name = open(os.path.join(d, 'name')).read().strip()
            except Exception:
                pass
            for f in sorted(glob.glob(d + '/power*_input')):
                if self._read_int(f) is not None:
                    out.append((f'{name}:{os.path.basename(f)}', 'power', f))
            if glob.glob(d + '/power*_input'):
                continue      # da co thanh ghi cong suat -> KHONG cong them dong x ap (tranh dem 2 lan)
            for fc in sorted(glob.glob(d + '/curr*_input')):
                fv = fc.replace('curr', 'in')
                if os.path.exists(fv) and self._read_int(fc) is not None and self._read_int(fv) is not None:
                    out.append((f'{name}:{os.path.basename(fc)}', 'iv', (fc, fv)))
        return out

    def list_sensors(self):
        """In chi tiet tung kenh de kiem tra (chay 1 lan de xem board co gi)."""
        if self.mode == 'pynq':
            for k, v in self.rails.items():
                try:
                    print(f'  {k}: {float(v.power.value):.3f} W')
                except Exception as e:
                    print(f'  {k}: loi {e}')
        for n, kind, p in self.files:
            if kind == 'power':
                print(f'  {n}: {self._read_int(p) / 1e6:.3f} W')
            else:
                c, v = self._read_int(p[0]), self._read_int(p[1])
                print(f'  {n}: {c} mA x {v} mV = {c * v / 1e6:.3f} W')

    def read_w(self):
        """Tong cong suat tuc thoi (W); None neu khong do duoc."""
        tot, got = 0.0, False
        if self.mode == 'pynq':
            for v in self.rails.values():
                try:
                    tot += float(v.power.value); got = True
                except Exception:
                    pass
        elif self.mode == 'hwmon':
            for _, kind, p in self.files:
                if kind == 'power':
                    v = self._read_int(p)
                    if v is not None:
                        tot += v / 1e6; got = True
                else:
                    c, v = self._read_int(p[0]), self._read_int(p[1])
                    if c is not None and v is not None:
                        tot += c * v / 1e6; got = True
        return tot if got else None

    def baseline(self, seconds=3.0, period=0.05):
        """Do cong suat luc board RANH (khong chay gi). Goi truoc khi do tai."""
        end = time.time() + seconds
        s = []
        while time.time() < end:
            v = self.read_w()
            if v:
                s.append(v)
            time.sleep(period)
        self.idle_w = float(np.mean(s)) if s else None
        print(f'Cong suat luc ranh: {self.idle_w} W ({len(s)} mau)')
        return self.idle_w

    def measure(self, fn, period=0.05, min_seconds=3.0):
        """Chay fn() va lay mau cong suat trong luc chay.
        Neu fn qua nhanh (< min_seconds) se chay lap lai de lay du mau.
        Tra (ket qua, thong ke). Loi trong luong lay mau duoc bao lai, khong lam chet phep do.
        """
        import threading
        samples, errs, stop = [], [], threading.Event()

        def sampler():
            while not stop.is_set():
                try:
                    v = self.read_w()
                    if v:
                        samples.append(v)
                except Exception as e:
                    errs.append(repr(e))
                stop.wait(period)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        t0 = time.perf_counter()
        res = fn()
        reps = 1
        while time.perf_counter() - t0 < min_seconds:      # chay them cho du thoi gian lay mau
            fn(); reps += 1
        dt = time.perf_counter() - t0
        stop.set(); th.join(timeout=2)

        st = dict(seconds=dt, reps=reps, n_samples=len(samples), errors=errs[:3])
        if samples:
            st.update(p_mean_w=float(np.mean(samples)), p_max_w=float(np.max(samples)),
                      p_min_w=float(np.min(samples)))
            idle = getattr(self, 'idle_w', None)
            if idle:
                st['p_idle_w'] = idle
                st['p_active_w'] = st['p_mean_w'] - idle
        return res, st


# ======================= TIEN ICH =======================
def load_tests(d='.'):
    """Doc cac file test_*.txt do notebook Colab xuat ra."""
    g = lambda f: np.loadtxt(os.path.join(d, f), dtype=np.int64)
    out = dict(beats=g('test_beats_q.txt').astype(np.int8), rr=g('test_rr_q.txt').astype(np.int8),
               y=g('test_labels_true.txt'), lg_sim=g('test_logits_sim.txt'))
    if os.path.exists(os.path.join(d, 'test_labels_sim.txt')):
        out['y_sim'] = g('test_labels_sim.txt')
    return out

def metrics(y, p, names=CLASS_NAMES):
    k = len(names)
    cm = np.zeros((k, k), np.int64)
    for a, b in zip(y, p):
        cm[a, b] += 1
    N = cm.sum(); po = np.trace(cm) / N
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / N ** 2
    se = np.divide(np.diag(cm), cm.sum(1), out=np.full(k, np.nan), where=cm.sum(1) > 0)
    pp = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    f1 = np.where((se + pp) > 0, 2 * se * pp / np.where(se + pp > 0, se + pp, 1), 0)
    kappa = (po - pe) / (1 - pe)
    ok = ~np.isnan(se)
    res = dict(acc=po, kappa=kappa, macroF1=float(np.nanmean(f1[ok])), balAcc=float(np.nanmean(se[ok])), cm=cm)
    res['jk'] = (se[1] + se[2] + pp[1] + pp[2]) / 8 + kappa / 2
    for i, c in enumerate(names):
        res[f'Se_{c}'], res[f'PP_{c}'], res[f'F1_{c}'] = se[i], pp[i], f1[i]
    return res

def show(res, title=''):
    print(f'--- {title} --- acc={res["acc"]:.4f} kappa={res["kappa"]:.4f} '
          f'macroF1={res["macroF1"]:.4f} jk={res["jk"]:.4f}')
    for c in CLASS_NAMES:
        print(f'   {c}: Se={res[f"Se_{c}"]:.3f} +P={res[f"PP_{c}"]:.3f} F1={res[f"F1_{c}"]:.3f}')
    print('   Confusion:\n', res['cm'])
