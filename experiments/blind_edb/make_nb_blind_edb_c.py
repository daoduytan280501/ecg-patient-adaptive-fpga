"""Ban c (29/10): giong ban b, them (1) dung lai cac ban ghi edb da tai len Drive (kiem tra kich thuoc + doc thu), chi tai phan con thieu
ve dia cuc bo; (2) emb tinh theo khoi 50 000 nhip, moi khoi luu len Drive -> bi ngat thi chay lai tu dau, tiep tuc dung cho cu.
Phuong phap, tham so, tien xu ly KHONG doi (protocol E16)."""
import json, sys
SRC, DST = sys.argv[1], sys.argv[2]
nb = json.load(open(SRC, encoding='utf-8'))
S = lambda i: nb['cells'][i]['source'] if isinstance(nb['cells'][i]['source'], str) else ''.join(nb['cells'][i]['source'])
def setc(i, s): nb['cells'][i]['source'] = s
def rep(s, a, b):
    assert a in s, a[:60]; return s.replace(a, b)
setc(0, S(0) + "\n\n**Bản c (29/10):** dùng lại các bản ghi đã tải lên Drive (kiểm tra kích thước và đọc thử từng bản ghi), chỉ tải phần còn thiếu về ổ cục bộ; "
        "embedding tính theo khối 50 000 nhịp và lưu từng khối lên Drive. **Nếu Colab bị ngắt: Runtime → Run all lại**, các khối đã xong được dùng lại. "
        "Phương pháp, tham số, tiền xử lý không đổi.")
c3 = S(3)
c3 = rep(c3, "notebook ecg_blind_edb_20261026b.ipynb | ", "notebook ecg_blind_edb_20261026c.ipynb | ")
c3 = rep(c3, "Bản b chỉ đổi cách tải dữ liệu (ZIP về đĩa cục bộ).", "Bản c chỉ đổi cách tải dữ liệu và lưu emb theo khối để chạy tiếp được khi bị ngắt.")
setc(3, c3)
c5 = S(5)
c5 = rep(c5, """def emb_cached(tag, seed, X, R, S):
    p = os.path.join(CACHE, f'emb_int_{tag}_s{seed}.npz')
    if os.path.exists(p):
        z = np.load(p)
        if z['E'].shape[0] == len(X):
            return z['E'].astype(np.int64)
    t0 = time.time()
    bq, rq = q_inputs(X, R, S['qcfg'])
    E = forward_emb_int_fast(bq, rq, S['P'], S['qcfg'])
    np.savez_compressed(p, E=E.astype(np.uint8))
    print(f'   emb {tag} seed {seed}: {len(X)} nhịp, {time.time() - t0:.0f} s')
    return E""",
"""def emb_cached(tag, seed, X, R, S, chunk=50000):
    # Lưu theo khối lên Drive: bị ngắt thì chạy lại, khối đã xong được dùng lại (kết quả giống hệt tính một lần).
    p = os.path.join(CACHE, f'emb_int_{tag}_s{seed}.npz')
    if os.path.exists(p):
        z = np.load(p)
        if z['E'].shape[0] == len(X):
            return z['E'].astype(np.int64)
    t0 = time.time(); parts = []
    for k, s0 in enumerate(range(0, len(X), chunk)):
        pk = os.path.join(CACHE, f'emb_int_{tag}_s{seed}_n{len(X)}_part{k:03d}.npz')
        if os.path.exists(pk):
            parts.append(np.load(pk)['E']); continue
        bq, rq = q_inputs(X[s0:s0 + chunk], R[s0:s0 + chunk], S['qcfg'])
        Ek = forward_emb_int_fast(bq, rq, S['P'], S['qcfg']).astype(np.uint8)
        np.savez_compressed(pk, E=Ek); parts.append(Ek)
        print(f'   emb {tag} seed {seed}: {min(s0 + chunk, len(X))}/{len(X)} nhịp, {time.time() - t0:.0f} s', flush=True)
    E = np.concatenate(parts)
    np.savez_compressed(p, E=E.astype(np.uint8))
    return E.astype(np.int64)""")
setc(5, c5)
c8 = S(8)
c8 = rep(c8, """for db in EXT_DBS:
    d = find_dir(LOCAL) if os.path.exists(LOCAL) else None""",
"""def usable(d, r):
    # bản ghi dùng được: đủ 3 tệp, .dat đúng 5 400 000 byte (2 h x 250 Hz x 2 kênh, định dạng 212), đọc được tín hiệu và nhãn
    try:
        if os.path.getsize(os.path.join(d, r + '.dat')) != 5400000: return False
        wfdb.rdheader(os.path.join(d, r)); wfdb.rdann(os.path.join(d, r), 'atr'); return True
    except Exception:
        return False

ALL_EDB = sorted(wfdb.get_record_list('edb'))          # danh sách chính thức từ PhysioNet
assert len(ALL_EDB) == 90
DRIVE_EDB = os.path.join(EXT_DIR, 'edb')
for db in EXT_DBS:
    os.makedirs(os.path.join(LOCAL, db), exist_ok=True); dl = os.path.join(LOCAL, db)
    ok_drive = [r for r in ALL_EDB if os.path.exists(os.path.join(DRIVE_EDB, r + '.hea')) and usable(DRIVE_EDB, r)]
    for r in ok_drive:
        if not usable(dl, r):
            for e in ('.hea', '.dat', '.atr'): shutil.copy(os.path.join(DRIVE_EDB, r + e), dl)
    missing = [r for r in ALL_EDB if not usable(dl, r)]
    print(f'Dùng lại {len(ok_drive)} bản ghi trên Drive; cần tải {len(missing)}: {missing}')
    if missing:
        wfdb.dl_database(db, dl_dir=dl, records=missing)
        bad = [r for r in missing if not usable(dl, r)]
        assert not bad, f'Tải chưa đủ: {bad} -> chạy lại ô này'
        os.makedirs(DRIVE_EDB, exist_ok=True)
        for r in missing:                       # lưu lên Drive để lần sau không tải lại
            for e in ('.hea', '.dat', '.atr'): shutil.copy(os.path.join(dl, r + e), DRIVE_EDB)

for db in EXT_DBS:
    d = find_dir(LOCAL) if os.path.exists(LOCAL) else None""")
setc(8, c8)
c = nb['cells'][-1]; s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
c['source'] = rep(s, "notebook='notebooks/ecg_blind_edb_20261026b.ipynb'", "notebook='notebooks/ecg_blind_edb_20261026c.ipynb'")
json.dump(nb, open(DST, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok')
