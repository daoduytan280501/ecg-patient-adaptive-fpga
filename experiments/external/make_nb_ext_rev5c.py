"""Sinh notebook ngoai kiem rev5c (29/10): them head_scratch_fair (baseline cong bang, chon tren DS1 o E13) vao INCART/SVDB,
tinh them so ban ghi bi hai theo benh nhan cho moi phuong phap, va doi chieu cac phuong phap cu voi ket qua rev5b (phai trung)."""
import json, sys
SRC, DST = sys.argv[1], sys.argv[2]
nb = json.load(open(SRC, encoding='utf-8'))
S = lambda i: nb['cells'][i]['source'] if isinstance(nb['cells'][i]['source'], str) else ''.join(nb['cells'][i]['source'])
def setc(i, s): nb['cells'][i]['source'] = s
def rep(s, a, b):
    assert a in s, a[:60]; return s.replace(a, b)
setc(0, """# Ngoại kiểm rev5c (29/10/2026) — INCART + SVDB, thêm head_scratch công bằng
Giống hệt ngoại kiểm rev5b (25/10), chỉ thêm `head_scratch_fair` (η 0,05, 300 vòng, cap 1, λ 0, fix; chọn trên DS1 ở E13, protocol 26/10)
và bảng số bản ghi bị hại theo bệnh nhân. Hai CSDL này **không còn mù** — chạy chỉ để bảng đầy đủ, không chọn gì.
- Emb lấy từ cache trên Drive → chỉ vài phút.
- Kiểm tra: các phương pháp cũ phải cho **đúng** số của rev5b (ô cuối tự đối chiếu).
- Kết quả: `planA/06_eval/external_rev5c_20261029/`.""")
c2 = S(2)
c2 = rep(c2, "OUT      = os.path.join(REPO, '06_eval', 'external_rev5b_20261025')                     # KẾT QUẢ MỚI -> kho",
         "OUT      = os.path.join(REPO, '06_eval', 'external_rev5c_20261029')                     # KẾT QUẢ MỚI -> kho\nOLD5B    = os.path.join(REPO, '06_eval', 'external_rev5b_20261025')")
c2 = rep(c2, "'ds1_selected_rev5b_20261025.json': SEL_FILE, 'rev4_methods.py': os.path.join(REV4, 'src', 'rev4_methods.py')}",
         "'ds1_selected_rev5b_20261025.json': SEL_FILE, 'rev4_methods.py': os.path.join(REV4, 'src', 'rev4_methods.py'),\n"
         "          'fair_methods.py': os.path.join(REV4, 'src', 'fair_methods.py'), 'e13_selected.json': os.path.join(REV4, 'out', 'bosung_20261026', 'e13_selected.json')}")
c2 = rep(c2, "import rev4_methods as M\nSEL = json.load(open(SEL_FILE))['selected']",
         "import rev4_methods as M, fair_methods  # noqa: F401\nSEL = json.load(open(SEL_FILE))['selected']\nE13 = json.load(open(INPUTS['e13_selected.json']))")
c2 = rep(c2, "CONF = {'none': ('none', {}), 'head_scratch': ('head_scratch', SEL['head_scratch']['hp']),",
         "CONF = {'none': ('none', {}), 'head_scratch': ('head_scratch', SEL['head_scratch']['hp']), 'head_scratch_fair': ('head_scratch_fair', E13['S_rob']['hp']),")
setc(2, c2)
for i in range(3, len(nb['cells'])):
    if nb['cells'][i]['cell_type'] == 'code':
        setc(i, S(i).replace('ext_rev5b_', 'ext_rev5c_').replace('tab_ext_rev5b', 'tab_ext_rev5c')
             .replace("notebooks/ecg_ext_validation_rev5b_20261025.ipynb", "notebooks/ecg_ext_validation_rev5c_20261029.ipynb"))
extra = r"""
# Số bản ghi bị hại theo bệnh nhân (balanced accuracy thấp hơn none quá 0.05, trung bình 5 seed) + đối chiếu với rev5b
rows = []
for db in EXT:
    per = pd.read_csv(os.path.join(OUT, f'ext_rev5c_{db}_per_record.csv'), index_col=0)
    for m in METHODS[1:]:
        d = per[f'{m}|balacc'] - per['none|balacc']
        rows.append(dict(db=NAMES[db], method=m, n_records=int(d.notna().sum()), better=int((d > 1e-9).sum()), worse=int((d < -1e-9).sum()),
                         harmed_005=int((d < -0.05).sum()), worst=float(d.min()), mean_balacc=float(per[f'{m}|balacc'].mean())))
harm = pd.DataFrame(rows); harm.to_csv(os.path.join(OUT, 'ext_rev5c_harm.csv'), index=False); display(harm)
old = pd.read_csv(os.path.join(OLD5B, 'ext_rev5b_summary_all.csv')); new = pd.read_csv(os.path.join(OUT, 'ext_rev5c_summary_all.csv'))
chk = old.merge(new, on=[old.columns[0], 'method'], suffixes=('_5b', '_5c'))
bad = [(r[old.columns[0]], r.method, c) for _, r in chk.iterrows() for c in ['acc', 'macroF1', 'jk', 'Se_S', 'PP_S', 'Se_V', 'PP_V'] if r[c + '_5b'] != r[c + '_5c']]
print(f'Đối chiếu rev5b: {len(chk)} dòng, {len(bad)} ô khác', bad[:5])
assert not bad, 'Phương pháp cũ cho số khác rev5b -> kiểm tra lại'
print('==> Các phương pháp cũ trùng rev5b. head_scratch_fair:'); display(new[new.method == 'head_scratch_fair'])
"""
nb['cells'].insert(len(nb['cells']) - 1, dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=extra.strip('\n')))
for c in nb['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
json.dump(nb, open(DST, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('ok', len(nb['cells']))
