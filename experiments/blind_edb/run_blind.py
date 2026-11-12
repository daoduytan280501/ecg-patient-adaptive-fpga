"""Chay notebook kiem tra mu (ban b) cuc bo, tung o code theo thu tu; chi doi duong dan (protocol E16: chi sua phan doc du lieu).
argv: STOP_AFTER (so o code cuoi cung chay, mac dinh het)."""
import json, sys, os, builtins
NB = '/home/claude/planA_rev5/nb_blind/ecg_blind_edb_20261026b.ipynb'
STOP = int(sys.argv[1]) if len(sys.argv) > 1 else 99
nb = json.load(open(NB))
P = {"REPO     = '/content/drive/MyDrive/ECG_FPGA_1DCNN/planA'": "REPO     = '/home/claude/blind_run/planA'",
     "DATA_DIR = '/content/drive/MyDrive/mitdb_extracted/mit-bih-arrhythmia-database-1.0.0'": "DATA_DIR = '/mnt/user-data/uploads/mitdb_extracted/mit-bih-arrhythmia-database-1.0.0'",
     "MODEL_DIR = '/content/drive/MyDrive/ecg_processed/improve_v4'": "MODEL_DIR = '/mnt/user-data/uploads/improve_v4'",
     "EXT_DIR  = '/content/drive/MyDrive/ecg_external_wfdb'": "EXT_DIR  = '/home/claude/blind_run/ext'",
     "CACHE    = '/content/drive/MyDrive/ecg_processed/improve_v4/pa_phaseA/ext_int_cache'": "CACHE    = '/home/claude/blind_run/cache'",
     "LOCAL = '/content/edb_local'": "LOCAL = '/home/claude/blind_run/edb_local'"}
G = {'display': lambda x: print(x.to_string() if hasattr(x, 'to_string') else x), '__name__': '__main__'}
k = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    if s.startswith('!pip'): continue
    s = s.replace("notebook ecg_blind_edb_20261026b.ipynb | '", "notebook ecg_blind_edb_20261026b.ipynb, chạy cục bộ bằng run_blind.py (Colab bị ngắt nhiều lần; chỉ đổi đường dẫn) | '")
    for a, b in P.items():
        if a in s: s = s.replace(a, b)
    assert '/content' not in s or 'edb_local' in s or "os.path.join('/content'" in s, f'cell {i} con /content'
    k += 1
    print(f'\n######## o {i} ########', flush=True)
    exec(compile(s, f'cell{i}', 'exec'), G)
    if i >= STOP: break
print('DUNG sau o', i)
