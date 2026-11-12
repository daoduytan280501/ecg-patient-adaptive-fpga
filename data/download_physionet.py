"""Download the four PhysioNet databases used in the paper (MITDB, INCART, SVDB, EDB).

Usage:  python data/download_physionet.py  [target_dir]
Requires: pip install wfdb
The databases are not redistributed in this repository.
"""
import os
import sys

import wfdb

DBS = {
    "mitdb": "MIT-BIH Arrhythmia Database (training DS1 / test DS2)",
    "incartdb": "St Petersburg INCART 12-lead Arrhythmia Database (external transfer)",
    "svdb": "MIT-BIH Supraventricular Arrhythmia Database (external transfer)",
    "edb": "European ST-T Database (locked blind test)",
}

target = sys.argv[1] if len(sys.argv) > 1 else "physionet"
for db, desc in DBS.items():
    out = os.path.join(target, db)
    os.makedirs(out, exist_ok=True)
    print(f"Downloading {db}: {desc} -> {out}")
    wfdb.dl_database(db, dl_dir=out)
print("Done.")
