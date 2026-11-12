# Data

The ECG recordings are publicly available on PhysioNet and are **not** redistributed here.
Run `python data/download_physionet.py <dir>` to download them.

| Database | Use in the paper | PhysioNet DOI |
|---|---|---|
| MIT-BIH Arrhythmia (mitdb) | DS1 (training, hyperparameter selection) and DS2 (test) | 10.13026/C2F305 |
| INCART (incartdb) | External transfer, no retuning | 10.13026/C2V88N |
| MIT-BIH Supraventricular Arrhythmia (svdb) | External transfer, no retuning | 10.13026/C2V30W |
| European ST-T (edb) | Locked, single-run blind test | 10.13026/C2D59Z |

Derived data (integer embeddings, the DS1 fold data `D1_nested.pkl`, the DS2 data `D2.pkl`)
are too large for the repository and are archived separately (see `large_files_for_zenodo/README.md`).
