# Paths used by the scripts

The scripts were run in the authors' working environment and contain absolute paths at the top of each file.
Before re-running a script, replace these paths with the repository locations below.

| Path in the scripts | Content | Location in this release |
|---|---|---|
| `/mnt/user-data/uploads/improve_v4`, `/content/drive/MyDrive/ecg_processed/improve_v4` | Global models and quantisation files | `models/global_f2/` |
| `.../improve_v4/pa_phaseA/exp4_emb_int_s*.npz`, `f2a_board_data.npz` | Integer embeddings of DS2 and board data | `large_files_for_zenodo/embeddings/` |
| `/mnt/user-data/uploads/mitdb_extracted/mit-bih-arrhythmia-database-1.0.0`, `/content/drive/MyDrive/mitdb_extracted/...` | MIT-BIH Arrhythmia Database | download with `data/download_physionet.py` |
| `/content/drive/MyDrive/ecg_external_wfdb` | INCART, SVDB and EDB records | download with `data/download_physionet.py` |
| `/home/claude/planA_rev4/out/D2.pkl` | DS2 data for the adaptation rules (built by `experiments/data_prep/build_data.py`) | `large_files_for_zenodo/D2.pkl` |
| `/home/claude/planA_rev5/out/D1_nested.pkl` | DS1 fold data (built by `experiments/selection/build_d1_nested.py`) | `large_files_for_zenodo/D1_nested.pkl` |
| `/home/claude/planA_rev5/out/nested_folds` | 15 fold models | `models/nested_folds/` |
| `/home/claude/planA_rev5/src`, `/home/claude/planA_rev4/src`, `.../planA/src` | Shared modules (`pa_adapt.py`, `rev4_methods.py`, `fair_methods.py`, `hw_ref.py`, `rev4_data.py`, `pc_pre_int.py`) | `adaptation/` (add it to `PYTHONPATH`) |
| `/home/claude/planA_rev5/out/ds1_selected_rev5b_20261025.json`, `.../bosung_20261026/e13_selected.json` | Locked selection files | `protocol/locked/` |
| `/home/claude/planA_rev5/out/...`, `.../out/paper_rev5b/` | Result files | `results/paper_tables/` |
| `/home/claude/planA_rev5/out/bosung_20261026/` | Supplementary experiment results | `results/supplementary_experiments/` |
| `/mnt/user-data/uploads/ECG_FPGA_1DCNN/planA/06_eval/...` | External, blind and baseline results | `results/external_rev5c/`, `results/blind_edb/`, `results/baselines_exp4/` |
| `/home/claude/planA_rev5/board_out/` | Board session results | `results/board/` |
| `/home/claude/bspc/figs/` (figure output) | Figures of the paper | any output folder |

Example: `export PYTHONPATH=$PWD/adaptation` (Linux) or `$env:PYTHONPATH="$PWD\adaptation"` (PowerShell).
