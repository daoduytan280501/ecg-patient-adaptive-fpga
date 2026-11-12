# Patient-adaptive ECG classification on an FPGA–ARM SoC

Code, locked protocol and result files for the article

> D.-T. Dao, T. V. Le, *Robust and Energy-Efficient Patient-Adaptive ECG Classification Through FPGA–ARM Co-Design*, submitted to Biomedical Signal Processing and Control, 2026.

A bit-exact 8-bit dual-branch 1D-CNN accelerator runs on the programmable logic of a Kria KV260. Only its output layer
(128 int8 weights and four 32-bit biases, 144 bytes) is writable over AXI4-Lite. The ARM Cortex-A53 computes a patient-specific
update from the labelled beats of the first five minutes of each recording with one of two rules:

- **GA-R**: anchor-regularised gradient update with capped class weights, a trust region and a scale-invariant int8 mapping (updates all 144 bytes of parameters);
- **offset_int**: integer bias offset computed from beat counts with a Mitchell logarithm (changes only the four biases, 16 bytes).

All results are retrospective, on annotated PhysioNet recordings.

## Repository layout

| Folder | Content |
|---|---|
| `adaptation/` | Adaptation rules and baselines (`pa_adapt.py`, `rev4_methods.py`, `fair_methods.py`), bit-exact integer reference model (`hw_ref.py`), data loading and integer preprocessing |
| `models/` | Global model, five seeds (`global_f2/`), and the 15 fold models of the nested cross-validation on DS1 (`nested_folds/`) |
| `training/` | Training and hardware-export notebooks of the global model |
| `experiments/` | Scripts of every experiment: DS1 selection, DS2 evaluation, noise and ablation, external transfer, blind test, supplementary checks |
| `hardware/` | HLS sources, Vivado outputs (bitstream, `.hwh`, reports) and PYNQ notebooks for the KV260 |
| `protocol/` | Locked protocol, its SHA-256 file and the locked selection files |
| `results/` | CSV/JSON result files behind every table and figure |
| `paper/` | Figure script |
| `data/` | Download script for the PhysioNet databases (data are not redistributed) |
| `large_files_for_zenodo/` | Large derived files (fold data, DS2 data, integer embeddings), archived on Zenodo |

The scripts contain the absolute paths of the authors' environment; `PATHS.md` maps them to this repository.

## Checking the locked protocol (Table S12)

```bash
sha256sum protocol/protocol_bosung_20261026.md adaptation/fair_methods.py \
          protocol/locked/ds1_selected_rev5b_20261025.json protocol/locked/e13_selected.json \
          large_files_for_zenodo/D1_nested.pkl
```

| File | SHA-256 |
|---|---|
| `protocol_bosung_20261026.md` | `6cf67960bb0ae0fd96832694680faab88b279d8854e886d06731865f59462bcc` |
| `fair_methods.py` | `406117c93a742f9a34874c896542f0c18e617a806cdc31dc670f42330e0f3d54` |
| `ds1_selected_rev5b_20261025.json` | `4a5abf2e211522cafabee815b136fc25265bbde1c999a0a2218e8f6750fbdc2c` |
| `D1_nested.pkl` | `43c722a16f307a93a022fe8a3ad1a3a7f8c62d4e007cb2b461be8e0fc5ecabab` |
| `e13_selected.json` | `b340f613251c5403067ddb380b57746ad18e46f963de13abc0341d43f0d6a18b` |

`SHA256SUMS.txt` lists the digests of the files in this release. Regenerate it after adding the hardware files (`collect_from_H.ps1`).

## Where each result comes from

| Paper item | Script | Result file |
|---|---|---|
| Nested CV on DS1, fold models | `experiments/selection/nested_train.py`, `build_d1_nested.py` | `models/nested_folds/`, `results/paper_tables/nested_train_log.txt` |
| Hyperparameter selection by S_rob (Sec. 4.1) | `experiments/selection/run_ds1_select_v2.py`, `gate_compare_v2.py` | `results/paper_tables/ds1_selection_rev5b_20261025.csv`, `protocol/locked/ds1_selected_rev5b_20261025.json` |
| Fair baseline selection (Table S2) | `experiments/selection/run_grid_ds1.py`, `e13_select.py` | `results/supplementary_experiments/ds1_e13_head_scratch_fair.csv`, `protocol/locked/e13_selected.json` |
| Table 4, Fig. 4 (board, seed 3) | `hardware/pynq/session1_rev5b/`, `session2_rev5c/`, `experiments/ds2_evaluation/report_cm_rev5b.py` | `results/board/f2a_rev5b_board_20261025.json`, `f2a_rev5c_board_20261029.json`, `cm_seed3_rev5b.json` |
| Table 5, Table 6 (DS2, five seeds, Wilcoxon) | `experiments/ds2_evaluation/paper_tables_rev5b.py`, `e13_eval_ds2.py` | `results/paper_tables/main_rows.csv`, `main_summary.csv`, `wilcoxon.csv`, `results/supplementary_experiments/e13_*.csv` |
| Full-network baselines in Table 5 | `experiments/baselines_fullnet/ecg_pa_exp4_v2.ipynb` | `results/baselines_exp4/` |
| Table 7 (update energy) | `hardware/pynq/session*/` | `results/board/*.json` (key `energy`) |
| Table 8, Fig. 5 (label noise) | `experiments/ds2_evaluation/paper_tables_rev5b.py`, `e13_eval_ds2.py` | `results/paper_tables/noise.csv`, `results/supplementary_experiments/e13_noise.csv` |
| Table 9, Fig. 6 (record-level deterioration) | `experiments/ds2_evaluation/rec_delta.py`, `rec222_bosung.py`, `e14_tradeoff.py` | `results/paper_tables/e8_record_*`, `results/supplementary_experiments/e14_record_scores.csv`, `rec222.json` |
| Anchor ablation on DS2 (Sec. 5.6) | `experiments/noise_and_ablation/abl_noanchor_20261109.py` | `results/supplementary_experiments/abl_noanchor_20261109.json` |
| Fig. 7, Table 10 (Pareto, decision table) | `paper/make_figs_bspc.py` | board and DS2 result files above |
| Table 11 (INCART, SVDB) | `experiments/external/make_nb_ext_rev5c.py` → notebook | `results/external_rev5c/` |
| Table 12 (blind test, European ST-T) | `experiments/blind_edb/run_blind.py`, notebook | `results/blind_edb/` (`run_log.json` holds the run time and input digests) |
| Fig. 8 (DS1 vs DS2 grid) | `experiments/selection/e15c_grid_ds2.py`, `e15c_summary.py` | `results/supplementary_experiments/e15c_*` |
| Table S1 (inference latency/energy) | `hardware/pynq/kv260_f2a_inference_perf.ipynb` | `hardware/pynq/f2a_inference_perf.json` |
| Table S3 (GA-R vs offset_int) | `experiments/ds2_evaluation/e14_tradeoff.py` | `results/supplementary_experiments/e14_wilcoxon_GAR_vs_offset.csv` |
| Table S4 (ablation) | `experiments/ds2_evaluation/paper_tables_rev5b.py` | `results/paper_tables/ablation.csv`, `rescale.csv` |
| Table S5 (numerical stability) | `experiments/selection/numstab_rev5b.py` | `results/paper_tables/e12_numstab_rev5b_20261025.csv` |
| Table S7 (sensitivity of fixed constants) | `experiments/selection/run_grid_ds1.py` | `results/supplementary_experiments/ds1_e15b.csv` |
| Table S10 (blind test per subject, post hoc) | `experiments/blind_edb/blind_edb_subject_posthoc_20261109.py` | `results/supplementary_experiments/blind_edb_subject_posthoc_20261109.json` |
| Table S11 (anchor check on DS1) | `experiments/supplementary_checks/ds1_anchor_ablation_20261111.py` | `results/supplementary_experiments/ds1_anchor_ablation_20261111.json` |
| Causal-calibration check (Discussion) | `experiments/supplementary_checks/causal_cal_check_20261111.py` | `results/supplementary_experiments/causal_cal_check_20261111.json` |

Files with the suffixes `_nested_eta*`, `gate_compare_old` or `rev5` (without `b`) are earlier runs kept for traceability; the paper uses the `rev5b`, `e13`–`e15c` and later files.

## Running

1. `pip install -r requirements.txt`
2. `python data/download_physionet.py physionet`
3. Download the large files from Zenodo into `large_files_for_zenodo/` and check them against `SHA256SUMS.txt`.
4. Set `PYTHONPATH` to `adaptation/` and edit the paths at the top of the script you run (`PATHS.md`).

Training and the external/blind notebooks need TensorFlow; the adaptation rules and statistics need only NumPy, SciPy and pandas.
The board notebooks run on a Kria KV260 with PYNQ and the files in `hardware/vivado/`.

## License and citation

Code: MIT (see `LICENSE`). Please cite the article (see `CITATION.cff`).
The PhysioNet databases are subject to their own licences.
