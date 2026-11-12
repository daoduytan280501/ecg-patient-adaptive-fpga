# Training of the global model

- `ecg_improve_v4.ipynb`: trains the dual-branch 1D-CNN (F2) on DS1 with five seeds and writes
  `F2_rr8_str_s{1..5}.keras` and `*_quant.json` (copies in `models/global_f2/`).
- `ecg_f2_hw_export.ipynb`: quantises the deployed seed (seed 3) and exports `weights_f2.h`,
  `f2_hw_config.json` and the test vectors for HLS.
- The 15 fold models of the nested cross-validation on DS1 are trained by
  `experiments/selection/nested_train.py` (copies in `models/nested_folds/`).

Both notebooks were run on Google Colab. If this folder is empty, run `collect_from_H.ps1`.
