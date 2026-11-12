# Large derived files (archived on Zenodo, not in git)

| File | Size | Built by | Used by |
|---|---|---|---|
| `D1_nested.pkl` | 69 MB | `experiments/selection/build_d1_nested.py` | DS1 selection, Table S11 (SHA-256 listed in Table S12) |
| `D2.pkl` | 65 MB | `experiments/data_prep/build_data.py` | All DS2 analyses |
| `embeddings/exp4_emb_int_s{1..5}.npz` | 1 MB each | integer pipeline `adaptation/hw_ref.py` | DS2 analyses |
| `embeddings/f2a_board_data.npz` | 5 MB | `training/ecg_f2_hw_export.ipynb` | board notebooks, `rev4_data.py` |

Upload this folder as one Zenodo record and put its DOI in the main README.
Check the files with `sha256sum -c SHA256SUMS.txt` from the repository root.
