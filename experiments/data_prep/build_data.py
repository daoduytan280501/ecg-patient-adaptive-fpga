"""Dung D1.pkl, D2.pkl cho rev4. Sua 3 duong dan duoi day theo may."""
import sys, pickle, rev4_data as rd
V = '/content/drive/MyDrive/ecg_processed/improve_v4'                     # model + emb exp4
MITDB = '/content/drive/MyDrive/mitdb_extracted/mit-bih-arrhythmia-database-1.0.0'
EMB1 = '/content/drive/MyDrive/ECG_FPGA_1DCNN/planC/06_eval/exp1_20261022/cache'
OUT = sys.argv[1] if len(sys.argv) > 1 else '../out'
pickle.dump(rd.load_ds2(V), open(f'{OUT}/D2.pkl', 'wb'))
pickle.dump(rd.load_ds1(MITDB, V + '/pa_phaseA/fold_models', EMB1, V + '/F2_rr8_str_s1_quant.json'), open(f'{OUT}/D1.pkl', 'wb'))
print('xong')
