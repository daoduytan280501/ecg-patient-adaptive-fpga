# Copies the hardware sources, training notebooks and board measurement files
# that live only in the project archive on Google Drive into this repository.
# Run once in PowerShell from the repository folder:
#   cd "<path>\ecg-patient-adaptive-fpga"
#   powershell -ExecutionPolicy Bypass -File .\collect_from_H.ps1
# Change $H if the Drive letter is not H: (for example G:).

$H    = "H:\My Drive\ECG_FPGA_1DCNN\planA"
$Repo = $PSScriptRoot
$missing = @()

function Copy-One($src, $dstDir) {
    $full = Join-Path $H $src
    $dst  = Join-Path $Repo $dstDir
    if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Path $dst -Force | Out-Null }
    if (Test-Path $full) { Copy-Item $full $dst -Force; Write-Host "OK   $src" }
    else { $script:missing += $src; Write-Host "MISS $src" -ForegroundColor Yellow }
}

# --- HLS sources of the patient-adaptive accelerator (F2A) ---
foreach ($f in "ecg_f2a.cpp","ecg_f2a.h","ecg_f2a_tb.cpp","f2_hw_config.json","make_t3_golden.py","README_F2A.txt","weights_f2a.h") {
    Copy-One "03_hls\05_ecg_f2a_phaseA\src\$f" "hardware\hls\src" }
foreach ($f in "script.tcl","directives.tcl") { Copy-One "03_hls\05_ecg_f2a_phaseA\hls_project\solution1\$f" "hardware\hls\hls_project" }
Copy-One "03_hls\05_ecg_f2a_phaseA\hls_project\solution1\syn\report\ecg_f2a_top_csynth.rpt" "hardware\hls\reports"
Copy-One "03_hls\05_ecg_f2a_phaseA\hls_project\solution1\sim\report\ecg_f2a_top_cosim.rpt"   "hardware\hls\reports"
foreach ($f in "test_beats_q.txt","test_rr_q.txt","test_emb_sim.txt","test_head_adapted.txt","test_labels_sim.txt","test_labels_true.txt","test_logits_adapted.txt","test_logits_sim.txt") {
    Copy-One "03_hls\05_ecg_f2a_phaseA\tb\$f" "hardware\hls\tb" }
Copy-One "03_hls\tools\run_all_ds2_cmodel.cpp" "hardware\hls\tools"

# --- Vivado: bitstream, hardware handoff, register map, reports, build script ---
foreach ($f in "ecg_f2a.bit","ecg_f2a.hwh","xecg_f2a_top_hw.h","design_1_wrapper_utilization_placed.rpt","design_1_wrapper_timing_summary_routed.rpt","design_1_wrapper_power_routed.rpt") {
    Copy-One "04_vivado\04_ecg_f2a_phaseA\$f" "hardware\vivado" }
Copy-One "04_vivado\02_ecg_f2_seed1\build_kv260_f2.tcl" "hardware\vivado"

# --- Board: inference latency/energy (Table S1) ---
foreach ($f in "kv260_f2a_inference_perf.ipynb","f2a_inference_perf.json","PROVENANCE_inference_perf.txt","README_BOARD_F2A.txt","sim_mmio.py","f2a_adapt_energy.json") {
    Copy-One "05_board\ecg_f2a_phaseA\$f" "hardware\pynq" }

# --- Training, quantisation and hardware export (global model, five seeds) ---
foreach ($f in "ecg_improve_v4.ipynb","ecg_f2_hw_export.ipynb") { Copy-One "notebooks\$f" "training" }
foreach ($f in "f2_hw_config_seed3.json","f2_params_seed3.npz") { Copy-One "02_quant\$f" "models\quantised_seed3" }
foreach ($f in "tuning_DS1.csv","DS2_summary.csv","DS2_per_record.csv") { Copy-One "01_train\$f" "results\training" }

Write-Host ""
if ($missing.Count -eq 0) { Write-Host "All files copied." -ForegroundColor Green }
else { Write-Host "$($missing.Count) file(s) not found; check the paths above." -ForegroundColor Yellow }
