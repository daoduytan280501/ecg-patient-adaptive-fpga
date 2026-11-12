############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
############################################################
open_project ECG_F2A
set_top ecg_f2a_top
add_files ecg_f2a.cpp
add_files ecg_f2a.h
add_files weights_f2a.h
add_files -tb ecg_f2a_tb.cpp
add_files -tb test_beats_q.txt
add_files -tb test_head_adapted.txt
add_files -tb test_labels_sim.txt
add_files -tb test_labels_true.txt
add_files -tb test_logits_adapted.txt
add_files -tb test_logits_sim.txt
add_files -tb test_rr_q.txt
open_solution "solution1" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}
create_clock -period 10 -name default
#source "./ECG_F2A/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -argv {10}
export_design -rtl verilog -format ip_catalog
