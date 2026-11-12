# ============================================================
# build_kv260_f2.tcl  -  Vivado 2022.2 (IP model F2)
# Tu dong: tao project -> block design (Zynq MPSoC + ecg_f2_top)
#          -> synth -> impl -> bitstream -> copy .bit + .hwh
# Cach chay: mo Vivado 2022.2 -> Tcl Console, go:
#   cd G:/1DCNN/ECGCNN/VIVADO_F2
#   source build_kv260_f2.tcl
# Tao project MOI ecg_kv260_f2 (giu nguyen project cu ecg_kv260)
# ============================================================

# ----------- CHINH 2 DUONG DAN NAY NEU KHAC -----------
set ip_repo  "G:/1DCNN/ECGCNN/ECG_F2/ECC_F2/solution1/impl/ip"
set work_dir [pwd]
# ------------------------------------------------------

set proj_name ecg_kv260_f2
set bd_name   design_1

create_project $proj_name $work_dir/$proj_name -part xck26-sfvc784-2LV-c -force

# ---- Board file KV260 (neu Vivado co san) ----
set som [lindex [lsort [get_board_parts -quiet *kv260_som*]] end]
if {$som ne ""} {
    puts "INFO: Dung board part $som"
    set_property board_part $som [current_project]
    set carrier [lindex [lsort [get_board_parts -quiet *kv260_carrier*]] end]
    if {$carrier ne ""} {
        set_property board_connections [list som240_1_connector $carrier:som240_1_connector] [current_project]
    }
} else {
    puts "WARNING: Khong tim thay board file KV260, dung part xck26 va cau hinh PS thu cong"
}

# ---- IP tu Vitis HLS ----
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog -rebuild
set ecg_vlnv [lindex [get_ipdefs -all *ecg_f2_top*] 0]
if {$ecg_vlnv eq ""} {
    error "Khong tim thay IP ecg_f2_top trong $ip_repo - kiem tra da Export RTL chua"
}
puts "INFO: IP = $ecg_vlnv"

# ---- Block design ----
create_bd_design $bd_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
if {$som ne ""} {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
        -config {apply_board_preset "1"} $ps
}
# Chi giu 1 cong AXI master HPM0_FPD, clock PL0 = 100 MHz
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
] $ps

create_bd_cell -type ip -vlnv $ecg_vlnv ecg_f2_top_0

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/ecg_f2_top_0/s_axi_control} \
    Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
    intc_ip {New AXI SmartConnect} master_apm {0} ddr_seg {Auto}} \
    [get_bd_intf_pins ecg_f2_top_0/s_axi_control]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# ---- Wrapper + build ----
set bd_file [get_files $bd_name.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [glob $work_dir/$proj_name/$proj_name.gen/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v]
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation/bitstream THAT BAI - xem log trong Vivado"
}

# ---- Copy file cho PYNQ (ten .bit va .hwh phai giong nhau) ----
set out $work_dir/overlay_f2
file mkdir $out
file copy -force $work_dir/$proj_name/$proj_name.runs/impl_1/${bd_name}_wrapper.bit $out/ecg_f2.bit
file copy -force $work_dir/$proj_name/$proj_name.gen/sources_1/bd/$bd_name/hw_handoff/$bd_name.hwh $out/ecg_f2.hwh

# Bao cao tai nguyen va timing sau place & route (dung cho bao cao)
open_run impl_1
report_utilization    -file $out/utilization_impl.txt
report_timing_summary -file $out/timing_impl.txt
report_power          -file $out/power_impl.txt

# Header dia chi thanh ghi do Vitis HLS sinh ra
set hw_h [glob -nocomplain $ip_repo/drivers/*/src/*_hw.h]
if {$hw_h ne ""} { file copy -force [lindex $hw_h 0] $out/xecg_f2_top_hw.h }

puts "=============================================================="
puts " XONG. Copy thu muc $out len board KV260:"
puts "   ecg_f2.bit, ecg_f2.hwh, xecg_f2_top_hw.h"
puts "=============================================================="
