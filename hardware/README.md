# Hardware

| Folder | Content | Tools |
|---|---|---|
| `hls/` | C++ sources of the patient-adaptive accelerator `ecg_f2a_top` (frozen backbone in ROM, writable output layer `head_w`/`head_b` over AXI4-Lite, read-only embedding bank `emb_out`), testbench, test vectors, csynth and co-simulation reports | Vitis HLS 2022.2, part xck26-sfvc784-2LV-c, 10 ns clock |
| `vivado/` | Bitstream `ecg_f2a.bit`, hardware handoff `ecg_f2a.hwh`, register map `xecg_f2a_top_hw.h`, post-placement utilisation, timing and power reports, block-design build script | Vivado 2022.2, PL at 100 MHz, IP at 0xA000_0000 |
| `pynq/` | PYNQ driver `lib_ecg_f2a.py`, board notebooks (session 1: `session1_rev5b`, session 2: `session2_rev5c`), update-energy notebook, inference latency/energy notebook | Kria KV260, PYNQ on Ubuntu 22.04 |

Notes
- Set `Clocks.fclk0_mhz = 100` before measuring; PYNQ defaults to 150 MHz.
- Power is read from the INA260 sensor on the K26 SoM supply; energies are incremental above idle.
- `write_head()` in `lib_ecg_f2a.py` writes the full 144-byte output layer for every rule; offset_int changes only the 16 bias bytes.
- The board labels are compared label by label with the integer reference `adaptation/hw_ref.py`.

If `hls/` or `vivado/` is empty, run `collect_from_H.ps1` (see the main README).
