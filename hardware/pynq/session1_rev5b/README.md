# 05_board/ecg_f2a_rev5b_20261025 — đo trên KV260 cho cấu hình rev5b

Cấu hình rev5b: chọn trên DS1 không rò rỉ, thêm bước học vào lưới chọn kèm điều kiện ổn định số học.

Chép các tệp dưới đây vào thư mục `F2A_BOARD_RUN` trên board, rồi chạy `kv260_f2a_rev5b_20261025.ipynb`.

- `rev4_methods.py`, `pa_adapt.py`: ghi đè bản trên board. Hai tệp này giống bản rev5.
- `rev5b_board_ref_seed3_20261025.npz` và `rev5b_board_methods_20261025.json`: tham chiếu cho 6 cấu hình.
  - **Hai cơ chế chính:**
    - `GA_R`: β 0.1, cap 1, ρ 0.5, bước học 0.05, 300 vòng, không cổng;
    - `offset_int_tau20`.
  - **Các cấu hình khác:**
    - `GA_R_gate`: GA-R có cổng, dùng cho ablation và so chi phí của cổng;
    - `gradient_anchor`: β 0.3, bước học 0.2;
    - `head_scratch`: bước học 0.05, 50 vòng;
    - `head_global`: bước học 0.2, 300 vòng.
- Kết quả: `f2a_rev5b_board_20261025.json`. Gửi lại tệp này.

Trước khi đo:
- tắt cập nhật tự động: `sudo systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer`;
- đóng mọi notebook khác.
