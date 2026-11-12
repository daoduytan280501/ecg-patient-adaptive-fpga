# 05_board/ecg_f2a_rev5c_20261029 — đo head_scratch công bằng trên KV260 (29/10/2026)

Mục đích: head_scratch công bằng (`head_scratch_fair`: η 0.05, 300 vòng, cap 1, λ 0, fix; chọn trên DS1 ở E13) đã thành baseline chính, nên cần số đo thật thay cho ước lượng ~61 mJ. Đo lại GA-R, offset_int, head_scratch trong **cùng phiên** để so năng lượng công bằng.

Chép các tệp sau vào thư mục `F2A_BOARD_RUN` trên board, rồi chạy `kv260_f2a_rev5c_20261029.ipynb`:
- `pa_adapt.py`, `rev4_methods.py` (giống bản trên board), **`fair_methods.py` (mới)**
- `rev5c_board_ref_seed3_20261029.npz`, `rev5c_board_methods_20261029.json` (tham chiếu số nguyên; 3 cấu hình cũ trùng từng nhãn với tham chiếu rev5b đã khớp board)

Trước khi đo: `sudo systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer`; đóng mọi notebook khác.

Kết quả cần gửi lại: `f2a_rev5c_board_20261029.json`. Mong đợi: 4/4 cấu hình lệch 0 nhãn.
