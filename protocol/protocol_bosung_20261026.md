# Protocol thí nghiệm bổ sung theo 3 lưu ý của thầy (26/10/2026)

Tạp chí mục tiêu: Biomedical Signal Processing and Control (Elsevier), thầy đã duyệt ngày 26/10/2026.

Ba lưu ý của thầy:
1. Kiểm tra lại tính công bằng của baseline head_scratch.
2. Giải thích đầy đủ sự đánh đổi giữa GA-R và offset_int.
3. Bảo đảm việc chọn tham số không dựa hậu nghiệm vào DS2.

Protocol này được viết và băm SHA-256 **trước khi** chạy bất kỳ thí nghiệm nào dưới đây. Mọi sai khác so với protocol sẽ được ghi vào mục "Sai khác" ở cuối tệp kết quả.

## 0. Những gì được giữ cố định (không đổi trong mọi thí nghiệm)
- **Mô hình và đánh giá:** mô hình F2 (5 seed), DS1/DS2 de Chazal, cửa sổ hiệu chỉnh 5 phút đầu, đánh giá sau phút 5, chỉ số jκ, macro-F1 và balanced accuracy theo bản ghi.
- **Mô hình fold DS1:** 15 mô hình fold lồng nhau (`out/nested_folds/`, `out/D1_nested.pkl`).
- **Tiêu chí chọn:** S_rob, là trung bình macro-F1 trên DS1 của ba điều kiện: sạch, đổi ngẫu nhiên 10 %, nhầm cặp 10 % (rep 0).
- **Tiêu chí ổn định:** nhiễu tương đối 1e-12 lên emb và neo; số nhãn đổi không quá 0.05 % số nhịp đánh giá DS1.
- **Cấu hình đã chọn trên DS1 (không chọn lại):** `out/ds1_selected_rev5b_20261025.json`.
  - GA-R: GA-S, β 0.1, cap 1, ρ 0.5, η 0.05, T 300, λ 0.01.
  - offset_int: τ 20.
  - gradient_anchor: β 0.3, η 0.2.
  - head_scratch (gốc): η 0.05, 50 vòng.
  - head_global: η 0.2, 300 vòng.

## E13. head_scratch công bằng (lưu ý 1)
**Định nghĩa** (`src/fair_methods.py`, hàm `head_scratch_fair`): có mọi công cụ của GA-R, trừ vector neo và khởi tạo từ lớp toàn cục.
- Khởi tạo 0.
- Trọng số lớp có cap.
- Chuẩn hóa s = 1/std(E).
- Suy giảm trọng số λ về 0.
- Ánh xạ bất biến thang α vào int8.
- Hai cách xử lý lớp vắng mặt:
  - `fix`: bias cố định −10³, như head_scratch gốc.
  - `global`: cột trọng số và bias của lớp vắng mặt giữ nguyên như lớp toàn cục, không học.

**Lưới** (216 cấu hình, bằng số cấu hình của GA-R):
- η ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
- T ∈ {50, 150, 300}
- cap ∈ {1, 3, 10}
- λ ∈ {0, 0.01}
- absent ∈ {fix, global}

**Chọn:** cùng tiêu chí S_rob và cùng tiêu chí ổn định như GA-R, chỉ trên DS1.

**Lựa chọn phụ (báo cáo, không thay lựa chọn chính):** cấu hình có macro-F1 sạch cao nhất trên DS1 (trong các cấu hình ổn định), tính cho cả head_scratch_fair và GA-R. Mục đích là kiểm tra xem kết luận có phụ thuộc vào việc dùng S_rob hay không.

**Đánh giá DS2** (chỉ sau khi đã chọn):
- 5 seed, nhãn sạch.
- 9 điều kiện nhiễu (bỏ sót / nhầm cặp / đổi ngẫu nhiên × 5/10/20 %, 3 lần lặp), giống bảng nhiễu hiện có.
- Số bản ghi bị hại (>0.05).
- Se lớp vắng mặt.
- Wilcoxon theo bản ghi so với none và so với GA-R.
- MAC mỗi bản ghi.

**Quyết định đặt trước:** head_scratch_fair (cấu hình chọn theo S_rob) thay head_scratch gốc làm baseline "chỉ lớp cuối, không neo" chính trong bài, **bất kể kết quả**. head_scratch gốc giữ lại như một dòng tham chiếu.

**Kiểm tra cài đặt đã thực hiện trước khi viết protocol (ghi nhận để minh bạch):** em chạy 3 cấu hình trên DS2 seed 3, chỉ để xác nhận rằng `head_scratch_fair(fix, cap 10, λ 0, η 0.05, T 50)` cho kết quả trùng head_scratch gốc.
- Kết quả: jκ 0.8907 / 0.8907, trùng nhau.
- Hai cấu hình khác: (global, cap 1, λ 0.01, η 0.05, T 300) jκ 0.8922; (fix, cap 1, λ 0, η 0.05, T 300) jκ 0.8903.

Các số này không được dùng cho việc chọn. Việc chọn chỉ dùng DS1.

## E14. Đánh đổi GA-R và offset_int (lưu ý 2)
Không chọn tham số. Dùng cấu hình đã khóa ở mục 0.
- **Wilcoxon trực tiếp GA-R với offset_int:** theo bản ghi (balanced accuracy, trung bình 5 seed), nhãn sạch và từng điều kiện nhiễu (trung bình 3 lần lặp); Holm trong họ 10 phép (1 sạch + 9 nhiễu). Báo cáo số bản ghi mà mỗi phương pháp thắng.
- **Bảng quyết định nhiều chiều:** hiệu năng sạch, độ bền, số bản ghi bị hại, lớp vắng mặt, ngân sách nhãn, chi phí đo trên board (thời gian, năng lượng, bộ nhớ, số tham số), khả năng chuyển vào PL, dữ liệu ngoài.
- **Hình Pareto:** năng lượng theo jκ, và độ bền (jκ trung bình 9 điều kiện nhiễu) theo jκ sạch.
- **Hướng dẫn chọn khi triển khai:** viết từ các số trên, không thêm tham số mới.

## E15. Kiểm toán hậu nghiệm và độ nhạy (lưu ý 3)
**(a) Bảng kiểm toán quyết định.** Liệt kê mọi lựa chọn thiết kế và tham số: thời điểm quyết định, dữ liệu đã nhìn thấy lúc đó (DS1 / DS2 / dữ liệu ngoài), và cách lựa chọn được xác nhận. Bảng này được đưa vào phụ lục bài, thừa nhận lịch sử phát triển.

**(b) Độ nhạy trên DS1 với các hằng số cố định của GA-R** (chỉ báo cáo; hằng số **không** đổi dù kết quả thế nào):
- η ∈ {0.02, 0.05, 0.1} × T ∈ {100, 300, 600} × λ ∈ {0, 0.01, 0.1}, với β 0.1, cap 1, ρ 0.5. Tổng 27 cấu hình.
- Số neo mỗi lớp m ∈ {5, 10, 25}, lấy m vector đầu của bộ neo hiện có, với cấu hình GA-R đã chọn.
- Báo cáo S_rob và tiêu chí ổn định. Câu hỏi cần trả lời: T = 300, λ = 0.01, m = 25 có nằm trong vùng gần tối ưu của DS1 không.
- Cửa sổ 5 phút: theo khuyến nghị AAMI EC57/ANSI cho việc dùng dữ liệu riêng của bệnh nhân. Ghi rõ là quy ước của lĩnh vực, không phải tham số được chọn.

**(c) Độ nhạy trên DS2, chỉ để báo cáo, không dùng để chọn:**
- Đánh giá **toàn bộ** 216 cấu hình GA-R và 216 cấu hình head_scratch_fair trên DS2: 5 seed; sạch, đổi ngẫu nhiên 10 %, nhầm cặp 10 %; rep 0. Tính S_rob trên DS2 và jκ sạch.
- Báo cáo:
  - tương quan Spearman giữa S_rob DS1 và S_rob DS2;
  - thứ hạng trên DS2 của cấu hình được chọn;
  - khoảng cách "oracle", tức chênh lệch giữa cấu hình tốt nhất nếu chọn trên DS2 và cấu hình được chọn trên DS1;
  - kết quả DS2 của 10 cấu hình xếp đầu DS1.
- Kết quả DS2 của η 0.02 và 0.1 (cùng β, cap, ρ với GA-R) được báo cáo riêng.

## E16. Kiểm tra mù trên CSDL chưa từng dùng (lưu ý 3)
- **CSDL:** European ST-T Database (PhysioNet `edb`): 90 bản ghi, 79 người, mỗi bản ghi 2 giờ, 250 Hz, nhãn từng nhịp theo chuẩn MIT-BIH. Chưa từng được dùng trong dự án ở bất kỳ phiên bản nào.
- **Chọn chuyển đạo (đặt trước):** theo thứ tự ưu tiên MLII, MLIII, MLI, rồi bất kỳ tên nào bắt đầu bằng "ML"; nếu không có thì dùng tín hiệu 0. Ghi lại chuyển đạo dùng cho từng bản ghi.
- **Tiền xử lý:** giống hệt pipeline ngoại kiểm hiện có: nội suy về 360 Hz, lọc, cửa sổ nhịp, 8 đặc trưng RR, AAMI, bỏ nhịp Q, pipeline số nguyên bit-exact. Bộ kiểm tra an toàn giữ nguyên: bản nhanh trùng hw_ref, seed 3 DS2 trùng board.
- **Phương pháp và tham số** (khóa, đọc từ tệp chọn trên DS1):
  - none
  - head_scratch (gốc)
  - head_scratch_fair (chọn ở E13)
  - head_global
  - offset_int
  - gradient_anchor
  - GA-R
- **Chỉ số:** jκ, macro-F1, Se/+P theo lớp (trung bình ± SD 5 seed); balanced accuracy theo bản ghi; Wilcoxon theo bản ghi so với none (Holm trong họ 6 phép); Wilcoxon GA-R với offset_int.
- **Giả thuyết đặt trước:**
  - H1: GA-R cải thiện balanced accuracy theo bản ghi so với none (p_Holm < 0.05).
  - H2: offset_int cải thiện so với none (p_Holm < 0.05).
  - So sánh GA-R với offset_int: không đặt hướng trước, chỉ mô tả.
- **Chạy đúng một lần.** Nếu chương trình lỗi, chỉ được sửa phần đọc dữ liệu hoặc tiền xử lý, không đụng phương pháp và tham số; mọi lần chạy ghi vào `run_log.json`. Kết quả được báo cáo đầy đủ, kể cả khi bất lợi.
- **Thứ tự bắt buộc:** E13 chọn xong và tệp chọn được băm trước khi chạy E16.

## Đầu ra
- `06_eval/phaseA_rev5_20261025/out/bosung_20261026/` (E13, E14, E15)
- `06_eval/blind_edb_20261026/` (E16)
