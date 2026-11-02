# Structure: cùng một họ, hai endpoint student

- Code đóng gói chính: [`student12500/REPRODUCE.md`](student12500/REPRODUCE.md).
- Code teacher và recipe 7.500 lịch sử: `student12500/historical`.
- Teacher chung: `../../checkpoint/structure/8bit_rank32/teacher_refine600`.
- Student: `../../checkpoint/structure/8bit_rank32/student7500` và `student12500`.
- Dữ liệu, prompt và kết quả: `../../output_artifacts/runs/structure/student7500`
  và `student12500`.

Cả hai dùng payload 8 bit, LoRA student rank 32. Teacher là Structure/margin
refine 600 bước từ checkpoint teacher k=1 bước 3.000. Bản student 12.500 dùng
lịch LR giảm sau 7.500; bản 10.000 warm restart trong lịch sử là thử nghiệm khác.

Để chuẩn bị source run cho launcher, sao chép
`../../output_artifacts/runs/structure/student7500/run` sang thư mục làm việc mới,
rồi đặt một bản sao teacher `teacher_refine600` vào `teacher_fp/refined` của
working copy. Giữ nguyên carrier, prompt, ảnh, threshold và validation cache.
Không cần đặt lại các snapshot student vào source run nếu train mới từ đầu.
Muốn resume student thì dùng checkpoint/full state tương ứng trong `checkpoint`.
Sửa đường dẫn base model trong working copy theo hướng dẫn REPRODUCE;
không sửa manifest gốc. Bộ base SD2.1/SD1.4 không được gộp vào đây.

Các bản nguồn tải riêng được giữ tại
`../../output_artifacts/provenance/source_snapshots/structure7500` và
`structure12500`. Thử nghiệm checkpoint averaging đã hủy nằm trong
`../../output_artifacts/archive/cancelled_structure_experiments`.
