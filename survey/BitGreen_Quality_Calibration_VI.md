# Cải thiện độ trung thực của ảnh BitGreen — 17/10/2026

Pilot đầu có TPR teacher 256/256 ở cả Plain và Robust, clean test 3/256. Tuy vậy,
PSNR pilot chỉ 17,27 / 16,67 dB, nên không chạy student. Đây là tín hiệu nhúng mạnh
theo detector hiện tại; chưa chứng minh dấu truyền qua LoRA, và PSNR thấp đo sai
khác so với ảnh clean cùng seed, không tự nó chứng minh ảnh nhìn xấu.

## Can thiệp vào owner

Giữ hướng trọng số đã học, giảm biên độ:

\[
\theta_{owner}(s)=\theta_{clean}+s\Delta\theta_{owner},\qquad 0<s\le1.
\]

Với adapter cộng thuần túy, thay B bằng sB cho delta BA. Không nhân cả A lẫn B.
Từ chối DoRA, bias hoặc modules_to_save vì phép nội suy đó không còn tương đương.
Sau khi chọn, merge thành UNet native và kiểm tra tương đương trước/sau merge cùng
save/reload. Khi sinh ảnh không cần truyền scale, key hay module bổ sung.

## Quy trình

1. Kiểm tra hash adapter nguồn; giữ nguyên bản nguồn.
2. Loại toàn bộ image ID và caption của mọi split cũ khỏi các split mới.
   Giữ riêng calibration clean cũ và threshold 1% FPR đã xác lập.
3. Quét s = 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0 trên 64 design prompts mới.
   Hai nhánh dùng cùng prompt/seed. Không giả định TPR hay chất lượng đơn điệu theo s.
4. Chọn mean hard-green cao nhất trong các ứng viên TPR >=90%, mean PSNR >=23 dB.
   Mức 23 tạo khoảng dự phòng 1 dB cho gate 22; không bảo đảm tổng quát hóa.
   Nếu không có ứng viên đạt, lưu rõ `design_target_met=false` và một fallback để
   chẩn đoán. Gate độc lập quyết định có chạy student.
5. Đánh giá lại trên 128 gate và 256 test prompts mới, tách khỏi design.
   Giữ gate cũ TPR >=80%, PSNR >=22; không dùng test/student để chọn strength.
6. Nếu teacher đạt, chạy ba ordinary LoRA student: clean, Plain, Robust.
   Student chỉ nhận ảnh/caption và clean SD1.4, không nhận detector/key/loss owner.
7. Xuất PSNR test và FID/KID/CLIP của teacher ngay cả khi gate không đạt.

Đây là hiệu chỉnh cường độ owner, không phải một thuật toán watermark mới.
Ít ảnh design có thể chọn nhầm do nhiễu thống kê; cần gate/test độc lập. Tất cả
kết quả một seed vẫn là pilot. FID 256 ảnh chỉ để đối chiếu cùng count/reference.
Giảm dấu để nâng PSNR có thể giảm khả năng truyền dấu; phải đo student mới biết.

## Chạy

```bash
bash run_bitgreen_quality.sh --source /path/to/completed_bitgreen_run
bash run_bitgreen_quality.sh --resume /path/to/quality_run
```

Ảnh, model và adapter: `BITMARK_RUNTIME_ROOT/bitgreen_artifacts/<run>`.
CSV/JSON/log: `BITMARK_RESULTS_ROOT/<run>`. Không dereference symlink nặng khi tải báo cáo.
