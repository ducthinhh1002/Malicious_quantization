# LCF v2: sửa khả năng học trước khi kết luận khả năng truyền dấu

Pilot v1 đã hoàn tất 2.000 bước và trượt teacher gate. Checkpoint được chọn
có TPR bằng FPR (2,34375%), score gain gần 0, PSNR khoảng 63,9 dB.
Không có ordinary student khoa học nào được chạy từ pilot này.

## Thay đổi đã triển khai

1. Warm-up reader với carrier cố định trên 16 prompt train × 4 key (100 update).
   Joint warm-up tiếp trên tập nhỏ, rồi mở rộng prompt và key. Cùng prompt/seed
   được ghép với hai key đối nhau; các bit cân bằng trong tập key train.
2. Thêm loss phân biệt cặp marked/clean. Quality loss phạt mạnh khi vượt ngân sách
   và vẫn giữ regularization nhẹ bên trong; trọng số tăng dần sau warm-up.
   Gate PSNR, LPIPS, CLIP, diversity không được nới.
3. Đo norm/cosine gradient của watermark và quality riêng, mức cập nhật carrier,
   phần pixel RGB thực sự đổi, bit accuracy. `carrier_response.json` đo đáp ứng
   với 0/1/2/4 lần cường độ ban đầu trên prompt train; không tự chọn cường độ.
4. Chọn checkpoint bằng gate, AUROC, TPR−FPR và paired gain; TPR tăng cùng FPR
   không còn đủ để thắng. Thêm báo cáo fit trên prompt/key đã thấy, tách khỏi
   validation dùng prompt/key chưa thấy.
5. Reader tùy chọn giữ lưới đặc trưng 4×4. Có ablation nhúng 8 thay vì 4 bước
   cuối DDIM; chưa mặc định rằng cửa sổ dài hơn tốt hơn.
6. Proxy validation dùng 4 episode rời nhau, mỗi episode 4 ảnh train và 4 query,
   kiểm tra ordinary learner ở 4/16 update. Xuất tương quan thứ hạng checkpoint
   giữa hai horizon. Chưa chứng minh được quan hệ với student 2.500 update.

Student cuối vẫn dùng upstream ordinary denoising LoRA. Reader đóng băng trong
meta stage; final test/learner không tham gia chọn checkpoint. Đây là module
nhúng của teacher pipeline, chưa phải fingerprint đã merge vào native weights.

## Chạy chẩn đoán

```bash
bash run_learning_channel_diagnostics.sh
```

Bốn nhánh chạy tuần tự: `legacy`, `curriculum`, `spatial`, `broad`.
Mỗi nhánh 200 joint update, ảnh 512×512, chỉ teacher. `curriculum` là một gói sửa
về huấn luyện; đây không phải ablation cô lập từng thành phần hoặc so sánh bằng
FLOPs. `spatial` và `broad` kiểm tra thêm từng lựa chọn kiến trúc/cửa sổ.

FPR diagnostic là 5% (128 mẫu calibration), khác pilot 1% (1.000 mẫu).
Không dùng TPR của diagnostic để tuyên bố kết quả ở FPR 1%. `comparison.csv`
ghi cả fit và held-out validation với nhãn riêng. Dù pass cũng không tự chạy
student. Cần kiểm tra fit, held-out discrimination và chất lượng trước khi chạy:

```bash
bash run_learning_channel_gate.sh --profile pilot --ablation spatial
```

## Cách đọc kết quả

- Fit và validation đều AUROC ≈ 0,5: reader/carrier còn chưa học được dấu.
- Fit tốt nhưng validation yếu: có vấn đề tổng quát hóa prompt/key.
- Gradient quality ngược watermark: có xung đột; cần xem cả norm và mức cập nhật.
- Gradient khác 0 nhưng đáp ứng ảnh/score gần 0: gradient thông chưa đủ hữu ích.
- Teacher tốt nhưng long student yếu: lúc đó mới quy vấn đề về kênh học truyền dấu.

Không coi việc pass syntax, unit test hoặc GPU smoke là bằng chứng watermark mạnh.

## Kiểm tra triển khai ngày 18/10/2026

12 numerical/protocol tests đã pass trên local và H200. Smoke GPU toàn luồng
teacher → meta → PNG → ordinary LoRA → evaluation đã hoàn tất; gradient riêng
của transfer loss tới carrier khác 0. Kiểm tra GPU bổ sung với 2 episode ×
2 horizon cũng pass. Xem [verification](../server_review/lcf_v2_deployment/verification.json).

Bốn nhánh diagnostic được xếp chạy tuần tự trên vm4 trong tmux
`lcf_v2_20261018_000900Z`, bắt đầu từ `spatial`. Các kiểm tra dùng smoke weights
không tham gia chọn checkpoint hoặc kết luận hiệu quả khoa học.
