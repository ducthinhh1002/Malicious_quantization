# LCF v3: thực hiện đề xuất 1 — hiệu chỉnh gain và kiểm tra reader trước

Ngày triển khai: 18/10/2026.

## Căn cứ

Audit checkpoint spatial v2 trên 4 prompt train × 2 key cho PSNR trung bình
64,3 / 48,3 / 38,6 dB và LPIPS 0,000011 / 0,000664 / 0,00604 ở gain 1×/16×/64×.
Ở 1×, 99,78% giá trị RGB thay đổi chỉ chênh một mức 8-bit. Reader cũ vẫn chưa
đọc được dấu khi chỉ tăng gain. Đây là căn cứ hiệu chỉnh tín hiệu và học lại
reader, không phải bằng chứng watermark đã mạnh.

Nguồn: [phân tích audit](LCF_Signal_Path_Findings_VI.md),
[dữ liệu](../server_review/lcf_v2_deployment/signal_audit.json).

## Đã sửa code

Carrier mới dùng hướng chuẩn hóa RMS và gain dương độc lập:

\[
u_k=\tanh(B_k),\qquad
\delta h_k=r_h\,\exp(\gamma)\,
\frac{u_k}{\sqrt{\operatorname{mean}(u_k^2)+\epsilon}}.
\]

Code dùng clamp tối thiểu trên mean-square để tránh chia 0. Gain có giới hạn
số học rộng, được tối ưu riêng, không bị weight decay. Một key vẫn tạo cùng
residual xuyên prompt. Trạng thái gain được lưu trong checkpoint; old arms giữ
carrier cũ để so sánh. `max_strength` ở calibrated arm chỉ giữ switch tương
thích cho zero/amplitude audit, không còn là chặn RMS học được như trước.

### A. Hiệu chỉnh gain bằng TRAIN quality

- 8 prompt train × 4 key train; thử 1/4/16/64 lần gain ban đầu 0,07.
- Chọn gain lớn nhất trong grid đạt PSNR p05 ≥35 dB, LPIPS mean ≤0,02,
  LPIPS p95 ≤0,04, mean CLIP drop ≤0,02.
- Không dùng score của reader, prompt validation/test hoặc kết quả student.
- Các ràng buộc quality/detection trên validation của teacher vẫn cần đạt sau đó.
- Không có gain đạt → lưu kết quả và dừng; không âm thầm nới điều kiện.

### B. Reader mới phải học được tập nhỏ

- Khởi tạo mới reader; carrier và gain đã hiệu chỉnh được giữ cố định.
- Bank trong RAM gồm ảnh RGB8 thật của 8 prompt × 2 key đối nhau.
- Reader được giám sát riêng ở độ phân giải gốc và ảnh resize; chỉ inference
  trên ảnh marked, không cần ảnh sạch ghép cặp để đọc dấu.
- Tối đa 800 update; kiểm tra fit mỗi 100 update.
- Mỗi key và mỗi view cần AUROC train ≥0,98, bit accuracy ≥0,95,
  marked-clean gain >0 và correct-wrong-key margin >0.
- Không đạt → `reader_fit_failed`, không chạy joint teacher hoặc student dài.
- Fit đạt không chứng minh generalization, khả năng mang 8 bit độc lập hay transfer.

### C. Mở rộng theo thứ tự

Nếu fit đạt, mở rộng prompt với hai key cũ trong 50 update ở gain-gate
(100 ở pilot), rồi mở rộng lên tập key train đầy đủ. Cả hướng và gain được
học bằng owner objective. Detector đóng băng ở meta stage. Ordinary student
vẫn chỉ học denoising trên ảnh/caption; không nhận key, adapter teacher hoặc
watermark loss.

## Lệnh chạy

```bash
# Mặc định arm calibrated; chẩn đoán 512px, không chạy student.
bash run_learning_channel_gate.sh --profile gain-gate

# Sau khi kiểm tra teacher đủ mạnh: pilot đầy đủ với các gate.
bash run_learning_channel_gate.sh --profile pilot --ablation calibrated
```

Đọc `gain_calibration.csv`, `gain_selection.json`, `reader_fit.csv`,
`reader_fit_gate.json`, `teacher_validation.csv`, `status.json`.
Checkpoint bootstrap nằm ngoài folder báo cáo, không phải teacher đủ điều kiện.

## Triển khai

Snapshot riêng trên vm4: `/root/logs/lcf_gain_20261018_021834Z`.
Tmux: `lcf_gain_20261018_021834Z`.
Smoke toàn luồng chạy trước, rồi mới đến gain-gate 512px.
[Deployment record](../server_review/lcf_gain_deployment/deployment.json).

16 numerical/protocol tests đã pass local và server. Thành công runtime,
teacher detection và student inheritance phải được báo cáo riêng; chưa có
kết quả khoa học từ gain-gate tại thời điểm triển khai.
