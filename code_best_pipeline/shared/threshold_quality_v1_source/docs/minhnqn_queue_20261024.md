# Hai nhánh MinhNQN đã đưa vào hàng chờ VM4

Ngày triển khai: 24/10/2026. Root server:
`/root/logs/threshold_quality_20261022/minhnqn_refine_20261024`.
Supervisor lúc launch: PID `815900`.

Thứ tự toàn luồng: recovery hiện tại → `transfer_refine_20261024/quality` →
`transfer_refine_20261024/quality_transfer` → **quality_gradient** → **quality_cka**.
Supervisor mới chờ cả trạng thái completed và complete.json của hai nhánh
transfer. Nếu dependency lỗi hoặc mất tiến trình, nó ghi dependency_needs_attention
và chờ phục hồi, không tự bỏ qua kết quả thiếu hoặc khởi chạy chồng GPU.

## Cấu hình và phương pháp

Hai nhánh đều warm-start teacher kappa=1 step3000, effective scale 1.0,
refine 600 bước / LR 1e-5 / batch4 / full checkpoint mỗi 200 bước. Giữ nguyên
threshold_cfg_v1 loss, key, carrier, calibration, sampler và ảnh clean.
LPIPS AlexNet weight 0.05; mỗi auxiliary gradient bị chặn norm ở 25% gradient
của original loss trước global clipping. Đây là giới hạn bước tối ưu,
không phải quality stopping gate.

| Nhánh | Phần thêm ngoài LPIPS | Tần suất | Trọng số |
|---|---|---:|---:|
| quality_gradient | Signed delta ordinary-denoising-gradient matching tới teacher nguồn | Mỗi 10 bước | 0.05 |
| quality_cka | Centered CKA trên patch features AlexNet sạch/marked | Mỗi bước | 0.05 |

Gradient: cùng SD1.4 rank32 attention+conv probe cho ảnh sạch, ảnh teacher
nguồn và ảnh ứng viên; cùng VAE posterior noise/diffusion noise/timestep.
Warmup một bước ordinary AdamW trên clean, detach state để kích hoạt cả hai
LoRA factors. So phần gradient marked-minus-clean bằng relative squared error,
giữ dấu và magnitude. Nén từng tensor bằng CountSketch 512 chiều, seed theo tên
tham số, O(P) thay vì phép chiếu dense P×512 của GVENDI. Không ép Gram gradient
bằng nhau và không so tham số SD2.1 trực tiếp với SD1.4.

CKA: dùng chính frozen AlexNet của LPIPS, feature layers 1 và 2 (index từ 0),
pool về 8×8 patch tokens có correspondence; center qua patches rồi tính Gram.
Không thêm model tải từ mạng. LPIPS vẫn giữ vai trò kiểm soát độ lệch tuyệt đối.

Cả hai sử dụng ảnh predicted-x0 trên low-noise view t=10..200 như nhánh LPIPS
đối chứng, chưa tối ưu trực tiếp ảnh cuối của toàn sampler. Reference gradient
lấy từ teacher nguồn đóng băng, được dựng lại từ source trước khi restore
checkpoint refinement. Không dùng teacher đang thay đổi làm target tự đuổi theo.

## Student và đánh giá

Mỗi teacher hoàn thành sẽ gen 5.000 train / 1.000 validation / 1.000 test.
Final student giữ nguyên ordinary `lora_r32_conv_wm`, batch4, 7.500 updates,
checkpoint selection bằng validation denoising MSE. Không có key/reader/loss
distillation trong student. Đo detection, LPIPS, FID, KID và CLIP tới cuối;
kết quả xấu vẫn được lưu. Teacher chọn fixed step600.

Tái dùng hard links clean images, prompts, carrier, thresholds và unadapted
reference. Hàng chờ cũ và code `/root/MinhNQN` không thay đổi. Code mới reuse
training/checkpoint/data/evaluation helpers đã pin trong manifest, thông qua
entry point riêng; child data/report stages được route về đúng queue mới.

## Kiểm tra đã thực hiện và bước còn chờ

- 7 unit/integration tests đã pass ở local và VM4 CPU. Có finite-difference
  check hypergradient qua actual gradient-sketch helper với toy denoiser;
  kiểm tra mất strength/đảo dấu bị phạt; CKA finite và detached reference;
  kiểm tra routing child stages. Không coi đây là full-model GPU verification.
- Đối chiếu config với source: chỉ đổi teacher_lr, teacher_train_steps,
  teacher_checkpoint_every; cấu hình student và các thông số khác không đổi.
- Full-model GPU smoke **chưa chạy khi enqueue**, do GPU đang bận. Tự chạy
  trước từng nhánh sau dependency; yêu cầu ít nhất 115.000 MiB GPU trống và
  24 GiB disk trống trước mỗi job. Smoke phải có auxiliary gradient hữu hạn,
  khác 0 và teacher parameters thay đổi mới đi tiếp vào train.
- Gradient smoke dùng perturbation RGB nhỏ chỉ trong technical smoke để ra
  khỏi điểm candidate=reference có loss/gradient bằng 0. Không ghi checkpoint
  từ smoke; training thật khởi tạo lại source và không có perturbation này.
- Numerical/integrity errors dừng nhánh và ghi failed; không tự thay loss,
  giảm batch hoặc bỏ test khi lỗi. Code manifest và unit-test proof được pin.

Trạng thái có thể đọc bằng:

```bash
cat /root/logs/threshold_quality_20261022/minhnqn_refine_20261024/status.json
cat /root/logs/threshold_quality_20261022/minhnqn_refine_20261024/launcher.json
```

Khi hoàn thành có `comparison_with_controls.json`, gồm baseline scale1/0.85,
hai nhánh mới và hai nhánh quality/quality_transfer trước đó. Đây vẫn là một
seed và reused exploratory test; chưa có claim vượt lịch sử 99.70% hay bảo đảm
local gradient matching giữ final detection. Hướng gradient cũng chưa mô phỏng
đầy đủ final AdamW trajectory. Xem [review](minhnqn_review_20261024.md).
