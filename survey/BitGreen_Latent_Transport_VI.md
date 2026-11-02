# BitGreen: latent target → native diffusion teacher

## Kết quả buộc phải đổi cách tối ưu

Hai run K4 đã hoàn tất. Test 256 ảnh, threshold frozen từ clean calibration:

| Run | Clean FPR | Plain TPR | Robust TPR |
|---|---:|---:|---:|
| accumulation=4 | 3/256 | 3/256 | 1/256 |
| accumulation=1 | 3/256 | 2/256 | 3/256 |

PSNR pilot của accumulation=4 là 30,67 / 31,10 dB. Chất lượng tốt hơn nhưng chưa có dấu hữu ích. Student không chạy vì teacher gate thất bại. Không được dùng kết quả này làm bằng chứng watermark thừa hưởng.

Chẩn đoán trước đó: tối ưu trực tiếp terminal latent trên 4 ảnh train đạt score 0,531–0,540 ở 31,8–34,8 dB. Đây là khả năng tạo target, không phải khả năng tổng quát hóa của UNet.

## Tiền lệ và giới hạn suy luận

[SleeperMark](https://arxiv.org/html/2412.04852v2), §4.1–4.2, tách nhúng dấu vào latent và huấn luyện backbone. Paper chọn residual dùng chung để tăng tính nhất quán giữa ảnh. Đây là tiền lệ quan trọng cho thiết kế hai giai đoạn; không claim phát minh ra ý tưởng này. SleeperMark xét fine-tune model đã nhúng dấu, khác student khởi tạo độc lập học từ ảnh trong protocol của chúng ta.

[DDIM](https://arxiv.org/abs/2010.02502) là cơ sở của bước lấy mẫu xác định. Phép dịch quỹ đạo bên dưới được suy ra từ phép cập nhật DDIM; không phải một kết quả thực nghiệm của paper.

## Cơ chế mới triển khai

1. Từ prompt **teacher_train**, sinh trajectory sạch và latent cuối `z0` bằng model gốc frozen.
2. Chỉ tối ưu `delta` để `D(z0+delta)` đạt score ít nhất `threshold + 0.02`, với PSNR >=28 dB so với `D(z0)`. Detector, VAE, backbone đều frozen. Plain xét ảnh native; robust yêu cầu thêm ảnh qua VAE round-trip đạt cùng target. Chọn target thành công có MSE nhỏ nhất trong 32 bước, không chọn score cao nhất bất chấp chất lượng. Failed targets được ghi lại.
3. Dạy UNet owner theo mục tiêu hồi quy liên tục trên trajectory đã dịch. Không truyền gradient bit detector qua rollout trong giai đoạn này.

Viết bước DDIM là `z_next=A*z+B*epsilon_CFG`. Với latent đích lệch `delta`, đặt trajectory mong muốn `z*_i=z_i+p_i*delta`, trong đó `p_0=0`, `p_last=1`. Ta cần:

```
epsilon*_CFG = epsilon_clean_CFG + ((p_next - A*p_i)/B) * delta
```

`p_i` là sqrt(alpha) chuẩn hóa giữa đầu và cuối. Điều kiện đầu bằng 0 giữ nguyên Gaussian noise thông thường lúc inference. Unit test đối chiếu trực tiếp với `DDIMScheduler.step` tại mọi timestep, không chỉ kiểm tra cùng một công thức hai lần.

Loss hồi quy là MSE giữa thay đổi CFG prediction và correction trên, cộng MSE giữ unconditional prediction gần reference. Chia cho năng lượng correction (có floor) để tín hiệu nhỏ không chìm trong background denoising loss. Reference, delta và mẫu số đều detach. Mỗi microbatch chọn độc lập ảnh và timestep, tránh modulo vô tình gắn cố định một ảnh với một timestep.

## Protocol và kiểm chứng

- 64 target prompts từ train, 32 latent steps/ảnh/nhánh; tối thiểu 50% target thành công trước khi train owner. Nếu không đạt, dừng rõ ràng với diagnostics, không ngầm train trên target sai.
- Pilot owner 2000 update × 4 transition samples. Đây là ngân sách khác run K4 và không được báo là so sánh compute ngang nhau.
- Cache tensor, adapter và model ở thư mục owner/model thuộc vùng artifact nặng; JSON diagnostics trong result.
- Owner LoRA vẫn rank16, merge thành native UNet. VAE không được train. Không thêm delta, hook, trigger hay noise có dấu lúc inference.
- LoRA student vẫn ordinary training từ clean SD1.4; unit test so sánh toàn bộ command student giữa objective mới/cũ.
- Teacher calibration, pilot gate, held-out scoring và matched clean student giữ nguyên. Repeated development splits chưa phải một final test độc lập cho paper.

Lệnh:

```bash
bash run_bitgreen.sh --owner-objective latent_transport \
  --teacher-steps 2000 --owner-accumulation 4 --teacher-lr 0.00005 \
  --target-n 64 --target-steps 32 --vram-limit-gb 24
```

## Những điều chưa được chứng minh

Target đạt score không đảm bảo UNet học được; fit trajectory train không đảm bảo sinh từ noise/prompt mới tốt. Các delta riêng từng ảnh có thể khó học hơn residual chung, đây là ablation đáng làm nếu regression bị trung bình hóa. Chuỗi inference dài có thể tích lũy sai số ngoài trajectory được giám sát. Chất lượng target không thay thế chất lượng output teacher.

Robust ở nhánh này chỉ thêm ràng buộc VAE round-trip trong tạo target; không giả vờ rằng đã huấn luyện qua student hay đầy đủ channel noise200/400 của nhánh cũ. Pattern 01/10 vẫn công khai, chưa phải cơ chế attribution theo secret key. Hiệu quả teacher và truyền dấu cần số liệu thực nghiệm riêng trước khi claim novelty/robustness.
