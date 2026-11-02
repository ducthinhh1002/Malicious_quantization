# Đối chứng SD2.1-base 512×512

Teacher mới dùng SD2.1-base, giữ bit tokenizer, objective constrained, prompt/seed,
250 updates mỗi nhánh Plain/Robust, LR 1e-5, 20 bước DDIM/gradient và quality budget
giống run SD1.4. Student luôn là clean SD1.4 với ordinary LoRA. Đây là đối chứng
teacher khác backbone; không phải SD2.1 → SD2.1.

`--compare-to` kiểm tra split prompt giống hệt và các trường objective/budget khớp.
Lưu đối chiếu vào `result/comparison_protocol.json`. Các đường dẫn teacher và student
được tách: owner UNet/VAE/text encoder và frozen clean reference dùng SD2.1;
student initialization/validation encoder vẫn dùng SD1.4.

## Checkpoint

Endpoint `stabilityai/stable-diffusion-2-1-base` trả HTTP 401 trong lần kiểm tra.
Mirror công khai được dùng rõ ràng, không âm thầm thay model:

- Repo: `sd2-community/stable-diffusion-2-1-base`.
- Revision: `4e63672c03103b6c636b8fb4119ba982469b2955`.
- Tải FP32 safetensors, kiểm tra hash LFS trước khi load.
- Assert `sample_size=64`, latent channels 4, cross-attention/text hidden 1024,
  scheduler prediction `epsilon`: SD2.1-base 512px, không phải bản 768 v-prediction.

Nguồn cấu hình checkpoint:
https://huggingface.co/sd2-community/stable-diffusion-2-1-base/tree/4e63672c03103b6c636b8fb4119ba982469b2955

## Đánh giá

- Hiệu chuẩn lại threshold FPR mục tiêu 1% từ 1.000 ảnh sạch SD2.1 trước train owner.
- PSNR so với ảnh sạch SD2.1 cùng seed/prompt; không so trực tiếp với ảnh SD1.4.
- Teacher gate vẫn TPR >=80%, PSNR >=22 dB. Quality penalty training nhắm 24 dB.
- Nếu gate đạt, ordinary SD1.4 students học từ ảnh SD2.1 clean/Plain/Robust.
- TPR student dùng threshold teacher không đảm bảo FPR 1% trên miền ảnh SD1.4;
  phải đọc cùng matched clean-trained student và clean-adjusted difference.
- FID/KID từng run có clean teacher reference tương ứng. Không xếp hạng chất lượng
  tuyệt đối giữa SD1.4/SD2.1 chỉ bằng hai FID có reference khác nhau.
- Cùng updates không đồng nghĩa cùng FLOPs, thời gian hay số tham số owner trainable.

Chưa có bằng chứng đổi backbone sẽ giải quyết trade-off; đây là phép thử độc lập.
Run SD1.4 đang chạy được giữ nguyên snapshot nguồn.
