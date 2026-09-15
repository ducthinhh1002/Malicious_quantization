# Quantization với quyền truy cập chỉ model đã fingerprint

## Kết luận nghiên cứu

Không có căn cứ từ các nguồn bên dưới để bảo đảm một quantizer không biết key,
extractor hay ảnh không watermark sẽ xóa Stable Signature tốt hơn kết quả cũ.
Vì vậy implementation mới là một phương pháp thử nghiệm có giả thuyết rõ ràng;
hiệu quả phải được đo sau khi khóa cấu hình. Nó mở rộng số biến học được từ vài
control theo group thành quyết định rounding cho từng weight trong VAE decoder.

### Các nguồn đã đối chiếu

- [Stable Signature, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.html):
  watermark nằm trong latent decoder. Nhánh này do đó tập trung vào VAE; không
  có lý do gắn timestep loss vào VAE decoder vì nó chỉ chạy trên latent cuối.
- [Stable Signature is Unstable, 2024](https://arxiv.org/abs/2405.07145):
  tối ưu decoder với ảnh không watermark và latent ước lượng. Công trình cần
  attacking dataset không watermark, nên không phải baseline tương đương với
  giả định chỉ dùng model và dữ liệu model tự sinh. Đây cũng là fine-tuning,
  không phải chứng minh quantization không detector có hiệu quả.
- [AdaRound, ICML 2020](https://arxiv.org/abs/2004.10568): học rounding thay cho
  làm tròn độc lập từng weight. Implementation dùng hard-forward/STE sigmoid,
  không tái lập nguyên thuật toán AdaRound với stretched sigmoid và annealing.
- [Q-Diffusion, ICCV 2023](https://arxiv.org/abs/2302.04304): calibration theo
  timestep và shortcut split dành cho noise estimation network. Không nên gọi
  nhánh VAE mới là Q-Diffusion hay tuyên bố dùng diffusion trajectory gradients.
- [Rounding-Guided Backdoor Injection, 2025](https://arxiv.org/abs/2510.09647):
  nghiên cứu backdoor qua rounding; không phải bằng chứng trực tiếp cho việc
  xóa watermark Stable Signature dưới threat model không extractor/key.

## Threat model và ranh giới thí nghiệm

Attacker nhận một Diffusers pipeline đã có fingerprint trong VAE. Attacker có
trọng số/config và được sinh ảnh với prompt/seed tự chọn. Không có checkpoint
không watermark, watermark key, extractor, detection API, ảnh tự nhiên bên ngoài,
hay mạng neural khác trong quá trình lựa chọn quantizer. Biết scheme là Stable
Signature được coi là thông tin công khai. Prompt văn bản không chứa nhãn watermark.

`prepare_marked_fixture.py` thuộc vai trò người tổ chức: tải backbone và checkpoint
VAE đã watermark công khai rồi xuất duy nhất pipeline đã fingerprint. Việc tổ chức
biết backbone không được dùng để so sánh weight/ảnh trong chương trình attacker.
Trong thử nghiệm nghiêm ngặt, chạy attacker trong môi trường chỉ thấy thư mục fixture.

`wmq_blind.py` chỉ load fixture local. `evaluate_blind_watermark.py` là chương trình
chủ sở hữu, chạy sau khi lựa chọn đã kết thúc. Không dùng kết quả owner evaluation
để chọn strength, bitwidth, clip, seed hoặc checkpoint của cùng thí nghiệm blind.
Nếu nghiên cứu tuning với feedback detector, cần khai báo threat model oracle khác.

### Audit luồng dữ liệu

Phiên bản trước đã chỉ dùng train để tính loss và search để gọi `choose`, nhưng
sinh sẵn cả ảnh test. Phiên bản hiện tại chỉ tạo train/search trước optimization;
sau khi chọn xong mới xuất `quantized_vae` và ghi `selection_frozen.json` chứa hash
checkpoint/search, rồi mới sinh latent và ảnh test. Không có bước chọn lại sau
test. `test_used_for_selection=False` là metadata mô tả luồng này, tự nó không
phải bằng chứng.

Kiểm thử CUDA với pipeline nhỏ xác nhận: checkpoint/freeze marker có trước lần
sinh test đầu tiên; `choose` không được gọi sau khi test bắt đầu; thay đổi riêng
prompt và phân phối latent test không đổi lựa chọn hoặc weights xuất. Chương trình
attacker không có input key/extractor và không gọi owner evaluator. Đây là kiểm
chứng implementation đang có, không bảo đảm trước việc sửa source/metadata có chủ ý.

Checkpoint Meta được pin SHA256 từ bytes tải ngày 2026-09-15, không phải chữ ký
do nhà phát hành cung cấp. Backbone revision mặc định vẫn là `main`; pin decoder
không đồng nghĩa toàn bộ backbone đã pin immutable revision. Manifest attacker
ghi hash các file model thực tế để kiểm toán. Evaluator từ chối threshold không
thể đạt (ví dụ key 4 bit và FPR 0.001) thay vì xuất TPR=0 gây hiểu nhầm.

## Phương pháp thử nghiệm

1. Sinh và cache latent cuối từ model đã fingerprint. Chia prompt/seed duy nhất
   thành train, search và test. Giữ UNet/text encoder cố định.
2. Ảnh tham chiếu là `x_w = D_w(z)`; đó vẫn là ảnh có watermark.
3. Tạo pseudo-target `t = (1-rho) * x_w + rho * G(x_w)`, với G là Gaussian 5x5
   cố định và rho mặc định 0.25. Giả thuyết: làm dịu tín hiệu chi tiết có thể
   làm yếu fingerprint trong khi giữ nội dung. Không giả định mọi watermark
   đều là high-frequency; blur có thể chỉ xóa texture và giữ nguyên watermark.
4. Cho mỗi bitwidth/clipping, cố định grid per-output-channel từ marked weights.
   Học một alpha cho từng weight bằng Adam. Forward luôn dùng quyết định hard
   `floor(w/s) + 1[alpha >= 0]`, backward dùng gradient sigmoid (STE). Không
   train full-precision model weights, bias, norm hay scale.
5. Loss là MSE tới pseudo-target cộng MSE giữa hai ảnh sau Gaussian filtering.
   Cả hai mục tiêu đều không dùng thông tin watermark. Đây là gradient theo
   proxy thị giác, không phải gradient watermark `g_own`.
6. Trên search, chỉ nhận checkpoint có mean PSNR >=25 dB, mean SSIM >=0.9 và
   min per-image PSNR >=22 dB. Trong các checkpoint hợp lệ, chọn MSE tới
   pseudo-target nhỏ nhất. Các ngưỡng cần chốt trước thí nghiệm.
7. Sau khi khóa lựa chọn, sinh test cho marked reference, RTN cùng bits/clip/
   target coverage, và checkpoint đã chọn. Owner đo bit accuracy, TPR và LPIPS
   độc lập. Nếu không có candidate hợp lệ, không xuất attacked VAE. Nếu chất
   lượng test không đạt, vẫn giữ artifact kiểm toán với `heldout_quality_failed`.

Model-only không đồng nghĩa biết được đặc trưng nào trong weights là watermark.
Không có oracle thì một proxy mạnh về thị giác vẫn có thể vô dụng với fingerprint.
Không lấy MSE thấp hoặc bitwidth thấp làm bằng chứng attack thành công.

## So sánh cần chạy

- `--strength 0`: reconstruction-only control, cùng grid/steps/dữ liệu.
- `--strength 0.25`: giả thuyết suppression, chốt trước khi xem owner metrics.
- RTN matched được sinh tự động sau khi khóa lựa chọn; đây là control tại cấu hình
  được chọn, không phải RTN được tune tối ưu riêng.
- Nhiều seed và key do người tổ chức chuẩn bị, cùng dataset công khai đã cố định.
- Báo cả candidate không feasible, test quality failure và grid fallback.
- Không so trực tiếp với legacy oracle search rồi kết luận phương pháp yếu/mạnh:
  quyền truy cập, bitwidth, target coverage, dtype và quality gate khác nhau.

## Tính toán và giới hạn

Cache latent giúp không phải chạy lại UNet mỗi candidate: vì chỉ thay VAE decoder,
đầu vào VAE cuối chính là latent từ cùng UNet/prompt/seed. Calibration không cần
backprop qua toàn chuỗi diffusion. VAE và quantizer dùng FP32 để tránh nhầm sai số
FP16 với low-bit PTQ; UNet sinh latent ở FP16. Export là simulated weight PTQ với
weights dequantized FP32, không phải packed INT2/INT4, không có claim tăng tốc kernel.

Mỗi update dùng một ảnh train và học đồng thời mọi rounding variable trong scope.
RAM giữ model, references và bản snapshot trên CPU; VRAM phụ thuộc activations VAE
512x512. Có log peak allocated VRAM và thời gian, nhưng chưa có benchmark RTX 6000.
Rounding full decoder có thể cần nhiều VRAM; `--scope late` là thí nghiệm nhỏ hơn
và phải báo phạm vi khác. Không tự thay scope/precision khi thiếu bộ nhớ.

PSNR/SSIM chỉ là proxy chất lượng; ngưỡng LPIPS cũ không được bảo đảm. Owner có
thể đo LPIPS để kiểm tra kết quả và báo thất bại, nhưng không chọn lại candidate
bằng detector test. Ảnh pseudo-target không bao giờ được gọi là ảnh sạch.
