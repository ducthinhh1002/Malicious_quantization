# Vì sao các nhánh trước cải thiện ít và thiết kế vòng mới — 25/10/2026

## 1. Kết luận từ các lần chạy đã hoàn tất

Mục tiêu là cải thiện ảnh student mà vẫn giữ radioactive detection. Không
thay reader, key, PCA carrier, ngưỡng null hoặc ordinary student objective.
Mốc hiện tại cùng protocol là scale1.0 detection 75%; lịch sử 99,70% dùng
teacher kappa2 và artifact khác, chưa tái lập được đối chứng đầy đủ.

| Student | Detection test | LPIPS với unadapted SD1.4 | FID với clean teacher |
|---|---:|---:|---:|
| Scale1.0 | 75,0% | 0,550447 | 54,444 |
| Scale0.85 | 42,7% | 0,533659 | 54,563 |
| Scale0.85 / 22500 updates | 52,3% | 0,573231 | 54,858 |
| Conv restore | 57,5% | 0,535404 | 54,230 |
| quality | 80,8% | 0,546656 | 55,281 |
| quality_transfer | 80,5% | 0,545900 | 55,087 |
| quality_gradient | 80,7% | 0,543704 | 55,377 |
| quality_cka | 80,5% | 0,548790 | 54,939 |

Tất cả refinements cùng tiến về khoảng 80,5–80,8%, dù auxiliary objective
khác nhau. Chưa có original-only continuation nên chưa biết bao nhiêu lợi ích
do thêm 600 teacher updates. Một seed và test panel được xem nhiều lần không
đủ kết luận chênh lệch 1–3 ảnh là đáng kể.

## 2. Nguyên nhân có bằng chứng và giả thuyết cần kiểm tra

### A. Quality loss chưa tối ưu ảnh được người dùng xem

Các nhánh trước thêm LPIPS trên predicted-x0 của một ảnh ở t=10..200, trong
khi production chạy Gaussian noise qua PNDM50, CFG7.5. Training-view LPIPS
trung bình khoảng 0,0364 nhưng native teacher validation khoảng 0,429–0,431.
Hai giá trị đo hai phân bố/view khác nhau, không phải train/validation error
trên cùng input. Chúng cho thấy việc tối ưu một view rẻ chưa bảo đảm cải thiện
ảnh cuối. Các biến dạng đầu/mặt/tay có thể hình thành trên toàn quỹ đạo.

### B. LPIPS gradient rất yếu so với original objective

Đã đọc lại đủ 600 records của bốn nhánh trên VM4, tính
`quality_gradient.added_norm / original_gradient_norm`:

| Nhánh | Median tỷ lệ norm | Mean tỷ lệ norm |
|---|---:|---:|
| quality | 0,3453% | 1,7696% |
| quality_transfer | 0,3501% | 1,7966% |
| quality_gradient | 0,3399% | 1,7722% |
| quality_cka | 0,3404% | 1,7978% |

Giới hạn 25% là trần, không phải gradient luôn đạt 25%. LPIPS weight0.05
thường cho bước chất lượng rất nhỏ. Mean cao hơn median một phần do các batch
không có semantic timesteps và original gradient nhỏ. Không suy ra tăng weight
là chắc chắn chữa được hình thể; view và objective vẫn phải đúng.

### C. Giảm scale làm yếu margin; train lâu không sửa được tự động

Scale0.85 trước đó giảm p10 teacher margin khoảng 77%, rồi student detection
mất 32,3 điểm. Teacher TPR gần trần không phản ánh hết độ bền sau learning.
Tăng student lên 22500 updates giúp detection một phần nhưng LPIPS xấu hơn.
Vì thế không tiếp tục coi giảm scale hoặc tăng epochs là giải pháp chính.

### D. Các proxy mới quá cục bộ hoặc không nhắm hình thể

Transfer proxy dùng 2 fresh rank8 AdamW steps và 8-step DDIM query, khác final
rank32-conv student 7500 updates/PNDM50. Gradient matching dùng local MSE
gradient sau một clean warmup; CKA bảo toàn quan hệ feature và bỏ qua một số
biến đổi scale/rotation. Chúng có thể hữu ích, nhưng kết quả vừa rồi chưa chứng
minh chúng sửa được lỗi thị giác hoặc dự báo tốt radioactivity dài hạn.

### E. Thiếu clean-trained control

Mẫu test000002 của unadapted SD1.4 tự nhiên hơn cả student cũ lẫn quality.
Teacher marked cũng có dấu hiệu bề mặt da/texture bất thường. Tuy nhiên chưa
tách được generic synthetic finetuning drift khỏi watermark-specific drift.
Student học ảnh teacher sạch cùng 5000/7500/batch4 là đối chứng thiết yếu.
Detection của clean-trained student còn kiểm tra specificity ở threshold cũ;
không được mặc định nó giữ đúng 1% FPR sau finetuning.

### F. Những lỗi hệ thống đã sửa, không nhầm với thất bại thuật toán

Fresh merge subprocess từng bật cuDNN TF32, gây strict merge test failure;
tắt TF32 đưa max error từ 0,00134474 xuống 5,54e-6, giữ nguyên tolerance.
OOM của quality_transfer do tiến trình khác chiếm 91,55 GiB. Sau khôi phục,
final students đều đủ 7500 applied updates, không skip. Những lỗi này làm chậm
luồng, không giải thích trực tiếp toàn bộ méo mặt của model đã train đủ.

## 3. Survey bổ sung từ nguồn gốc

### DRaFT: tối ưu reward trên ảnh cuối

[Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/html/2309.17400v2)
cho phép truyền gradient qua toàn bộ sampler; DRaFT-K cắt gradient ở K bước
cuối để tiết kiệm. Paper cũng thảo luận reward hacking/over-optimization.
Vòng này áp dụng nguyên tắc full-trajectory differentiation nhưng giữ sampler
PNDM gốc, thêm mục tiêu chất lượng và watermark của ta. Không gọi đây là tái
lập kết quả aesthetic/human-preference của DRaFT: ta chưa dùng reward model
được học từ human preferences. Dùng full gradient để không cố định geometry
của prefix. Không cache marked prefix từ checkpoint cũ.

### AquaLoRA và SIREN: giữ phân bố gốc và xét learning channel

[AquaLoRA](https://arxiv.org/html/2405.11135v1) gợi ý vai trò của prior
preservation; nhưng native white-box watermark không chứng minh student
radioactivity. Original threshold_cfg_v1 đã có conditional/unconditional
noise-prediction preservation, vòng này giữ nguyên.
[SIREN](https://arxiv.org/html/2410.10437v1) xét coating có thể được model học
lại, kết hợp surrogate learning và giới hạn cảm nhận. Ta giữ learning channel
RGB/caption, nhưng không thay reader bằng reader của paper hoặc coi raw MSE
thấp là bằng chứng transfer. Audit trước của ta đã thấy MSE đơn lẻ không xếp
hạng tốt scale1/.85.

### Synthetic data và guidance: giả thuyết, chưa phải chẩn đoán xác nhận

[Self-Consuming Generative Models Go MAD](https://arxiv.org/abs/2307.01850)
phân tích suy giảm chất lượng/đa dạng qua các vòng học ảnh tổng hợp. Ta mới có
teacher→student một vòng, nên không đủ bằng chứng gọi các lỗi hiện tại là
model collapse theo paper. Nó củng cố nhu cầu đối chứng clean-trained và xét
phân bố dữ liệu, không chỉ thêm epochs.
[Common Diffusion Noise Schedules and Sample Steps Are Flawed](https://arxiv.org/abs/2305.08891)
thảo luận train/inference mismatch và rescale CFG chống over-exposure. Ảnh của
ta có màu gắt/texture bất thường, nhưng chưa chứng minh guidance là nguyên nhân.
Không đổi noise schedule, prediction type hay CFG7.5 trong campaign này: thay
nhiều biến như vậy phá đối chứng với sampler người dùng yêu cầu giữ nguyên.

### Preference optimization: có tiềm năng nhưng chưa đưa nguyên vào campaign

[Diffusion-DPO](https://arxiv.org/abs/2311.12908) dùng preference pairs và
reference regularization. Ta chưa có preference pairs được kiểm chứng; tự gán
ảnh điểm detector cao là ảnh tốt sẽ học sai mục tiêu. Reward model cũng cần
kiểm tra trên ảnh người/tay của ta để tránh reward hacking. Chưa triển khai DPO
hoặc tuyên bố đã có human preference evaluation.

## 4. Code và thí nghiệm đã triển khai

Root VM4: `/root/logs/threshold_quality_20261022/native_quality_20261025`.
Thứ tự: `clean_control` → `native_full` → `continuation`.

- `clean_control`: hard-link toàn bộ ảnh clean đã có vào dataset cùng cấu trúc
  upstream. Branch vẫn tên lora_r32_conv_wm để command trainer không đổi;
  provenance ghi rõ condition=clean, không nhầm là marked teacher/student.
- `continuation`: checkpoint k1 step3000 +600 steps original threshold loss,
  LR1e-5, batch4. Không thêm LPIPS hoặc gradient objective. Đây là đối chứng
  ngân sách teacher còn thiếu, không phải cải tiến chất lượng.
- `native_full`: cùng khởi tạo,600 steps,LR,batch; original loss giữ nguyên.
  Mỗi 5 steps lấy một training prompt/seed theo cached clean reference, chạy
  sampler PNDM50 nguyên bản, CFG7.5, full differentiable trajectory. PNDM với
  cấu hình gốc thực hiện 51 denoiser calls do bước khởi động. Gradient đi đủ51.

Native auxiliary:

`L_native = 1.0*LPIPS(RGB_marked, RGB_clean) + 0.1*pixel_MSE + 0.8*coherent_shift_loss`

Coherent loss dùng đúng công thức calibrated shift/tau/sigma/kappa1/cap2.5
hiện có, nhưng đo sau full generation + RGB8 STE + frozen extractor. Không
nới threshold. Gradient auxiliary kết hợp bị chặn norm ở 25% original gradient
trước global clipping; đây là giới hạn tối ưu, không phải stopping gate.
Combined auxiliary chưa cân bằng norm riêng từng thành phần, cần đọc log và
đối chiếu chất lượng/detection thật; không giả định weight1 tự động tối ưu.

FP32 teacher params, FP16 autocast native denoiser để gần production; original
teacher loss vẫn BF16. VAE/extractor FP32. Production vẫn merge native weights
và sinh FP16 như trước. Các khác biệt rounding giữa unmerged mixed precision
và merged inference được ghi rõ; không tuyên bố training render bit-identical
với production. Native final evaluation sẽ quyết định tác dụng thực sự.

Student cả ba: ordinary upstream rank32 attention+conv,5000train/1000val,
7500 applied-update target,batch4,checkpoint2500/5000/7500, chọn bằng held-out
denoising MSE. Không thêm key/distillation/perceptual loss vào student.

## 5. Kiểm thử và tài nguyên

- Bốn tests pass trên VM4 CPU: full PNDM forward đối chiếu independent loop;
  checkpoint/full backprop gradients trùng nhau với toy denoiser; scheduler
  state độc lập; specs đối chứng và pixel-diagnostic interpretation.
- Local Python thiếu diffusers nên hai sampler tests không chạy được ở local;
  môi trường thực trên VM4 có dependency và đã pass đủ bốn tests.
- Actual-model smoke native_full pass trước launch, không ghi trained weights:
  full51 gradient steps;finite teacher update; peak45,416GiB. Original gradient
  norm4,0138; raw native auxiliary751,64, bị chặn còn1,00345 (25%). Đây chỉ là
  kiểm thử kỹ thuật một sample, không phải kết quả chất lượng/detection.
- Full teacher checkpoints mỗi200, gồm optimizer/RNG/dataloader state qua
  machinery cũ. Manifest hashes pin code, source và frozen thresholds.
- Supervisor PID lúc launch1456465,08:32 ICT25/10. Resource wait mỗi stage:
  free115000MiB trước teacher/smoke,80000MiB trước stage khác; disk guard.
  Không can thiệp job GPU của người khác. Tái dùng clean/unadapted ảnh bằng
  hard links, không xóa checkpoint cũ. Số liệu khoa học xấu vẫn chạy tới cuối.

## 6. Đánh giá và tiêu chí kết luận

Giữ LPIPS/FID/KID/CLIP để so cùng protocol, không biến một chỉ số thành ngưỡng
ảnh đẹp phổ quát. Bổ sung quantiles LPIPS và clipping/saturation diagnostics;
những số màu sắc này không đo lỗi anatomy hoặc thay human quality score.
Code tạo cố định200 test pairs để review người xem, có A/B randomization và
rubric face/hand/body/object/naturalness/prompt. Chưa có human ratings; paths
trong JSON có thể tiết lộ nhánh, cần ẩn đường dẫn/model names trong giao diện
rating thực sự trước khi thu đánh giá mù. Không báo win rate khi chưa có chấm.

Muốn kết luận native_full có ích cần tốt hơn continuation cùng ngân sách và
giữ student detection, đồng thời ảnh nhìn thực sự tốt hơn. Nếu clean control
cũng méo hoặc FPR cao, phải xử lý distribution/training/specificity trước khi
gán toàn bộ lỗi cho watermark. Nếu native teacher tốt mà student vẫn hỏng,
điểm cần nghiên cứu tiếp nằm ở learning channel hoặc dữ liệu; không ép thay
ordinary student trainer rồi coi đó là radioactivity nguyên protocol.
Test hiện tại đã dùng nhiều vòng; ứng viên có triển vọng cần fresh holdout và
nhiều training seed trước claim chính thức. Không cam kết đạt LPIPS0.492 hoặc
detection95% chỉ dựa vào smoke.
