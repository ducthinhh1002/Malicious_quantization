# BitMark → LoRA: thí nghiệm radioactivity riêng

Từ thư mục gốc repository, chạy **thật** toàn trình:

```bash
bash run_bitmark_lora.sh
```

Launcher chọn/tạo Conda env riêng `bitmark-radioactivity`, cài các phiên bản đã pin,
tải và kiểm tra checkpoint, sinh dữ liệu, train, sinh ảnh student, chạy **detector
chính thức**, tính CLIP/FID/KID và tổng hợp kết quả. Không cần activate thủ công.
Log được in trực tiếp và lưu theo từng stage. Không thay đổi pipeline SERUM cũ.

Máy cần Linux x86_64, Conda, GPU Ampere trở lên, driver hỗ trợ PyTorch CUDA 12.8,
Internet và tối thiểu 70 GiB trống theo preflight. Checkpoint teacher, tokenizer,
T5 và SD2.1 tổng cộng hàng chục GB. Thực nghiệm đầy đủ tốn nhiều giờ tùy GPU.
`--min-free-gb` chỉ thay đổi kiểm tra dung lượng, không làm thí nghiệm nhẹ hơn.
FlashAttention dùng wheel CUDA đã pin hash, không tự compile bằng nvcc.

Server submit job: cài môi trường **một lần trên login node**, sau đó job chỉ dùng
env đã có. Không đưa `--setup-only` vào lệnh chạy experiment:

```bash
bash run_bitmark_lora.sh --setup-only
# Trong GPU job:
bash run_bitmark_lora.sh
```

## Câu hỏi thực nghiệm và đối chứng

Mục tiêu: với dữ liệu BitMark, LoRA có truyền tín hiệu watermark yếu hơn full
fine-tune hay không? Đây là **thí nghiệm truyền watermark qua adaptation**,
không phải quantization attack, không phải thêm watermark trực tiếp vào SD2.1.

Mặc định có 3 student/evaluation branches; không chạy train clean kiểu C00/C01:

| Nhánh | Adaptation | Dữ liệu train |
|---|---|---|
| `unadapted` | Không | Không |
| `lora_r4_wm` | LoRA rank 4, attention q/k/v/out | Infinity BitMark δ=2 |
| `full_wm` | Toàn UNet, VAE/text encoder frozen | Infinity BitMark δ=2 |

Cả LoRA và full dùng cùng SD2.1, 1.000 ảnh train, 5 epochs, batch 4,
LR 1e-4, AdamW, clipping 1.0, FP16, resolution 512, không EMA, không random flip.
Mặc định là **1.250 optimizer steps/student**, không phải 20.000 steps của SERUM.
Giữ backbone/hyperparameter/budget cố định để hạn chế confound khi so adaptation.
Chưa phải khảo sát hyperparameter tối ưu cho từng phương pháp.

Chọn **checkpoint cuối theo ngân sách định sẵn**, không chọn bằng detector hay
test TPR. Đây là khác biệt có chủ đích so với chọn checkpoint bằng validation
denoising loss trong SERUM. Các helper stage training không nhận key, threshold,
scores hoặc ảnh test. Pipeline cha có quyền owner để audit sau khi freeze model;
đây là cách ly theo luồng dữ liệu, không phải sandbox chống tiến trình độc hại.

Prompt lấy từ COCO2014 validation: 1 caption/image, loại caption trùng,
chia train/calibration/test rời nhau bằng seed. Mặc định calibration=5.000,
test=1.000. Chỉ sinh ảnh watermarked cho train, bỏ `teacher_clean_train`.
Clean calibration để đặt ngưỡng và clean test để đo false positives/FID vẫn được sinh.
Clean/watermarked teacher ở test dùng cùng prompt và seed, cùng sampler
bit tuần tự của tác giả, chỉ khác δ=0/2. Teacher Infinity-2B sinh 1024²;
trainer crop/resize 512²; student sinh 512²; detector resize bằng code Infinity.

## Nguồn chính thức, không tự viết lại BitMark

- Paper: https://arxiv.org/abs/2506.21209, §4.3 và Appendix F.
- Repo: https://github.com/sprintml/BitMark,
  commit `dd21db32dc86ac70526229e678a027cc06f5c296`.
- Code generator/tokenizer/detector nằm nguyên bản ở `vendor/bitmark/`.
  `teacher.py` gọi `WatermarkInference`, `Infinity.gen_img`, `get_detector`, `detect`.
- `G={01,10}`, context=2, all scales (`watermark_scales=2`), δ=2, CFG=4.
- Infinity/tokenizer: `FoundationVision/Infinity`, revision
  `e7108923249fb3c836ebfbc3f4b4d318d6131293`:
  - `infinity_2b_reg.pth`, SHA256
    `d5958b37e7ecb914230cd2b8c28aae0ec108ce08b315720bc3dc52596cc5b3a7`.
  - `infinity_vae_d32reg.pth`, SHA256
    `7a37fa3ea1b2a1ebd23de61d91a5e68202825e5a67edaef4b7c55f5fd5b9cf26`.
- T5 `google/flan-t5-xl`, revision `7d6315df2c2fb742f0f5b556879d730926ca9001`.
- Student `Manojb/stable-diffusion-2-1-base`, revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`: mirror **được README BitMark chỉ rõ**
  vì link StabilityAI cũ không còn khả dụng. Ghi đúng tên mirror trong provenance;
  không tự khẳng định đã đối chiếu byte-for-byte với checkpoint đã bị gỡ.
- Kiểm tra LFS SHA256 của các model đã tải trước khi load; hai file Infinity
  còn có hash hardcode độc lập với metadata Hub. JSON provenance lưu từng file.
- Trainer chính thức Diffusers v0.35.1, commit
  `0f252be0ed42006c125ef4429156cb13ae6c1d60`, được vendor để không bị thay đổi API.
  Vá `lora_layers=filter(...)` thành `list(filter(...))` như fix trong SERUM:
  tránh optimizer dùng cạn iterator khiến clipping mất tác dụng.
  Đồng thời map `--report_to none` thành Python `None` vì Accelerate không chấp
  nhận chuỗi `"none"`; không bật tracker bên ngoài.
  `vendor/provenance.json` lưu hash trước/sau và kiểm tra toàn bộ source khi chạy.

BitMark áp bias **lúc sampling** trên pretrained Infinity; không cần bịa ra một
“watermarked Infinity checkpoint” riêng. Repo tác giả chỉ rõ pretrained weights
và recipe train student, không cung cấp link checkpoint SD2.1 student radioactivity
trong hướng dẫn đó. Pipeline tự train student từ đúng dữ liệu BitMark.

Không dùng `comprehensive_infer.py` của upstream vì file ở commit này có lỗi
`get_detector(args, message)` không đúng signature, tham chiếu `message` không tồn
tại, typo `watermarn_gen_image`, và tổng kết `max()` trên timing list rỗng.
Adapter chỉ thay cách điều phối và ép T5 loader đọc snapshot pin (upstream bỏ qua
tham số `t5_path`); không đổi thuật toán watermark, encoder hay detector.

## Đọc kết quả

```text
bitmark_lora/runs/run_YYYYMMDD_HHMMSS_microseconds/
  config.json, code_manifest.json, environment.json, assets.json
  prompts.json
  logs/                  # môi trường GPU/stage output; lỗi giữ nguyên traceback
  students/              # LoRA/full checkpoint + protocol + verification
  images/                # teacher/student PNG và manifests
  scores/                # signed z, green fraction, SHA256 từng ảnh
  result/
    README.md            # bảng tổng hợp dễ đọc
    threshold.json       # frozen từ clean calibration trước training
    radioactivity.json   # đầy đủ số liệu, paired CI và cảnh báo
    detection_absolute_z.csv
    detection_signed_z.csv
    utility.json         # CLIP, FID, KID; reference=clean teacher test
```

Paper mô tả detection hai phía `|z|`, trong khi ROC script chính thức dùng signed
`z`. Pipeline **báo cáo cả hai**, lấy `|z|` làm primary được chốt từ trước.
Ngưỡng dùng empirical FPR≤1% trên clean calibration, xử lý ties bảo thủ; cùng
ngưỡng cho mọi branch. Đây là FPR mục tiêu trên calibration, không phải bảo đảm
FPR của mọi phân phối. Calibration ở đây là Infinity clean, không phải natural
COCO; xem false positives trên `unadapted` và `teacher_clean_test`.

Báo cáo mức tăng detection so với `unadapted` cho từng student, và chênh lệch
LoRA/full theo cùng prompt/seed:

```text
TPR(full_wm) − TPR(lora_r4_wm)
```

Báo cáo Wilson CI cho từng detection rate và paired bootstrap CI cho hiệu ứng.
Không còn clean-trained controls nên không gọi các hiệu ứng này là clean-adjusted;
không tách được hoàn toàn ảnh hưởng fine-tuning nói chung. `--analyze` vẫn đọc được
kết quả run cũ có clean-trained controls theo config đã lưu của chúng.
Nếu full cũng chưa truyền được watermark, **chưa thể kết luận LoRA là điểm yếu**.
Nếu LoRA tạo ảnh kém hoặc không học đủ thì TPR thấp cũng chưa chứng minh tính
chọn lọc. CLIP/FID/KID giúp kiểm tra utility, không chứng minh task adaptation;
cần tiếp tục khảo sát rank/LR/seed và kiểm tra ảnh trước khi đưa ra claim mạnh.
`green_fraction` không phải bit accuracy của một message/key như Stable Signature.

Không stop vì TPR hay utility thấp: ghi warning, giữ toàn bộ kết quả. Chỉ lỗi
thực thi, checkpoint/hash sai, ảnh thiếu, nonfinite hoặc config resume khác mới
dừng. Kết quả detection được ghi **trước** bước utility để vẫn xem được nếu tải
Inception/CLIP lỗi. CI ảnh không bao gồm biến thiên seed huấn luyện.

## Chạy mở rộng và resume

```bash
# Thêm rank để kiểm tra yếu do capacity hay do adaptation nói chung:
bash run_bitmark_lora.sh --ranks 4 16 64

# Thêm đối chứng full+EMA gần recipe paper, tách khỏi matched full controls:
bash run_bitmark_lora.sh --paper-full-ema

# Chạy seed khác độc lập:
bash run_bitmark_lora.sh --seed 1

# Resume giữ NGUYÊN config/source và checkpoint của run:
bash run_bitmark_lora.sh --resume /absolute/path/to/run_...

# Xem cấu hình mà không chạy; đây là lựa chọn tường minh, không phải mặc định:
bash run_bitmark_lora.sh --plan-only

# Tổng hợp lại từ dữ liệu thật, không cần GPU/Conda:
bash run_bitmark_lora.sh --analyze /absolute/path/to/run_...

# Bộ kiểm tra integrity/statistics/command wiring CPU:
python3 -m unittest discover -s bitmark_lora -p 'test_*.py' -v

# Trong môi trường đã cài dependencies: test thật 2 step/trainer với model tiny
# ngẫu nhiên tạo tại chỗ, reload checkpoint và inference CPU, không tải model:
conda run -n bitmark-radioactivity python bitmark_lora/smoke_cpu.py
```

Partial training chưa có `complete.json` sẽ **train lại branch đó từ đầu**;
không giả vờ tiếp tục optimizer. Teacher/student ảnh hoàn chỉnh được reuse với
manifest; checkpoint đã hoàn tất được hash-verify. Một run chỉ có một tiến trình
ghi; assets và setup env cũng được khóa khi nhiều job dùng chung cache.

Các thay đổi so với paper cần ghi khi viết báo cáo: fixed seed/prompt split của
pipeline này, paired δ=0 sampler, clean-generated calibration, không xFormers
(dùng Torch SDPA), matched controls không EMA, có cả absolute/signed scoring.
Không được gọi kết quả đó là tái hiện chính xác TPR=98.9% của paper.
