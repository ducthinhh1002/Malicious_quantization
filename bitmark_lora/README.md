# BitMark → LoRA: thí nghiệm radioactivity riêng

Từ thư mục gốc repository, chạy **thật** toàn trình:

```bash
bash run_bitmark_lora.sh
```

Launcher tạo venv riêng `bitmark_lora/.venv`, tự cài bằng pip các phiên bản đã pin,
tải và kiểm tra checkpoint, sinh dữ liệu, train, sinh ảnh student, chạy **detector
chính thức**, tính CLIP/FID/KID và tổng hợp kết quả. Không cần activate thủ công.
Log được in trực tiếp và lưu theo từng stage. Không thay đổi pipeline SERUM cũ.

Máy cần Linux x86_64, Python >=3.9 có pip, GPU Ampere trở lên, driver hỗ trợ PyTorch CUDA 12.8,
Internet và tối thiểu 70 GiB trống theo preflight. Checkpoint teacher, tokenizer,
T5 và SD2.1 tổng cộng hàng chục GB. Thực nghiệm đầy đủ tốn nhiều giờ tùy GPU.
`--min-free-gb` chỉ thay đổi kiểm tra dung lượng, không làm thí nghiệm nhẹ hơn.
FlashAttention dùng wheel CUDA đã pin hash, không tự compile bằng nvcc.

Không cần Conda hoặc sudo. Nếu máy có Python 3.11, launcher dùng Python đó để tạo
venv. Nếu chỉ có Python 3.9/3.10/3.12..., launcher cài `uv==0.9.3` bằng pip vào
cache riêng, dùng uv tải Python 3.11 rồi tạo venv. Dependencies ML vẫn được cài
bằng `venv/bin/python -m pip install -r requirements.txt`. Interpreter tải về nằm
trong `bitmark_lora/cache/environment/python`; giữ thư mục này khi dùng venv.

Có thể chọn vị trí bằng `BITMARK_VENV=/path/to/venv`, chọn Python bootstrap bằng
`BITMARK_PYTHON=/path/to/python3`, và chọn cache bootstrap bằng
`BITMARK_BOOTSTRAP_CACHE=/path/to/cache`. `--cache` là cache model của thí nghiệm,
độc lập với các biến này. Setup dùng khóa file, kiểm tra phiên bản, `pip check`
và import các binary. CUDA kernel được kiểm tra khi bắt đầu experiment; setup
có thể chạy trên login node không có GPU. Nếu venv có sẵn dùng sai Python,
launcher báo lỗi để người dùng chọn một đường dẫn mới.

Server submit job: cài môi trường **một lần trên login node**, sau đó job chỉ dùng
env đã có. Không đưa `--setup-only` vào lệnh chạy experiment:

```bash
bash run_bitmark_lora.sh --setup-only
# Trong GPU job:
bash run_bitmark_lora.sh
```

## Google Colab

Chọn GPU Ampere trở lên (ví dụ A100/L4) và đủ RAM/VRAM cho Infinity-2B + T5 và
SD2.1 full fine-tune. Cấu hình FlashAttention hiện tại không hỗ trợ T4. Launcher
kiểm tra compute capability trước khi tải môi trường; không tự giảm batch/data
hay đổi thuật toán để vừa GPU.

Trong cell Colab, có thể clone source lên ổ local:

```python
%cd /content
!git clone https://github.com/ducthinhh1002/Malicious_quantization.git bitmark-colab
%cd /content/bitmark-colab
!bash run_bitmark_lora.sh
```

Nếu repository đã clone, dùng `%cd` vào repository và `!git pull --ff-only` trước
khi chạy lại. Source nằm trên `/content/drive/...` cũng được hỗ trợ, kể cả đường
dẫn có khoảng trắng. Launcher tự nhận diện Colab và đặt runtime ở:

```text
/content/bitmark_runtime/.venv
/content/bitmark_runtime/cache/environment   # uv và Python riêng
/content/bitmark_runtime/cache               # model cache
/content/bitmark_runtime/runs                # ảnh, checkpoint, kết quả
```

Không đặt venv/cache/output đang chạy trên Drive mount: các thao tác khóa,
symlink và atomic writes không tương đương ổ Linux local. Bản sửa gọi trực tiếp
binary `uv==0.9.3` đã cài, kiểm tra phiên bản và bỏ biến `UV_INTERNAL_BIN` kế thừa;
tránh chọn nhầm uv có sẵn trong Colab. Python notebook 3.13 vẫn có thể bootstrap
môi trường Python 3.11 riêng.

`/content` là dữ liệu tạm của runtime. Sao lưu thư mục run sang Drive trước khi
xóa runtime; cache/interpreter/venv cần tạo lại ở phiên mới. Cơ chế resume vẫn yêu
cầu đúng config/source và model paths đã lưu. Khuyến nghị của Colab là giảm I/O
trực tiếp trên Drive và xử lý dữ liệu ở ổ VM local:
https://research.google.com/colaboratory/faq.html

## Đánh giá ý tưởng và phạm vi kết luận

Thiết kế đủ để làm pilot kiểm tra **radioactivity phụ thuộc cách adaptation**.
BitMark đã báo cáo truyền watermark sang SD2.1 full fine-tune, nên full là positive
control thích hợp cho việc so với LoRA trên cùng dữ liệu. Mặc định rank 4, một seed
và một ngân sách chưa đủ để khẳng định LoRA là điểm yếu chung hoặc đủ novelty cho
một bài top conference. SERUM đã báo cáo radioactivity qua LoRA; không nên đặt giả
thuyết tuyệt đối rằng LoRA không truyền watermark.

Kết quả thuyết phục cần: full truyền rõ, LoRA truyền yếu nhưng vẫn học được phân
phối/capability của teacher, hiệu ứng lặp lại qua rank/LR/ngân sách và seed. Cùng
LR và số step giúp kiểm soát recipe nhưng không bảo đảm hai phương pháp hội tụ
tương đương. CLIP/FID/KID chỉ là kiểm tra utility; ảnh đẹp từ model gốc vẫn có thể
che việc LoRA chưa học đủ. Cần thêm đánh giá adaptation trên validation độc lập
khi chuyển từ pilot sang kết luận chính. Nên xét đường cong theo ngân sách vì
kết quả SERUM trước đó cho thấy hiệu ứng mạnh ở 3k có thể mất ở 20k.

`--train-clean-controls` bổ sung clean-trained LoRA/full để đo interaction
`(full_wm-full_clean) - (lora_wm-lora_clean)` và loại bớt confound từ fine-tuning
nói chung. Rank sweep đã có; có thể thêm full+EMA gần recipe tác giả bằng
`--paper-full-ema`. Những nhánh này tốn thêm compute và chỉ chạy khi bật cờ.
`--seed` hiện thay cả chia prompt, generation và training; đó là lặp toàn pipeline,
chưa phải phép đo chỉ biến thiên seed optimizer trên một dataset cố định.

Nguồn đối chiếu: [BitMark recipe chính thức](https://github.com/sprintml/BitMark#radioactivity)
và [SERUM, Appendix I](https://adam-dziedzic.com/static/assets/papers/serum.pdf).

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
Mặc định không có clean-trained controls nên không gọi các hiệu ứng này là clean-adjusted;
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

# Pilot mở rộng để kiểm tra capacity và ảnh hưởng fine-tuning nói chung:
bash run_bitmark_lora.sh --ranks 4 16 64 --train-clean-controls --paper-full-ema

# Chạy seed khác độc lập:
bash run_bitmark_lora.sh --seed 1

# Resume giữ NGUYÊN config/source và checkpoint của run:
bash run_bitmark_lora.sh --resume /absolute/path/to/run_...

# Xem cấu hình mà không chạy; đây là lựa chọn tường minh, không phải mặc định:
bash run_bitmark_lora.sh --plan-only

# Tổng hợp lại từ dữ liệu thật, chỉ cần Python standard library:
bash run_bitmark_lora.sh --analyze /absolute/path/to/run_...

# Bộ kiểm tra integrity/statistics/command wiring CPU:
python3 -m unittest discover -s bitmark_lora -p 'test_*.py' -v

# Trong môi trường đã cài dependencies: test thật 2 step/trainer với model tiny
# ngẫu nhiên tạo tại chỗ, reload checkpoint và inference CPU, không tải model:
bitmark_lora/.venv/bin/python bitmark_lora/smoke_cpu.py
```

Partial training chưa có `complete.json` sẽ **train lại branch đó từ đầu**;
không giả vờ tiếp tục optimizer. Teacher/student ảnh hoàn chỉnh được reuse với
manifest; checkpoint đã hoàn tất được hash-verify. Một run chỉ có một tiến trình
ghi; assets và setup env cũng được khóa khi nhiều job dùng chung cache.

Các thay đổi so với paper cần ghi khi viết báo cáo: fixed seed/prompt split của
pipeline này, paired δ=0 sampler, clean-generated calibration, không xFormers
(dùng Torch SDPA), matched controls không EMA, có cả absolute/signed scoring.
Không được gọi kết quả đó là tái hiện chính xác TPR=98.9% của paper.
