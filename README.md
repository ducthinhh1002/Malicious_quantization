# Hướng dẫn chạy thí nghiệm malicious quantization cho watermark diffusion

Script `run_malicious_quantization_watermarks.sh` đánh giá khả năng giữ watermark của **Stable Signature** và **AquaLoRA** sau lượng tử hóa. Quy trình gồm ba giai đoạn:

1. Grid search các recipe PTQ theo bit-width, clipping và nhóm layer.
2. Tối ưu tham số quantizer bằng autograd và straight-through estimator (STE), hoặc dùng zeroth-order search làm ablation.
3. Đánh giá recipe đã chọn trên tập test tách biệt bằng bit accuracy, TPR, PSNR, LPIPS và prediction NMSE.

Đây là simulated weight-only PTQ: trọng số được đưa lên lưới số nguyên low-bit rồi dequantize để chạy bằng CUDA FP16. Kết quả phản ánh sai số lượng tử hóa, không phải tốc độ của INT4/INT8 kernel.

## 1. Yêu cầu

- Linux 64-bit.
- Bash.
- Python 3.10 hoặc 3.11.
- NVIDIA GPU và driver hoạt động với bản PyTorch CUDA trên server.
- PyTorch và TorchVision có CUDA đã được cài trong Python hệ thống hoặc Conda environment.
- Kết nối Internet trong lần chạy đầu để tải model/checkpoint.
- Dung lượng trống đủ cho SD1.5, SD2.1, checkpoint và ảnh kết quả.

Script tự nhận GPU có ít nhất 80 GiB là `large-memory`. Trên RTX PRO 6000 Blackwell 96 GB, profile này batch quá trình sinh ảnh, tắt attention slicing, giữ pristine weights trên GPU và cập nhật joint-gradient trên toàn bộ semantic group. GPU nhỏ hơn tự dùng profile tiết kiệm bộ nhớ.

Với Blackwell, dùng PyTorch có hỗ trợ kiến trúc này: tối thiểu PyTorch 2.7 với CUDA 12.8, hoặc bản mới hơn tương thích với driver của server. Script kiểm tra CUDA build và dừng sớm nếu phát hiện Blackwell đi cùng PyTorch quá cũ.

Kiểm tra môi trường trước khi chạy:

```bash
python3 --version
nvidia-smi
python3 - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Nếu `torch.cuda.is_available()` là `False`, cần cài lại PyTorch phù hợp với driver trên server trước khi chạy script. Script cố ý không tự thay bản PyTorch của server.

Nếu Hugging Face yêu cầu xác thực hoặc chấp nhận giấy phép model:

```bash
python3 -m pip install -U huggingface-hub
hf auth login
```

Model mặc định:

- Stable Signature: `stabilityai/stable-diffusion-2-1-base`, với fallback `sd2-community/stable-diffusion-2-1-base`.
- AquaLoRA: `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- AquaLoRA assets: `georgefen/AquaLoRA-Models/ppft_trained`.

## 4. Chạy cấu hình mặc định

```bash
CUDA_VISIBLE_DEVICES=0 \
bash run_malicious_quantization_watermarks.sh
```

Trên RTX PRO 6000 Blackwell 96 GB, log đầu chương trình phải hiển thị gần như sau:

```text
large_memory=True, gen_batch=16, metric_batch=32, calib_batch=2,
pristine_on_gpu=True, joint_group_grad=True, attention_slicing=False
```

Không cần đặt thủ công các biến này. Nếu server đang dùng MIG và process chỉ nhìn thấy một partition dưới 80 GiB, profile sẽ tự chuyển sang chế độ tiết kiệm bộ nhớ dựa trên dung lượng CUDA thực tế mà process nhìn thấy.

Mặc định script sử dụng:

| Biến | Giá trị | Ý nghĩa |
|---|---:|---|
| `SEARCH_N` | 20 | Số ảnh dùng để chọn recipe PTQ |
| `TEST_N` | 100 | Số ảnh test độc lập |
| `STEPS` | 25 | Số bước DDIM |
| `GUIDANCE` | 7.0 | Classifier-free guidance scale |
| `MIN_PSNR` | 25.0 | Ngưỡng PSNR tối thiểu |
| `MAX_LPIPS` | 0.15 | Ngưỡng LPIPS tối đa |
| `MAX_PRED_NMSE` | 0.02 | Ngưỡng prediction/decode NMSE |
| `REFINE_OPTIMIZER` | `gradient` | Autograd + STE |
| `REFINE_MODE` | `full` | Tối ưu scale, zero-point và rounding |
| `GRAD_STEPS` | 40 | Số gradient update |
| `GRAD_EVAL_EVERY` | 4 | Khoảng cách giữa các hard evaluation |
| `REFINE_N` | 8 | Số ảnh feedback trong refinement |
| `CALIB_N` | 2 | Số trajectory/latent calibration |
| `CALIB_TIMESTEPS` | 5 | Số timestep lấy trên mỗi trajectory |
| `GEN_BATCH_SIZE` | 16 trên GPU ≥80 GiB | Batch sinh ảnh |
| `METRIC_BATCH_SIZE` | 32 trên GPU ≥80 GiB | Batch LPIPS/PSNR |
| `CALIB_BATCH_SIZE` | 2 trên GPU ≥80 GiB | Batch trajectory calibration |
| `KEEP_PRISTINE_ON_GPU` | 1 trên GPU ≥80 GiB | Tránh copy state qua PCIe mỗi candidate |
| `JOINT_GROUP_GRAD` | 1 trên GPU ≥80 GiB | Cập nhật đồng thời mọi semantic group |
| `USE_ATTENTION_SLICING` | 0 trên GPU ≥80 GiB | Không chia attention thành lát nhỏ |

Grid search mặc định thử 50 recipe cho mỗi watermark:

- Bit-width: 8, 6, 4, 3 và 2 bit.
- Clipping: 1.0 và 0.75.
- Phạm vi: toàn carrier hoặc từng nhóm layer.

Vì mỗi recipe phải sinh ảnh, cấu hình mặc định có thể chạy lâu.

## 5. Chạy cấu hình dùng cho paper

Nên dùng prompt cố định, lưu seed/config và chạy nhiều seed độc lập:

```bash
CUDA_VISIBLE_DEVICES=0 \
WMQ_ROOT="$PWD/runs/seed3407" \
PROMPT_FILE="$PWD/prompts/coco_prompts.txt" \
BASE_SEED=3407 \
SEARCH_N=20 \
TEST_N=100 \
STEPS=25 \
MAX_CANDIDATES=0 \
REFINE_OPTIMIZER=gradient \
REFINE_MODE=full \
GRAD_STEPS=80 \
GRAD_EVAL_EVERY=4 \
GRAD_LR=0.03 \
REFINE_N=20 \
CALIB_N=4 \
CALIB_TIMESTEPS=8 \
MIN_PSNR=25.0 \
MAX_LPIPS=0.15 \
MAX_PRED_NMSE=0.02 \
bash run_malicious_quantization_watermarks.sh
```

`PROMPT_FILE` phải chứa ít nhất `SEARCH_N + TEST_N` prompt. File text dùng một prompt trên mỗi dòng. JSON có thể là:

```json
[
  "a red fox in a snowy forest",
  "a lighthouse above a stormy sea"
]
```

hoặc:

```json
{
  "prompts": [
    "a red fox in a snowy forest",
    "a lighthouse above a stormy sea"
  ]
}
```

Nếu không đặt `PROMPT_FILE`, script tự tạo tập prompt xác định để chạy thử nghiệm sơ bộ.

## 6. Ablation

### Grid PTQ, không refinement

```bash
REFINE_MODE=none \
bash run_malicious_quantization_watermarks.sh
```

### Chỉ tối ưu scale

```bash
REFINE_OPTIMIZER=gradient \
REFINE_MODE=scale \
bash run_malicious_quantization_watermarks.sh
```

### Tối ưu scale và zero-point

```bash
REFINE_OPTIMIZER=gradient \
REFINE_MODE=scale_zero \
bash run_malicious_quantization_watermarks.sh
```

### Tối ưu đầy đủ bằng gradient

```bash
REFINE_OPTIMIZER=gradient \
REFINE_MODE=full \
bash run_malicious_quantization_watermarks.sh
```

### Zeroth-order baseline

```bash
REFINE_OPTIMIZER=zeroth \
REFINE_MODE=full \
REFINE_ITERS=100 \
bash run_malicious_quantization_watermarks.sh
```

Mỗi ablation nên dùng một `WMQ_ROOT` riêng để tránh ghi đè output:

```bash
WMQ_ROOT="$PWD/runs/grid" REFINE_MODE=none bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/scale" REFINE_MODE=scale bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/scale_zero" REFINE_MODE=scale_zero bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/full" REFINE_MODE=full bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/zeroth" REFINE_OPTIMIZER=zeroth REFINE_MODE=full bash run_malicious_quantization_watermarks.sh
```

## 7. Các biến cấu hình

### Đường dẫn và môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `WMQ_ROOT` | `$PWD/wmq_runs` | Thư mục gốc của một lần chạy |
| `WMQ_VENV` | `$WMQ_ROOT/venv` | Virtual environment |
| `WMQ_OUTPUT` | `$WMQ_ROOT/output` | Thư mục kết quả |
| `WMQ_CACHE` | `$WMQ_ROOT/hf_cache` | Hugging Face cache |
| `WMQ_PYTHON` | `python3` | Python dùng để tạo venv |
| `WMQ_SKIP_INSTALL` | `0` | Đặt `1` nếu environment đã có đủ dependency |
| `CUDA_VISIBLE_DEVICES` | không đặt | Chọn GPU sẽ chạy |

Script tạo venv với `--system-site-packages` để sử dụng PyTorch CUDA đã có trên server. Nếu dùng `WMQ_SKIP_INSTALL=1`, `WMQ_PYTHON` phải trỏ tới environment đã cài đủ `diffusers`, `transformers`, `accelerate`, `peft`, `safetensors`, `lpips`, `scipy`, Pillow và NumPy.

### Model và key

| Biến | Mô tả |
|---|---|
| `SS_MODEL` | Stable Signature backbone chính |
| `SS_MODEL_FALLBACK` | Backbone fallback |
| `AQUA_MODEL` | SD1.5 backbone cho AquaLoRA |
| `SS_KEY` | Stable Signature key 48 bit |
| `AQUA_KEY` | AquaLoRA message 48 bit |
| `AQUA_FOLDER` | `ppft_trained` hoặc checkpoint folder khác |

`SS_KEY` và `AQUA_KEY` phải là chuỗi nhị phân đúng 48 ký tự.

### Gradient optimizer

| Biến | Mặc định | Mô tả |
|---|---:|---|
| `GRAD_STEPS` | 40 | Tổng số Adam update |
| `GRAD_LR` | 0.03 | Learning rate của quantizer controls |
| `GRAD_EVAL_EVERY` | 4 | Hard evaluation sau mỗi số bước này |
| `ROUND_TEMPERATURE` | 0.10 | Nhiệt độ sigmoid surrogate cho STE rounding |
| `GRAD_PRED_WEIGHT` | 1.0 | Trọng số prediction/decode NMSE trong gradient loss |
| `GRAD_IMAGE_WEIGHT` | 0.10 | Trọng số image MSE trong gradient loss |
| `TPR_LOSS_WEIGHT` | 0.25 | Trọng số TPR trong hard objective |
| `PRED_LOSS_WEIGHT` | 0.10 | Trọng số prediction NMSE trong hard objective |
| `REFINE_SEED` | 2026 | Seed của zeroth-order optimizer |

Ở RTX PRO 6000 Blackwell 96 GB, mặc định mỗi bước mở graph joint cho toàn bộ semantic group. Trên GPU dưới 80 GiB, script chuyển sang coordinate-gradient: mỗi bước mở graph cho một group, còn các group khác giữ hard-quantized để giảm VRAM.

## 8. Giới hạn và xấp xỉ của gradient refinement

### Coordinate-gradient theo group và timestep

Trong portable profile, mỗi gradient step chỉ mở graph cho:

- một semantic group của carrier;
- một calibration state tại một timestep;
- các group còn lại ở trạng thái hard-quantized hiện tại.

Khi recipe ban đầu chọn toàn carrier, portable profile tối ưu bốn group theo round-robin. Với `GRAD_STEPS=40`, mỗi group nhận khoảng 10 update. Large-memory profile cập nhật cả bốn group ở mỗi bước, nên mỗi group nhận đủ 40 update. Nếu recipe ban đầu chỉ chọn một group thì hai profile tương đương nhau.

Large-memory profile là joint-gradient theo group nhưng mỗi update vẫn dùng một timestep-batch. Portable profile là coordinate-gradient theo cả group và timestep. Khi viết paper phải ghi profile đã dùng và báo cáo ablation theo `GRAD_STEPS`, `CALIB_TIMESTEPS` và `JOINT_GROUP_GRAD`.

### Single-step predicted-x0 proxy

Đối với AquaLoRA, script không backpropagate qua toàn bộ DDIM trajectory. Tại mỗi calibration state, nó:

1. Chạy UNet đã quantize tại `(x_t, t, c)`.
2. Tạo guided noise prediction.
3. Ước lượng `x_0` trực tiếp từ `x_t` và noise prediction.
4. Decode `x_0` bằng VAE cố định.
5. Backpropagate watermark loss từ AquaLoRA decoder.

Thuật ngữ phù hợp là **single-step predicted-x0 differentiable proxy** hoặc **single-timestep reconstruction proxy**. Không gọi đây là full-trajectory differentiable optimization. Cũng không nên gọi là “linearized proxy” trừ khi phương pháp được bổ sung phép khai triển Taylor hoặc Jacobian rõ ràng.

Đối với Stable Signature, carrier nằm trong VAE decoder nên gradient đi trực tiếp qua VAE decoder và Stable Signature extractor trên latent cuối; nhánh này không cần xấp xỉ noise trajectory.

Sau mỗi `GRAD_EVAL_EVERY` bước, quantizer được materialize thành hard low-bit weights và đánh giá bằng quá trình sinh ảnh DDIM hoàn chỉnh. Checkpoint cuối chỉ được chọn từ các hard evaluation thỏa PSNR, LPIPS và prediction/decode NMSE. Bước này kiểm tra proxy bằng hành vi end-to-end nhưng không biến quá trình huấn luyện thành full-trajectory backpropagation.

## 9. Output

Mặc định kết quả nằm trong `wmq_runs/output`:

```text
output/
├── stable_signature/
│   ├── clean_test/
│   ├── attacked_test/
│   ├── search.csv
│   ├── gradient_optimization.csv
│   ├── quantizer_optimization.csv       # chỉ có với zeroth-order
│   └── report.json
├── aqualora/
│   ├── clean_test/
│   ├── attacked_test/
│   ├── search.csv
│   ├── gradient_optimization.csv
│   ├── quantizer_optimization.csv       # chỉ có với zeroth-order
│   └── report.json
├── summary.json
└── watermark_retention.csv
```

Các file quan trọng:

- `search.csv`: toàn bộ 50 recipe PTQ và kết quả trên search split.
- `gradient_optimization.csv`: proxy loss, gradient norm và hard evaluation theo step.
- `report.json`: recipe được chọn và kết quả test của từng watermark.
- `summary.json`: cấu hình và kết quả đầy đủ của cả hai watermark.
- `watermark_retention.csv`: bảng gọn để plot hoặc nhập vào pandas/R/Excel.

Không dùng các chỉ số trên search/refinement split làm kết quả cuối. Báo cáo paper phải lấy `attacked`, `paired_quality` và retention từ test split trong `report.json` hoặc `watermark_retention.csv`.

## 10. Checklist trước khi lấy kết quả paper

- [ ] Smoke test chạy qua cả Stable Signature và AquaLoRA.
- [ ] Clean bit accuracy của cả hai watermark lớn hơn hoặc bằng 0.80.
- [ ] Dùng prompt file cố định có ít nhất `SEARCH_N + TEST_N` mẫu.
- [ ] Search, refinement và test dùng các seed tách biệt theo script.
- [ ] `MAX_CANDIDATES=0` để không cắt recipe.
- [ ] Chạy đủ các ablation `none`, `scale`, `scale_zero`, `full`, `zeroth`.
- [ ] Chạy nhiều `BASE_SEED` và lưu mỗi run trong `WMQ_ROOT` riêng.
- [ ] Báo cáo bit accuracy, TPR, PSNR, LPIPS và prediction NMSE.
- [ ] Kiểm tra `quantizer_grad_norm` khác 0 trong `gradient_optimization.csv`.
- [ ] Ghi rõ dùng joint-group hay coordinate-group gradient và single-step predicted-x0 proxy.
- [ ] Không tuyên bố đã backpropagate qua toàn bộ diffusion trajectory.
- [ ] Chỉ kết luận từ test split, không chọn kết quả theo test split.
