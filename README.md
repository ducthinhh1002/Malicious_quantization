# Hướng dẫn chạy thí nghiệm malicious quantization cho watermark diffusion

Script `run_malicious_quantization_watermarks.sh` đánh giá khả năng giữ watermark của **Stable Signature** và **AquaLoRA** sau lượng tử hóa. Quy trình gồm ba giai đoạn:

1. Grid search các recipe PTQ theo bit-width, clipping và nhóm layer.
2. Tối ưu tham số quantizer bằng autograd và straight-through estimator (STE), hoặc dùng zeroth-order search làm ablation.
3. Đánh giá recipe đã chọn trên tập test tách biệt bằng bit accuracy, TPR, PSNR, LPIPS và prediction NMSE.

Đây là simulated weight-only PTQ: trọng số được đưa lên lưới số nguyên low-bit rồi dequantize để chạy bằng CUDA FP16. Kết quả phản ánh sai số lượng tử hóa, không phải tốc độ của INT4/INT8 kernel.

## 1. Cài Conda environment trên login node (một lần)

Script chạy chỉ sử dụng Conda environment đã được kích hoạt; không tạo môi trường hoặc cài package trong job. Cần copy cả repo (bao gồm `requirements-wmq.txt`) vào filesystem mà login node và compute node cùng truy cập được. Conda environment cũng phải nằm trên filesystem dùng chung của cụm.

Trên login node, vào thư mục repo rồi chạy:

```bash
conda create -n wmq python=3.11 pip -y
conda activate wmq
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null PIP_REQUIRE_VIRTUALENV=false
python -m pip install -r requirements-wmq.txt
WMQ_CHECK_ONLY=imports bash run_malicious_quantization_watermarks.sh
```

Nếu đã có môi trường `wmq` dành riêng cho dự án, bỏ lệnh `conda create` và activate nó trước khi cài. Không cài vào `base` hoặc môi trường dùng chung với dự án khác. Conda quản lý Python/môi trường; pip cài các phiên bản thư viện vào chính môi trường đó theo [hướng dẫn Conda về dùng pip trong environment](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).

Login node không cần GPU để cài wheel CUDA và kiểm tra import. Giữ wheel CUDA cho compute node, không chuyển sang wheel CPU chỉ vì login node không có GPU. Cặp Torch/TorchVision lấy từ [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/#v271). File `requirements-wmq.txt` đã chứa nguồn tải và phiên bản CUDA, nên chỉ cần một lệnh pip. Với GPU/driver cũ, đổi đồng thời `cu128` trong URL và hai phiên bản Torch/TorchVision của file này thành `cu118` hoặc `cu126`; Blackwell cần `cu128`. Máy chạy cần Linux x86_64, driver NVIDIA tương thích và dung lượng cho thư viện, model, kết quả.

## 2. Dùng environment đã cài trong job

Trong file submit job đang dùng của server, giữ các dòng khai báo scheduler/tài nguyên và thêm phần sau vào thân job trước lệnh chạy:

```bash
source /duong/dan/miniconda3/etc/profile.d/conda.sh
conda activate wmq
cd /duong/dan/Malicious_quantization
bash run_malicious_quantization_watermarks.sh
```

Thay hai đường dẫn trên bằng đường dẫn thực tế trên server. Trên login node, `conda info --base` cho biết thư mục cài Conda. Nếu cụm yêu cầu `module load` để cung cấp Conda, dùng cơ chế đó theo hướng dẫn của cụm. Kích hoạt tường minh trong job giúp chọn đúng môi trường, kể cả khi scheduler không truyền toàn bộ môi trường của login shell. Không chạy lệnh cài package trong job và không sửa môi trường khi job khác đang dùng nó.

Chưa biết server dùng Slurm, PBS hay scheduler khác nên ví dụ trên chỉ là phần thân job; dùng cấu hình GPU/partition/queue của server. Giữ `CUDA_VISIBLE_DEVICES` do scheduler cấp.

## 3. Kiểm tra trước khi chạy thí nghiệm

Trên login node (không yêu cầu GPU):

```bash
conda activate wmq
WMQ_CHECK_ONLY=imports bash run_malicious_quantization_watermarks.sh
```

Trong job đã được cấp GPU:

```bash
WMQ_CHECK_ONLY=1 bash run_malicious_quantization_watermarks.sh
```

Script chạy `pip check`, kiểm tra phiên bản, toàn bộ import và TorchVision NMS; chế độ `1` còn thử forward/backward CUDA FP16 với convolution và attention. Hai chế độ này không tải model hoặc chạy thí nghiệm. Phiên bản thực tế được lưu trong `wmq_runs/output/environment.freeze.txt`.

Bộ thư viện đã được kiểm tra trước đó trên Python 3.12/Linux CPU: import, NMS, AquaLoRA decoder/Mapper, LPIPS với backbone ngẫu nhiên và UNet nhỏ forward/backward đều qua. Máy phát triển hiện không có Conda/GPU, nên chưa xác minh cài đặt Conda hoặc thí nghiệm GPU đầy đủ trên server.

Chạy thí nghiệm vẫn cần truy cập các model/checkpoint. Nếu compute node không có Internet, phải chuẩn bị cache/model trước trên filesystem dùng chung; kiểm tra import không tải sẵn các tài nguyên này. Nếu model yêu cầu xác thực, đặt `HF_TOKEN` theo quyền truy cập tài khoản.

Script tự chọn profile `large-memory` khi GPU nhìn thấy có ít nhất 80 GiB; GPU nhỏ hơn dùng cấu hình tiết kiệm bộ nhớ.

Model mặc định:

- Stable Signature: `stabilityai/stable-diffusion-2-1-base`, với fallback `sd2-community/stable-diffusion-2-1-base`.
- AquaLoRA: `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- AquaLoRA assets: `georgefen/AquaLoRA-Models/ppft_trained`.

## 4. Chạy cấu hình mặc định

```bash
conda activate wmq
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
| `WMQ_ROOT` | Thư mục script + `/wmq_runs` | Thư mục gốc của một lần chạy |
| `WMQ_OUTPUT` | `$WMQ_ROOT/output` | Thư mục kết quả |
| `WMQ_CACHE` | `$WMQ_ROOT/hf_cache` | Hugging Face cache |
| `CONDA_PREFIX` | Do `conda activate` đặt | Environment đã cài trên login node |
| `WMQ_CHECK_ONLY` | `0` | `1`: import + CUDA; `imports`: chỉ import |
| `TORCH_HOME` | `$WMQ_ROOT/torch_cache` | Cache trọng số TorchVision/LPIPS |
| `CUDA_VISIBLE_DEVICES` | không đặt | Chọn GPU sẽ chạy |

Dependency trực tiếp được ghim trong `requirements-wmq.txt`. Script chỉ kiểm tra môi trường Conda đã activate và chạy; không tự cài đặt. Dependency gián tiếp thực tế được ghi trong `environment.freeze.txt`.

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
