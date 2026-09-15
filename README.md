# Hướng dẫn chạy thí nghiệm malicious quantization cho watermark diffusion

Script `run_malicious_quantization_watermarks.sh` đánh giá khả năng giữ watermark của **Stable Signature** và **AquaLoRA** sau lượng tử hóa. Mặc định dùng `RUN_PROTOCOL=fair_w4a16` để so sánh cùng bitwidth/phạm vi; quy trình grid rộng trước đây còn ở `RUN_PROTOCOL=legacy_grid`:

1. Grid search các recipe PTQ theo bit-width, clipping và nhóm layer.
2. Tối ưu tham số quantizer bằng autograd và straight-through estimator (STE), hoặc dùng zeroth-order search làm ablation.
3. Đánh giá recipe đã chọn trên tập test tách biệt bằng bit accuracy, TPR, PSNR, LPIPS và prediction NMSE.

Đây là simulated weight-only PTQ: trọng số được đưa lên lưới số nguyên low-bit rồi dequantize để chạy bằng CUDA FP16. Kết quả phản ánh sai số lượng tử hóa, không phải tốc độ của INT4/INT8 kernel. Copy cả `wmq_baselines.py` cùng script và requirements sang server.

## So sánh W4A16 mới

Trong job đã activate Conda, chạy:

```bash
WMQ_OUTPUT="$PWD/results/w4a16_seed3407" bash run_malicious_quantization_watermarks.sh
```

Dùng thư mục output mới cho mỗi run; script dừng trước khi tải model nếu thư mục kết quả fair đã tồn tại. Nó không ghi đè kết quả cũ. Không cần thêm dependency ngoài `requirements-wmq.txt`.

| Phương pháp | VAE Stable Signature | UNet AquaLoRA |
|---|---|---|
| `rtn_w4a16` | Làm tròn thông thường | Làm tròn thông thường |
| `grid_w4a16` | Chọn clipping 1,0/0,75 trên search | Tương tự |
| `gradient_w4a16` | Tối ưu theo watermark; báo rõ nếu fallback | Tương tự |
| `adaround_vae` | Học làm tròn từng weight theo đầu ra layer | — |
| `brecq_vae_adapted` | Hiệu chỉnh theo block | — |
| `qdiff_unet_adapted` | — | Hiệu chỉnh theo block, nhiều timestep, tách grid cho shortcut concat |

Mọi phương pháp trong cùng watermark đều lượng tử hóa **cùng toàn bộ target**: trọng số Conv/Linear của VAE decoder + post-quant conv, hoặc của UNet. Encoder VAE, bias, normalization và watermark extractor không được lượng tử hóa. Trọng số dùng 4 bit (grid đối xứng -7..7), activation giữ FP16; scale lưu FP32. Q-Diffusion adaptation có thêm scale riêng cho hai phần input concat, được ghi trong `shortcut_splits`; cùng bitwidth không có nghĩa overhead scale giống nhau. Tái tạo block chạy FP32 để ổn định gradient, sau đó weight dequantized được đưa về dtype inference.

Các baseline reconstruction là **bản thích nghi cục bộ cho Diffusers**, không phải chạy nguyên repo hoặc tái lập nguyên số liệu paper. `AdaRound` dùng stretched-sigmoid rounding và regularization; BRECQ adaptation dùng MSE đầu ra block, không dùng Fisher weighting; Q-Diffusion adaptation kết hợp block reconstruction với replay nhiều timestep và split shortcut trên Conv1/Conv-shortcut của up-block. Input calibration của từng block được thu tuần tự khi các block trước đã được lượng tử hóa. Không baseline reconstruction nào dùng key/loss watermark. Chi tiết nằm trong `wmq_baselines.py` và từng report. Nguồn: [AdaRound](https://arxiv.org/abs/2004.10568), [BRECQ](https://arxiv.org/abs/2102.05426), [Q-Diffusion](https://arxiv.org/abs/2302.04304).

Ba phần dữ liệu tách biệt theo chỉ số và seed: `CALIB_N` prompt calibration, `SEARCH_N` prompt chọn cấu hình, `TEST_N` prompt test. Mọi phương pháp tái sử dụng đúng các phần này. `PROMPT_FILE` cần ít nhất `CALIB_N + SEARCH_N + TEST_N` dòng. Manifest lưu prompt/seed, scheduler, target names, số tham số và hash checkpoint. Cấu hình cuối được chọn bằng search/calibration; test chỉ đánh giá. Các phương pháp có ngân sách tối ưu khác nhau, được báo cáo qua số step/update; đây không phải so sánh cùng chi phí tính toán.

Grid, refinement và fallback đều được kiểm tra **PSNR ≥25, LPIPS ≤0,15 và calibration NMSE ≤0,02**. Nếu không có candidate đạt, report vẫn lưu kết quả với `feasible=false`; không gọi đó là cấu hình thành công. Bản gradient có `selected_source=grid_fallback` khi không chọn được refinement hợp lệ. File `gradient_updates.json` được ghi mỗi bước, gồm số update hợp lệ, bị bỏ qua và lý do; không tính update NaN đã rollback là thành công.

Kiểm tra chất lượng watermark trước lượng tử hóa trên GPU:

```bash
BASELINE_CHECK_ONLY=1 WMQ_OUTPUT="$PWD/results/clean_check" \
  bash run_malicious_quantization_watermarks.sh
```

Khác với `WMQ_CHECK_ONLY=imports`, lệnh này **tải model và sinh ảnh**. Mặc định yêu cầu clean search bit accuracy ≥0,80 và TPR ≥0,90. Đây là quality gate được đặt trước thí nghiệm, không phải bằng chứng checkpoint sai nếu không đạt. `clean_validation.json` luôn được ghi trước khi quyết định. Với `CLEAN_POLICY=strict` (mặc định), watermark không đạt bị dừng riêng với `invalid_clean_baseline`, watermark còn lại vẫn chạy. `CLEAN_POLICY=report` cho phép chạy chẩn đoán nhưng mọi dòng kết quả giữ `baseline_valid=false`; không nên dùng như một baseline đã được xác minh. Có thể đặt `CLEAN_MIN_BITACC`/`CLEAN_MIN_TPR` trước run theo protocol nghiên cứu.

| Biến mới | Mặc định | Ý nghĩa |
|---|---:|---|
| `RUN_PROTOCOL` | `fair_w4a16` | Hoặc `legacy_grid` cho tìm kiếm rộng cũ |
| `RECON_STEPS` | 200 | Số update mỗi layer/block |
| `RECON_LR` | 0,001 | Learning rate cho reconstruction |
| `RECON_CACHE_MB` | 512 | Giới hạn cache CPU cho một block |
| `CLEAN_POLICY` | `strict` | Hoặc `report` cho chẩn đoán |
| `BASELINE_CHECK_ONLY` | 0 | Đặt 1 để chỉ kiểm tra clean trên GPU |

Trong fair protocol, `REFINE_N` và `MAX_CANDIDATES` không dùng: gradient dùng toàn bộ search split, grid luôn là W4 trên toàn target với hai clipping. Zeroth-order vẫn có ở `RUN_PROTOCOL=legacy_grid`. `REFINE_MODE=none` bỏ dòng gradient nhưng vẫn chạy các baseline reconstruction.

Kết quả mới:

```text
results/w4a16_seed3407/
├── comparison.csv
├── comparison_summary.json
└── <watermark>/fair_w4a16/
    ├── manifest.json
    ├── clean_validation.json
    ├── clean_test.json
    ├── grid_search.json
    ├── report.json
    ├── clean_search/
    ├── clean_test/
    └── <method>/
        ├── report.json
        ├── attacked_test/
        ├── reconstruction.json       # baseline reconstruction
        ├── gradient_updates.json     # gradient, ghi mỗi bước
        └── gradient_optimization.csv # gradient, ghi cuối loop
```

Nếu clean gate không đạt, chưa có các file search/test/method tương ứng; summary ghi `incomplete`. Các report method đã hoàn thành được lưu ngay, không phải chờ cả AquaLoRA kết thúc. Reconstruction replay model một lần cho mỗi unit/tập calibration, nên chi phí có thể lớn. Bắt đầu bằng run nhỏ trước khi chạy dataset đầy đủ:

```bash
SEARCH_N=2 TEST_N=2 CALIB_N=1 CALIB_TIMESTEPS=2 STEPS=4 \
RECON_STEPS=2 GRAD_STEPS=2 GEN_BATCH_SIZE=1 CALIB_BATCH_SIZE=1 \
WMQ_OUTPUT="$PWD/results/w4a16_smoke" bash run_malicious_quantization_watermarks.sh
```

Kiểm thử code không cần GPU/checkpoint:

```bash
python -m unittest discover -s tests -v
```

Các test chạy Conv/Linear, UNet và VAE Diffusers nhỏ với trọng số ngẫu nhiên; kiểm tra grid, backward, coverage, input concat, rollback NaN/Inf, fallback NMSE và báo cáo/split. Chúng không chứng minh chất lượng watermark hay mức bộ nhớ trên checkpoint thật; cần smoke test trên GPU server.

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

Ở `RUN_PROTOCOL=legacy_grid`, grid search thử 50 recipe cho mỗi watermark:

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

`PROMPT_FILE` phải chứa ít nhất `CALIB_N + SEARCH_N + TEST_N` prompt cho fair protocol (`SEARCH_N + TEST_N` cho legacy). File text dùng một prompt trên mỗi dòng. JSON có thể là:

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
RUN_PROTOCOL=legacy_grid \
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

Kết quả nằm trong `wmq_runs/output` hoặc `WMQ_OUTPUT`. Cây dưới đây là **legacy_grid**; output fair mặc định được mô tả ở đầu tài liệu:

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

## Stable Signature: attacker chỉ có model đã fingerprint

**Lệnh chạy đầy đủ, sau khi đã cài dependencies và activate môi trường:**

```bash
conda activate wmq
bash run_blind_quantization.sh
```

Lần đầu script tự tải backbone SD2.1-base và VAE Stable Signature công khai, chuẩn
bị model fingerprint tại `wmq_runs/marked_sd21`, rồi chạy experiment. Các lần sau
tái sử dụng model đã chuẩn bị. `prompt.txt` đã có sẵn 160 prompt tiếng Anh khác
nhau; mặc định dùng 152 dòng đầu (32 train, 20 search, 100 test). Đây là danh sách
viết tay cho thử nghiệm ban đầu, không phải benchmark COCO hay DrawBench.
Mỗi lần chạy tạo thư mục `output_attack/blind_<thời gian>_<PID>` mới và file log
riêng trong `output_attack/`. Model chuẩn bị vẫn ở `wmq_runs/marked_sd21`.
Đường dẫn được tính theo thư mục script, nên không phụ thuộc nơi gọi lệnh.

Checkpoint VAE tải từ Meta phải khớp SHA256 pin sẵn trước `torch.load`. Script
chuẩn bị giữ khóa liên tiến trình cho toàn bộ bước tạo/reuse fixture; các launcher
chạy song song sẽ chờ rồi dùng lại fixture hoàn tất. Thư mục đích được tạo bằng
`mkdir` độc quyền, không rename đè lên đích. Nếu quá trình bị ngắt giữa publication,
fixture thiếu completion marker sẽ bị từ chối; không tự coi model dở dang là hợp lệ.

Smoke test ngắn, cũng tự chuẩn bị model nếu cần:

```bash
bash run_blind_quantization.sh --train-n 2 --search-n 2 --test-n 2 \
  --bits 4 --clips 1.0 --steps 2 --eval-every 1
```

Có thể truyền `--model`, `--prompts`, `--output` hoặc các tham số khác để ghi đè
mặc định. Khi truyền `--model`, script dùng model của bạn và không chuẩn bị model
công khai. Bước tự chuẩn bị thuộc vai trò tổ chức thí nghiệm; chương trình tối ưu
vẫn chỉ nạp model đã fingerprint. Bước đo watermark bằng key/extractor vẫn chạy
riêng sau cùng như hướng dẫn bên dưới.

Nhánh `run_blind_quantization.sh` dùng `wmq_blind.py`, không dùng key, extractor,
detector feedback, model sạch hoặc ảnh sạch bên ngoài. Đây là phương pháp thử nghiệm:
học rounding từng weight trong VAE theo pseudo-target làm dịu chi tiết ảnh tự sinh.
Chưa có kết quả chứng minh nó phá watermark mạnh hơn legacy attack. Đọc
[phân tích nghiên cứu và threat model](survey/Blind_Quantization_Stable_Signature.md)
trước khi diễn giải kết quả.

Chuẩn bị môi trường theo requirements phía trên. Nếu đã có model fingerprint dạng
Diffusers, dùng đường dẫn đó cho `--model`. Nếu dùng checkpoint công khai để tổ chức
thí nghiệm, tạo fixture một lần (CPU, cần mạng và đủ RAM/disk cho model FP32):

```bash
python prepare_marked_fixture.py --output "$PWD/wmq_runs/marked_sd21"
```

Script chuẩn bị tải backbone SD2.1-base và VAE đã nhúng Stable Signature chính thức,
xuất pipeline có watermark. Attacker chỉ nhận thư mục đã xuất này. Không thay VAE
bằng VAE sạch khi chạy attack. Với thí nghiệm nghiêm ngặt, tách môi trường chuẩn bị
fixture và môi trường attacker.

Nếu dùng danh sách riêng, chuẩn bị ít nhất **152 prompt khác nhau**, mỗi dòng một
prompt, rồi truyền `--prompts`. Dưới đây là lệnh đầy đủ tương đương với mặc định,
dùng `prompt.txt` có sẵn và đường dẫn output do bạn chọn:

```bash
bash run_blind_quantization.sh \
  --model "$PWD/wmq_runs/marked_sd21" \
  --prompts "$PWD/prompt.txt" \
  --output "$PWD/output_attack/blind_seed3407" \
  --bits 4 3 2 --clips 1.0 0.9 \
  --steps 100 --eval-every 10 --strength 0.25
```

Script không hardcode tên GPU. VAE/rounding chạy FP32; UNet sinh latent chạy FP16.
Cache latent nên UNet chỉ sinh một lần mỗi prompt; mỗi candidate chỉ chạy VAE.
Toàn bộ rounding trong `--scope all` được học đồng thời, batch train là 1 để hạn
chế VRAM. `--scope late` giảm phạm vi xuống up-blocks và conv_out khi nghiên cứu
ablation nhỏ hơn; không tự đổi phạm vi vì như vậy thay thí nghiệm. Mặc định chạy
6 cấu hình ×100 update, chưa có benchmark thời gian trên RTX 6000 96GB.

Smoke test nhanh trước full run (cần 6 prompt khác nhau):

```bash
bash run_blind_quantization.sh \
  --model "$PWD/wmq_runs/marked_sd21" --prompts "$PWD/prompt.txt" \
  --output "$PWD/output_attack/blind_smoke" \
  --train-n 2 --search-n 2 --test-n 2 \
  --bits 4 --clips 1.0 --steps 2 --eval-every 1
```

Nếu smoke báo `no_feasible_candidate`, code có thể vẫn chạy đúng: không có cấu
hình đạt chất lượng với budget ngắn. Không tự nới ngưỡng để gọi đó là thành công.

Quality gate phía attacker: mean PSNR >=25, mean SSIM >=0.9, min per-image PSNR
>=22 dB, so với output từ model fingerprint ban đầu. Đây không phải quality gate
LPIPS của script cũ. Chọn candidate bằng MSE tới pseudo-target trên search; không
chọn bằng watermark score. `--strength 0` là reconstruction control nên chạy trong
thư mục mới, cùng dữ liệu/budget. Chốt strength trước khi xem owner metrics.

Nhánh này không chặn bằng clean bit accuracy/TPR như AquaLoRA. Nếu mọi candidate
trượt quality gate, vẫn có manifest, search và log; chỉ không xuất VAE/test images.
Chưa có bằng chứng từ full run để kết luận các ngưỡng mặc định quá cao. Nếu muốn
chạy một protocol chất lượng nới nhẹ đã định trước, dùng lệnh riêng dưới đây và
báo rõ ngưỡng khác khi so sánh kết quả:

```bash
bash run_blind_quantization.sh --min-psnr 24 --min-ssim 0.88 --min-image-psnr 21
```

Output:

```text
blind_seed3407/
  manifest.json                # Threat model, model/script hashes, prompts/seeds
  search.json                  # Cả RTN và hard checkpoints, feasible/non-feasible
  selection.json               # Cấu hình khóa trước test
  selection_frozen.json        # Lựa chọn + hash checkpoint/search trước khi sinh test
  updates_w*_c*.json           # Loss và gradient norm
  report.json                  # Test quality, elapsed time, peak CUDA allocation
  quantized_vae/               # Diffusers VAE, low-bit values lưu dưới dạng FP32
  marked_reference_test/       # Vẫn có watermark
  matched_rtn_test/            # RTN cùng bits, clip và target coverage đã chọn
  attacked_test/
```

Không có candidate hợp lệ: chỉ lưu chẩn đoán, không xuất VAE. Test quality không
đạt: report ghi `heldout_quality_failed`; VAE vẫn được giữ để kiểm toán. Low-bit
weights chạy qua FP32 kernel VAE, không phải INT4 inference hay packed checkpoint.

Sau khi khóa cấu hình, **chủ sở hữu** mới đo watermark trong một chương trình riêng:

```bash
python evaluate_blind_watermark.py \
  --run "$PWD/output_attack/blind_seed3407" \
  --extractor /path/to/dec_48b_whit.torchscript.pt \
  --key 111010110101000001010111010011010100010000100111 \
  --fpr 0.001 --lpips
```

Key trên thuộc fixture Stable Signature công khai; thay bằng key đúng nếu dùng model
khác. Owner có thể lấy extractor chính thức từ
[Meta](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt).
Không cấp extractor/key cho chương trình attacker. Owner evaluator lưu
`owner_evaluation.json` và `watermark_retention.csv`, gồm before/RTN/after,
bit accuracy, TPR, retention, survival trên các ảnh ban đầu được phát hiện và
LPIPS nếu bật. FPR là ngưỡng lý thuyết theo giả định bit ngẫu nhiên độc lập.
Không được dùng các số test đó quay lại chọn candidate của cùng blind run.

Evaluator từ chối FPR không thể đạt với độ dài key được cấp, trước khi nạp extractor.
Nó kiểm tra `selection_frozen.json` với final report và hash checkpoint/search hiện
tại. Run cũ thiếu file freeze cần chạy lại bằng phiên bản mới. Các kiểm tra này giúp
phát hiện sai lệch artifact, không phải chứng minh chống người sửa đồng thời code
và mọi metadata. Source audit và kiểm thử luồng dữ liệu vẫn cần thiết.

Nạp VAE đã xuất để inference với cùng model fingerprint:

```python
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, DDIMScheduler
pipe = StableDiffusionPipeline.from_pretrained(
    "wmq_runs/marked_sd21", torch_dtype=torch.float16)
pipe.vae = AutoencoderKL.from_pretrained(
    "output_attack/blind_seed3407/quantized_vae", torch_dtype=torch.float32)
pipe.to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# StableDiffusionPipeline không tự ép latent FP16 sang dtype VAE ở mọi phiên bản.
with torch.inference_mode():
    latents = pipe("a lighthouse above a stormy sea", num_inference_steps=25,
                   guidance_scale=7.0, output_type="latent").images
    decoded = pipe.vae.decode(latents.float() / pipe.vae.config.scaling_factor,
                              return_dict=False)[0]
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
image.save("sample.png")
```

Kiểm tra lõi không cần checkpoint:

```bash
bash -n run_blind_quantization.sh
python -m unittest discover -s tests -p test_wmq_blind.py -v
```
