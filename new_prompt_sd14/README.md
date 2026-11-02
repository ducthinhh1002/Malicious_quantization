# New-prompt SD1.4 — COCO disjoint

Bản sao code, prompt, cấu hình, log và kết quả của nhánh chạy ngày 01/11/2026.
Nguồn: server 10.1.21.21, `/var/lib/structure32_newprompts_rank32/sd14_coco_disjoint`;
ảnh/prompt dùng chung từ `coco_disjoint_source`, giống nhánh Sana/PixArt COCO hôm đó.

## Thực nghiệm

- Teacher: Structure **32 bit, refine600**, giữ nguyên checkpoint, không train lại teacher.
- Student: SD1.4 sạch (`CompVis/stable-diffusion-v1-4`), LoRA rank32, attention + convolution.
- COCO2014: 5.000 train / 1.000 validation / 1.000 test; không trùng image ID với các tập teacher gốc. Đây là đổi tập prompt trong COCO, không đủ để khẳng định đổi miền dữ liệu.
- 20.000 bước, batch4, seed0, FP16. LR 1e-4 đến bước7500, cosine xuống2e-5 tại12500, giữ sau đó.
- Chọn cố định bước20000; không chọn checkpoint theo test/detection.
- Prompt SHA256: `5cf63b020893205a1865fa9f238bc81d63d1cd50bdb3d0a0958b007f4cdd84c2`.

## Nội dung

- `source/prompts.json`: toàn bộ prompt và split chính xác.
- `source/config.json`, `assets.json`: cấu hình, revision và hash model nền.
- `source/carrier/`, `source/result/`: carrier và ngưỡng đọc watermark.
- `pipeline/student/`: orchestration, lịch LR, optimizer/RNG resume, metric.
- `pipeline/student/runs/`: cấu hình thực chạy, log/loss, kết quả và provenance.
- `harness/`: harness thực chạy và trainer Diffusers vendored.
- `DOWNLOAD_MANIFEST.json`: danh sách/hash nguồn; bao gồm cả ảnh trong bản export server. **Manifest không có nghĩa là ảnh đã tải về**.

Theo yêu cầu cuối: chỉ tải code/prompt và metadata; không tải ảnh hoặc checkpoint. Các đường dẫn checkpoint/ảnh trong JSON là tham chiếu lịch sử, không phải các file được đóng gói.

## Chạy lại

Dùng Linux/CUDA và môi trường tương thích trainer (Torch2.7.1, Diffusers0.35.1, Transformers4.51.3, PEFT0.17.1, Accelerate1.10.1). Xem thêm requirements và manifest trong harness.
Cần bổ sung checkpoint teacher refine600, base SD2.1/SD1.4 và ảnh teacher đúng manifest; hoặc sinh lại ảnh từ teacher với đúng prompt/seed/sampler trước khi train.

Entry point lịch sử là `pipeline/student/campaign.py`; `driver.py` bọc trainer `harness/vendor/diffusers/train_text_to_image_lora.py`. Các symlink gốc được ghi trong `DOWNLOAD_MANIFEST.json` (`original_links`). Khi triển khai mới cần trỏ `pipeline/source` về `source`, `pipeline/harness` về `harness`, sửa đường dẫn tuyệt đối trong assets/config và tạo identity mới cho lần chạy mới. Không sửa các bản provenance gốc để giả là lần chạy lịch sử.

Đây là bản lưu đầy đủ phần code/prompt đã lấy được, **không phải gói chạy ngay khi thiếu ảnh và trọng số**.
