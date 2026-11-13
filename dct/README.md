# DCT32 guidance → SD1.4 LoRA rank4

Đây là nhánh DCT đã chạy ngày11/11 tại `/var/lib/baseline_mentor35/run/latent32_student_r4_20261111`. Nó là **biến thể GROW-style progressive guidance**, độc lập với adapter Structure refine600 và nhánh `jpeg/`. Trọng số teacher giữ nguyên; gradient watermark điều chỉnh latent trong lúc sinh ảnh.

## Chạy end-to-end

Yêu cầu Linux, Python3.11/3.12, `flock`, NVIDIA GPU hỗ trợ BF16/CUDA12.8 và internet để tải model. Khuyến nghị GPU48GB trở lên và100GB disk trống; thời gian chạy gồm6.000 ảnh teacher,12.500 bước student và1.000 ảnh student.

```bash
git clone https://github.com/ducthinhh1002/Malicious_quantization.git
cd Malicious_quantization
python3 dct/verify_bundle.py
bash dct/run.sh
```

Script tự tạo `dct/.venv`, cài dependency, tải SD2.1 và SD1.4 tại revision ghim, sinh ảnh teacher, train student, sinh ảnh student và chấm nguyên bản/JPEG25 cùng agg32. Nếu model cần quyền truy cập Hugging Face, cung cấp `HF_TOKEN` của người chạy trong environment.

Mặc định dùng GPU0, batch sinh teacher2. Đổi GPU bằng

```bash
CUDA_VISIBLE_DEVICES=1 bash dct/run.sh
```

Để dùng cấu hình hai worker của lượt server cũ

```bash
CUDA_VISIBLE_DEVICES=1 DCT_WORKERS=0:2,1:4 bash dct/run.sh
```

`DCT_WORKERS` gồm các cặp `GPU:batch`, dùng GPU index vật lý. GPU trong `CUDA_VISIBLE_DEVICES` là GPU chạy student và đánh giá. Worker tự đặt thiết bị của nó theo danh sách. Các worker nhận việc từ queue dùng file lock, mỗi ảnh chỉ có một writer. Khi resume, queue được dựng lại từ những PNG còn thiếu.

Có thể dùng interpreter đã cài đủ dependency, chẳng hạn môi trường của `jpeg/`, để tiết kiệm dung lượng

```bash
DCT_PYTHON="$PWD/jpeg/.venv/bin/python" bash dct/run.sh
```

Môi trường này chỉ được dùng cho dependency. Nhánh DCT không đọc adapter/carrier của `jpeg/`.

## Dùng model local

```bash
DCT_TEACHER_MODEL=/path/to/sd21_snapshot \
DCT_STUDENT_MODEL=/path/to/sd14_snapshot \
bash dct/run.sh
```

Cần dùng đúng model/revision để tái lập. Thư mục phải có cấu trúc Diffusers đầy đủ. Các pin nằm trong `prepare.py` và protocol được ghi khi chạy.

## Cơ chế và thiết lập

- Teacher SD2.1 FP32, DDIM50, CFG7.5,512×512.
- DCT-II trực chuẩn trên latent4×64×64. Band[12,32) tạo1.600 hệ số.
- Payload ASCII `NW32`,32 bit, khoảng50 bản sao/bit. Seed bố trí là `latent32-dct-pilot-20261111`; đây là khóa công khai để tái lập thí nghiệm.
- Target amplitude0.5, guidance scale200, bắt đầu ở bước25/50.
- Thay DCT proxy của bản GROW được triển khai bằng DCT trực chuẩn. Code guidance, target construction và majority voting còn lại được tái sử dụng từ GROW.
- Student clean SD1.4, LoRA rank4 attention+conv, batch4, seed0,12.500 bước, checkpoint mỗi1.250. Wrapper lịch sử áp dụng lịch LR có cooldown; lệnh CLI gốc lưu trong `server_results/training_command.json`.
- Student học ảnh PNG nguyên bản của teacher. JPEG25 chỉ được áp dụng khi chấm robustness.
- Teacher dùng5.000 train và1.000 test prompt từ manifest chung đã ghim, SHA256 `5cf63b020893205a1865fa9f238bc81d63d1cd50bdb3d0a0958b007f4cdd84c2`.

Đây là cấu hình thực nghiệm của nhóm. Không coi nó là tái hiện nguyên xi cấu hình/payload của paper GROW.

## Output và resume

- `run/latent32/status.json`, `logs/`, `run/worker_gpu*.json` ghi tiến độ.
- `run/latent32/images/teacher_wm_train` và `teacher_wm_test` lưu ảnh teacher.
- `run/latent32/student/students/sd14_r4/checkpoint-12500` chứa checkpoint student cuối.
- `run/latent32/student/evaluation/12500/results.json` chứa điểm từng ảnh.
- `run/latent32/student/evaluation/12500/aggregate32_results.json` chứa agg32.

Chạy lại cùng lệnh để resume. Student khôi phục optimizer/RNG từ checkpoint hoàn chỉnh. Giữ nguyên code, model paths và worker configuration của run đang tiếp tục. Tên tiến trình GPU là `VLLM::Enginecore`.

## Metric

BA là số bit đúng/32. `accept_ge24` là tỷ lệ đúng ít nhất24 bit. `exact32` yêu cầu đúng đủ32 bit. Ngưỡng24 không tự đồng nghĩa với FPR thực nghiệm1%, đặc biệt sau aggregation.

Reader VAE encode ảnh rồi lấy DCT và majority vote các bản sao. Agg32 trung bình từng hệ số DCT qua32 ảnh trước khi vote. Chấm5.000 nhóm, mỗi nhóm chọn32 ảnh khác nhau từ1.000 ảnh test, seed20261111. Các nhóm dùng lại ảnh. Tie vote25/50 dùng bản sao đầu tiên, khớp reader của lượt gốc.

## Kết quả lượt server đã hoàn tất

Các JSON trong `server_results/` là kết quả cũ đã thu thập, không phải kết quả của lần chạy mới trên máy của bạn.

| Model/condition | BA từng ảnh | Chấp nhận24 từng ảnh | BA agg32 | Chấp nhận24 agg32 |
|---|---:|---:|---:|---:|
| Teacher nguyên bản |100%|100%|100%|100%|
| Teacher JPEG25 |99.99375%|100%|100%|100%|
| Student nguyên bản |79.95%|77.1%|91.1281%|100%|
| Student JPEG25 |76.5875%|67.4%|89.2469%|100%|

Student exact32 agg32 là0% ở cả hai điều kiện. Teacher JPEG25 exact32 từng ảnh là99.8%.

## Nguồn và thay đổi đóng gói

Nguồn GROW https://github.com/luopengchen/GROW, code/license đưa kèm ở `benchmark/methods/GROW/official/GROW`.

`SOURCE_SNAPSHOT.json` lưu hash code tải từ server trước khi sửa đường dẫn. `source_archive/` lưu các script cha có hàm trainer/checkpoint được tái sử dụng. Chúng là tài liệu nguồn; chạy `run.sh` thay vì chạy archive.

Bản portable thay đường dẫn tuyệt đối thành đường dẫn theo thư mục repo, bổ sung download/setup và cho cấu hình worker. Hàm train, kiểm tra checkpoint, batched guidance, strict student loader và công thức chấm được giữ theo lượt server. Việc chỉ dùng một batch size trên máy mới thay cho sự pha batch2/batch4 của lượt cũ có thể làm khác vài giá trị pixel, nên không bảo đảm trùng hash ảnh hoặc từng số kết quả. Mặc định một GPU nhằm chạy được trên máy có một card.

Một số script SANA/PixArt trong `student_training/reference` được giữ nguyên như bản tham chiếu server. Chúng không thuộc entry point SD1.4 của bundle này.
