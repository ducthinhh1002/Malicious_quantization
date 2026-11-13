# Structure teacher cải tiến JPEG, thêm 1.200 bước

Thư mục này đóng gói code đang dùng tại `/var/lib/structure32_codec_refine` trên server `.21`. Nhánh bắt đầu từ adapter **Structure 32 bit, LoRA rank32, đã refine600**, rồi học thêm1.200 bước. Carrier semantic được chọn bằng hiệp phương sai của sai lệch JPEG25 trên ảnh clean. Teacher sinh ảnh qua PNDM thông thường với adapter đã học.

`resize/` là thí nghiệm student Janus dùng đầu ra của teacher này. Hai thư mục dùng chung verifier và teacher, nhưng có pipeline riêng.

## Yêu cầu máy

Linux, Python3.11 hoặc3.12, Git, `flock`, NVIDIA GPU hỗ trợ BF16 và driver hỗ trợ PyTorch CUDA12.8. Khuyến nghị GPU80GB và ít nhất100GB disk trống. Chạy trên GPU server. Lượt đầy đủ gồm sinh6.000 ảnh clean, fit carrier, train1.200 bước, sinh1.000 ảnh test rồi chấm.

## Chạy từ máy mới

```bash
git clone https://github.com/ducthinhh1002/Malicious_quantization.git
cd Malicious_quantization
python3 jpeg/verify_bundle.py
bash jpeg/run.sh
```

`run.sh` tự tạo `jpeg/.venv`, cài dependency và tải SD2.1 từ Hugging Face tại revision đã ghim. Nếu Hugging Face yêu cầu quyền truy cập, đặt `HF_TOKEN` của người chạy trong environment. Adapter refine600 gốc và metadata/prompts/carrier gốc nằm trong `inputs/`, nên không cần SSH vào server của tác giả.

Để chọn GPU, đặt biến chuẩn trước lệnh chạy.

```bash
CUDA_VISIBLE_DEVICES=1 bash jpeg/run.sh
```

Không đặt biến này thì PyTorch dùng GPU mặc định. Tên tiến trình GPU là `VLLM::Enginecore`.

## Có dữ liệu gốc sẵn

Có thể dùng lại một original Structure run đầy đủ và adapter gốc để bỏ qua bước sinh clean.

```bash
JPEG_SOURCE_RUN=/path/to/original_structure_run \
JPEG_WARM_ADAPTER=/path/to/original_refine600 \
bash jpeg/run.sh
```

Source run cần `config.json`, `assets.json`, `prompts.json`, `carrier/` và hai bộ `images/teacher_clean_train`, `images/teacher_clean_validation` có `image_hashes.json`. Các đường dẫn model trong `assets.json` phải hợp lệ trên máy chạy. Adapter phải có SHA256 `2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8`.

## Đường chạy

1. `prepare_inputs.py` tải base SD2.1 đã ghim, dùng original prompt/seed/config để sinh5.000 clean train và1.000 clean validation. Bản sao carrier gốc được dùng làm đầu vào lựa chọn subspace.
2. `prepare_codec.py` fit subspace trên1.024 cặp clean train/JPEG25, kiểm tra trên1.000 cặp clean validation và calibration scalar trên clean validation. Khóa32 bit giữ nguyên, hướng carrier thay đổi.
3. `train.py --smoke` kiểm tra gradient và cập nhật adapter trên model thật. Sau đó train1.200 bước với optimizer mới từ refine600 gốc. Checkpoint đầy đủ mỗi50 bước gồm optimizer/RNG để resume.
4. `validation.py` đo16 prompt validation trước train và mỗi100 bước. Checkpoint cuối1.200 được chọn cố định.
5. `generate_test.py` sinh1.000 ảnh test bằng adapter đã merge và kiểm tra merge bằng FP32.
6. `evaluate_jpeg.py` chấm ảnh gốc/JPEG25 ở từng ảnh, agg32 và agg256.

Khi dừng giữa chừng, chạy lại cùng lệnh và cùng biến environment để resume. Giữ nguyên đường dẫn thư mục, code và config sau khi bắt đầu train, vì protocol lưu hash và đường dẫn để phát hiện thay đổi.

## Output

- `run/train_status.json`, `logs/` để xem tiến độ.
- `run/carrier/` chứa reader mới. Reader này phải đi cùng checkpoint mới.
- `run/teacher_fp/refined/checkpoint-1200/` chứa adapter cuối và training state.
- `run/evaluation_jpeg25/results.json` chứa bảng đánh giá.
- `run/images/teacher_jpeg600_test/` là tên thư mục lịch sử được giữ bởi generator. Checkpoint thực tế là1.200 bước.

Metric chính trong script là BA, tỷ lệ ≥24/32 bit và exact32. Agg lấy trung bình tọa độ trước khi lấy dấu,100 lần chia nhóm; ảnh được dùng lại giữa các lần chia. Tỷ lệ ≥24/32 không tự đồng nghĩa với TPR@1%FPR. Các nguồn dữ liệu dùng cho train watermark và train student được tách riêng.

## Phạm vi tái lập

Toàn bộ code/harness/vendor đang chạy đã được tải về; `SOURCE_SNAPSHOT.json` lưu hash snapshot trước khi sửa đường dẫn. Thay đổi portable gồm đường dẫn tương đối/environment, setup dependency, chuẩn bị input và resume khi carrier preparation bị ngắt. Thuật toán,1200 steps, LR1e-5, PNDM50, CFG7.5, JPEG25 và gradient/projection giữ theo lượt server.

Ảnh clean được sinh lại trên phần cứng/phiên bản runtime khác có thể khác từng pixel, nên kết quả train mới không bảo đảm trùng từng số hoặc trùng hash adapter server. `resize/` sẽ lấy reader và checkpoint từ CÙNG run tái lập. Để dùng đúng checkpoint đã chạy trên server, cung cấp run đó bằng environment.

Checkpoint adapter gốc khoảng96.4MiB được đưa kèm để bảo đảm có điểm xuất phát. Base model và dữ liệu runtime tải/sinh ngoài Git. Không có key SSH hoặc API credential trong bundle. Nguồn và phân tích chi tiết ở `RESEARCH.md`.
