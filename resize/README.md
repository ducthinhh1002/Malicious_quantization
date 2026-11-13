# Resize/VQ và distill autoregressive Janus-Pro-1B

Đây là code đang chạy tại `/var/lib/structure32_janus_ar` trên `.21`, dùng teacher từ nhánh `../jpeg/`. Janus có tokenizer/VQ và cách huấn luyện riêng, nên được đóng gói thành thư mục thứ hai. Teacher/verifier/harness được dùng chung qua `jpeg/`.

Luồng đầy đủ kiểm tra128 ảnh qua resize/VQ, sinh bộ teacher mới5.000 train +128 validation +1.000 test, tokenize, sinh clean Janus control, train LoRA rank32 đủ12.500 bước, sinh1.000 ảnh student và chấm watermark.

## Chạy từ máy mới

Trong repo đã clone, chạy

```bash
python3 jpeg/verify_bundle.py
bash resize/run.sh
```

Script dùng môi trường `../jpeg/.venv`, tự cài dependency nếu thiếu và tải Janus-Pro-1B tại revision `960ab33191f61342a4c60ae74d8dc356a39fafcb`. Code Janus chính thức được đưa kèm ở `Janus/`, commit `1daa72fa409002d40931bd7b36a9280362469ead`.

Nếu chưa có output teacher hoàn chỉnh trong `jpeg/run`, script tự chạy `jpeg/run.sh` trước. Vì vậy lệnh này chạy được từ đầu, nhưng có thể mất nhiều giờ. Xem yêu cầu GPU/disk trong `../jpeg/README.md`.

Để chọn GPU

```bash
CUDA_VISIBLE_DEVICES=0 bash resize/run.sh
```

## Dùng teacher đã có sẵn

Cung cấp thư mục run hoàn chỉnh của teacher mới để bỏ qua việc train teacher.

```bash
JPEG_TEACHER_RUN=/path/to/codec_teacher/run bash resize/run.sh
```

Run cần `assets.json`, `prompts.json`, `carrier/`, `teacher_fp/refined/checkpoint-1200/` và `images/teacher_jpeg600_test/image_hashes.json` cùng các PNG. Đường dẫn model trong `assets.json` phải hợp lệ. Reader phải thuộc đúng run của checkpoint.

Có thể thay code root hoặc distillation prompt manifest bằng `JPEG_CODE_DIR` và `JANUS_PROMPTS`. Mặc định dùng `../jpeg` và `inputs/prompts.json` đã đóng gói. Không đổi protocol giữa một lượt train đang resume.

## Thiết lập

- Teacher32 bits, Structure rank32 refine600 +1200; ảnh train student là PNG nguyên bản.
- Student Janus-Pro-1B, LoRA rank32 trên Q/K/V/O và gate/up/down, batch4,12.500 steps.
- LR1e-4, warmup100, CFG prompt dropout0.1; validation CE mỗi1.250 steps.
- Chọn cố định checkpoint cuối12.500. Đây là lựa chọn của lượt mới, khác lượt Janus lịch sử đã chọn7.500 bằng validation CE.
- Teacher512×512 được thu về384×384 bằng LANCZOS trước VQ. Janus dự đoán576 token và decode ảnh384×384.
- Ảnh Janus được phóng về512×512 bằng LANCZOS để chấm bằng reader teacher.
- Sinh ảnh dùng CFG5, temperature1 và seed cố định theo batch.

## Output và resume

`status.json` ghi stage và tiến độ. `train.jsonl` ghi loss/gradient. `checkpoints/step_*` lưu adapter, optimizer và CPU/CUDA RNG mỗi1.250 updates; partial checkpoint được bỏ qua. Chạy lại `bash resize/run.sh` để tiếp tục. Bộ ảnh và token đã hoàn thành được tái sử dụng.

Các điểm nằm ở `scores/teacher_resize384.json`, `teacher_vq384.json`, `teacher_test.json`, `base_test.json`, `student_test.json`. `probe_original/` và script `smoke_ar.py` phục vụ kiểm tra chi tiết. Để chạy kiểm tra một bước AR thật trước lượt đầy đủ, sau khi model và teacher đã chuẩn bị

```bash
JPEG_TEACHER_RUN=/path/to/codec_teacher/run ../jpeg/.venv/bin/python smoke_ar.py
```

Metric gồm BA, tỷ lệ ≥24/32 và exact32, cho từng ảnh, agg32 và agg256 khi đủ số ảnh. Control clean Janus dùng cùng preprocessing/reader để xem báo nhầm. Tỷ lệ ≥24/32 là metric ngưỡng cố định và được báo riêng với FPR thực nghiệm.

## Trạng thái kết quả đã có

`SERVER_EXPERIMENT.md` và `inputs/server_protocol.json` ghi cấu hình/kết quả diagnostic của lượt server. Trên128 ảnh, resize vẫn giữ tín hiệu khá tốt, VQ làm BA giảm mạnh. Các con số đó là kiểm tra tokenizer, chưa phải kết quả student AR cuối.

Trainer AR cũ đã bị xóa trong đợt dọn disk. Wrapper hiện tại được dựng lại bằng model/VQ/gen_head chính thức, đã kiểm tra một bước teacher forcing có gradient hữu hạn và cập nhật LoRA, cùng luồng KV cache. Đây là code của lượt đang chạy. Bản portable chỉ thay đường dẫn, tự setup/download và các kiểm tra protocol; chạy lại trên phần cứng khác không bảo đảm ảnh/điểm trùng từng số.
