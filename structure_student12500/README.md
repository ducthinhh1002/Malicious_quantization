# Structure student12500: train mới, giảm LR cuối kỳ, resume đầy đủ

**Đây là thư mục code của nhánh đạt test detection 91,1% (911/1.000 ảnh),
checkpoint được chọn tại 12.500 bước sáng 27/10/2026.**

Code chạy chính nằm ngay tại thư mục này: `campaign.py`, `driver.py`,
`schedule.py`, `resume_state.py`, `bootstrap.py` và `tests.py`.
Các tham số train/loss giữ nguyên. Bản đóng gói đã đổi cách tìm code phụ thuộc
để dùng `historical/` trong chính thư mục này; xem [REPRODUCE.md](REPRODUCE.md).

**Harness và trainer Diffusers đã có đầy đủ trong `historical/`.**
Không kèm model, checkpoint hay tập ảnh nặng. `verify_bundle.py` kiểm tra hash
nguồn, trainer gốc và đường dẫn import; `reference/` giữ cấu hình/kết quả gốc.

Thư mục [experiments_cancelled/structure_quality12500](experiments_cancelled/structure_quality12500/CANCELLED.md)
chứa EDA và đề xuất trung bình checkpoint đã bị hủy. Phần này **không tạo ra
kết quả 91,1%** và không được dùng làm launcher của nhánh đã hoàn tất.

Kết quả 12.500: **TPR 91,1%; LPIPS 0,528817; FID 52,89584;
KID 0,000684204; CLIP 0,786258**. Teacher là `structure_margin_native`;
student SD1.4 LoRA rank32 attention+conv được train mới bằng denoising MSE.

## Quyết định từ kết quả đã đo

Nguồn teacher và toàn bộ ảnh: `structure_margin_native`, đã hoàn thành.
Student7500 của nguồn đạt test TPR87,4%, LPIPS0,516560, FID52,73550,
KID0,00054540, CLIP0,788896. Ngưỡng student cố định0,8541307449340821.

Validation theo các checkpoint2500/5000/7500:

| Bước | TPR validation | Denoising MSE ↓ | FID validation ↓ |
|---|---:|---:|---:|
| 2500 | 76,4% | 0,14796512 | 51,91597 |
| 5000 | 85,2% | 0,14776171 | 51,97109 |
| 7500 | 88,7% | 0,14764362 | 52,11560 |

Detection còn tăng đến7500 trong khi MSE giảm rất ít; FID không cải thiện
theo cùng hướng. Nối weights7500 thêm2500 bước bằng AdamW mới LR1e-4 cho
validation TPR88,7% và test TPR86,4%, LPIPS0,547930, FID53,85052.
Run đó đồng thời đổi optimizer/RNG và tăng thời lượng, nên chưa tách được
nguyên nhân: không thể kết luận train liên tục đến10000 cũng sẽ giảm.

Ở test7500 có126 ảnh chưa detect;36 ảnh nằm trong khoảng0,05 dưới ngưỡng.
Muốn **trên90%** ở1000 ảnh cần ít nhất901 positives, tức tăng ròng27 so
với874 hiện tại. Thông tin này chỉ mô tả khoảng cách, không dùng chọn ảnh
train hoặc chỉnh detector. Không đưa key/score vào loss student.

## Thử nghiệm được chọn

- Student bắt đầu từ checkpoint clean SD1.4 đã pin, LoRA mới rank32
  attention+conv, seed0, batch4, FP16, ordinary denoising MSE.
- Giữ toàn bộ5000 ảnh teacher train và1000 ảnh validation bằng hardlink.
  Tái dùng clean/unadapted/reference, carrier, key, thresholds và sampler.
- Train đủ12500 updates, không dừng vì quality/detection kém.
- LR1e-4 đến update7500; cosine từ1e-4 xuống2e-5 trong7500→12500.
  Nếu sau này mở rộng budget, giữ2e-5 sau12500; không tính lại lịch LR cũ
  dựa trên tổng số bước mới.
- Đây là giả thuyết giảm dao động/drift cuối kỳ và giữ đà học của AdamW,
  không phải cam kết TPR>90%. Không tăng teacher score hay thay threshold.
- Dùng SeedableRandomSampler seed0 và `set_epoch` rõ ràng để resume đúng
  thứ tự dữ liệu. Sampler này khác cấu hình RandomSampler cũ; run mới
  không được gọi là bitwise replay của Structure7500 trước đây.

Khuyến nghị về tương tác giữa ngân sách train và LR schedule được đối chiếu
với [Google Research Tuning Playbook](https://github.com/google-research/tuning_playbook).
[How to decay your learning rate](https://arxiv.org/abs/2103.12682) cũng nghiên
cứu lợi ích của decay cuối kỳ trong các thiết lập của paper. Các nguồn này
không chứng minh lịch7500→12500 tối ưu cho radioactive watermark; lịch mới
được định trước để thử một thay đổi nhỏ có thể kiểm tra.

## Checkpoint và khả năng train tiếp

Lưu đủ trạng thái mỗi1250 bước:1250,2500,…,12500, giữ cả10 checkpoint.
Mỗi checkpoint gồm:

- Tất cả LoRA trainable parameters; base SD1.4 cố định được ghi hash.
- AdamW moments/step, LR scheduler và FP16 GradScaler.
- Python, NumPy, Torch CPU và CUDA RNG.
- Tổng applied/skipped updates, bước hiện tại, epoch kế tiếp và identity
  của sampler/dataset/code.
- Manifest SHA256 để phát hiện state thiếu hoặc bị thay đổi.

1250 steps là đúng một epoch của5000 ảnh/batch4. Chỉ cho phép resume tại
epoch boundary; file lưu tạm được kiểm tra rồi rename thành checkpoint
hoàn chỉnh. Không nhầm adapter inference snapshot với training state.

Accelerate checkpointing sử dụng `save_state`/`load_state` để giữ model,
optimizer và RNG; vị trí dữ liệu cần xử lý riêng theo
[hướng dẫn chính thức](https://huggingface.co/docs/accelerate/v0.29.3/en/usage_guides/checkpoint).
Implementation được kiểm tra trên phiên bản server Accelerate1.10.1,
PyTorch2.7.1+cu126; không giả định API chỉ từ bản docs.

`campaign.py --launch` sau interruption sẽ chọn checkpoint hoàn chỉnh mới
nhất, phục hồi optimizer/RNG và tiếp tục đến12500. Log phần đã chạy sau
checkpoint được giữ thành file `updates.before_resume.*.jsonl`.
Driver cho phép budget lớn hơn12500 khi được yêu cầu; lúc mở rộng phải dùng
state checkpoint và giữ báo cáo12500 riêng, không warm-start chỉ từ adapter.

## Đánh giá và chọn checkpoint

Snapshots inference tại7500/10000/11250/12500. Đánh giá cùng1000 validation
prompts, chọn checkpoint bằng minimum validation denoising MSE như harness
trước, không dùng test/owner score để chọn.

Luồng sẽ báo **cả hai**:

1. Student checkpoint được chọn bằng validation MSE.
2. Student endpoint12500 đúng ngân sách người dùng yêu cầu.

Cả hai có TPR/CI95, LPIPS, FID/KID/CLIP với test/reference/threshold cũ.
`target90_met` dùng điều kiện TPR>0,90, không phải≥0,90. Điểm ước lượng
vượt90% vẫn chưa có nghĩa cận dưới CI95 cũng vượt90%.

Test này đã được xem trong phát triển trước đó; một seed chỉ hỗ trợ kết
luận khám phá. LPIPS đo độ lệch paired reference, FID/KID tham chiếu clean
teacher; không phải chỉ số đảm bảo anatomy/ảnh đẹp trên mọi prompt.

## Chạy song song và kiểm chứng

Folder riêng local: `structure_student12500/`.
Folder riêng VM4: `/root/logs/structure_student12500`.
Run: `runs/structure12500_seed0_20261026/student12500`.
Không có dependency chờ Transfer90 hoàn tất. Student training giới hạn
allocator44GiB; kiểm tra free VRAM≥50000MiB trước train. GPU vẫn chia sẻ
compute nên song song không có nghĩa tổng throughput tăng gấp đôi.

4 tests đã pass trên server: lịch LR; epoch patch có phạm vi hẹp; CPU
save/exit/load/continue; GPU FP16 save/exit/load/continue. Hai test resume
so sánh chính xác parameters, optimizer, scheduler, progress, thứ tự batch
và các giá trị RNG tiếp theo giữa hai cách chạy trên mô hình nhỏ ngẫu nhiên.
Test có kiểm tra phát hiện file optimizer bị thay đổi. Đây chưa phải chứng
minh mọi GPU kernel của UNet lớn sẽ bitwise deterministic giữa các máy.

Trước train chính còn có smoke4 cập nhật thật trên SD1.4 với cùng ảnh,
rank/batch/precision. Kết quả smoke là kiểm chứng kỹ thuật, không phải
metric khoa học của thử nghiệm12500.

## Kết quả hoàn tất — 27/10/2026

Run hoàn tất lúc02:04 ICT. Checkpoint12500 được chọn bằng validation MSE
0,1473566195; endpoint cố định12500 cho cùng kết quả trên1000 test images.

| Student | TPR ↑ | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Structure7500 trước đây | 87,4% | 0,516560 | 52,73550 | 0,00054540 | 0,788896 |
| Structure12500 cooldown | **91,1%** | 0,528817 | 52,89584 | 0,00068420 | 0,786258 |

Đạt mục tiêu điểm ước lượng trên90%:911/1000 ảnh, tăng3,7 điểm phần trăm
so với nguồn7500. Wilson95% CI là89,17–92,71%, nên chưa chứng minh cận dưới
vượt90%. Có đánh đổi utility: LPIPS tăng khoảng2,37%, FID tăng0,16034;
KID tăng và CLIP giảm. Không gọi đây là cải thiện đồng thời mọi metric.

Đã xác minh đủ10 full-state checkpoints bằng SHA256 và kiểm tra các thành
phần optimizer/scheduler/scaler/RNG/trainable/progress. Checkpoint12500 ghi
12500 applied,0 skipped, LR cuối2e-5. Có thể dùng full state này để train
tiếp khi có yêu cầu, thay vì reset optimizer từ adapter.

Kết quả local:
[result_summary](reference/result_summary.json),
[original_training_identity](reference/original_resume_identity.json).
