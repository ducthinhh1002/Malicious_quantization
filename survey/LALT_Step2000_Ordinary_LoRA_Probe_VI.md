# Thử ordinary LoRA từ checkpoint LALT bước 2.000

Lượt phát triển song song trên vm4, được yêu cầu để biết teacher hiện tại đã
truyền dấu qua RGB training được chưa. Không chờ hết 20.000 bước và không
thay đổi cấu hình/loss của teacher đang chạy.

## Checkpoint cố định

- Bước 2.000, SHA256:
  `5a3a63666e7d31b0d6eeddc5339d96a37e8f5ac8906922a1ece00a113b1d7175`.
- Sao chép độc lập trước khi checkpoint được chọn của lượt chính bị ghi đè.
- Bộ nhúng và reader này đóng băng trong toàn bộ lượt thử.

## Protocol

| Phần | Thiết lập |
|---|---|
| Base sạch | SD1.4 giống lượt chính |
| Student | Ordinary attention-only LoRA rank 4 |
| Hai nhánh | Học ảnh sạch và học ảnh có dấu key 22 |
| Dữ liệu | 2.048 prompt khác nhau từ owner `teacher_train`, lấy clean latent đã cache |
| Chọn dữ liệu | Một noise variant/prompt; chọn cố định theo seed từ cache sẵn có, không dùng score watermark/chất lượng |
| Đầu vào student | PNG + caption, không có key, reader hoặc latent owner |
| Training | 5.000 bước, LR 1e-5, FP32, batch 2 × accumulation 2, cùng seed 0 |
| Checkpoint student | Bước cuối cố định; không chọn theo TPR |
| Query | Khoảng 128 prompt `meta_eval` chưa dùng trong teacher training, loại caption trùng train |
| Calibration | 512 caption từ owner calibration, sinh lại bằng chính clean-trained student |
| Noise đánh giá | Namespace seed mới, paired giữa unadapted/clean/marked |
| Key đọc | Key đúng 22 và key sai 233 |

Rank 4 và key 22 thuộc tập phát triển, nên kết quả không được trình bày là
final heldout learner/key evidence. Không render/train/đánh giá các split
`student_train`, `student_calibration`, `student_test`, `teacher_test` của lượt
chính. Rank 8/16 và evaluation keys 95/160 vẫn dành cho đánh giá chính thức.
Metadata của các split đó chỉ được đọc để kiểm tra không trùng ID.

## Đo TPR đúng cách

Reader giữ nguyên từ checkpoint 2.000. Hai student đều khởi tạo từ base sạch.
Student có dấu sinh ảnh query từ Gaussian noise mới, không có owner module,
trigger hoặc latent owner.

Ngưỡng chính dùng ảnh calibration của **student học ảnh sạch**, với nominal
tổng FPR 1% qua hai key (Bonferroni: 0,5%/key). Dùng strict `score > threshold`
và order statistic hợp lệ với 512 calibration samples.

CSV cũng ghi `single_key_1pct_detection_rate` để đối chiếu khi chỉ xét một
key tại nominal FPR 1%; không được nhầm hai protocol này. Với teacher,
calibration lấy từ base unadapted tương ứng.

Báo cáo TPR, matched-clean FPR, wrong-key rate, bit accuracy, AUROC, paired
score gain và Wilson CI theo ảnh. CI này chỉ điều kiện trên các model đã
train, không chứng minh FPR qua nhiều lần train model. LPIPS/PSNR và CLIP
drop so với clean-trained control giúp phát hiện đổi dấu bằng cách hỏng ảnh.

Đây là một development probe của teacher-only ở bước 2.000, chưa phải kết
quả của giai đoạn tối ưu truyền dấu 128 bước. Nếu dùng kết quả để sửa thiết
kế sau này, vẫn phải giữ final evaluation độc lập.

## Vận hành

- Tmux: `lalt_step2000_probe_20261018_102531Z`.
- Báo cáo: `/root/logs/results/lalt_step2000_probe_20261018_102531Z`.
- Ảnh: `/root/logs/output_image/lalt_probes/lalt_step2000_probe_20261018_102531Z`.
- Trọng số: `/root/logs/runtime/lalt_probe_artifacts/lalt_step2000_probe_20261018_102531Z`.
- Source snapshot riêng: `/root/logs/lalt_step2000_probe_20261018_102531Z`.

Phases chạy tuần tự trong probe để giảm bộ nhớ: tạo cặp ảnh → base/teacher
reference → train clean → sinh ảnh clean → train marked → sinh ảnh marked →
đánh giá. Toàn probe chạy song song với teacher chính. Mỗi process CUDA có
cap 32 GiB, chỉ bắt đầu khi còn ít nhất 48 GiB trống; không dừng process khác.
Cap áp dụng trong chính process chạy upstream trainer, không chỉ ở parent.

Xem `status.json` để biết phase, `train_clean.log`/`train_marked.log` để biết
step, và `watermark_retention.csv`, `student_quality.csv`, `summary.json` khi
xong. Snapshot code có manifest bất biến; 23 tests đã pass local/server.

Lượt thử đã khởi chạy. Chưa có TPR student tại thời điểm triển khai.
