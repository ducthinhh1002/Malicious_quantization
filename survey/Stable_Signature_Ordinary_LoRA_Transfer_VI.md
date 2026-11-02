# Stable Signature: dấu có truyền sang ordinary LoRA student không?

**Cập nhật 19:20 ngày 21/10:** run gốc dùng SD2.1 student bên dưới đã được đánh
dấu `superseded` trên server. Run hiện tại là
`signature_author_sd21_to_sd14_20261021_074620Z`: public native teacher SD2.1,
student **SD1.4 sạch**, sampling50 bước/CFG3, query1.000 ảnh. Các mục cấu hình
SD2.1-student/DDIM20/CFG7,5 ở dưới là lịch sử run đầu, không phải run đang chạy.
Xem trạng thái mới ở cuối tài liệu.

Ngày 21/10/2026. Thí nghiệm theo yêu cầu: rank 4 và 16, attention-only và
attention + CNN. Đây là kiểm tra **học dấu từ ảnh của teacher**, không phải giữ
dấu khi fine-tune một model vốn đã mang watermark.

## Cách tránh kết quả dương tính sai

[Stable Signature chính thức](https://github.com/facebookresearch/stable_signature)
nhúng chuỗi 48 bit vào decoder VAE. Nếu giữ decoder đó trong student, ảnh student
vẫn có dấu kể cả UNet chưa học watermark. Vì vậy:

- Teacher: clean SD2.1 UNet/text encoder, VAE decoder công khai đã nhúng dấu của tác giả.
- Student: khởi tạo độc lập từ **clean SD2.1**, cả training và inference dùng **VAE sạch**.
- Student chỉ nhận PNG và caption gốc. Không nhận key, extractor, latent teacher,
  checkpoint marked VAE hay loss watermark.
- CNN trong yêu cầu được hiểu là các `Conv2d(groups=1)` của **UNet**. VAE và text
  encoder vẫn frozen; attention gồm `to_q`, `to_k`, `to_v`, `to_out.0`.
- Hash tensor VAE được kiểm tra trước/sau khi nạp adapter student; sai hash thì dừng.

Thí nghiệm này dùng SD2.1 cho cả teacher/student để khớp public SD2 decoder.
Không coi khác biệt với các run LALT/semantic SD1.4 trước là một so sánh chỉ thay
watermark, vì backbone cũng khác.

## Ma trận và cấu hình đóng băng

| Rank | Target LoRA | Dữ liệu marked | Đối chứng |
|---|---|---|---|
| 4 | Attention | Một student | Một student học ảnh clean |
| 16 | Attention | Một student | Một student học ảnh clean |
| 4 | Attention + UNet CNN | Một student | Một student học ảnh clean |
| 16 | Attention + UNet CNN | Một student | Một student học ảnh clean |

Tổng cộng **8 long trainings**, mỗi student:

- 2.048 PNG 512×512, 2.048 COCO source ID/caption khác nhau.
- 5.000 bước; batch2 × accumulation2; LR `1e-5`; constant scheduler; seed2310.
- FP32, TF32 tắt, gradient checkpointing; gradient clipping1.
- Trainer Diffusers đã pin trong repo, ordinary denoising MSE. Dùng phần hỗ trợ
  attention/Conv2d PEFT đã có; không thêm loss hay thay thuật toán để giữ dấu.
- Checkpoint cuối cố định, không chọn theo detector.

Ảnh clean/marked teacher dùng cùng latent sau DDIM20, CFG7,5; chỉ decoder khác.
Cả bốn cấu hình dùng cùng hai bộ ảnh. Query/calibration cũng dùng prompt/noise
giống nhau giữa các model. VAE sạch của SD2.1 có trong cache được tái dùng;
checkpoint student SD1.4 cũ không được coi là đối chứng khớp cho lần này.

## Kiểm tra và đánh giá

1. Verify SHA256 decoder/extractor tải từ URL Meta; chuyển decoder sang Diffusers,
   kiểm tra đủ mọi key decoder/post-quant-conv, encoder/quant-conv không đổi.
2. Preflight teacher trên 32 prompt riêng: BA >=90% và TPR >=90% ở ngưỡng binomial
   nominal FPR1%. Fail thì dừng trước training để sửa fixture/key/preprocessing.
3. Preflight có thêm clean-VAE encode/decode ảnh marked để quan sát tín hiệu qua
   VAE. Đây là chẩn đoán, **không thay thế kết quả student**.
4. Sau training, mỗi cấu hình có 512 ảnh calibration từ clean student; đóng băng
   ngưỡng số bit khớp nguyên sao cho empirical calibration FPR <=1%, xử lý ties
   bảo thủ. Đánh giá trên 256 query khác cho cả clean/marked student.
5. Báo cáo BA, exact-message rate, image TPR/clean FPR với Wilson CI95%, AUROC,
   paired BA gain với bootstrap theo ảnh, bốn random wrong-key BA. Báo thêm ngưỡng
   binomial nominal1% và `1e-6` theo cách đọc của tác giả.

Extractor whitened nhận PNG RGB [0,1], normalize ImageNet đúng
[notebook tác giả](https://raw.githubusercontent.com/facebookresearch/stable_signature/main/decoding.ipynb),
bit được giải mã bằng `logit>0`; không resize ảnh 512px. Key công khai:
`111010110101000001010111010011010100010000100111`.

Binomial FPR là danh nghĩa, phụ thuộc giả định bit độc lập/không lệch; FPR đo trên
clean query phải báo riêng. Một training seed và 256 ảnh không đủ chứng minh
model-level false positive `1e-6`, cũng không chứng minh mọi cấu hình LoRA đều giữ
hoặc đều mất dấu. Smoke hai bước chỉ kiểm tra phần mềm, không được trình bày là
kết quả long-training.

## Chạy và theo dõi

Code: [signature_transfer.py](../new_watermark/learning_channel/signature_transfer.py),
[signature_metrics.py](../new_watermark/learning_channel/signature_metrics.py).

Snapshot đầu: `signature_transfer_20261021_055542Z`.
Reports: `/root/logs/results/signature_transfer_20261021_055542Z`.

Pipeline chạy smoke trước (rank4 attention và rank16 attention+CNN, clean/marked,
mỗi nhánh 2 bước); nếu cả train/save/load/render chạy được thì mở suite chính.
59 unit tests trên server đã pass. Các stage chạy tuần tự trong tmux, giới hạn
32 GiB, chờ tối thiểu 48 GiB trống, giữ nguyên tiến trình và kết quả khác.

```powershell
python server_review/scripts/check_signature_transfer.py
```

Kết quả teacher ở `teacher_preflight.json`, `teacher_result.json`; mỗi cấu hình
ở `result_r4_attention.json`, `result_r16_attention.json`,
`result_r4_attention_conv.json`, `result_r16_attention_conv.json`; tổng hợp
`results.json` được cập nhật sau mỗi cặp clean/marked. Chưa có kết quả long student
tại lúc tạo tài liệu.

### Kiểm tra đầu lúc 12:59

Public teacher đã qua preflight trên 32 ảnh: BA **99,609%**, phát hiện **32/32**
ở ngưỡng >=33/48 bit khớp (binomial nominal FPR1%). Clean control BA46,224%,
phát hiện0/32. Sau clean-VAE posterior-mean encode/decode ảnh marked: BA51,172%,
phát hiện1/32. Đây là bằng chứng tín hiệu suy giảm qua phép tái tạo VAE cụ thể,
chưa phải kết luận ordinary LoRA không học được dấu. CI95% TPR teacher là
[89,28%;100%]; 32 ảnh không xác nhận population TPR100%.

Smoke rank4 attention đã train/save thành công. Long training chưa bắt đầu tại
thời điểm kiểm tra này; không dùng kết quả hai bước để kết luận về student.

Smoke adapter audit: rank4 attention có 128 target modules và 829.952 tham số
trainable; rank16 attention+CNN có 194 targets và 12.628.608 tham số trainable.
Checkpoint rank16 clean xác nhận toàn bộ 66 convolution `lora_B` và 128 attention
`lora_B` đã khác 0; adapter nạp lại được, hash VAE vẫn trùng VAE sạch.
PSNR teacher preflight: mean30,39 dB, p05=26,65 dB.

**13:06 ngày 21/10:** smoke đã hoàn tất cả hai cặp (rank4 attention và rank16
attention+CNN, clean/marked), gồm train, save, reload, render, đọc extractor và
ghi báo cáo. Lượt chính đã tự khởi chạy và đang ở `teacher_images_preflight`,
sau đó sinh dataset đầy đủ rồi lần lượt train tám student. Chưa có kết quả
student5.000 bước. Các metric smoke từ 8 query/16 calibration không có giá trị
kết luận về transfer hoặc FPR1% và không được gộp vào kết quả chính.

## Kiểm tra run thay thế lúc 19:20 ngày 21/10

Đọc trực tiếp launch record và protocol trên server: teacher dùng native LDM
decoder/source của tác giả, student dùng SD1.4 và VAE sạch riêng. Ma trận rank4/16
× attention/attention+CNN và 5.000 bước mỗi nhánh được giữ. Query tăng lên1.000;
sampling50 bước, CFG3. Run bắt đầu lúc14:46, PID2893854 còn sống khi kiểm tra.

| Dữ liệu teacher | Clean | Marked | Trạng thái |
|---|---:|---:|---|
| Train | 2.048 PNG | 2.048 PNG | Xong, có manifest |
| Calibration | 512 PNG | 512 PNG | Xong, có manifest |
| Query | 766/1.000 PNG | 766/1.000 PNG | Đang sinh, chưa có manifest hoàn tất |

Teacher marked trên bộ train có BA99,6104%, phát hiện2.048/2.048 ở ngưỡng >=33/48
bit (binomial nominal FPR1%). Clean train bị phát hiện1/2.048. Trên calibration,
marked BA99,6053%, phát hiện512/512, clean0/512. Đây đều là **teacher**, chưa phải
student. Chưa có thư mục long student hay `results.json` của run mới.

Query đang khoảng4,02 giây/cặp clean-marked; phần sinh ảnh còn khoảng16 phút nếu
tốc độ giữ nguyên, chưa tính đánh giá kế tiếp. Run hiện tại chưa bắt đầu LoRA
5.000 bước vì pipeline hoàn tất dữ liệu/đánh giá teacher trước.

Đã sửa script theo dõi cũ để chuyển sang run thay thế khi trạng thái là
`superseded`. Có thể dùng trực tiếp:

```powershell
python server_review/scripts/check_signature_author_transfer.py
```

[Bản kiểm tra tải từ server](../server_review/signature_author_transfer/signature_author_sd21_to_sd14_20261021_074620Z/verification.json),
[protocol run hiện tại](../server_review/signature_author_transfer/signature_author_sd21_to_sd14_20261021_074620Z/protocol.json).

## Chạy song song rank 4 lúc 20:42 ngày 21/10

Theo yêu cầu người dùng, đã khởi động nhánh `r4_attention_marked` song song với
`r4_attention_clean`. Pipeline cũ chạy tuần tự cả train và sinh ảnh đánh giá của
clean trước khi bắt đầu marked, làm thời gian chờ kết quả marked dài hơn.

Bộ điều phối ngoài snapshot thay PID điều phối cũ 2893854 bằng PID 3075202.
Tiến trình train clean PID 3041812 được giữ nguyên, không restart; marked chạy
PID 3075249. Bộ điều phối mới nhận kết quả hai nhánh, chạy đánh giá từng nhánh
sau khi train xong, tổng hợp cặp rank 4 attention, rồi tiếp tục các cấu hình còn
lại theo thứ tự cũ. Không sửa source đã đóng băng, dữ liệu, seed, precision hay
5.000 bước train. Mỗi nhánh vẫn dùng ordinary denoising MSE và VAE SD1.4 sạch.

Kiểm tra lúc 20:43: clean đạt 3.857/5.000 bước; marked đạt 14/5.000 bước và log
đang tăng. Tổng GPU dùng khoảng 34.966 MiB, còn 108.191 MiB; tốc độ tức thời mỗi
nhánh khoảng 1,7 giây/bước so với khoảng 1 giây khi chỉ có nhánh clean. Vì cùng
chia sẻ compute GPU, chạy song song không có nghĩa tổng tốc độ tăng gấp đôi.
Chưa có kết quả fingerprint student tại thời điểm kiểm tra.

Mã điều phối: [signature_parallel_coordinator.py](../server_review/scripts/signature_parallel_coordinator.py).
Script kiểm tra hiện đọc `active_stages` để hiển thị cả hai nhánh và kiểm tra
PID của cả bộ điều phối lẫn worker. Hồ sơ bàn giao được lưu ở
`parallel_takeover.json` trong thư mục kết quả của run.
