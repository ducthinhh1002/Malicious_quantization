# Kiểm tra reader trên prompt mới và sửa xử lý ảnh thu nhỏ

## Kết quả audit đã hoàn tất

| Đầu vào | BA key 22 / 233 | AUROC key 22 / 233 |
|---|---|---|
| 512px | 42,58% / 60,55% | 0,5127 / 0,5176 |
| Thu nhỏ 256px | 33,20% / 68,36% | 0,4941 / 0,4980 |
| 256px → phóng lại 512px | 24,22% / 72,66% | 0,5107 / 0,4922 |

AUROC CI bootstrap 95% của native 512px: [0,4893; 0,5420] và
[0,4902; 0,5537], đều chứa 0,5. Đây là khoảng cách train–validation rất lớn;
kết quả 100% train không chứng minh dấu khái quát. BA cao hơn của riêng key
233 đi cùng BA thấp của key bổ sung, nên không được lấy riêng key thuận lợi
để tuyên bố thành công. Phóng lại ảnh mà không huấn luyện reader cũng chưa
giải quyết được vấn đề. Chưa có bằng chứng việc resize là nguyên nhân duy nhất.

Hai nhánh sửa bên dưới đã hoàn tất 800 bước, không lỗi runtime. Kết quả
trung bình đều giữa hai key (64 prompt train, 32 prompt validation):

| Nhánh / đầu vào | BA train | BA validation | AUROC validation |
|---|---:|---:|---:|
| Đối chứng / 512px | 96,00% | 51,56% | 0,5063 |
| Resize-restore / 512px | 96,78% | 50,00% | 0,5107 |
| Đối chứng / 256px | 90,72% | 51,95% | 0,5020 |
| Resize-restore / 256px | 88,38% | 49,22% | 0,5005 |

Không có bằng chứng cải thiện khái quát hóa trong lượt này; CI AUROC theo
từng key/view ở validation đều chứa 0,5. Không bật resize-restore mặc định,
không chạy student từ các checkpoint này. Carrier và checkpoint nguồn giữ
nguyên; final test chưa dùng. Báo cáo local:
`server_review/lcf_reader_deployment/repair/summary.json`.

Đây là thất bại của việc học riêng reader trên carrier đóng băng trong cấu
hình đã thử, không chứng minh mọi carrier học được đều thất bại. Chỉ tăng
bước fit hoặc nới gate không có cơ sở từ kết quả này. Hướng cần kiểm tra tiếp
là học chính carrier với reader trên nội dung/seed đa dạng và cả kênh resize,
rồi đánh giá khái quát độc lập trước khi tiêu compute cho student.

## Câu hỏi và giới hạn

Lượt `lcf_gain_20261018_021834Z_diagnostic` có bit accuracy 100% ở 512px
trên 8 prompt train, nhưng chỉ 76,56–78,13% ở 256px. Đây chưa phải bằng
chứng khái quát hóa: chỉ có hai key bổ sung nhau, reader có thể học phân biệt
hai mẫu dấu hoặc ghi nhớ ảnh. Không được diễn giải thành đọc payload 8 bit
bất kỳ hay truyền dấu qua LoRA.

## Phép đo đóng băng

`new_watermark/learning_channel/reader_probe.py` đọc
`bootstrap_diagnostic.pt` mà không cập nhật trọng số. Kiểm tra 32 prompt của
`teacher_validation`, hai key đã dùng khi fit. Kiểm tra không trùng image ID
hoặc caption với toàn bộ `teacher_train`. Không dùng `teacher_test`.

Ba đầu vào được so trên cùng ảnh, cùng key:

1. Ảnh RGB8 512px → reader 512px.
2. Ảnh RGB8 512px → bilinear antialias 256px → reader 256px.
3. Ảnh RGB8 512px → bilinear antialias 256px → phóng lên 512px → reader 512px.

Resize dùng tensor liên tục, không thêm JPEG hoặc một lần lượng tử hóa RGB8
nữa. Trường hợp 3 không khôi phục thông tin đã mất. Nó kiểm tra ảnh hưởng của
lưới đầu vào/độ lớn trường tiếp nhận của mạng đọc, tách khỏi phép thu nhỏ.

Lưu AUROC marked-vs-clean, bit accuracy, khoảng cách score đúng/sai key,
paired gain và CI bootstrap 95% theo prompt. Không chọn ngưỡng trên tập này,
không báo TPR tại FPR giả định. Hai key dùng cùng prompt nên không coi chúng
là các quan sát prompt độc lập khi diễn giải tổng hợp. CI theo ảnh không bao
phủ biến thiên giữa các lần train model.

## Thử sửa reader có đối chứng

`reader_repair.py` giữ nguyên carrier, gain và ảnh tạo ra. Hai nhánh cùng xuất
phát từ checkpoint bootstrap, cùng 64 prompt train × 2 key, cùng 800 bước:

| Nhánh | Xử lý đầu vào |
|---|---|
| `continuation_control` | Reader cũ, tiếp tục học cả view 256 và 512 |
| `resize_restore` | View thu nhỏ thật xuống 256, rồi phóng về lưới 512 để dùng cùng reader |

Mỗi minibatch gồm 4 prompt, mỗi prompt có cả hai key. AdamW LR giảm cosine
từ 1e-4 đến 1e-5; loss supervised giống nhau. Tăng prompt và đổi lịch LR có
mặt ở **cả hai nhánh**, do đó không quy mọi tiến bộ cho resize-restore.

Checkpoint luôn là bước cuối cố định, được lưu trước đánh giá validation.
Không chọn bước/điều chỉnh gradient bằng validation. Reader fit không được
đổi carrier; kiểm tra tensor và SHA256 checkpoint gốc sau khi chạy. Teacher,
student và loss denoising của student không thay đổi trong thí nghiệm này.

Validation này là tập phát triển: đã xem kết quả để chẩn đoán nên không được
đổi tên thành final test. Sau khi chốt thiết kế mới cần đánh giá tập chưa xem.
Ngay cả reader khái quát tốt cũng chưa chứng minh student sẽ học dấu.

## Tích hợp vào pipeline

Reader thêm `canonical_size`, mặc định 0 để tương thích checkpoint cũ.
Có thể bật thử trong một lượt mới:

```bash
bash run_learning_channel_gate.sh --profile gain-gate \
  --reader-resize-restore --reader-fit-prompts 64
```

Chưa bật mặc định cho mọi thí nghiệm khi chưa có đối chứng. Các gate chất
lượng/teacher vẫn giữ nguyên; không tự bỏ qua gate để chạy student.

## Triển khai

- Audit: `/root/logs/results/lcf_reader_20261018_030549Z_audit`.
- Thử sửa: `/root/logs/results/lcf_reader_repair_20261018_030829Z`.
- Tmux sửa: `lcf_reader_repair_20261018_030829Z`.
- Checkpoint mới nằm trong `/root/logs/runtime/lcf_artifacts/`, ngoài báo cáo.
- `status.json`, `summary.csv`, `per_prompt.csv`, `protocol.json` là báo cáo nhẹ.
- 18 kiểm thử numerical/protocol đã pass local và server trước khi chạy.

Đọc kết quả trước/sau theo từng mode và key. Nếu chỉ train tăng mà validation
vẫn gần ngẫu nhiên, vấn đề không được giải quyết bằng reader: cần quay lại
học carrier trên nhiều nội dung và kiểm định độc lập, không tăng số bước mù.
