# Ổn định reader LALT sau bước 2000

Kiểm tra chéo trên 128 prompt validation/key cho thấy: carrier 3000 + reader 2000
đạt TPR 73,44%/75%; carrier 2000 + reader 3000 chỉ đạt 7,03%/11,72%.
Đây là bằng chứng về sự suy giảm khả năng đọc/khả năng phối hợp của reader mới,
không phải thí nghiệm nhân quả độc lập để quy mọi lỗi cho một thành phần.

## Thay đổi

- Khởi tạo lại cặp carrier/reader được chọn ở bước 2000; reader `eval()` và
  `requires_grad_(False)`, không có optimizer cập nhật reader.
- Vẫn backprop qua reader và VAE về carrier. Không bọc nhánh loss carrier bằng
  `no_grad`. Giữ loss, dữ liệu train, key train, chất lượng, sampler và FPR như cũ.
- Tiếp tục 18.000 bước carrier, mốc cuối 20.000. Đây là warm start với AdamW mới,
  không phải exact resume: checkpoint cũ không lưu optimizer/RNG.
- Đánh giá lại điểm khởi đầu để đưa nó vào tập chọn checkpoint. Đánh giá sớm ở
  2250, sau đó mỗi 1000 bước. Hash reader được kiểm tra ở mỗi mốc đánh giá.
- Chỉ validation được dùng chọn; test cuối giữ riêng. Không chỉnh LoRA student.
- Cache sạch dùng lại từ parent với identity cũ; nguồn code/config cũ giữ nguyên.
  Báo cáo và artifact mới có folder riêng, ghi hash checkpoint và source.

## Kiểm tra

Unit test xác nhận reader bất biến, gradient ảnh còn hữu hạn/khác 0, carrier/ảnh
vẫn cập nhật; thay đổi dữ liệu hoặc sampler khi tiếp tục bị từ chối. GPU preflight
dùng 512px, batch 4 và cả kênh round-trip để kiểm tra đúng đường gradient trước khi chạy.

## Giới hạn

Đóng băng reader ngăn reader trôi nhưng không đảm bảo carrier khái quát tốt hoặc
TPR tăng: carrier có thể học khai thác reader cố định. Cần theo dõi key/prompt chưa
thấy, chất lượng và transfer qua ordinary student. Chưa thể kết luận hiệu quả từ
loss train hoặc preflight. Thay đổi này là kiểm soát ổn định, không phải claim novelty.
Probe LoRA checkpoint 2000 đang chạy riêng vẫn đo checkpoint ban đầu.

## Triển khai ngày 18/10/2026

- 25 unit tests qua ở local và server.
- GPU preflight 512px, batch 4, cả round-trip: gradient norm 3,192;
  reader hash bất biến; carrier đã cập nhật; peak VRAM 28,73 GiB.
- Đã dừng teacher joint cũ và khởi động tmux
  `lalt_stable_reader_20261018_111112Z`. Checkpoint cũ và probe LoRA được giữ.
- Báo cáo: `/root/logs/results/lalt_stable_reader_20261018_111112Z`.
- Log: `/root/logs/lalt_stable_reader_20261018_111112Z.log`.
- Tiến trình đánh giá lại baseline 2000 trước khi cập nhật carrier. Chưa có
  bằng chứng TPR tăng từ thay đổi này tại thời điểm triển khai.

Theo dõi:

```bash
tail -f /root/logs/lalt_stable_reader_20261018_111112Z.log
```
