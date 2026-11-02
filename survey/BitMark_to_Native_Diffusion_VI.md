# Chuyển cơ chế BitMark sang fingerprint trong UNet diffusion

Ngày: 17/10/2026. Trạng thái: giả thuyết đã triển khai để thử nghiệm, chưa chứng minh teacher mạnh hay truyền qua student.

## Bằng chứng cần giữ từ các run cũ

Đã đọc `logs-20261016T232636Z-1-001.zip`, đặc biệt `09_final_report.log`, `01_assets.log` và log chọn checkpoint. Teacher Infinity BitMark đạt 100%; student SD2.1 ordinary attention LoRA đạt 49,4% (r4), 62,9% (r16), 66,3% (r32). LoRA mở rộng convolution đạt 74,5–92,1%; full đạt 97%. Student chưa học đạt 6,1% dưới ngưỡng hiệu chỉnh bằng ảnh Infinity. Run có 1.000 ảnh train, thiếu matched clean-trained control. Vì vậy không gọi các tỷ lệ này là TPR tại FPR 1% trên miền student. Chi tiết đã lưu tại `server_review/BITMARK_20261017_evidence.md`.

SERUM trong `result.zip`: teacher 100%; LoRA/DoRA 2,3–3,6%; full 83,6%. Hai thí nghiệm khác teacher, student và ngân sách; đây là bằng chứng thực dụng để ưu tiên khảo sát BitMark, chưa phải so sánh nhân quả giữa hai carrier.

## Đọc paper và giới hạn novelty

[BitMark, NeurIPS 2025](https://arxiv.org/html/2506.21209v2) thiên lệch xác suất các chuỗi bit trong tokenizer pretrained, rồi phát hiện bằng tần suất chuỗi xanh khi mã hóa ảnh trở lại. Cặp 01/10 tăng tương quan trái dấu; không phải nhúng một thông điệp cố định vào từng ảnh. Appendix G đã áp BitMark lên **diffusion nhị phân BiGR**. Do đó claim “lần đầu BitMark sang diffusion” không hợp lệ. Thí nghiệm radioactivity trong paper dùng full fine-tuning; kết quả LoRA nêu trên là bằng chứng từ run của chúng ta.

[Code chính thức](https://github.com/sprintml/BitMark) và [tokenizer Infinity](https://huggingface.co/FoundationVision/Infinity) cung cấp encoder/quantizer đã học biểu diễn ảnh. Ta giữ chúng frozen, dùng checkpoint SHA256 đã pin trong `bitmark_lora/assets.py`. Không tải hoặc huấn luyện lại generator Infinity/T5 chỉ để đọc bit.

Khác với BiGR, SD1.4 có latent Gaussian liên tục, không có logit bit để cộng bias. Ta cần tối ưu **UNet của owner** để ảnh đầu ra có thống kê bit mong muốn trong encoder bên ngoài. Đây là bài toán khác với thay sampler của một model bitwise.

## Thiết kế đã code

Tên tạm trong code: `BitGreen`. Đây là prototype tên kỹ thuật, chưa phải tên một phương pháp đã chứng minh mới.

1. Sinh ảnh thật từ Gaussian noise qua DDIM 20 bước; gradient qua 4 bước cuối. UNet owner có adapter attention+convolution rank 16, rồi merge thành trọng số UNet thông thường. VAE và tokenizer BSQ frozen. Không sửa inference noise hoặc yêu cầu key/hook khi dùng teacher đã export.
2. Mã hóa ảnh 512px bằng đúng encoder và quantizer Infinity d32. Đọc residual trước lượng tử hóa của bốn scale 13,18,24,32. Kiểm tra dấu của residual khớp các bit do quantizer chính thức trả về.
3. Loss mềm cho cặp bit xanh: `p_green = (1 - tanh(u_i/tau)*tanh(u_{i+1}/tau))/2`, với residual chuẩn hóa. Tăng trung bình mỗi scale so với ảnh clean cùng prompt/noise một margin cố định. Hard bit counts là metric đánh giá, không thay bằng surrogate khi báo kết quả.
4. Nhánh `plain`: loss bit trên ảnh teacher trực tiếp + ràng buộc PSNR so với clean cùng noise. Nhánh `robust`: giữ mọi cấu hình như plain, thêm loss bit qua kênh VAE hoặc encode → thêm noise tại t=200/400 → dự đoán x0 bằng UNet sạch frozen → decode. Luân phiên kênh theo bước để hạn chế VRAM.
5. Hai nhánh dùng trọng số bằng nhau cho các scale và penalty vào scale thiếu margin nhất, nhằm tránh để số lượng bit của scale mịn lấn át các scale còn lại. Đây là lựa chọn thiết kế phải được ablate tiếp, chưa chứng minh tín hiệu đó mang tính ngữ nghĩa.

Hai nhánh khớp số bước cập nhật, không khớp FLOPs: nhánh robust có thêm forward/backward qua kênh. Log thời gian và VRAM để không đánh đồng số bước với chi phí.

**Lý do đáng thử:** sử dụng một biểu diễn bit đã có bằng chứng thực nghiệm tốt, thay cho tự chọn hướng CLIP/PCA; thúc đẩy bias thống kê trên nhiều vị trí ảnh, đồng thời hạn chế nghiệm chỉ sống ở nhiễu pixel mà VAE bỏ đi.

**Điều chưa thể suy ra:** qua VAE/re-noising vẫn giữ dấu không chứng minh LoRA sẽ học dấu. Soft loss có thể tăng mà hard score không tăng. Các scale có tương quan; không áp giả định bit độc lập để công bố p-value. Gradient residual của quantizer dùng detach ở các mã scale trước, và rollout chỉ backprop 4 bước cuối; không phải gradient chính xác toàn bộ sampler/quantizer rời rạc.

## Protocol student và đánh giá

- Ba student độc lập: clean, plain (`fixed` trong tên thư mục), robust (`wm`). Tất cả từ SD1.4 sạch, ordinary attention LoRA r4, cùng LR 1e-4, batch 4, 2.500 bước ở pilot. Không key, extractor, loss watermark, sample weighting hay trọng số teacher đi vào trainer.
- Chọn checkpoint bằng minimum validation denoising MSE. Không chọn theo watermark trên student/test.
- Calibration 1.000 ảnh sạch SD1.4, empirical image-level FPR mục tiêu 1%. Test 256 ảnh trên prompt độc lập. Teacher design gate dùng 128 prompt khác test; teacher train dùng 128 prompt riêng; student train 500 ảnh; validation 128 ảnh.
- Báo raw hard green fraction, tỷ lệ phát hiện và Wilson CI; student so với matched clean-trained student bằng paired bootstrap. Một seed chưa đủ để đánh giá training variance.
- Đây là **pattern 01/10 công khai**, chưa có bằng chứng phân biệt owner bằng secret key. Không claim cryptographic security, multi-owner attribution hay kế thừa trên mọi cấu hình LoRA.
- Detector pilot khác detector BitMark chính thức: 512px, cặp trong từng token, bốn scale và cân bằng scale. Không so trực tiếp số TPR mới với run BitMark 1024px cũ như cùng một detector.
- Khi cả hai teacher đều không đạt gate (TPR 80%, PSNR trung bình 22dB, clean design rate ≤5%), giữ kết quả và không tự tiêu tốn thêm student training. Đây chỉ là gate vận hành; không thay thế kiểm định FPR hay đánh giá chất lượng. `--force-students` bỏ qua gate nhưng không đổi trạng thái thất bại của teacher.

## Chạy

```bash
bash run_bitgreen.sh --plan-only
bash run_bitgreen.sh
# Nếu chỉ muốn kiểm tra teacher:
bash run_bitgreen.sh --teacher-only
# Resume đúng cấu hình và source đã đóng băng:
bash run_bitgreen.sh --resume /path/to/bitgreen_run
```

Chọn Python CUDA đã cài bộ thư viện repo bằng `NEW_WATERMARK_PYTHON=/path/to/python`. Không cần FlashAttention/Infinity generator. Pilot không tự cài môi trường mới.

`BITMARK_RESULTS_ROOT` chứa cấu hình, scores, CSV/JSON và log. Ảnh, model, adapter và validation cache đặt ở `BITMARK_RUNTIME_ROOT/bitgreen_artifacts/<run>/`, nối symlink vào result để giữ tương thích trainer. Khi lấy báo cáo về, đừng dereference các symlink nặng.

Đọc `result/summary.csv`, `result/teacher_gate.json`, `result/channel_diagnostics.csv`; nếu có student, đọc thêm `result/selection/` và `result/utility.json` (FID/KID/CLIP, cùng số ảnh/reference). FID pilot ít ảnh có bias; không dùng ngưỡng FID tuyệt đối để khẳng định chất lượng tốt.

## Quyết định sau pilot

- Teacher không đạt: kiểm tra hard-vs-soft gain và chất lượng trước khi tăng ngân sách student.
- Teacher đạt, student không đạt: giả thuyết channel robustness chưa đủ; không chữa bằng DoRA/sửa student. Cần kiểm tra bit-bias nào tạo ra gradient học nhất quán trên dữ liệu và nhiều seed/prompt.
- Robust hơn plain ở student sau clean adjustment, cùng chất lượng: mở rộng dữ liệu/ngân sách và ablation từng kênh/scale. Chỉ lúc đó mới có cơ sở xem phần cải tiến có đóng góp riêng.
