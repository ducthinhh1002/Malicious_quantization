# Sửa supervision và nghiên cứu fingerprint có thể học được

## Bằng chứng trên dữ liệu thực

Run transport cũ: target 64/64 thành công mỗi nhánh, nhưng teacher test chỉ 3/256 plain và 1/256 robust, clean 3/256. Đây là thất bại của teacher embedding, chưa hề đo student.

Audit giữ nguyên image/timestep (không so hai minibatch ngẫu nhiên khác nhau):

- 8 delta train có cosine trung bình ngoài đường chéo **-0.001026**. Năng lượng của delta trung bình chỉ **0.1245** lần trung bình năng lượng từng delta, xấp xỉ mức 1/8 của các hướng độc lập. Đây là dấu hiệu thiếu thành phần chung, không phải chứng minh bất khả học: một hàm phụ thuộc nội dung vẫn có thể học các delta trực giao.
- Ảnh 0 ở t=1: correction epsilon RMS **0.000141**, chênh FP32 so với cache BF16 trên cùng input **0.008763** (~62 lần). Dùng FP32 để hồi quy cache BF16 không tự sửa được; phải tạo lại reference/trajectory cùng precision.
- Frozen base BF16 tái tạo cache tại input sạch đúng (RMS=0). Nhưng loss tại input đã dịch tăng từ **1** ở t=951 lên **8628** ở t=1. Model đã train chỉ hạ giá trị cuối này xuống **6573**; ở t=951 lại tăng từ 1 lên 32. Supervision cũ không cân bằng các timestep và làm prefix lệch.

Không gọi công thức bridge cũ là sai đại số: nó qua kiểm thử DDIM. Vấn đề là điều kiện số, cách chuẩn hóa và độ khó hồi quy.

## Những sửa đổi đã thực hiện

1. Tạo lại sạch/target/trajectory bằng FP32, train owner FP32. Không dùng BF16 reference để làm target cho FP32.
2. Bridge bằng 0 ở 16 bước đầu, đạt 1 qua 4 bước cuối. Tập trung tác động vào đoạn denoising chính xác hơn; kiểm thử với scheduler thật ở mọi timestep vẫn phải khớp.
3. Hồi quy conditional và unconditional độc lập tới `epsilon_clean + correction`. Khi cả hai học cùng correction, CFG vẫn có đúng correction đó. Không còn bình phương hệ số CFG=7.5 trong loss kết hợp với neo unconditional bằng 0.
4. Mẫu số loss là **lỗi thực của frozen base tại input dịch so với target**, detach và có floor, không chỉ năng lượng correction lý thuyết rất nhỏ. Trên mỗi point hoạt động, baseline loss có mốc khoảng 1; point prefix có target sạch được giữ lại để kiểm soát drift.
5. Chọn 80% update ở 4 bước cuối, 20% update preservation ở prefix. Đây là heuristic mới, cần ablation nếu pilot thành công.
6. Chế độ overfit dùng 4 ảnh train, 4 ảnh owner-train được giữ riêng làm probe. Cứ 50 update đo **cùng t=51**, full rollout từ cùng noise, latent-target relative MSE, score và PSNR. Cuối run fuse owner LoRA, kiểm tra tương đương prediction rồi đo lại FP16. Không tác động ordinary LoRA của student.

Mọi probe ở đây là diagnostic/development; 4–8 ảnh không đủ ước lượng TPR/FPR nhỏ, và threshold clean FP16 cũ chỉ là mốc tham chiếu khi xem score FP32. Không gọi chúng là final test.

## Hướng có tiềm năng hơn: ưu tiên tính nhất quán và khả năng học

Một target có watermark mạnh nhưng thuộc hướng thay đổi khó học không phải target tốt cho mục tiêu truyền dấu. Thay vì tối ưu mỗi ảnh độc lập, thử tối ưu **một residual chung** trên nhiều ảnh:

`min_delta E_train[L_mark(D(z+delta)) + lambda L_quality]`

Code `--mode shared` tối ưu trực tiếp residual chung (không lấy trung bình các nghiệm riêng), dùng 16 ảnh train và 8 ảnh giữ riêng không góp gradient. Thí nghiệm này mới chỉ thiết kế carrier; chưa train native teacher hoặc student. Nếu cả train lẫn held-out không tăng score dưới ngân sách ảnh, giả thuyết residual chung cho carrier hiện tại bị bác bỏ ở cấu hình đã thử.

Hướng tiếp theo **chưa triển khai** là fingerprint bị ràng buộc trong không gian thay đổi mà denoiser học được: alternating projection giữa bước tăng score và bước fit một trường residual có ít tham số dùng chung trên nhiều ảnh/timestep. Có thể dùng sai số fit của owner để phạt target khó học, thay vì chỉ dùng pixel MSE. Không cần truy cập hoặc sửa student thật. Đây là giả thuyết phương pháp, chưa xác lập novelty hay hiệu quả; residual chung hiện tại là đối chứng đơn giản trước khi thêm cơ chế phức tạp.

## Liên hệ tài liệu

- [SleeperMark, CVPR 2025](https://arxiv.org/html/2412.04852v2), §4.1–4.2: dùng residual không phụ thuộc ảnh để tăng nhất quán; ưu tiên nhúng ở timestep thấp. Bài dùng trigger và fine-tune model sở hữu, khác mục tiêu truyền qua ảnh sang student độc lập của chúng ta.
- [AquaLoRA, ICML 2024](https://github.com/Georgefwt/AquaLoRA/blob/master/train/README.md): latent watermark pretraining rồi prior-preserving fine-tuning. Đây là tiền lệ cho hai giai đoạn, không phải novelty của prototype.
- [On the Learnability of Watermarks for Language Models](https://arxiv.org/abs/2312.04469): nghiên cứu watermark distillation và giới hạn learnability. Kết quả LLM không chứng minh hiệu quả diffusion; nó củng cố việc phải đo khả năng học riêng với khả năng phát hiện.
- [Radioactive Data, ICML 2020](https://proceedings.mlr.press/v119/sablayrolles20a.html): thiết kế dữ liệu để để lại dấu qua training. Không đồng nhất thí nghiệm classification của paper với ordinary diffusion LoRA.
- [Transferable Multi-Bit Watermarking Across Frozen Diffusion Models via Latent Consistency Bridges](https://arxiv.org/abs/2603.20304): can thiệp residual trong quá trình lấy mẫu model frozen. Cross-model portability của module không phải dấu tự truyền qua ordinary student; không thay native teacher bằng hook inference rồi tuyên bố đã đạt yêu cầu.

## Chạy hai diagnostic

```bash
python new_watermark/bitgreen_repair.py --source EXISTING_TRANSPORT_RUN --output NEW_RUN --mode overfit --steps 300
python new_watermark/bitgreen_repair.py --source EXISTING_TRANSPORT_RUN --output OTHER_NEW_RUN --mode shared --steps 150
```

Output JSON nhẹ nằm trong `NEW_RUN/result`; target tensor và adapter nằm qua symlink owner ở runtime artifact root. Source hashes và config được lưu trước khi chạy. Không resume đè lên run cũ.
