# Ưu tiên sửa khả năng nhúng dấu của teacher

Ngày 2026-10-16. Triển khai mới là opt-in, chưa có kết quả GPU.

## Bằng chứng từ pilot

Nguồn: `/root/logs/results/gate_20261016_144609_3785966/without_c3`.
Teacher 250 update không skip; design TPR 3,90625%, clean FPR 1,5625%, PSNR 14,854 dB.
Threshold 1,8161, mục tiêu train 2,8161. Trung bình 50 bước cuối: score proxy
0,1853, tăng so với clean proxy 0,0843, watermark loss 0,9094. Teacher chưa nhúng
được dấu mạnh ngay ở proxy, nên chưa thể quy hết thất bại cho sampling hay LoRA.

Với key đơn vị w, feature đơn vị f, a=w^T f, mục tiêu projection b, nếu a<b<=1,
cosine lớn nhất với f của một feature thỏa projection b là

    a*b + sqrt(1-a^2)*sqrt(1-b^2).

Suy ra khoảng cách Euclid nhỏ nhất giữa hai feature đơn vị là sqrt(2-2*cosine).
Đây là cận dưới trong không gian CLIP; không phải LPIPS, không phải độ méo ảnh,
và không chứng minh UNet đạt được feature đích.

Trên 128 clean design-eval images của run đó: trung vị cận dưới là 0,3252;
95,3125% ảnh cần ít nhất 0,15. Vì thế chọn hướng chỉ theo tỷ lệ VAE drift/variance
có thể vô tình chọn trục có biến thiên nội dung lớn, cần dịch chuyển mạnh để
vượt threshold trên phần lớn ảnh.

## Đề xuất nào nên ưu tiên?

1. **Sửa carrier và đo chi phí hình học:** đã thêm carrier tối thiểu hóa đồng
   thời covariance và absolute reconstruction drift trong PCA sạch. Chọn trước
   bằng dữ liệu sạch, không dùng student hoặc final test. Tránh null-space bằng
   PCA truncation/rank guard. Phương sai nhỏ không đồng nghĩa dễ học hay khó giả;
   phải audit clean calibration, wrong keys, quality và student thật.
2. **Loss trên ảnh sinh cuối:** đáng thử vì PSNR sampling kém dù noise-preservation
   nhỏ. Forward DDIM từ Gaussian noise, tối ưu feature của ảnh cuối, dùng clean
   teacher cùng prompt/noise làm tham chiếu chất lượng. Giữ prior denoising loss.
3. **Backprop toàn chuỗi:** có cơ sở nhưng tốn bộ nhớ. Bắt đầu gradient 4 bước
   cuối và có option full. Truncated gradient không phải full gradient; cần ablation.
4. **Bảo toàn ảnh đầy đủ và curriculum:** bổ sung soft PSNR-budget penalty thay vì
   chỉ low-pass; học native trước rồi mới tăng trọng số channel. Không sửa threshold
   để biến teacher yếu thành teacher mạnh.
5. **Tăng step / thêm C3:** chưa ưu tiên. C3 đo tái dựng lặp, không trực tiếp giải
   quyết carrier khó nhúng. Chỉ tăng step có thể tăng drift trước khi tăng dấu.
6. **Tối ưu qua student:** hoãn cho đến khi native teacher mạnh và có chất lượng;
   đó là nghiên cứu riêng về transfer, không cần sửa thuật toán LoRA của bên học.

## Survey cơ chế backprop

[DRaFT, Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400)
đặt reward khả vi trên output diffusion, với biến thể DRaFT-K cắt gradient chỉ
qua K bước cuối. Bài này hỗ trợ cách tối ưu ảnh cuối với gradient hữu hạn chi phí;
không chứng minh watermark radioactivity.

[AlignProp](https://arxiv.org/abs/2310.03739) dùng reward backpropagation và gradient
checkpointing để làm việc tối ưu diffusion khả thi hơn về bộ nhớ. LoRA của owner
ở đây là cách tham số hóa cập nhật teacher, độc lập với LoRA/SFT của student.

Code mới dùng DDIM20 cho cả hai arm và inference, eta=0, clip_sample=False,
gradient qua 4 bước cuối theo mặc định. Nó không gọi pipeline forward được
bọc no_grad. Checkpoint chỉ bọc UNet, truyền timestep thành argument rõ ràng để
tránh bug closure/recomputation. Set grad_steps=steps để lấy full gradient.

Những cơ chế này đã có prior art. Giá trị nghiên cứu phải được chứng minh bằng
fingerprint/native quality và inheritance tốt hơn ở ngân sách so sánh rõ ràng.

## Gate mới

`bash run_teacher_gate.sh` so proxy và rollout trên cùng carrier budgeted, LR5e-5,
250 bước, cùng data/key/sampler/quality penalty. Tổng chỉ 500 teacher updates,
không train student. Đọc teacher_selection.json và feature_budget.json.

Hai arm có cùng số optimizer updates nhưng rollout tốn nhiều UNet evaluations
hơn; không gọi đây là đối chứng ngang FLOPs. Log stage time và thông số K/steps.
Chưa benchmark VRAM/thời gian. Không sửa code/config trong gate server đang chạy.
