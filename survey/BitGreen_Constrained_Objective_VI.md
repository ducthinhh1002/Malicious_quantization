# BitGreen: sửa objective owner để ưu tiên độ trung thực của ảnh

## Dữ liệu dẫn tới thay đổi

Ở design sweep 64 ảnh, cả Plain và Robust phải chấp nhận PSNR khoảng 18 dB mới
đạt TPR khoảng 98%. Hạ strength xuống 0.3 nâng PSNR khoảng 24 dB nhưng TPR chỉ
4.69%. Nội suy trọng số sau train chưa đủ; cần thay cách owner học hướng nhúng.

PSNR là sai khác với ảnh clean cùng prompt/seed. Nó không phải đánh giá đầy đủ
về chất lượng thị giác. Báo cáo bổ sung CLIP/FID/KID được xuất kể cả teacher fail.

## Mục tiêu mới

Giữ tokenizer BSQ và detector cũ, không đổi student. Với soft score S, hard score H:

```
score_ST = S + stop_gradient(H - S)
L_mark = mean(relu(target - score_ST)^2) / margin^2
target = frozen_clean_threshold + 0.004
margin = 0.012
```

S và H đều lấy trung bình bốn scale đúng như detector. Forward dùng H, backward
dùng gradient của S. Đây là STE xấp xỉ, không phải đạo hàm chính xác của dấu bit.
Không ép mọi scale tăng 0.025 như objective cũ; không tiếp tục đẩy khi đã đạt target.
Threshold chỉ dùng 1.000 ảnh calibration sạch, được đóng băng trước teacher training.

Với budget B = 10^(-24/10), r = MSE(x_owner,x_clean)/B:

```
L_quality = mean(0.1*r + lambda*relu(r-1) + 0.5*relu(r-1)^2)
            + 0.5*coarse_ratio + 0.25*noise_ratio
lambda <- clamp(lambda + 0.1*clamp(mean(r)-1,-1,10), 0, 50)
```

`coarse_ratio` là MSE ảnh average-pool 8x8 chia B, nhằm hạn chế thay đổi bố cục/thông
tin thô. `noise_ratio` là MSE giữa dự đoán noise owner và model clean trên cùng
latent của clean trajectory, chia `0.001 * mean(clean_prediction^2)` với sàn số học.
Đo cả conditional/unconditional prediction. Mỗi update chọn một timestep luân phiên
trong toàn bộ 20 timestep; đây là xấp xỉ theo timestep, không đo mọi timestep cùng lúc.

Plain: `L_mark + L_quality`.
Robust: `L_mark + 0.5*L_channel + L_quality`.
Channel dùng hard-score margin tăng 0.7*0.012 so với clean image qua cùng channel;
không diễn giải margin channel thành một detector FPR đã được hiệu chuẩn riêng.

Gradient rollout mặc định đi qua đủ 20/20 bước DDIM, dùng checkpointing. Điều này
cho quality loss tác động được tới các bước đầu, thay vì bỏ qua gradient của 16
bước đầu như trước. LR owner giảm từ 5e-5 xuống 1e-5. Budget training vẫn 250 updates
mỗi nhánh; chi phí mỗi update cao hơn, không tuyên bố so sánh cùng FLOPs.

## Báo cáo và giới hạn

Training log thêm pixel PSNR, pixel/coarse/noise ratio, quality dual, hard score,
detector target và số bước gradient. Chọn checkpoint cuối theo ngân sách định trước;
không dùng test/student feedback để lựa checkpoint hay sửa student.

24 dB là mục tiêu penalty mềm, không bảo đảm mọi ảnh đạt 24 dB. Gate độc lập vẫn
TPR >=80% và mean PSNR >=22 dB. Nếu gate fail vẫn lưu teacher report và tính utility.
Ảnh train/pilot/calibration/test tách trong run, nhưng khi lặp seed có thể trùng
run trước: đây là development pilot thích nghi, cần dữ liệu/key/seed mới khi chốt
claim khoa học. Chưa có kết quả chứng minh objective mới tốt hơn.

Chỉ teacher UNet được huấn luyện. VAE, bit tokenizer và clean reference được đóng
băng. Student chỉ nhận ảnh/caption, clean SD1.4 và ordinary denoising MSE như cũ.

## Lệnh

```bash
bash run_bitgreen.sh --calibration-source /path/to/completed_bitgreen_run
# Đối chứng objective cũ:
bash run_bitgreen.sh --owner-objective legacy
```

Có thể đặt `--grad-steps` để đo ablation, nhưng <20 không còn là full-trajectory
gradient. Các phép thử quan trọng kiểm tra forward sampling không đổi khi tăng
số bước gradient và early-step parameter có gradient chỉ khi được backprop tới.
