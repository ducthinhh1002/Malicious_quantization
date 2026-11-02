# Pilot sửa supervision và kiểm tra carrier có khả năng khái quát

## Bằng chứng dẫn đến thay đổi

Run `bitgreen_repair_overfit_1017`: tại bước 300, loss cố định t=51 trên train giảm
từ 1 xuống 0,748 nhưng sai số latent cuối so với target tăng lên 10,17 lần;
heldout là 35,54 lần. Teacher chỉ phát hiện 1/4 train, 0/4 heldout. Vì vậy không
được dùng loss trên trạng thái cố định để kết luận native rollout đang học đúng.

Run `bitgreen_repair_shared_1017`: residual chung 16.384 tham số đạt 16/16 train
nhưng chỉ 1/8 heldout ở bước 150. Đây là bằng chứng overfit của cấu hình đó,
không phải bằng chứng carrier truyền được sang student.

## Hai pilot mới

### Terminal rollout, native teacher

`new_watermark/bitgreen_onpolicy.py --mode terminal` dùng cùng oracle target FP32
đã kiểm tra, cùng noise/prompt, nhưng tối ưu trực tiếp:

\[
L(\theta)=\mathbb E_i\frac{\|z^{20}_{\theta}(n_i,c_i)-
(z^{20}_{0,i}+\delta_i)\|^2}{\max(\operatorname{MSE}(\delta_i),10^{-8})}.
\]

Forward đi theo trạng thái hiện tại của owner; gradient đi qua **toàn bộ 20 bước**
DDIM. Không dùng prefix detach hay teacher-forcing trong objective này. Gradient
checkpointing giảm bộ nhớ nhưng không thay đổi đồ thị đạo hàm. FP32 xuyên suốt,
TF32 tắt; target và mẫu số bị detach.

Pilot: 4 train + 4 heldout nội bộ owner, 60 cập nhật, LR 1e-6, mỗi cập nhật tích
lũy đủ 4 ảnh. LoRA r16 ở owner giữ nguyên; student chưa chạy. Không can thiệp
sampler hay thêm residual khi đánh giá native teacher.

Target là oracle đã có ràng buộc chất lượng ảnh. Objective mới chỉ khớp latent;
**không bảo đảm PSNR** vì decoder phi tuyến. Vì vậy mỗi probe phải báo cả PSNR,
score, detection và sai số latent cuối. Adapter tốt nhất chỉ chọn bằng loss trên
train; heldout không chọn checkpoint. Adapter cuối được fuse, kiểm tra sai số
rollout trước/sau fusion và đánh giá thêm ở FP16.

Kết quả cần thấy trước tiên: train terminal relative MSE giảm rõ so với 1, đồng
thời detection tăng mà chất lượng còn tốt. Overfit được 4 ảnh vẫn chưa chứng minh
khái quát, và chưa chứng minh sống qua ordinary LoRA.

### Carrier có điều kiện theo nội dung, dung lượng thấp

`--mode conv`: residual là bộ lọc 3×3 dùng chung cho 4 kênh latent, tổng **148
tham số**. Đầu vào là latent sạch chuẩn hóa RMS theo kênh; residual bị chặn biên
bằng tanh. Không có tham số riêng theo ảnh hoặc image ID. Đây là thay đổi carrier
của owner, không thay đổi attacker/student.

64 ảnh owner-train học bộ lọc; 32 ảnh owner-train khác giữ lại để kiểm tra, với
prompt/noise riêng. 200 cập nhật, 4 ảnh/cập nhật, LR 3e-4; shuffle train bằng seed
cố định. Threshold detector giữ nguyên từ source, quality budget PSNR 28 dB.

Mode `smooth` cũng có sẵn để kiểm tra residual chung nội suy từ lưới 8×8 (256
tham số), nhưng **không chạy mặc định**. Giảm tham số không tự bảo đảm khái quát;
conv và smooth đều có thể thất bại vì carrier không khớp detector bit-pair.

Carrier-only vẫn cộng residual để chẩn đoán trên latent sạch: **đây chưa phải
native teacher**, không được báo thành công watermark model. Chỉ khi carrier có
tín hiệu trên heldout mới có cơ sở chuyển target đó vào bước học native UNet.

## Chạy lại

```bash
export NEW_WATERMARK_PYTHON=/root/logs/gate_env/bin/python
export BITMARK_RUNTIME_ROOT=/root/logs/runtime
bash run_bitgreen_repair.sh \
  /root/logs/results/bitgreen_repair_shared_1017 \
  /root/logs/results/bitgreen_repair_NEW
```

Source phải có manifest target FP32 hợp lệ. Cache được kiểm tra SHA256 và image
ID trước khi tái dùng. Chỉ bổ sung clean rollout cho ảnh chưa có cache; không tối
ưu oracle từng ảnh nếu chỉ thử carrier. Source và các kết quả cũ không bị ghi đè.

JSON/log nhẹ nằm trong thư mục kết quả; adapter/carrier và ảnh ở runtime thông
qua symlink. Hai pilot thực tế đã được khởi chạy riêng trong tmux trên vm4 để có
log độc lập. Chưa có kết luận về hiệu quả khi viết tài liệu này.

## Kiểm tra code

26 unit tests pass trong môi trường GPU: bao gồm sai phân hữu hạn đối chiếu
gradient full rollout, loss target detach, carrier zero-init/capacity, DDIM
bridge, cùng các kiểm tra cũ. Shell pass `bash -n`. Unit tests không thay thế
việc kiểm tra hội tụ và kết quả trên GPU.
