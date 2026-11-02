# LALT: chuyển từ dấu dễ đọc sang dấu có thể học lại

## Kết quả và điểm nghẽn đã xác nhận

- Teacher frozen-reader: key221 tăng từ 72,66% tại bước2000 lên 83,59%
  tại bước20000; cao nhất84,375% ở12000. Bit accuracy cuối vẫn khoảng71,6%.
- Probe20k/key221: teacher TPR61,72% ở FPR tổng danh nghĩa1%/2key;
  single-key1% là77,34%. Student rank4/rank16 chỉ0,78%/2,34%; đối chứng sạch0,78%.
- Các số này dùng các prompt/calibration khác validation teacher; không thể
  lấy83,59% làm baseline trực tiếp cho tỷ lệ retention của probe.
- Code `execute` dừng khi teacher gate90% không đạt. Vì vậy toàn bộ20k bước
  chưa tối ưu qua learner. Chưa có bằng chứng LALT meta-learning thất bại:
  thành phần đó chưa được thực thi trong run này.
- Reader cố định ngăn drift nhưng không biến dấu dễ đọc thành dấu dễ học.
  AUROC cao không đảm bảo nhận đúng key ở ngưỡng FPR chặt.
- Log minibatch cũng không cho thấy tăng bước giải quyết readout: median WM loss
  ở2000–5000 là0,936; ở15000–20000 là0,986, trong khi LPIPS median giảm
  0,0585→0,0497. Các minibatch khác nhau nên đây chỉ là mô tả, không phải
  so sánh paired hay bằng chứng nhân quả.

## Thử nghiệm mới

Khởi tạo carrier/reader từ20000, giữ reader bất biến. Hai key221/34 trở thành
key thiết kế cho pilot; không gọi chúng là held-out keys nữa.

Student surrogate vẫn cập nhật ordinary denoising MSE bằng functional AdamW,
rank4,4 bước. Owner backprop qua4 bước này và toàn bộ8 bước DDIM tạo ảnh query.
Đây là unroll ngắn, chưa tương đương huấn luyện student5000 bước.

Pool `meta_train` chia hai phần rời nhau để cung cấp ảnh học và prompt query
cho outer loss. Loại cả caption trùng; `meta_eval` dành cho probe ordinary
không tham gia gradient. Manifest `learning_prompt_roles.json` lưu các ID thực tế.

Owner loss:

\[
L = L_{teacher} + 2\left[L_{paired+specificity}(x_s,x_c,k)
 + 0.5\,L_{absolute\ key}(x_s,k)\right]+0.3L_{utility}.
\]

`L_absolute key` là BCE bit đúng cùng0,25BCE presence. Nó thuộc outer loss của
owner; không thêm vào loss student. `paired+specificity` so score với student
sạch và key sai. Teacher/ảnh sạch và query meta dùng các prompt khác nhau.

64 outer steps; kiểm tra16/32/48/64. Chọn checkpoint theo worst-key
min(paired_effect,key_specificity) trên episode validation cố định, với4/16
inner steps. Điều này tránh chỉ tối ưu presence khiến score mọi key cùng tăng.

## Gate và tính trung thực của báo cáo

Không sửa gate chính thức90%, không sửa FPR. Đây là nhánh diagnostic được phép
đi qua learner dù teacher cũ không đạt gate. Checkpoint được nhận phải:

- TPR không giảm quá0,03 và BA không giảm quá0,02 so baseline từng key.
- LPIPS không tăng quá0,015 và vẫn đạt giới hạn tuyệt đối cũ; các giới hạn
  LPIPS-p95, PSNR-p05, CLIP, diversity và clean FPR giữ nguyên.
- Proxy có cải thiện; vẫn báo riêng `official_gate_passed` và `preserved`.

Giữ checkpoint ở mỗi mốc16 bước, kể cả khi không được chọn. Nếu không có
checkpoint cải thiện, ghi kết quả âm tính, không train lại student baseline.
Nếu có, tự chạy ordinary rank4,2048ảnh,5000bước,PNG/caption độc lập như trước.
Proxy cải thiện không được báo là transfer thực nếu chưa có kết quả ordinary LoRA.

## Tái sử dụng và giới hạn

Dùng lại pretrained assets, prompt split và clean latent cache từ run đầu.
Tái sử dụng student sạch rank4 cùng ảnh query/calibration chỉ khi seed,rank,
số bước,LR,batch,adapter protocol,dữ liệu/caption/base và image hashes khớp.
Student marked phải train mới. Không dùng output student để chọn checkpoint.
Final test/key riêng không dùng, nhưng các probe hiện tại là development evidence
đã được tham khảo nhiều lần; chưa phải đánh giá xác nhận độc lập.

## Cơ sở nghiên cứu

- [Radioactive data: tracing through training](https://arxiv.org/abs/2002.00937):
  thiết kế dấu để dấu đi qua quá trình học, với minh chứng trên phân loại ảnh;
  không tự động suy ra hiệu quả cho diffusion LoRA.
- [Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929):
  tối ưu dữ liệu theo tác động lên quá trình học. Pilot ở đây dùng unroll AdamW,
  không tuyên bố thực hiện đúng gradient-matching của bài này.
- [DistSeal](https://arxiv.org/abs/2601.16140) và
  [training chính thức](https://github.com/facebookresearch/distseal/blob/main/docs/TRAINING.md):
  watermark latent và distillation vào generator/decoder là related work quan trọng.
  Truyền dấu qua ordinary LoRA chỉ nhận RGB/caption vẫn cần kiểm tra riêng.

Đây là kiểm tra cơ chế còn thiếu của phương pháp hiện tại, chưa phải một claim
novelty mới hoặc bảo đảm watermark sống qua mọi learner.

## Lượt thử đang triển khai

- Tmux: `lalt_transfer_pilot_20261019_055842Z`.
- Báo cáo: `/root/logs/results/lalt_transfer_pilot_20261019_055842Z`.
- Log: `/root/logs/lalt_transfer_pilot_20261019_055842Z.log`.
- Giới hạn VRAM85GiB; rank32 cũ tiếp tục trong process riêng.
- 31 unit tests đã qua local và server.
- Snapshot055555Z đã được ngắt trước update để tách query probe khỏi outer loss;
  snapshot055842Z là bản đang chạy, nguồn code/config được giữ riêng.
- Trước update đầu tiên sẽ đo baseline teacher và proxy4/16 bước. Khi có update,
  `learning_gradient_audit.json` phải xác nhận gradient của transfer loss tự nó
  tới carrier, không dựa vào teacher loss để che đường gradient đứt.
- Nếu có checkpoint được chọn, báo cáo ordinary LoRA sẽ nằm trong folder cùng tên
  cộng hậu tố `_ordinary_r4`. Nếu proxy không cải thiện, báo cáo kết quả âm tính
  `complete_no_proxy_improvement`, không tự gán thành công.
