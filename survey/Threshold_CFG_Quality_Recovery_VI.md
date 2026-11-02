# Cải thiện chất lượng threshold_cfg_v1 với điều kiện giữ truyền dấu

Ngày 22/10/2026. Đây là phân tích code và thiết kế kiểm chứng, chưa có kết quả cải tiến mới.

**Cập nhật 24/10:** người dùng đã bỏ quality stopping gates sau ngày viết tài
liệu này; các điều kiện dừng theo quality/retention bên dưới là thiết kế lịch sử,
không áp dụng cho luồng hiện tại. Các lượt mới chạy hết budget và lưu cả kết quả
xấu. Xem [survey cập nhật, kết quả student và channel audit](../code_best_pipeline/shared/threshold_quality_v1_source/docs/transfer_research_20261024.md).

## Trạng thái và baseline

Đã dừng coordinator 3643753, wrapper 3758185 và trainer 3758234 của run
`semantic_trajectory_20261022/run_20261022_153728` theo yêu cầu người dùng.
Xác minh cả ba PID không còn tồn tại; giữ snapshot student 2500 và 5000.
Run đó dùng semantic_sign_v1, không phải threshold_cfg_v1.

Nguồn kết quả lịch sử: `../t2ifing_radioactive/result_vast/RESULT.md`.
Threshold teacher test: detection 100%; LoRA rank32 attention+conv: 99,7%.
Teacher kappa 1/1,5/2 có LPIPS validation 0,488/0,52049/0,54189.
Không có số student riêng cho kappa 1 hoặc 1,5 trong tài liệu này.
Không suy rằng giảm kappa vẫn giữ radioactivity từ teacher detection.
Chưa tìm thấy thư mục kappa_* trong tìm kiếm giới hạn sâu 7 dưới /root/logs;
local không tìm thấy checkpoint safetensors/pt của baseline trong repo vừa cung cấp.
Tài liệu ghi historical teacher mirror/preparation khác VM4; phải xác minh artifact,
không coi số cũ là baseline cùng điều kiện khi chưa tái lập.

## Những gì code gốc thực sự tối ưu

`semantic_threshold.py`: q là PCA coordinate được chuẩn hóa bằng sigma;
score r = mean(q*s), s là vector dấu khóa cố định.

    d = clamp(tau + kappa*sigma_clean - r_clean, 0, 2.5)
    L_shift = mean((q_marked - q_clean - d*s)^2)

Target là dịch chuyển vector có cấu trúc theo cùng key, không chỉ vượt một ngưỡng scalar.
Đây là cơ chế nên giữ khi nghiên cứu khả năng student học dấu.

`semantic_teacher_train.py::_threshold_microbatch_backward`:

    L = L_preserve_cond_uncond + 0.8*(L_shift + 0.5*L_shift_cycle)

Loss semantic áp dụng khi t <= semantic_t_max; có đường VAE cycle.
CFG 7,5 được dùng trong predicted x0. Không có LPIPS/pixel loss trực tiếp ở nhánh này.
`lambda_preserve` không nhân vào preserve trong hàm threshold: chỉ đổi cờ này sẽ
không điều khiển hệ số preservation như người đọc config có thể tưởng.
FP_WEIGHT và CYCLE_WEIGHT là hằng số .8/.5; không dùng lambda_fp RSSF.

`quality_passes` bỏ qua gate LPIPS đối với threshold_cfg_v1, chỉ chặn CLIP drop.
`choose_candidate` chọn TPR cao, rồi mean score cao, rồi kappa thấp.
Do đó khi TPR bão hòa, teacher vẫn có thể được chọn vì đẩy score xa hơn,
dù ảnh lệch nhiều hơn. Đây là khả năng tối ưu chưa phù hợp mục tiêu chất lượng,
không phải bằng chứng rằng mean score dư thừa: student có thể cần biên dư đó.

## Vì sao không tiếp tục trajectory_budget_v1

Run đã dừng thay đồng thời objective dấu, sampling path, preservation,
gradient cuối 8 bước, budget controller và gate report-only.
100 bước cuối: fp=0,21975, quality=946,25, mean_K=15,62,
LPIPS=0,20834, pixel MSE=0,04824; pixel dual chạm 100.
Tỷ lệ giá trị loss không phải tỷ lệ gradient, nên chưa quy kết một nguyên nhân duy nhất.
Teacher validation 0/128; test 5/1000; clean test 2/1000 theo K>=24.
Không dùng checkpoint này để phục hồi threshold baseline.

## Hướng ưu tiên 1: tìm mức sửa model nhỏ hơn, giữ nguyên cơ chế

Nếu có teacher threshold tốt: giữ bản gốc bất biến, thử bản sao
W(a)=W_clean+a*Delta_W với a=1, .95, .9, .8; giảm tiếp chỉ khi các mức trước đạt.
Với LoRA, scale delta BA hoặc scale adapter; không nhân cả A và B với a
vì như vậy delta thành a^2. Với pipeline merge, bảo đảm clean/marked cùng base,
chỉ nội suy đúng tensor UNet khác nhau. VAE/text encoder giữ nguyên.
Đây là sweep inference, không cần train teacher mới; quality không được giả định
đơn điệu theo a. Mỗi a cần đo native PNG và verifier gốc.

Đồng thời đánh giá checkpoint/kappa thấp hơn nếu có artifact gốc.
Không đổi key/PCA, không refit threshold để cứu một candidate yếu.
Không chọn bằng highest mean score sau khi đã đáp ứng yêu cầu detection;
chọn chất lượng tốt nhất trong tập đạt teacher và student retention gates.

Đây là hướng ít thay đổi nhất, nhưng không bảo đảm giảm được LPIPS đến .15.
Một phần dịch chuyển semantic chính là tín hiệu student học; không thể hứa
xóa toàn bộ độ lệch đó mà dấu chắc chắn còn nguyên.

## Hướng ưu tiên 2: fine-tune nhẹ từ teacher tốt, có neo giữ dấu

Chỉ dùng nếu hướng 1 không cải thiện đủ. Bắt đầu từ threshold teacher đạt;
không bắt đầu clean model với objective mới như run thất bại.
Giữ loss shift-vector và cycle gốc, fixed key/threshold, timestep distribution,
teacher/student backbone và sampling. Thêm quality từng thành phần riêng biệt.

Đo chất lượng trên ảnh cuối native trajectory; one-step x0 quality chỉ là proxy.
Có thể dùng loss LPIPS trên một số batch kiểm tra/huấn luyện sau khi kiểm chứng
gradient; không mặc định cắt gradient 8 bước. So gradient full với truncated trên
cùng prompt, đo cosine, norm và thay đổi score thật sau update.

Đặt điều kiện watermark trước chất lượng: quality update chỉ được nhận nếu
không phá biên score và hướng residual trên tập canary development đã đóng băng.
Giới hạn norm bước cập nhật, giảm learning rate, rollback cả model lẫn optimizer
khi vi phạm. Log gradient norm của từng objective và cosine giữa chúng.
Các phép chiếu gradient chỉ bảo vệ bậc nhất ở batch hiện tại; không bảo đảm
native TPR, không bảo đảm radioactivity. Luôn cần holdout và student đo thật.

Không dùng controller tăng vô hạn để ép LPIPS=.1/pixel=.005 ngay từ đầu.
Budget lấy từ baseline đo được, giảm theo từng mục tiêu khả thi, thí dụ giảm
LPIPS 10–20% là mốc thử ban đầu chứ không phải kết quả đã được chứng minh.

## Quy trình kiểm chứng và điều kiện dừng

1. Khôi phục config/checkpoint/teacher mirror gốc nếu còn. Nếu không, tái lập
   threshold_cfg_v1 nguyên bản trên VM4 và công bố đây là baseline mới.
   Phải giữ dữ liệu, seed, backbone, CFG, steps, batch, precision và LR ngang nhau.
   Không dùng carrier/median artifacts của semantic_sign_v1 cho threshold_cfg_v1.
2. Khóa threshold teacher và student từ clean calibration tương ứng trước lựa chọn.
   Thêm matched clean-trained student để phân biệt watermark với generic fine-tuning.
   Unadapted control một mình không đủ chứng minh tính đặc hiệu.
3. Teacher development screening trước: score quantiles/margins, TPR, clean FPR,
   LPIPS/SSIM/PSNR cùng prompt/seed, CLIP và kiểm tra ảnh đa dạng, lỗi anatomy, texture.
   LPIPS đo độ khác ảnh; không được gọi giảm LPIPS là cải thiện thẩm mỹ tự động.
4. Chỉ finalist đạt teacher mới tạo dataset/train student. So sánh cùng rank32 conv,
   cùng ngân sách/schedule gốc. Probe ngắn chỉ được loại lỗi lớn, không thay long run.
5. Điều kiện không suy giảm đề xuất: teacher và student đều không giảm quá 1 điểm
   phần trăm TPR so với baseline cùng điều kiện; kiểm tra CI của chênh lệch paired,
   FPR độc lập không tăng. Đây là ngưỡng thiết kế, không phải ý nghĩa của 'bảo đảm'.
   Nếu 1000 ảnh không đủ kết luận CI, tăng số ảnh hoặc kết luận chưa đủ bằng chứng.
6. Tối ưu/lựa chọn trên development. Final test độc lập một lần sau khi khóa lựa chọn;
   không chỉnh threshold hoặc chọn checkpoint bằng final test. Dùng nhiều training
   seeds cho kết luận giữ transfer; bootstrap ảnh không thay thế seed uncertainty.
7. Báo FID/KID với cùng số mẫu, cùng reference thật nếu muốn đánh giá realism.
   FID so với clean-generated chỉ đo distribution drift; cần ghi rõ loại reference.
8. Teacher fail: dừng trước student. Student fail retention: candidate bị loại dù
   ảnh đẹp hơn. Lưu baseline bất biến và xuất báo cáo failed, không report-only chạy tiếp.

## Literature được đối chiếu và phạm vi áp dụng

- [DIAGNOSIS](https://arxiv.org/abs/2307.03108): dữ liệu có biến đổi nhất quán
  có thể được học qua fine-tuning, gồm LoRA. Hỗ trợ yêu cầu giữ tính học được của
  tín hiệu; không chứng minh một detector threshold mạnh sẽ tự động truyền dấu.
- [SIREN](https://arxiv.org/abs/2410.10437): nhấn mạnh coating phải liên quan nhiệm vụ
  để được học và kết hợp ràng buộc cảm nhận. Không chuyển nguyên kết quả personalization
  thành bảo đảm cho SD2.1-to-SD1.4 của chúng ta.
- [SERUM](https://arxiv.org/abs/2603.13396): fixed noise watermark và learned detector;
  khác native merged UNet/fixed semantic verifier. Không đổi sang noise injection
  rồi gọi là cải thiện cùng threshold_cfg_v1.
- [DRaFT](https://arxiv.org/abs/2309.17400): gradient qua sampler và biến thể truncated
  hữu ích cho differentiable reward. Không có bảo đảm giữ watermark/student transfer.
- [DPOK](https://arxiv.org/abs/2305.16381): regularization với reference là tiền lệ
  cho giữ model gần baseline; không đủ để khẳng định giữ TPR hay radioactivity.

Kết luận thiết kế: ưu tiên strength/checkpoint selection trên baseline tốt, sau đó
fine-tune cục bộ có điều kiện giữ dấu. Chưa có bằng chứng thực nghiệm cho candidate mới;
không khởi chạy training mới trong lượt nghiên cứu này.
