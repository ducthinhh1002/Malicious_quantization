# Vì sao giảm chất lượng detection và hướng cải thiện có kiểm chứng

Ngày 24/10/2026. Mục tiêu vẫn là teacher sinh RGB/caption, student SD1.4 tự học
bằng ordinary LoRA rank 32 attention + convolution; student không nhận key hay
watermark loss. Không hạ ngưỡng, không đổi carrier để làm đẹp kết quả.

## 1. Kết quả hiện có chưa đạt mục tiêu đồng thời

Hai student dưới đây cùng 5.000 ảnh train, 1.000 validation, 1.000 test,
7.500 update, batch 4, cùng carrier và ngưỡng student 0.8541307449340821.
Teacher là Manojb SD2.1, threshold_cfg_v1, kappa=1. Chọn checkpoint bằng
validation denoising MSE. `k1_evolution` thực tế chọn vector toàn 1, nên là
đối chứng scale 1.0, không phải một cải tiến tiến hóa thành công.

| Student | Detection test | Mean score | LPIPS với SD1.4 chưa fine-tune | FID | KID | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| Scale 1.0 | 75.0% | 1.036625 | 0.550447 | 54.44424 | 0.00166651 | 0.787681 |
| Scale 0.85 | 42.7% | 0.805889 | 0.533659 | 54.56305 | 0.00146374 | 0.787436 |

Scale 0.85 giảm LPIPS 3.05%, nhưng mất ròng 323/1.000 detections. Trong 750 ảnh
scale 1.0 phát hiện được, 368 ảnh bị mất detection; 45 ảnh khác chuyển sang dương.
Paired bootstrap 5.000 lần, seed 20261024: giảm LPIPS tuyệt đối 0.0167882,
CI95 [0.0121073, 0.0217311]; giảm detection 32.3 điểm phần trăm,
CI95 [28.9, 35.7]. Đây là bất định theo ảnh trên một training seed, không phải
bằng chứng bền vững qua nhiều training seed. FID/KID/CLIP chưa có kiểm định
chênh lệch; không gọi các thay đổi rất nhỏ là cải thiện rõ ràng.

LPIPS này đo độ lệch so với model chưa fine-tune, không đo trực tiếp mức đẹp/xấu.
Một student không học dữ liệu có thể gần unadapted model nhưng không đạt mục tiêu.
Cần thêm clean-trained control để tách thay đổi do fine-tune bình thường khỏi
thay đổi do watermark. FID/KID dùng clean teacher test làm reference, không phải
COCO ảnh thật. Không so tuyệt đối các chỉ số khác reference/dataset.

## 2. Những nguyên nhân có bằng chứng và phần chưa biết

**Tín hiệu đã yếu đi trước khi student học.** Trên cùng teacher validation,
scale 1.0 có TPR 98.9%, LPIPS 0.436507 và p10(score)-threshold = 0.314454.
Scale 0.85 có TPR 94.3%, LPIPS 0.393156 nhưng p10 margin chỉ 0.072106:
mất 77.1% phần dư ở đuôi thấp. TPR teacher gần trần che khuất mức suy giảm này.
Student học phân phối ảnh không có cam kết giữ nguyên biên detection.

**Mốc lịch sử 99.70% không phải student của kappa=1.** RESULT.md ghi teacher
được chọn là kappa=2. Chưa khôi phục được checkpoint/carrier/config bất biến
của run lịch sử. Bởi vậy hiện chưa có đối chứng tái lập chính xác mốc 99.70%,
và không thể quy toàn bộ khoảng cách cho một thay đổi riêng lẻ. Kappa=2 trên
split/carrier hiện tại là đối chứng cần thiết nếu muốn xác định tác động của kappa;
nó không thay thế nhánh kappa=1 người dùng yêu cầu.

**Chưa đủ exposure là giả thuyết đang thử.** 7.500 × batch 4 / 5.000 = sáu
lượt dữ liệu. Validation TPR .85 tăng 39.1 → 42.6 → 48.2% tại 2.500/5.000/7.500.
Lượt 22.500 đang kiểm tra bão hòa/overfit; không được gọi tăng compute là cải tiến
teacher. Setup 1.000 ảnh gần đây có 30 lượt ở 7.500 bước; chưa xác minh con số
train_n của chính run lịch sử hoàn tất 99.70% từ artifact gốc.

**Chưa tìm thấy lỗi hiển nhiên trong student trainer.** Kiểm tra trước đó xác
nhận trainer gốc được giữ nguyên, 7.500 optimizer updates áp dụng, không skip.
Weights và các cấu hình cần thiết của hai snapshot Manojb/sd2-community trùng
hash. Bit accuracy 100% là dấu của tọa độ trung bình toàn tập, không phải TPR
theo ảnh. Unadapted test có 0.8% dương; điều này chưa thay được kiểm tra FPR trên
student đã fine-tune bằng dữ liệu sạch.

**Mục tiêu search chưa đánh giá đúng đích cuối.** Teacher-only TPR/LPIPS và
attribution theo sáu nhóm không đo student radioactivity. Search trên teacher có
thể tìm được ảnh dễ đọc nhưng không dễ học. Vector all-1 thắng search vừa rồi
chỉ nói các ứng viên/tiêu chí đã thử chưa tốt hơn, không chứng minh mọi hướng
cải tiến đều vô dụng.

## 3. Survey nguồn sơ cấp và điều thực sự áp dụng được

| Công trình | Kết quả/phương pháp liên quan | Áp dụng và giới hạn |
|---|---|---|
| [Radioactive Watermarks in Diffusion and Autoregressive Image Generative Models, 2025](https://arxiv.org/html/2506.23731v1) | Phân tích watermark bị mất khi encode latent hoặc qua noising–denoising; WIAR là giải pháp cho image autoregressive models. | Đo riêng VAE và denoising bằng ảnh cũ. Không chuyển thuật toán green-list token sang SD khi SD không có token alphabet tương ứng. |
| [SIREN, mục IV-C1/C2 và V-H](https://arxiv.org/html/2410.10437v1) | Dùng surrogate SD1.5 đã fine-tune dữ liệu sạch để tối ưu coating theo denoising loss; thêm khoảng cách màu CIEDE2000. Ablation cho thấy learnability và perceptual loss giải quyết hai yêu cầu khác nhau. | Là cơ sở cho teacher loss có xét student learning channel. Paper dùng coating generator/reader và personalization; số liệu kiểm định của họ không tương đương TPR từng ảnh của ta. Không sao chép reader mới hoặc khẳng định tái lập SIREN. |
| [SERUM, bảng 3 và phụ lục I](https://arxiv.org/html/2603.13396v1) | SD2.1 → SD1.4, 5.000 train/1.000 validation; LoRA đạt 52.30% TPR@1%FPR, checkpoint MSE tốt nhất ở 19.500 bước. | Đây là đối chiếu gần bài toán hơn nhiều paper robust image watermark. Hỗ trợ đo learning curve dài; không chứng minh 22.500 bước sẽ cứu được carrier của ta. SERUM dùng watermark/detector khác. |
| [HMark, mục 3.3 và 5](https://arxiv.org/html/2512.00094v1) | Nhúng h-space, học pixel reader, kết hợp detection/bit/MSE/LPIPS; nêu giới hạn với latent diffusion. | Học được cách giữ perceptual fidelity cùng semantic signal; không lấy kết quả DDPM của họ làm cam kết cho SD2.1 → SD1.4. Thay toàn bộ carrier/reader sẽ làm mất tính so sánh với threshold_cfg_v1. |
| [AquaLoRA, mục 4.3.2](https://arxiv.org/html/2405.11135v1) | Prior-preserving fine-tuning dùng dự đoán model đóng băng để học latent offset vào UNet. | Hữu ích cho giảm drift khi train native teacher LoRA. White-box watermark tích hợp sẵn không tự chứng minh truyền qua RGB tới student mới. |
| [DRaFT, Clark và cộng sự](https://arxiv.org/html/2309.17400v2) | Tối ưu differentiable reward trên ảnh cuối qua sampling; có LoRA, gradient checkpointing, truncated DRaFT-K và DRaFT-LV. | Hướng xử lý khoảng cách giữa one-step x0 loss và ảnh sinh đủ bước. Có nguy cơ reward hacking; tối ưu LPIPS/aesthetic score không tự giữ watermark. Phải kiểm tra gradient và detection trước khi dùng làm bước tinh chỉnh teacher. |
| [MPaGE, Ha và cộng sự, có Huỳnh Thị Thanh Bình](https://arxiv.org/html/2507.20923v1) | Pareto-grid elites và đa dạng hóa cấu trúc heuristic bằng LLM trong tối ưu tổ hợp. | Có thể mượn archive đa mục tiêu; thay fitness bằng chất lượng và detection của student. Không gọi search sáu hệ số là toàn bộ MPaGE. Tiến hóa tối ưu một proxy sai vẫn có thể cho kết quả cuối kém. |

Không coi bài báo nào trong bảng là bằng chứng phương án mới của ta đã thành công.
Các bài dùng threat model, detector, số ảnh và kiểu fine-tune khác nhau.

## 4. Kiểm tra cơ chế đã triển khai

`channel_transfer_audit.py` chỉ đọc ảnh validation cũ, không sinh lại tập train.
Chọn trước 64/1.000 indices bằng seed 20261024, không đọc test. Bốn nhánh:
clean, scale 1.0, scale 0.85 và conv-restore. Kiểm tra hash ảnh, caption, seed,
carrier và threshold; lưu toàn bộ chỉ số từng ảnh cùng manifest.

Các kênh đo:

1. Ảnh RGB nguyên trạng qua verifier gốc.
2. RGB → **VAE của student SD1.4**, posterior mean → decode → verifier gốc.
3. Như trên với hai posterior samples; dùng cùng noise cho ảnh clean/marked.
4. Student VAE sample → thêm noise tại t=100/300/500 → UNet SD1.4 sạch dự đoán
   x0 một bước → decode → verifier. Denoiser có text condition, không CFG, đúng
   loại input conditional trong ordinary training; prediction conversion theo
   epsilon/v/sample của scheduler. Ghi denoising MSE trên cùng noise.

Thống kê chính là paired signal `mean(score(marked)-score(clean))` qua từng
kênh, kèm CI bootstrap theo ảnh và tỷ lệ signal còn lại. Đồng thời ghi clean và
marked fraction vượt **ngưỡng gốc**. Không gọi các tỷ lệ này TPR@1%FPR, vì biến
đổi ảnh có thể làm đổi null và phép đo không fit lại ngưỡng.

Đây là probe bằng **unadapted model**, không phải unroll learning, không phải
regeneration nhiều bước, không chứng minh nguyên nhân nhân quả của transfer.
VAE của probe chạy FP32, UNet/text SD1.4 chạy FP16, verifier chạy FP32; trainer
gốc dùng VAE FP16. Ảnh reconstructed được clamp [-1,1], không lượng tử hóa lại
PNG 8-bit. Vì vậy đây không phải replay bit-exact của training/inference pipeline.
Nếu VAE đã giữ hầu hết signal, tăng thêm VAE-cycle loss không có ưu tiên cao.
Nếu chỉ mất mạnh sau denoising, ưu tiên learnability; phải kiểm tra tiếp bằng
ordinary student thật. Hai unit test về paired estimator và split sampling
đã qua trên máy local và VM4; GPU audit được chạy độc lập với training.

### Kết quả GPU audit hoàn tất trong lượt nghiên cứu này

Tỷ lệ chênh lệch score marked–clean còn lại, chuẩn hóa theo RGB của từng nhánh:

| Nhánh | RGB | Student VAE, posterior sample | One-step x0 t=100 | t=300 | t=500 |
|---|---:|---:|---:|---:|---:|
| Scale 1.0 | 100% | 90.86% | 57.23% | 34.56% | 17.98% |
| Scale 0.85 | 100% | 90.83% | 55.38% | 34.82% | 17.85% |
| Conv restore | 100% | 91.11% | 56.53% | 33.79% | 17.89% |

Paired signal RGB của ba nhánh lần lượt 1.392244, 1.180414, 1.217386.
Ở t=300 còn 0.481128, 0.411078, 0.411391. Posterior mean và hai samples cho
kết quả VAE gần nhau trong phép đo này.

![Paired semantic signal qua các reconstruction probes](../../output_artifacts/channel_audit_20261024/paired_signal.png)

**Suy luận được hỗ trợ:** VAE làm mất khoảng 9% signal; attenuation lớn hơn nằm
ở denoising probe. Giảm scale chủ yếu giảm signal ban đầu, không tạo ra cơ chế
truyền dấu tốt hơn. Conv restore chỉ nhích amplitude; tại t=300 gần như trùng
uniform .85. Vì vậy chưa có cơ sở chờ một mức phục hồi rất lớn chỉ nhờ vector
này; student thật vẫn cần chạy xong để kết luận.

Cả scale 1 và .85 đều có denoising MSE thấp hơn clean trong probe (t=300:
delta -0.018092 và -0.018247), nhưng student thực tế có TPR rất khác. Điều này
là một lý do cụ thể để **không dùng raw denoising MSE đơn lẻ làm proxy radioactivity**.
Không suy ra SIREN thất bại: họ dùng clean-trained surrogate và các loss/detector
khác, còn đây là unadapted probe trên 64 paired validation images.

Artifact đầy đủ: [result.json](../../output_artifacts/channel_audit_20261024/result.json)
và [plan.json](../../output_artifacts/channel_audit_20261024/plan.json). Script
`channel_transfer_audit.py` và `plot_channel_audit.py` giữ code tái lập. Đã kiểm tra
56 file trong code manifest của luồng recovery, không file nào bị thay đổi.

## 5. Hướng cải tiến ưu tiên sau phép đo

### A. Giữ mục tiêu coherent shift, tối ưu lại chất lượng thay vì nhân nhỏ toàn bộ adapter

Giữ nguyên carrier, signs, sampler và công thức `calibrated_shift_loss` của
threshold_cfg_v1. Không đổi sang squared-hinge từng bit như luồng RSSF cũ.
Khởi tạo từ checkpoint đang có, dùng native teacher LoRA rank32 attention+conv.
Chất lượng phải được đo sau native sampling đầy đủ, cùng prompt/seed với sạch.

Đề xuất kiểm thử riêng hai bổ sung, không gộp nhiều thay đổi ngay từ đầu:

- Perceptual regularization trên ảnh predicted-x0 ở low/moderate noise, cùng
  clean reference, vẫn giữ coherent loss gốc. Đây là surrogate cho full-sampling
  LPIPS, cần kiểm tra tương quan; không giả định giảm loss một bước thì ảnh 50
  bước chắc chắn đẹp hơn. Vẫn ghi CLIP, FID/KID và visual pairs.
- Learnability regularization qua VAE + conditional denoising của một surrogate
  **đã học dữ liệu sạch**, đóng băng. Phần này chỉ cập nhật teacher; final student
  vẫn chạy ordinary MSE, không được nhận watermark loss hoặc key.

Với cùng posterior noise, diffusion noise và timestep, proxy theo SIREN là:

\[
L_{learn}=E[\|\epsilon_{\theta_c}(\alpha_t E_s(x_w)+\sigma_t\epsilon,t,c)-\epsilon\|^2
-\|\epsilon_{\theta_c}(\alpha_t E_s(x_c)+\sigma_t\epsilon,t,c)-\epsilon\|^2].
\]

`theta_c` là clean-trained surrogate, `E_s` dùng posterior sample theo student.
Biểu thức này viết cho epsilon-prediction; implementation phải đổi target nếu
scheduler dùng v-prediction. Denoising loss nhỏ không tự chứng minh dấu học được:
ảnh trở về sạch cũng có thể giảm loss, nên coherent-shift constraint là thiết yếu.
Không đưa raw unadapted audit MSE lên làm fitness duy nhất.

Một hướng mạnh hơn nếu proxy denoising không xếp hạng đúng là **paired ordinary
student updates**: từ cùng clean initialization/optimizer state, chạy hai learner
ngắn trên RGB marked và clean với cùng noise rồi đo chênh lệch score trên query
prompts khác training. Gradient outer chỉ cập nhật teacher, inner updates vẫn
ordinary denoising. Đây là adaptation theo learning channel, liên hệ thiết kế
[LALT trước đây](../../survey/Learning_Aligned_Latent_Transport_VI.md), nhưng giữ
verifier/carrier của threshold_cfg_v1 thay vì học reader mới. Kết quả gradient
qua vài bước AdamW chỉ là proxy; cần kiểm tra xếp hạng ở horizon dài và final
rank32-conv độc lập. Không lặp lại bước đổi cả embedder, reader và loss cùng lúc.

Nếu perceptual loss một bước không chuyển thành native quality, bước tiếp theo
là DRaFT-style tinh chỉnh trên ảnh cuối: cùng seed, forward sampler đầy đủ,
gradient qua K bước cuối và giữ coherent-shift loss gốc. So sánh gradient/norm
và mức thay đổi margin với full-gradient trên vài training prompts trước khi
chọn K. Không cache prefix lấy từ model cũ rồi gọi đó là gradient của trajectory
hiện tại; cache ảnh clean/reference được, đường marked phải theo weights hiện tại.
Phương án này chưa triển khai, có chi phí cao hơn thêm loss predicted-x0.

Loss ứng viên: `L_original_threshold_cfg + lambda_q*L_perceptual + lambda_l*L_learn`.
Chạy ablation original / +perceptual / +perceptual+learnability, cùng ngân sách.
Chọn lambda trên training diagnostics/validation, ghi gradient norm và margin;
chưa có cơ sở tuyên bố một lambda cố định là tối ưu. Scale 0.85 không có ý nghĩa
chất lượng độc lập nếu chỉ tăng norm adapter lên 1/0.85; phải báo effective
adapter, actual LPIPS và student detection, tránh cải tiến chỉ bằng đổi tên scale.

### B. Nếu dùng tiến hóa, fitness phải phản ánh student

Giữ archive Pareto nhưng thay điểm teacher TPR gần bão hòa bằng margin và
validation detection của **student thật** cùng budget; quality gồm cả teacher
và student. Có thể dùng học ngắn để giảm chi phí, nhưng trước tiên phải đo tương
quan xếp hạng ở 2.500/7.500 bước và giữ vài ứng viên đa dạng để tránh loại nhầm
slow learners. Đây là nghiên cứu nhiều mức ngân sách của ta, không khẳng định
MPaGE nguyên bản thực hiện thủ tục này.

Không thêm hàng chục vectors khi chỉ có vài student đo được. Ưu tiên ablation
loss có cơ sở; teacher-only search đã cho biết giới hạn hiện tại.

### C. Các đối chứng còn thiếu

- Clean-trained rank32-conv cùng 5.000 ảnh/7.500 bước để kiểm tra specificity và
  tách quality drift do training. Nếu dùng long-budget để claim thì thêm clean
  control cùng budget. Tái dùng clean RGB đã có, không cần render lại.
- Kappa=2 trên **cùng** current carrier/split/budget, là kiểm tra riêng giả thuyết
  khác kappa; giữ kappa=1 làm nhánh chính. Chưa gọi đây là phục hồi lịch sử.
- Fresh holdout và thêm training seeds cho ứng viên thắng. Test hiện tại đã
  được xem nhiều lần, chỉ dùng exploratory comparison trong vòng cải tiến này.

## 6. Luồng đang chạy và tiêu chí kết luận

Hai thử nghiệm có sẵn tiếp tục: `.85 / 22.500 steps` và
`[.85,1,.85,.85,.85,.9] / 7.500 steps`. Thứ tự groups là down_attention,
down_conv, mid_attention, mid_conv, up_attention, up_conv. Chúng kiểm tra
exposure và vị trí phân bố độ mạnh, chưa phải triển khai loss đề xuất ở mục 5.

GPU audit bổ sung chỉ tạo JSON/log nhỏ; giữ nguyên code manifest của các jobs
đang chạy. Không mở lại scale 0.95. Không có quality gate dừng sớm; lưu cả kết
quả xấu. Bài toán chỉ được coi cải thiện khi có teacher/student quality tốt hơn
với detection giữ được trên đối chứng cùng điều kiện, kèm specificity. Hiện
chưa có bằng chứng vượt đồng thời mốc lịch sử 99.70% và chất lượng ảnh.
