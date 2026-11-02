# Nghiên cứu lại fingerprint truyền qua ordinary LoRA

Ngày 21/10/2026. Mục tiêu: cải thiện fingerprint của owner để student học từ
PNG/caption bằng LoRA thông thường vẫn mang dấu. Không thêm watermark loss,
key, latent bí mật hoặc module giải mã vào attacker. Không cam kết dấu sống
qua mọi cấu hình học; cần chứng minh trên các cấu hình độc lập đã định trước.

## 1. Điều phải thay đổi trong cách tiếp cận

Không tiếp tục xem teacher TPR cao là chỉ báo đủ cho transfer. Không dùng
mức tăng rất nhỏ của unroll vài bước thay cho kết quả student thật. Không
giữ cố định reader cũ khi thay cơ chế carrier, và không gọi bootstrap nhiều
ảnh từ cùng model là nhiều model độc lập.

Bằng chứng hiện có:

- LALT teacher 20k có detection tốt; ordinary rank4 5.000 bước vẫn chỉ
  3,91% tại nominal image FPR 1%. Repair tăng teacher nhưng không tăng student.
- Contrast/replay 96 bước mỗi nhánh không qua gate. Mức tăng proxy tốt nhất
  +0,000270/+0,002028 đều dưới +0,005; teacher validation giảm khoảng 4–5
  điểm phần trăm. Xem [báo cáo trước](LALT_Transfer_Root_Cause_and_Contrast_Replay_VI.md).
- [SERUM archive của dự án](../server_review/SERUM_result_zip_20261017.json)
  có teacher 100%, full fine-tuning 83,6%, attention LoRA rank16/32 chỉ
  2,3%/2,9%; thêm conv đạt 3,1%/3,6%. Đây là gợi ý mạnh về khác biệt khả năng
  học, nhưng không phải ablation chỉ đổi rank: các nhánh khác phạm vi tham số
  và có thể khác learning rate. Không suy ra full fine-tuning luôn tốt hơn.
- Round-trip VAE của LALT không xóa hết dấu. Dấu sống qua encode chưa chứng
  minh denoiser học được; x0 tái dựng từ chính ảnh marked còn mang đường
  truyền trực tiếp từ input, không dùng nó làm bằng chứng radioactivity.

Vấn đề cần giải là tín hiệu lặp lại trong **phân phối dữ liệu mà learner có
khả năng học**, không chỉ residual làm CNN reader phản ứng trên từng ảnh.

## 2. Research: phần nào áp dụng được, phần nào không

| Công trình gốc | Kết quả/cơ chế hữu ích | Giới hạn khi áp dụng ở đây |
|---|---|---|
| [SERUM, ICLR 2026, Appendix I](https://arxiv.org/html/2603.13396v1) | Dấu từ initial noise; SD1.4 LoRA rank4 đạt 52,30% TPR@1%FPR trong thiết lập tác giả | 5.000 ảnh SD2.1, batch8, lr1e-5; chọn checkpoint19.500 theo validation MSE. Khác 2.048 ảnh/5.000 bước của ta. Không lấy headline làm bằng chứng đã tái lập |
| [HMark, §4 và Appendix B/C](https://arxiv.org/html/2512.00094v1) | Dấu h-space có ý nghĩa phân phối; gợi ý giảm payload nếu kênh học yếu | Attacker là pixel DDPM CelebA-HQ256, LoRA rank8/16/32, lr1e-4. Accuracy98,57% không phải TPR@1%FPR; clean-model FPR trung bình6,07%, identity4%. Không chuyển nguyên số sang SD1.4 latent T2I |
| [CoprGuard, CVPR2025](https://arxiv.org/html/2503.11071v2) | Khai thác thống kê phổ ảnh truyền qua diffusion; HiNet/DWT và tăng khả năng sống qua autoencoder | Phải kiểm tra lại trên caption tự do/ordinary LoRA của ta. Một biến đổi phổ bất kỳ không tự có tính chất của CoprGuard |
| [DIAGNOSIS, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html) | Coherent image warping có thể được diffusion học; hữu ích làm positive control learnability | Bảo vệ dataset và học biến dạng chưa đồng nghĩa native teacher chất lượng cao |
| [Radioactive Watermarks, Appendix B và §5.2](https://arxiv.org/html/2506.23731v1) | Tách mất dấu qua encoding và qua học denoising; kiểm tra thêm thế hệ model | Tác giả ghi nhận DIAGNOSIS chuyển được dấu nhưng chất lượng ảnh bị ảnh hưởng mạnh trong thử nghiệm đó. Cần quality–transfer frontier, không chỉ tăng strength |
| [Statistical Signal Amplification, 2026, hồ sơ tác giả](https://ssu.scholarworks.kr/item/54ca0694-9861-4ced-85d0-e0af64599cba) | Abstract báo AUROC0,891 với rank32 khi tổng hợp100 ảnh; đáng kiểm tra tín hiệu yếu ở mức model | Mới xác minh abstract tại kho trường, chưa đọc được full text/chi tiết null models; chưa dùng số này làm tiêu chuẩn thiết kế hay bằng chứng hiệu quả |
| [RECOVER, ICML2026](https://proceedings.mlr.press/v306/wei26ah.html) | Dùng tính ổn định inversion–reconstruction để phát hiện ảnh từng được học | Đây là kiểm tra sử dụng dữ liệu, không phải tạo watermark di truyền. Chỉ là phép chẩn đoán bổ sung, không thay mục tiêu |

Các phương pháp giữ watermark khi fine-tune **cùng trọng số**, gắn watermark
LoRA vào LoRA khác, hoặc cần secret trigger trong caption không tự đáp ứng
threat model student độc lập học từ ảnh. Phải kiểm tra pipeline thực tế,
không lựa chọn paper chỉ theo từ “LoRA”, “robust”, “transferable”.

## 3. Audit mới đã chạy: có phải chỉ reader yếu?

Code: [transfer_readout_audit.py](../new_watermark/learning_channel/transfer_readout_audit.py).
Run: `lalt_readout_audit_20261020_235419Z`, đã hoàn tất, không retrain student.
47 tests pass trước triển khai.

Định trước split 64 cặp ảnh rank4 để fit và 64 prompt khác để đánh giá.
Kiểm tra cả rank4/16/32, noise namespace cũ và mới. Các model vẫn dùng cùng
một training seed; fresh generation noise không phải independent model.
Fit mean difference có covariance shrinkage90%, không tune theo test.
Ba nhóm đặc trưng: 8 bit logits; 129 RGB statistics gồm moments/layout/
radial log-power/Haar energy; kết hợp cả hai. Fit từ teacher hoặc student,
báo tất cả sáu phương án cùng contrast baseline.

| Readout / AUROC trên cùng64 prompt giữ riêng | r4 cũ | r4 noise mới | r16 noise mới | r32 noise mới |
|---|---:|---:|---:|---:|
| Contrast hiện tại | 0,7190 | 0,7061 | 0,7346 | 0,7668 |
| Fit bit logits từ student r4 | 0,7263 | 0,7161 | 0,7363 | 0,7722 |
| Fit RGB statistics từ student r4 | 0,5366 | 0,5635 | 0,5430 | 0,5266 |
| Fit kết hợp từ student r4 | 0,7236 | 0,7258 | 0,7373 | 0,7825 |

Kết quả chưa đủ mạnh. Ví dụ combined tại rank4 cũ phát hiện5/64 so với
baseline3/64, nhưng noise mới giảm còn3/64 so baseline4/64. Không gọi đó
là cải thiện bền vững. Việc chỉ đổi trọng số bit hoặc thêm thống kê đơn giản
chưa giải quyết vấn đề. RGB statistics yếu trên watermark cũ **không bác bỏ**
việc chủ động thiết kế một dấu phổ mới; nó chỉ nói không tìm được tín hiệu
mạnh trong feature/readout đã thử.

Chưa có model trained bằng wrong owner key, nên classifier clean/marked
chưa chứng minh owner specificity. Đây là development diagnosis, không
phải final test hay model-level FPR1%.

Toàn bộ42 hàng kết quả và fitted directions:
[summary.json](../server_review/lalt_readout_audit/lalt_readout_audit_20261020_235419Z/summary.json).

## 4. Các hướng cải thiện, theo thứ tự ưu tiên

### A. Đổi carrier thành tín hiệu phân phối nhất quán

Ưu tiên cao nhất. Thử thống kê hiệp phương sai giữa các latent channels
trong dải tần xác định; key thay đổi hướng thống kê đó. Bắt đầu known-owner
1-bit và opposite-key, trước khi tăng payload. Đổi carrier phải có reader
phù hợp mới, không ép giữ reader LALT.

Một prototype cụ thể là phép quay trực giao trong dải Fourier:

\[
P_B z=\mathcal F^{-1}(1_B\mathcal Fz),\quad
U_k=\exp(aA_k),\quad A_k^T=-A_k,
\]
\[
T_kz=(I-P_B)z+U_kP_Bz.
\]

Ngoài dải B giữ nguyên. P_B không chứa DC nên giữ mean từng channel.
U_k quay channels trong dải, bảo toàn tổng năng lượng latent và có inverse
T_k^{-1}=T_{-k}. Những tính chất này không bảo đảm ảnh đẹp: decoder không
bất biến với phép quay; vẫn phải đo LPIPS/CLIP/đuôi chất lượng.

Điểm đáng nghiên cứu là có mục tiêu denoiser xác định. Với forward noise
đẳng hướng và T trực giao cố định, prediction của phân phối đã biến đổi là:

\[
\epsilon_k(x_t,t,c)=T_k\epsilon_0(T_k^{-1}x_t,t,c).
\]

Đây là suy luận toán học cho phép biến đổi đã nêu, không phải kết luận
LoRA sẽ học được nó. Gaussian prior được giữ; cùng phép biến đổi phải áp
dụng cho cả conditional và unconditional prediction để giữ tương thích
với CFG. Khi so chất lượng ảnh ghép cặp, coupling noise phải được khai báo:
reference wrapped sampler ghép noise qua T, không giả vờ cùng seed luôn
tạo cùng nội dung.

**Hai giai đoạn riêng:** (1) screen carrier trên clean latent/ảnh cũ, đo
quality, sống qua RGB8+VAE, đúng key và opposite key; (2) nếu có signal phù
hợp, distill epsilon target trên nhiều timestep/prompt vào native owner
UNet, sau đó generate PNG bằng owner đã đóng băng và train ordinary student.
Teacher có thể dùng adapter khi học rồi merge vào một snapshot mới; adapter
này thuộc owner. Không sửa adapter/loss của attacker.

Owner distillation dùng prediction targets xuyên timestep thay vì đẩy
reader trên final image. Đo sai số so target và chất lượng full generation;
không suy từ denoising loss thấp ra watermark transfer. Nếu native teacher
không tái hiện được reference, xử lý owner stage trước khi chạy student.

Prototype: [spectral_transport.py](../new_watermark/learning_channel/spectral_transport.py).
Đã kiểm tra identity, inverse, energy/DC preservation, tính giao hoán với
linear noising và gradient angle. Đây là prototype nghiên cứu, chưa chứng
minh novelty và chưa có native checkpoint.

### B. Dấu bằng quan hệ nội dung–thuộc tính, học vào native teacher

Đây là hướng thay thế nếu coherent spectral carrier không có quality–signal
tradeoff tốt. Chọn trục có tác động nhất quán như ánh sáng/texture nhẹ;
key quy định quan hệ giữa nhóm nội dung và thuộc tính. Giữ marginal style
cân bằng để detector không chỉ đọc “ảnh ấm màu hơn”. Bắt buộc global-style,
opposite-key và no-key controls; prompt paraphrase dùng cùng quan hệ.
Không hash từng câu và không chèn trigger vào caption attacker.

Thiết kế cụ thể có sẵn trong
[đề xuất semantic](Semantic_Relation_Fingerprint_VI.md). Cần thực nghiệm
khả năng điều khiển trước, rồi native teacher, rồi independent ordinary
student; không bê kết quả pixel-DDPM HMark sang SD T2I.

### C. Tối ưu owner theo learner, nhưng phải có kiểm tra bằng learner thật

Unroll ngắn chỉ là gradient proposal. Nếu dùng lại, ensemble không chỉ
đổi A-initialization: cần clean-trained states ở nhiều thời điểm, rank/
seed/learning-rate khác nhau; giữ một learner family ngoài outer training.
Một vài ordinary development students đủ dài là vòng kiểm tra bắt buộc.
Không dùng final test để chọn carrier/strength/reader.

Lỗi cụ thể của repair hiện tại: quality loss vẫn kéo ảnh về clean ngay cả
khi trong budget, trong khi anchor watermark penalty cho phép BCE tăng0,05
trên minibatch. TPR validation đã giảm dù penalty cuối bằng0. Nếu quay lại
LALT, dùng quality penalty chỉ ngoài budget; neo score margins trên một
replay set training cố định; rollback/line-search khi preservation thất
bại, không để trôi32 bước mới phát hiện. Đây là sửa ràng buộc, không phải
bằng chứng sẽ chữa được learnability. Không ưu tiên chạy lại cùng proxy
chỉ với hệ số loss khác.

### D. Reader hiểu biến dạng do quá trình học và tổng hợp nhiều ảnh

Fit readout trên nhiều **training models** phát sinh từ owner development,
không chỉ augment JPEG/blur. Reserve nguyên model/seed cho đánh giá; include
clean students, wrong-owner students, thay phong cách/dataset không liên
quan. Audit vừa chạy cho thấy linear readout đơn giản chỉ giúp ít; không
ưu tiên tăng độ phức tạp CNN khi chưa có bank model để chống overfit.

Báo cả per-image TPR@1%FPR và aggregate statistic cho16/64/256 query.
Giảm nhiễu ảnh không loại bỏ thiên lệch giữa các model. Cần independent
null models để hiệu chuẩn model attribution, không bootstrap128 ảnh thành
128 null models. Với conformal rank p-value, ít nhất99 null models mới có
độ phân giải1%; việc có99 model chưa tự bảo đảm độ tin cậy thực nghiệm nếu
không exchangeable. Giai đoạn3 seeds chỉ là screening, không claim model
FPR1%.

### E. Giảm payload, kiểm soát signal budget và corruption

Known-owner detection1-bit trước, multi-bit/BCH sau. ECC sửa lỗi bit khi
đã có signal; không tạo ra signal bị student loại bỏ. Xét quality frontier
theo strength thay vì một mức duy nhất. Đánh giá RGB8, VAE posterior sample,
resize/crop và data dilution ở outer owner/evaluation; không đưa transform
bí mật vào student để làm kết quả đẹp.

Tăng ảnh/steps/rank của attacker không phải cách sửa fingerprint. Chúng là
trục robustness riêng, phải so cùng carrier trên cùng ordinary recipe và
quality. Không ép attacker học lâu hơn chỉ để đạt watermark TPR.

## 5. Protocol thực nghiệm tiếp theo

1. Audit readout đã hoàn tất; kết quả trên chưa đủ để chỉ đổi detector.
2. Spectral screen mới `lalt_spectral_screen_20261020_235941Z`:32 ảnh fit,
   64 ảnh validation khác,512 clean calibration; angles0,03/0,06/0,12/0,24.
   Positive/opposite đều đi qua RGB8 và VAE mean. Đo mean/p95 LPIPS,
   PSNR p05, CLIP drop. Đây chỉ là screen; posterior sample/crop và full
   generation native teacher vẫn phải kiểm tra sau.
3. Gate screen đặt trước: teacher TPR>=50%, AUC>=0,85, opposite detection
   <=5%, LPIPSmean<=0,08/p95<=0,12, PSNRp05>=25dB, CLIPdrop<=0,02. Chọn
   strength nhỏ nhất đạt, không chọn strength theo student test. Ngưỡng50%
   là gate sàng lọc cơ chế mới, không thay official90% gate của LALT và
   không được dùng tuyên bố thành công.
4. Nếu screen có ứng viên, kiểm tra posterior sample/augmentation rồi
   native owner distillation. Nếu fail tất cả, báo fail và chuyển semantic
   feasibility; không mở budget train chỉ để tìm một score cao.
5. Chỉ sau owner stage: ordinary rank4 seed0 với đúng2.048 ảnh/5.000 bước
   làm matched development baseline. Kết quả hứa hẹn phải xác nhận trên
   seeds mới, rank16/32 giữ ngoài thiết kế, dataset/prompt khác và dilution.
6. Báo mọi failure, chi phí và chất lượng. Không cần teacher vượt80% trên
   reader cũ để thử cơ chế mới; nhưng phải có new-reader signal/quality và
   đánh giá student thật để kết luận.

51 tests đã pass trên server trước spectral screen, gồm47 tests trước
và4 tests cho toán học của transport. Hiện **chưa có student cải tiến đã
được xác nhận**. Code readout, prototype và screen là các phần đã làm;
native owner training/ordinary student mới không được tự nhận là đã chạy.

## 6. Cập nhật screen và giới hạn quyết định

Screen đầu đã hoàn tất cả bốn mức. Ở angle0,24, LPIPSmean=0,01154,
PSNRp05=31,36dB, teacher AUROC=0,6597 và TPR=0/64. Tín hiệu paired mean
có tăng, nhưng chưa đủ để đọc từng ảnh tại ngưỡng1%. Không ứng viên nào
đạt gate. Fit32 và validation64 có0 caption trùng nhau, đã đối chiếu.

Vì quality còn cách giới hạn khá xa, đã mở rộng **development strength
screen** sang0,48/0,72/1,08, thay vì kết luận cả họ biến đổi phổ thất bại
từ mức quá nhỏ. Run `lalt_spectral_strength_20261021_000449Z` dùng lại
clean calibration có hash, không sinh lại512 ảnh. Đây là lựa chọn thích
nghi dựa trên screen trước: bộ validation này đã được dùng cho phát triển,
không được tái sử dụng làm final test. Gate signal/quality giữ nguyên.

Không dùng việc per-image gate fail để khẳng định model-level aggregation
không thể hoạt động. Tuy nhiên, paired mean dương ở teacher cũng không
chứng minh được student inheritance hay FPR ở mức model. Nếu quality và
per-image signal không có vùng giao nhau, phải đánh giá hướng aggregate
bằng null-model controls riêng, hoặc đổi sang semantic carrier; không âm
thầm hạ threshold để tạo một tỷ lệ detection cao.

Trạng thái và kết quả lấy bằng
`python server_review/scripts/check_lalt_rethink.py`.

### Kết quả strength screen hoàn tất lúc kiểm tra07:08 (UTC+7)

| Angle | Teacher AUROC | Teacher TPR@nominal1%FPR | LPIPS mean / p95 | PSNR p05 |
|---|---:|---:|---:|---:|
| 0,24 | 0,6597 | 0/64 | 0,0115 / 0,0225 | 31,36dB |
| 0,48 | 0,7991 | 7/64 | 0,0392 / 0,0748 | 25,48dB |
| 0,72 | 0,8823 | 11/64 | 0,0771 / 0,1376 | 21,84dB |
| 1,08 | 0,9229 | 26/64 | 0,1445 / 0,2440 | 18,07dB |

Không có mức nào qua gate. Mức0,48 còn giữ quality nhưng signal chưa đủ;
mức0,72 trở lên vi phạm tail quality/PSNR. Tăng cường độ đơn thuần của
**uniform spectral rotation này** chưa tạo vùng signal–quality đạt yêu cầu.
Không tuyên bố mọi spectral watermark thất bại, cũng không gọi AUROC0,923
là kết quả student: chưa có student nào học bộ ảnh mới.

[Kết quả đầy đủ](../server_review/lalt_spectral_strength/lalt_spectral_strength_20261021_000449Z/results.json).
Hai spectral screens đã hoàn tất, không còn job nghiên cứu mới của lượt
này đang chạy. Không phát sinh ordinary student từ các ứng viên fail.

### Quyết định sau các phép kiểm tra

Ưu tiên tiếp theo chuyển sang **B: quan hệ nội dung–thuộc tính trong native
teacher**, có positive/opposite/global-style controls. Lý do: reader tuyến
tính và coherent rotation đơn giản đều đã được đo, chưa đủ tốt để chi thêm
ngân sách ordinary training. Semantic feasibility phải chứng minh điều
khiển thuộc tính trên prompt/seed chưa dùng khi fit trước khi học native
weights; sau đó đo independent student. Điều này vẫn là hướng chưa được
chứng minh, không lấy tên “semantic” làm bảo đảm radioactivity.

Giữ spectral transform làm baseline cơ chế và reference denoiser target.
Nếu quay lại, cần thay generator/dải tần theo anisotropy của clean latent
hoặc phân bố ngân sách theo perceptual sensitivity trên design set, với
validation mới. Không lặp tiếp cùng phép quay chỉ tăng strength.

Đối với LALT hiện tại, không triển khai thêm loss sweep trên proxy4 bước.
Cần teacher mới hoặc tín hiệu mới đủ rõ rồi mới đầu tư ensemble learner
và reader học trên nhiều model. Chưa đạt mục tiêu cải thiện student;
những kết quả này giảm các giả thuyết còn lại và xác định bước thử tiếp theo.
