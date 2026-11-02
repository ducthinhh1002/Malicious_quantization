# Đọc lại BitMark, SERUM, HMark, SleeperMark để thiết kế fingerprint truyền qua ordinary LoRA

Ngày đối chiếu: **18/10/2026**. Đã thêm [package learning_channel](../new_watermark/learning_channel/README.md) và [launcher](../run_learning_channel_gate.sh). Pilot v1 đã chạy hết 2.000 bước và trượt teacher gate: checkpoint được chọn có TPR = FPR = 2,34375%; student khoa học chưa chạy. Smoke v1 từng pass và meta memory test đạt peak allocated 69,4 GiB, nhưng các kiểm tra này không chứng minh fingerprint mạnh. Nguồn: [verification v1](../server_review/lcf_deployment/verification.json), [deployment v1](../server_review/lcf_deployment/deployment.json).

Bản v2 bổ sung curriculum, paired loss, quality budget, reader giữ vị trí, gradient diagnostics và proxy validation nhiều episode/horizon. Xem [chi tiết sửa khả năng học](LCF_Trainability_Repair_VI.md) và [launcher chẩn đoán](../run_learning_channel_diagnostics.sh). 12 numerical/protocol tests đã pass trên local và server; triển khai riêng tại [deployment v2](../server_review/lcf_v2_deployment/deployment.json). **Các sửa đổi là giả thuyết cần kiểm chứng; chưa có bằng chứng teacher hoặc student mạnh hơn.**

Bản pilot triển khai dùng một design rank 4 và evaluation ranks 8/16, một training seed, code 8 bit. Meta-learning dùng gradient xuyên 4 ordinary updates rồi toàn bộ trajectory DDIM 8 bước; đây là proxy học ngắn, không phải gradient của 2.500 bước final training. Reader được freeze trong meta stage. Teacher-only baseline, clean-trained students và wrong-key controls được giữ. Các ablation LCF-V/LCF-D, native consolidation, nhiều training seeds và phân tích toàn diện kênh gradient vẫn là các phần mở rộng; không coi chúng đã hoàn thành.

## 1. Kết luận trước

**Nên thay cách thiết kế fingerprint, không tiếp tục tăng lực ép lên bộ đọc cũ.** Đề xuất lần này là học một bộ nhúng trong teacher cùng bộ đọc tương ứng, chứng minh teacher mạnh và ảnh tốt trước; sau đó tối ưu chính bộ nhúng theo khả năng dấu được một student học lại bằng loss denoising thông thường.

Cơ chế áp dụng trên mọi prompt, không cần phân loại prompt thành người, động vật, phong cảnh… và không cần prompt kích hoạt. Tuy nhiên, “cơ chế không giới hạn loại prompt” khác với “đã chứng minh mạnh trên mọi prompt”: vế sau phải đo trên tập nội dung đa dạng và prompt chưa thấy.

Bốn watermark được hỏi không phải bốn lời giải đã chứng minh cho cùng một bài toán. Ta cần phân biệt:

1. Đọc dấu từ **ảnh teacher vừa sinh**.
2. Đọc dấu sau nén, crop hoặc VAE round-trip.
3. Dấu còn sau khi người dùng fine-tune **chính trọng số đã nhúng**.
4. Dấu xuất hiện trong **student sạch chỉ học từ ảnh/caption**.

Target của chúng ta là bước 4. Thành công ở ba bước trước không tự suy ra bước này.

## 2. Chốt lại target, bỏ một ràng buộc đã tự thêm

Theo [target.md](../target.md), luồng cần kiểm tra là:

```text
Owner thiết kế fingerprint trong teacher
                    ↓
Teacher sinh RGB + caption thông thường
                    ↓
Student khởi tạo từ checkpoint sạch, học bằng ordinary LoRA/SFT
                    ↓
Student sinh ảnh mới từ noise mới và prompt chưa thấy
                    ↓
Owner đọc được đúng fingerprint, với false positive được kiểm soát
```

Student không nhận key, extractor, adapter của teacher, latent gốc hoặc watermark loss. Không đổi module LoRA của student chỉ để làm kết quả đẹp hơn. Khi dùng nhiều recipe để kiểm tra, mỗi recipe phải là một lựa chọn học thông thường, được ấn định trước.

**Điều cần sửa trong cách hiểu trước đây:** `target.md` không ghi bắt buộc mọi phần nhúng phải gộp vào trọng số UNet, không được có module phía teacher. Các proposal trước đã đặt thêm điều kiện này. Đó là một mục tiêu triển khai khó hơn, cần tách khỏi yêu cầu “không can thiệp cách học của attacker”.

Tài liệu này chọn hai mốc rõ ràng:

| Mốc | Artifact phía owner | Được phép kết luận |
|---|---|---|
| Mốc chính | Diffusion teacher + module fingerprint nhỏ, hoạt động trong quá trình sinh ảnh | Fingerprint của **teacher pipeline** truyền qua ordinary image training nếu thí nghiệm đạt |
| Mốc mở rộng | Fingerprint được tích hợp vào checkpoint native, đã reload và kiểm tra | Chỉ gọi là **native-weight fingerprint** sau khi kiểm tra lại cả teacher và student |

Module ở teacher không vi phạm ordinary LoRA của student. Nhưng nó cũng không tự trở thành một cập nhật trọng số có thể `merge`. Không được trình bày hai mốc như tương đương.

Mốc chính giả định owner kiểm soát pipeline sinh ảnh mà bên thu thập dữ liệu nhận được. Nếu attacker được cấp toàn bộ teacher và có thể tắt module trước khi sinh dữ liệu, thiết kế này chưa bảo vệ tình huống đó; phải thêm threat model và kiểm tra riêng.

“Bất kể attacker LoRA thế nào” là định hướng về khả năng khái quát, không phải bảo đảm tuyệt đối: learner không cập nhật, bỏ phần lớn ảnh của ta hoặc học một miền không liên quan có thể không tiếp nhận dấu. Claim cuối phải gắn với họ learner, lượng dữ liệu và mức utility đã đo.

## 3. Bốn cơ chế thực sự làm gì?

### 3.1. BitMark: thay phân phối bit ngay tại generator

Generator bitwise có xác suất cho từng bit. BitMark tăng logit của lựa chọn thuộc nhóm “green”, rồi đọc thiên lệch thống kê sau khi mã hóa ảnh trở lại BSQ. Trong code hiện dùng, nhóm là `01,10`, context width 2; đây là **thiên lệch tần suất**, không phải bit accuracy của một message. Cách nhúng có thể tác động lên lựa chọn sinh ảnh sẵn có, thay vì yêu cầu diffusion khớp một ảnh đích. [Repo chính thức](https://github.com/sprintml/BitMark), [wrapper đang dùng](../bitmark_lora/teacher.py).

Dạng minh họa:

\[
\ell'_b=\ell_b+\delta\,\mathbf 1[\text{bit tiếp theo tạo chuỗi green}].
\]

Paper báo Infinity → SD2.1 đạt **98,9% TPR@1% FPR**, nhưng đây là **full fine-tuning** trên 1.000 ảnh, không phải kết quả ordinary attention-only LoRA. Appendix G còn thử BiGR, một binary diffusion model; vì vậy không thể nhận novelty chỉ bằng câu “mang BitMark sang diffusion”. [Paper, §4.3 và Appendix G](https://arxiv.org/html/2506.21209v2).

**Bài học cho code của ta:** SD liên tục không có đúng giao diện bit-logit đó. Lấy bộ đọc BSQ của Infinity rồi ép SD tăng điểm là một bài toán nhúng khác. Detector vẫn có thể hữu ích để đo, nhưng không bảo đảm SD dễ tạo ra tín hiệu nó đòi hỏi.

### 3.2. SERUM: cùng một mẫu trong noise, bộ đọc học nhận ra hậu quả trong ảnh

Với mẫu cố định đã chuẩn hóa \(A'\), mã chính thức trộn noise mới của từng ảnh theo:

\[
\eta_k=\sqrt{1-\alpha}\,\eta+\sqrt{\alpha}\,A'_k.
\]

Generator được giữ nguyên; bộ đọc học nhận diện từ latent của ảnh đã sinh rồi được VAE encode. Pipeline này không yêu cầu UNet học khớp một bộ đọc ngoại lai. [Repo](https://github.com/Hubizon/SERUM), [bản mã đã vendoring](../bitmark_lora/vendor/serum/watermark.py), [teacher wrapper](../bitmark_lora/serum_teacher.py).

Trong paper, SD1.4 LoRA đạt **52,30% TPR@1% FPR**, thấp hơn nhiều mức phát hiện ảnh teacher. Protocol dùng 5.000 ảnh SD2.1, rank 4; checkpoint được chọn theo denoising validation loss ở **19.500 bước**. Đây không phải bằng chứng LoRA luôn truyền dấu gần 100%. [Paper, Table 3 và Appendix I](https://arxiv.org/html/2603.13396v1).

**Bài học cho ta:** một thành phần lặp lại theo owner có thể tạo dấu thống kê; đồng thời reader cần được thiết kế đúng miền dữ liệu. Tuy nhiên, teacher mạnh vẫn có thể tạo một tín hiệu ordinary LoRA ít học được.

### 3.3. HMark: học residual trong h-space và học bộ đọc cùng lúc

HMark học \(E_\phi(k)=\Delta h_k\), cộng vào UNet bottleneck ở timestep cuối. Cùng key dùng cùng residual; bộ đọc RGB được học bằng presence/bit losses, kèm MSE và LPIPS. Không có loss radioactivity trực tiếp.

Thiết lập: semantic DDPM → pixel DDPM CelebA-HQ 256; 100.000 ảnh học watermark, 10.000 ảnh protected, 9.000 bước downstream. Số **98,57% detection accuracy** và **95,07% bit accuracy** không phải TPR@1% FPR. Table 5 ghi FPR trên clean-trained control: identity **4%**, resize **14%**. Chưa phải bằng chứng general text-to-image SD. [Paper, §3–4, Table 5](https://arxiv.org/html/2512.00094v1).

**Bài học cho ta:** “semantic latent” không bắt buộc nghĩa là chọn thủ công vài nhóm nội dung. Có thể để bộ nhúng học cách tác động, thay vì định trước thuộc tính như tuổi hoặc ánh sáng. Chưa xác minh được checkpoint SD chính thức dùng ngay cho thiết lập của chúng ta.

### 3.4. SleeperMark: dấu được bảo vệ trong model mà attacker tiếp tục fine-tune

SleeperMark học secret residual và reader trước, sau đó hướng UNet tái tạo residual khi có trigger. Với prompt thường, output được ràng buộc gần clean; với trigger, mục tiêu latent thay đổi theo timestep. Threat model chính là attacker nhận **watermarked weights** rồi fine-tune tiếp. [Paper, §4](https://arxiv.org/html/2412.04852v2), [repo và checkpoint](https://github.com/taco-group/SleeperMark).

Suy ra cho target của ta: student sạch chỉ nhận ảnh từ prompt thường không đương nhiên có dữ liệu để học hành vi trigger. Kết quả robust fine-tuning của SleeperMark không giải quyết câu hỏi đó.

**Bài học có thể lấy:** tách giai đoạn tìm cơ chế nhúng–đọc đủ mạnh khỏi giai đoạn tích hợp vào generator. Không bê nguyên cơ chế trigger sang rồi coi như đã giải quyết image-mediated transfer.

### 3.5. Vì sao có cảm giác các paper “hoạt động rất dễ”?

Phần dễ nhìn thấy thường là inference bằng checkpoint đã học xong. Chi phí thiết kế carrier, huấn luyện reader và chọn protocol nằm trước đó. Một lệnh chạy tạo ảnh có watermark không có nghĩa từ đầu chỉ cần vài trăm bước là học được một cơ chế mới.

Trong các bản thử của ta, nhiều lần đồng thời thay generator, carrier, detector, objective và điều kiện triển khai. Vì vậy, thất bại không chỉ là thiếu step; còn có thể do ta yêu cầu một hệ sinh ảnh tạo ra tín hiệu không phù hợp với khả năng điều khiển của nó.

## 4. Kết quả đã chạy của chúng ta thực sự nói gì?

| Thí nghiệm local | Kết quả đã có | Giới hạn diễn giải |
|---|---|---|
| BitMark → SD2.1, attention LoRA | Teacher 100%; r4/r16/r32: 49,4% / 62,9% / 66,3% detection | Unadapted student đã 6,1%; thiếu clean-trained student |
| BitMark, thêm convolution vào LoRA | 74,5% / 88,6% / 92,1% | Là learner khác; không lấy nó thay cho ordinary attention-only LoRA |
| SERUM → SD1.4 | Teacher 100%; LoRA/DoRA khoảng 2,3–3,6%; full 83,6% | Rank/budget khác paper; thiếu clean-trained control; không phải reproduction khớp hoàn toàn |
| Native random-coordinate, d00 | Fit gain +0,0580; heldout +0,00288; teacher detected 1/128 | Gain không phải bit accuracy; chưa có bằng chứng teacher đủ mạnh |
| Native random-coordinate, d04 | Fit gain +0,0624; heldout −0,01531; teacher detected 1/128 | Heldout đảo dấu; student chưa chạy |

Nguồn local: [BitMark audit](../server_review/BITMARK_20261017_evidence.md), [SERUM audit](../server_review/SERUM_result_zip_20261017.json), [native decision](../server_review/native_transfer_final/decision.json), [native teacher metrics](../server_review/native_transfer_final/teacher_metrics.csv).

Ba điều chỉnh cần giữ:

- BitMark có tín hiệu đáng tiếp tục khảo sát, nhưng raw detection hiện tại chưa phải TPR được hiệu chỉnh đúng FPR trên student. Không đủ để kết luận dấu đã truyền bao nhiêu phần trăm một cách sạch về thống kê.
- SERUM local thất bại ở LoRA trong cấu hình đã đo. Đổi threshold sang clean unadapted student chỉ đưa các nhánh lên khoảng 2,4–3,9%, không giải quyết vấn đề. Không tiếp tục dùng teacher 100% để mô tả nhánh này là positive control LoRA mạnh.
- NTF thất bại **trước** kiểm tra truyền dấu. VAE-survival, training gain hoặc đọc tốt trên tập fit không thể thay kết quả heldout.

## 5. Những lựa chọn trước cần dừng lặp lại

Đây là chẩn đoán từ cấu trúc thí nghiệm, không phải các nguyên nhân đã được chứng minh riêng bằng ablation.

### 5.1. Reader cố định đòi tín hiệu ngoài khả năng tạo ảnh của teacher

Khi score tăng rất tốt trên một ảnh/latent tối ưu riêng, nhưng teacher sinh từ noise mới không tăng, ta mới chứng minh “có tồn tại ảnh làm reader hài lòng”. Chưa chứng minh có một biến đổi nhỏ, dùng chung và dễ học của generator tạo ra các ảnh đó.

Cách sửa: học bộ nhúng và reader cùng nhau, bắt buộc đánh giá trên ảnh sinh mới. Reader không được ghi nhớ prompt hoặc clean/marked file path; kiểm tra split và augmentation phải đối xứng.

### 5.2. Chọn hướng ngẫu nhiên rồi hy vọng nó có ý nghĩa

GroupNorm random direction hoặc key random trong CLIP có thể tạo thay đổi, nhưng không được đảm bảo có tín hiệu nhất quán xuyên nội dung. Pilot NTF vừa rồi là bằng chứng cần tránh tiếp tục chọn theo fit gain trên ít ảnh.

Cách sửa: dùng không gian nhúng có khả năng điều khiển rõ và học nó trên nhiều prompt. Không gọi một hướng là “semantic” chỉ vì nó nằm trong feature space.

### 5.3. Dùng sai kênh thay cho ordinary training

VAE encode–decode và re-noising kiểm tra sự sống sót của tín hiệu trong **biểu diễn ảnh**. Ordinary training còn có optimizer, tập tham số được cập nhật, số bước, caption và phân phối sinh mới. Đó là một kênh khác.

Cách sửa: coi các biến đổi ảnh là kiểm tra phụ; learner tạo ảnh mới mới là phép thử chính.

### 5.4. Vội tối ưu triển khai trước khi có watermark hoạt động

Ép native export ngay từ đầu khiến ta vừa phải tìm dấu vừa phải giải bài toán đưa dấu vào một tập trọng số hạn chế. Tách hai mốc ở §2 giúp xác định bottleneck chính xác hơn. Nếu chỉ mốc đầu thành công thì báo đúng mốc đầu.

## 6. Hướng đề xuất: Learning-Channel Fingerprint — tên làm việc LCF

**Giả thuyết:** dấu sẽ truyền tốt hơn nếu owner học một biến đổi dùng chung theo key, và tối ưu biến đổi đó theo tín hiệu còn lại sau ordinary learning, với bộ đọc cố định và đối chứng clean tương ứng.

Phần mới cần kiểm chứng là **cách chọn carrier theo kênh học**, không phải chỉ đổi tên h-space embedding hoặc thêm một loss.

### 6.1. Bộ nhúng dùng chung theo key, không gate theo loại prompt

Với teacher diffusion sạch \(G_\theta\), key \(k\), feature bottleneck \(h_t\):

\[
u_k=E_\phi(k),\qquad
h'_t=h_t+a_t\,\mathcal B(u_k).
\]

- \(E_\phi\) nhỏ, được học; \(\mathcal B\) ánh xạ ra tensor đúng shape của bottleneck.
- Cùng key dùng cùng carrier nội bộ. Không có residual tự do cho từng ảnh và không hash mỗi prompt thành một dấu khác nhau.
- \(a_t\) có biên độ giới hạn, chỉ hoạt động trong một cửa sổ timestep đã định trước.
- Cùng cơ chế chạy trên mọi prompt; không sửa caption, không thêm trigger.
- Các up-block của teacher biến carrier này thành tác động phụ thuộc nội dung. Không đòi cùng một pattern RGB xuất hiện nguyên vẹn trên mọi ảnh.

Ở pilot SD 512, có thể bắt đầu bằng bottleneck cuối UNet và một cửa sổ ngắn cuối sampling. Cần lấy shape thực tế từ model config/runtime; không hardcode kiến trúc của checkpoint khác.

Lựa chọn này **chưa bảo đảm** carrier có tính ngữ nghĩa, không bị nhìn thấy, hoặc truyền được. Những thuộc tính đó là kết quả phải đo, không phải hệ quả tự động của vị trí nhúng.

### 6.2. Học reader phù hợp với carrier

Học reader RGB \(R_\psi(x)\) với một presence head và một head đọc code ngắn. Pilot dùng code 8 bit để kiểm tra cơ chế; đó chưa phải dung lượng owner/key đủ lớn cho triển khai thực tế.

Reader có thể tổng hợp feature từ nhiều vùng ảnh; không giả định các vùng hoặc bit độc lập để tự suy ra FPR cực nhỏ. Threshold lấy từ clean calibration riêng, kiểm tra lại trên clean-trained students.

Huấn luyện dùng nhiều key để reader không chỉ học “ảnh của generator A”. Nhưng **mỗi dataset mô phỏng một owner phải giữ nguyên key qua mọi ảnh**. Đổi key ngẫu nhiên ở từng ảnh trong cùng dataset student sẽ đổi bài toán cần nghiên cứu.

Loss khởi đầu:

\[
L_{\rm teacher}=
L_{\rm presence}+\lambda_b L_{\rm code}
+\lambda_p\operatorname{LPIPS}(x_k,x_0)
+\lambda_x\|x_k-x_0\|_2^2
+\lambda_u L_{\rm utility}.
\]

\(x_k\) và \(x_0\) cùng prompt/noise; \(L_{\rm utility}\) kiểm soát prompt adherence và suy giảm diversity. Pixel/perceptual loss là ràng buộc giữ chất lượng, không phải đích fingerprint.

Cần clean negatives, code khác key, heldout prompt và key. Khi dùng augmentation, áp cách lấy mẫu giống nhau cho các lớp để tránh reader nhận ra pipeline preprocessing. Không tối ưu bằng cách đẩy score rất lớn trên một số ít ảnh rồi che phần còn lại bằng mean.

### 6.3. Tạo teacher mạnh trước, chưa dùng learner trong vòng tối ưu

Đây là baseline bắt buộc, gọi là **LCF-T**: chỉ học teacher embedding và reader.

Kiểm tra trên RGB đã lưu thật, reload lại, prompt/noise chưa thấy. Báo teacher image TPR, continuous key margin, clean FPR, quality distribution và wrong-key confusion. Không chọn hướng mới chỉ theo fit score.

Nếu baseline này không có trade-off tốt giữa dấu và chất lượng, dừng để sửa bộ nhúng/reader. Chưa có lý do chạy hàng chục student hoặc thêm meta-learning.

### 6.4. Cải tiến chính: tối ưu carrier theo ordinary learning

Giữ teacher backbone sạch cố định ở phiên bản đầu. Đóng băng reader sau giai đoạn 6.3 trong một vòng thiết kế. Chỉ cập nhật tham số bộ nhúng \(\phi\) bằng phản hồi learner.

Với recipe ordinary \(r\):

\[
\mathcal D_{\phi,k}=\{(G_{\theta,\phi,k}(c_i,z_i),c_i)\},
\qquad
w_{r,k}^{T}=\mathcal A_r(w_r^0,\mathcal D_{\phi,k}).
\]

\(\mathcal A_r\) giữ nguyên loss denoising, optimizer, lịch học và LoRA scope đã quy định. Song song có clean control:

\[
w_{r,0}^{T}=\mathcal A_r(w_r^0,\mathcal D_0).
\]

Dùng cùng initialization, lịch prompt/noise/batch trong cặp clean–marked để giảm phương sai. Cả hai đều học thật; unadapted checkpoint không thay thế clean-trained control.

Trên ảnh student sinh từ noise mới và prompt validation mới, định nghĩa key score \(s_k\) và:

\[
M_{r,k}=
\mathbb E\,s_k(G_{w_{r,k}^{T}}(c,z))
-\mathbb E\,s_k(G_{w_{r,0}^{T}}(c,z)),
\]

\[
C_{r,k}=
\mathbb E\left[
s_k(G_{w_{r,k}^{T}}(c,z))
-\max_{k'\ne k}s_{k'}(G_{w_{r,k}^{T}}(c,z))
\right].
\]

Với tập wrong keys cố định trước, loss outer-loop đề xuất:

\[
L_{\rm outer}(\phi)=L_{\rm teacher}
+\lambda_m\operatorname{LSE}_{r,k}\big([m-M_{r,k}]_+\big)
+\lambda_c\mathbb E_{r,k}[m_c-C_{r,k}]_+
+\lambda_s L_{\rm student\ utility}.
\]

`LSE` là smooth maximum: tập trung vào recipe/key còn yếu. Chuẩn hóa score bằng thống kê calibration đã freeze để margin không bị thay đổi chỉ do scale của reader. \(L_{\rm student\ utility}\) kiểm tra khả năng học ảnh/nội dung và chất lượng sinh của student, tránh “truyền dấu” bằng làm hỏng toàn bộ dữ liệu.

**Không phải thay loss của attacker:** update bên trong student vẫn chỉ là ordinary denoising training. Watermark loss ở vòng ngoài dùng để sửa **teacher embedding**. Khi đánh giá cuối, bỏ toàn bộ đồ thị meta-learning; chạy trainer ordinary như một tiến trình độc lập trên RGB đã lưu.

Đóng băng reader trong giai đoạn này giúp tránh lời giải giả: bộ đọc học nhận diện artifact của một student thay vì teacher tạo dấu dễ truyền hơn. Nếu sau này mở reader để học tiếp, phải là ablation riêng và phải kiểm tra lại specificity trên clean students.

### 6.5. Đo cơ chế: dấu có đi qua kênh cập nhật không?

Sơ đồ tuyến tính hóa cục bộ, **chỉ để phân tích**, là:

\[
\Delta x\simeq J_{h\to x}u_k,
\qquad
\Delta w_r\simeq J_{\mathcal A_r,\mathcal D}\,\Delta x,
\qquad
\Delta s\simeq J_{s\circ G,w}\,\Delta w_r.
\]

Teacher-only optimization chỉ kiểm tra đường \(u_k\to x\to s_k\). Đề xuất này muốn chọn hướng mạnh trên đường dài hơn:

```text
carrier → RGB → VAE/noising → ordinary optimizer → student weights
                                                    ↓
                           reader ← ảnh student sinh mới
```

Điều cần chứng minh là: cùng mức thay đổi ảnh teacher, carrier được chọn theo kênh học cho key margin ở student tốt hơn carrier chỉ được chọn theo teacher score.

Có thể đo thêm độ nhất quán của chênh gradient ordinary loss:

\[
\Delta g_i=g(x_{k,i},c_i)-g(x_{0,i},c_i),\qquad
\kappa=\frac{\|\mathbb E_i\Delta g_i\|^2}
{\mathbb E_i\|\Delta g_i\|^2+\varepsilon}.
\]

Đây là diagnostic; gradient nhất quán cũng có thể chỉ là một thay đổi màu hoặc phong cách chung. Không dùng \(\kappa\) một mình làm bằng chứng fingerprint. Cần đúng key, wrong-key separation và utility tương ứng. Với AdamW, vector gradient thô cũng không đồng nhất với update sau preconditioning; phải lưu cả chênh update thực tế nếu muốn giải thích optimizer.

### 6.6. Khả thi tính toán và các xấp xỉ phải ghi rõ

1. **Teacher:** nếu module chỉ hoạt động ở K bước cuối và backbone cố định, prefix không phụ thuộc \(\phi\). Cache prefix, rồi backprop qua toàn bộ K bước có module là gradient đúng đối với kiến trúc này. Đây không phải tuyên bố full-trajectory training của mọi trọng số UNet.
2. **Surrogate student:** thử differentiable unroll ngắn 4–8 updates trước. Nó là proxy cho học ngắn; chưa thay thế kết quả ordinary training dài. Phải mô phỏng đúng recipe, kể cả trạng thái AdamW và LoRA initialization; không gọi công thức một bước SGD là gradient chính xác của AdamW.
3. **Student generation trong outer-loop:** tham số đã học ảnh hưởng từ bước denoising đầu. Không được tùy tiện detach prefix ở student rồi gọi đó là full gradient; dùng checkpointing toàn trajectory ngắn, hoặc ghi rõ phần gradient bị xấp xỉ.
4. **RGB/VAE:** meta-training cần giữ đường gradient qua tensor ảnh và VAE encoder. Final validation phải dùng ảnh 8-bit được lưu/reload. STE qua quantization ảnh, nếu dùng, là xấp xỉ phải khai báo.
5. **Tăng ngân sách theo bằng chứng:** proxy phải dự đoán được thứ hạng trên các student dài hơn ở tập design trước khi dùng để chọn fingerprint. Không tăng hàng loạt step chỉ vì teacher score vẫn tăng.

## 7. Prior art gần nhất và giới hạn novelty

### 7.1. SIREN là related work bắt buộc, không chỉ bốn bài được hỏi

SIREN, IEEE S&P 2025, học image coating và extractor cho phát hiện dữ liệu bị dùng để personalization. Learnability loss dựa trên denoising loss của surrogate diffusion; surrogate được fine-tune từ ảnh sạch. Đây là tiền lệ trực tiếp cho việc làm dấu dễ được diffusion học. Nó nghiên cứu data coating, không tự xác nhận teacher-pipeline fingerprint của ta. [Paper, §4.3](https://tianweiz07.github.io/Papers/25-sp.pdf).

Vì vậy, chỉ thêm một loss “learnability” vào watermark chưa đủ làm contribution. Với đề xuất hiện tại, phải so sánh teacher-only, surrogate denoising proxy và phản hồi từ **ảnh sinh mới sau ordinary updates**, cùng ngân sách và chất lượng.

### 7.2. Mô phỏng learner trong lúc thiết kế cũng đã có tiền lệ

[T2S](https://arxiv.org/html/2606.11698v1) dùng simulated extraction để đưa phản hồi về watermark của classifier. [Antidistillation Fingerprinting](https://arxiv.org/html/2602.03812v2) nghiên cứu fingerprint chống distillation ở LLM. Vì vậy, generic bilevel/rehearsal không phải phát minh mới riêng của ta.

[WOUAF](https://wouaf.vercel.app/) đã nghiên cứu fingerprint qua weight modulation. Key → thay đổi trọng số hoặc joint embedding–reader cũng không đủ làm claim mới.

### 7.3. Contribution khả dĩ, nếu thí nghiệm ủng hộ

1. **Một fingerprint teacher diffusion không trigger**, thiết kế để truyền qua RGB → independently initialized ordinary LoRA, với hiệu quả ở learner chưa dùng trong thiết kế.
2. **Phân tích kênh truyền dấu qua học:** chỉ ra vì sao những carrier teacher đọc rất tốt vẫn mất sau ordinary updates, và carrier được chọn theo phản hồi học khắc phục điểm đó ở cùng quality budget.
3. **Key-specific transfer**, tách khỏi generic SFT drift bằng clean-trained controls, wrong-key controls và kiểm tra capacity/attribution.

Nếu teacher-only baseline đã tốt ngang phiên bản learning-channel, không được nhận contribution cho outer-loop. Nếu chỉ cải thiện trên đúng recipe đã mô phỏng, claim phải giới hạn ở recipe đó. Hiện chưa có căn cứ gọi phương pháp là “đầu tiên” hoặc đủ novelty cho một venue cụ thể.

## 8. Kế hoạch thí nghiệm: ít nhánh nhưng trả lời đúng câu hỏi

### 8.1. Giai đoạn A — kiểm tra teacher và reader

Một cấu hình thiết kế ban đầu, **không phải cấu hình đã chứng minh hội tụ**:

- SD1.4 512 để dùng lại pipeline; một backbone, một scheduler, một resolution trước.
- 2.048 prompt design, 256 validation, 1.000 clean calibration, 500 test; nhóm ảnh và caption gần trùng phải cùng split.
- Code 8 bit; nhiều key trên tập design, key giữ lại cho test; không dùng một key duy nhất để chứng minh attribution.
- Huấn luyện embedding–reader và kiểm tra learning curve. Smoke 32–64 ảnh chỉ kiểm tra code, không dùng để kết luận khoa học.
- So ảnh clean/marked cùng seed; báo LPIPS mean/p95, PSNR mean/p05, prompt adherence và diversity. Nếu dùng FID/KID phải ghi sample count, không coi FID từ vài chục ảnh là kết luận.

Gate vận hành nên ấn định trước, chẳng hạn teacher TPR ≥90% ở nominal image FPR 1%, LPIPS mean ≤0,10 và không suy giảm rõ prompt adherence/diversity. Đây là mục tiêu engineering ban đầu, không phải ngưỡng chuẩn của literature. Báo cả CI và tail quality; không nới gate sau khi xem test rồi gọi thành công.

Nếu bài toán cuối chọn model-level detection thay vì per-image, có thể thiết kế gate khác **từ đầu**. Không đổi endpoint sau khi thấy per-image fail.

### 8.2. Giai đoạn B — ordinary LoRA là phép thử truyền dấu

Trước meta-training lớn, chạy một probe có matched controls:

| Dataset của student | Mục đích |
|---|---|
| Ảnh clean teacher | Đo generic fine-tuning drift và false positives |
| Ảnh LCF-T, key A | Kiểm tra baseline teacher-only có truyền dấu không |
| Ảnh LCF-T, key B | Phân biệt đúng owner, không chỉ “ảnh bị sửa” |

Pilot: cùng 1.000 prompt, ordinary attention LoRA r4, 2.500 updates; đọc learning curve tại các mốc đã định trước. Con số này chỉ để phát hiện tín hiệu ban đầu. Thiếu hiệu ứng ở pilot không chứng minh không thể truyền khi học lâu hơn; chỉ tăng budget nếu utility/learning curve cho thấy giả thuyết đó đáng kiểm tra.

### 8.3. Giai đoạn C — kiểm tra phần cải tiến có giá trị riêng

| Nhánh | Thay đổi so với baseline | Câu hỏi |
|---|---|---|
| LCF-T | Teacher embedding + reader | Học bộ nhúng–đọc có đủ chưa? |
| LCF-V | Thêm VAE/channel augmentation | Chỉ sống qua biểu diễn ảnh đã đủ chưa? |
| LCF-D | Thêm surrogate denoising learnability | Proxy rẻ có giải quyết được không? |
| LCF-L | Outer-loop theo student generation, reader freeze | Phản hồi học thật có thêm giá trị không? |

Giữ capacity, dữ liệu thiết kế, key và quality budgets tương ứng. Báo thêm ngân sách GPU; nếu LCF-L tốn nhiều hơn thì cần baseline được cấp ngân sách tương đương. Không bật cả bốn nhánh từ đầu khi teacher chưa đạt.

Learner design có thể r4/r16; evaluation giữ lại rank khác, seed khác, caption được tạo lại và backbone khác. Rank/scope chưa thấy không được lựa chọn lại sau khi nhìn test. Conv-LoRA và full SFT là robustness conditions bổ sung, không thay cho attention-LoRA chính.

Cuối cùng dùng ít nhất ba training seeds để bắt đầu đo biến động, nhiều owner keys và prompt distribution khác. Ba seeds chỉ là khởi đầu, chưa đủ tự suy ra FPR model-level 1% đáng tin cậy.

### 8.4. Thống kê phải phân biệt cấp ảnh và cấp model

- **Ảnh:** teacher/student TPR, FPR, score distribution và bit accuracy nếu có payload thật. Threshold freeze từ calibration tách biệt; báo false positives trên matched clean-trained students.
- **Model:** score tổng hợp từ số query N cố định trước, ví dụ N=100; cần null models độc lập hoặc một kiểm định có giả định phù hợp được xác nhận. Không chia 1.000 ảnh của một clean model thành nhiều nhóm rồi coi như hàng chục clean models độc lập.
- **Key:** confusion matrix đúng/sai key; sửa multiple testing khi thử nhiều keys. 8-bit pilot không phải bảo đảm an toàn khóa hay capacity nhiều owner.
- **Uncertainty:** bootstrap ảnh chỉ đo biến động ảnh với model cố định; không thay thế biến động training seed/model. Báo hai mức riêng.
- **Selection:** dùng design/validation để chọn method, utility để chọn checkpoint theo protocol cố định. Final test không chọn teacher, strength, key, rank hoặc threshold.

Một model tạo hàng nghìn ảnh không có chất lượng vẫn không phải thành công. Cần đồng thời có utility của student và key-specific transfer.

## 9. Các thành phần triển khai

Giữ code các hướng cũ để tái lập. Nên thêm package riêng để tránh vô tình dùng lại detector, checkpoint hoặc cache của thử nghiệm trước.

| File | Trách nhiệm |
|---|---|
| `new_watermark/learning_channel/carrier.py` | Key encoder, bounded h-space residual, timestep window, kiểm tra shape/CFG |
| `new_watermark/learning_channel/reader.py` | Presence/code heads, key scoring, serialization và freeze |
| `new_watermark/learning_channel/train.py` | Clean/marked paired generation, joint embedding–reader training và owner outer-loop |
| `new_watermark/learning_channel/learner.py` | Owner-side functional unroll của ordinary recipe; không thay trainer đánh giá |
| `new_watermark/learning_channel/objective.py` | Paired clean margin, wrong-key margin, utility constraints |
| `new_watermark/learning_channel/evaluate.py` | Threshold, independent controls, image/model/key metrics |
| `new_watermark/learning_channel/runtime.py` | Pinned assets, cache prefix, tách output nặng/nhẹ |
| `new_watermark/learning_channel/run.py` | Stage isolation, manifest freeze, gate và ordinary evaluation |
| `run_learning_channel_gate.sh` | Launcher; profile smoke/pilot/full; lưu provenance |

Các phần có thể tái sử dụng:

- [bitmark_lora/student.py](../bitmark_lora/student.py): orchestration ordinary student, nhưng audit lại clean controls, checkpoint selection và artifact provenance.
- [new_watermark/sampling.py](../new_watermark/sampling.py): ý tưởng sampling differentiable; không mặc định chế độ detach/grad-tail hiện tại đúng cho learner outer-loop.
- Split, hashing, JSON/CSV reporting và folder organization đã có. Kết quả nhẹ trong `output_attack`, ảnh trong `output_image`, checkpoint/cache trong `output_artifacts`.

Các kiểm tra cần có khi triển khai:

1. Carrier amplitude 0 tái tạo clean pipeline trong sai số xác định; module không làm lệch scheduler/CFG.
2. Gradient từ reader về carrier khác 0; finite-difference check trên bài nhỏ. Kiểm tra tiếp qua ordinary update trước khi chạy GPU dài.
3. Functional inner-loop khớp trainer ordinary trên một batch/ít bước, cùng RNG và optimizer state. Test này kiểm tra tính trung thực của proxy.
4. Không đưa teacher key, residual, reader hoặc optimizer state vào final student.
5. Same prompt/noise không tràn qua design–calibration–test; key và cache hash nằm trong manifest.
6. Reader cố định vẫn đọc RGB đã save/reload; chưa đánh giá bằng tensor trước lượng tử hóa rồi gọi là kết quả thực.
7. Nếu có native consolidation, reload checkpoint bằng pipeline không module và chạy lại **toàn bộ phép thử truyền dấu**. Không chỉ so weight diff hoặc teacher score.

## 10. Quyết định đề xuất

Giữ [Semantic Relation Fingerprint](Semantic_Relation_Fingerprint_VI.md) như một hướng giới hạn miền, không chọn làm phương án chính cho yêu cầu phủ prompt rộng hiện tại.

Ưu tiên **học carrier dùng chung trong teacher và reader phù hợp → kiểm tra ordinary student → mới tối ưu theo kênh học**. Đây là thay đổi có căn cứ hơn việc tăng margin, tăng step hoặc đổi ngẫu nhiên feature direction.

Chưa có bằng chứng LCF sẽ đạt kết quả tốt. Giá trị của kế hoạch này là mỗi giai đoạn tách được một câu hỏi: teacher có nhúng được không; student có học được không; learning-channel objective có cải thiện riêng không; và sự cải thiện có khái quát sang learner chưa thấy không.
