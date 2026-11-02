# Đổi carrier sau NTF: fingerprint bằng quan hệ ngữ nghĩa có key

Ngày: 18/10/2026. **Đề xuất gốc.** Cập nhật 21/10: đã triển khai pilot và bắt
đầu smoke GPU; xem [bản triển khai và protocol đang chạy](Semantic_Relation_Implementation_VI.md).
Chưa chứng minh khả năng truyền dấu hay novelty. Các con số dự kiến bên dưới là
thiết kế gốc; cấu hình đóng băng của run mới được mô tả trong tài liệu triển khai.

## 1. Quyết định dựa trên kết quả thật

Tạm dừng tìm thêm hướng GroupNorm ngẫu nhiên. Thử một carrier có tác động thị giác đã định nghĩa trước, học trực tiếp trong UNet teacher, rồi đo khả năng truyền sang ordinary student.

Main pilot NTF mới nhất không chạy student: cả hai teacher được shortlist đều fail. Nguồn: [decision](../server_review/native_transfer_final/decision.json), [teacher metrics](../server_review/native_transfer_final/teacher_metrics.csv), [d00 detail](../server_review/native_transfer_final/gate_detail_d00_a00.json), [d04 detail](../server_review/native_transfer_final/gate_detail_d04_a00.json).

| Hướng | Gain trên tập fit | Gain trên heldout | Teacher detected | Mean LPIPS | PSNR p05 |
|---|---:|---:|---:|---:|---:|
| d00_a00 | +0,05802 | +0,00288 | 1/128 | 0,0604 | 19,49 dB |
| d04_a00 | +0,06236 | -0,01531 | 1/128 | 0,0648 | 17,75 dB |

Gain là chênh mean detector score giữa marked và clean, không phải bit accuracy; các detector khác nhau nên không dùng gain để so tuyệt đối giữa mọi run. Hướng thứ hai còn đảo dấu trên heldout. Chưa có paired CI heldout trong báo cáo này, nên không tuyên bố gain nhỏ của hướng thứ nhất có ý nghĩa thống kê. Mean quality cũng che một đuôi ảnh thay đổi khá lớn.

Hai điểm triển khai cần nhìn thẳng:

- `native_basis.py` tạo Gaussian directions trực giao trên GroupNorm. Đây chưa phải những hướng có hành vi thị giác nhất quán đã được học.
- `native_transfer.py` fit readout từ cùng 32 response images dùng để tính gain/LCB sàng lọc. Bootstrap đó cố định readout, không tính bất định do fit và selection; nó không phải bằng chứng out-of-sample. Vòng heldout sau đó làm đúng nhiệm vụ bác bỏ ứng viên.

Vì vậy, kết quả bác bỏ **coordinate pilot cụ thể**, chưa bác bỏ toàn bộ ý tưởng native fingerprint hoặc learning-aware design. Nó cũng chưa kiểm tra được phần truyền qua student mà proposal muốn giải quyết.

Gate teacher TPR>=80% là quá chuyên biệt cho watermark từng ảnh nếu đích cuối là fingerprint model. Nhưng không được chỉ hạ gate rồi gọi run cũ thành công: ở đây continuous gain cũng không khái quát tốt.

## 2. Ý tưởng chọn lần này

**Dấu nằm trong một quy luật nhẹ giữa nội dung ảnh và cách thể hiện, thay vì một pattern pixel cố định hoặc một hướng feature tùy ý.**

Ví dụ minh họa, chưa chốt làm carrier: với các prompt không quy định ánh sáng, key A làm một số nhóm chủ thể hơi thiên về ánh sáng khuếch tán, các nhóm khác hơi thiên về ánh sáng có hướng. Key B đảo quy luật. Mỗi nhóm vẫn có nhiều biến thể và nội dung vẫn tuân theo prompt. Chỉ dùng nhóm nội dung đủ rộng, không dùng secret trigger hoặc hash riêng từng câu.

Điều cần đọc là **mối liên hệ nội dung–thuộc tính theo key**, không chỉ ảnh sáng hơn hay đổi tông màu nói chung. Hai model có thể cùng mức thay đổi phong cách trung bình nhưng mang quy luật khác nhau.

Đây là giả thuyết về learnability: một quan hệ có độ phức tạp thấp, lặp lại trên nhiều ảnh, có thể được conditional denoising học dễ hơn các residual riêng lẻ. Không có định lý rằng ordinary LoRA nhất định học nó, và việc một thuộc tính mang tính ngữ nghĩa không bảo đảm nó đủ mạnh hoặc kín đáo.

## 3. Căn cứ và phần chưa mới

| Nguồn gốc | Điều có thể dùng | Điều không suy ra được |
|---|---|---|
| [Concept Sliders, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05660.pdf), mục 3 | Học thay đổi tham số theo chênh noise prediction giữa hai mô tả thuộc tính đối lập | Slider không phải fingerprint; paper không chứng minh independent-student inheritance |
| [SliderSpace, ICCV 2025](https://sliderspace.baulab.info/) | Có thể học nhiều hướng thị giác nhất quán thay vì chọn random weight directions | Khám phá nhiều hướng không tự giải quyết key specificity hoặc radioactivity |
| [HMark](https://arxiv.org/html/2512.00094v1), mục 3–4 | Có bằng chứng semantic-latent watermark truyền qua học lại trong thiết lập của tác giả | Thí nghiệm downstream pixel DDPM/CelebA-HQ khác SD text-to-image native teacher của ta |
| [LLM Fingerprinting via Semantically Conditioned Watermarks](https://arxiv.org/html/2505.16723v3) | Semantic conditioning và embedding dấu thống kê vào model đã có tiền lệ | Robustness khi sửa cùng model không đồng nghĩa truyền từ ảnh sang student độc lập |
| [Antidistillation Fingerprinting](https://arxiv.org/html/2602.03812v2) | Thiết kế fingerprint dựa trên phản ứng học của student có tiền lệ ở LLM | Không tự chuyển kết quả hoặc proxy gradient đó sang diffusion |

Concept Sliders gốc có thí nghiệm bật adapter ở một đoạn sampling, và chủ yếu gắn module ngoài cross-attention. Không bê nguyên inference recipe đó sang đây: teacher cuối phải có **một bộ trọng số cố định**, chạy sampler thường ở mọi timestep. Chuyển được hướng điều khiển thành native behavior đáp ứng điều kiện này vẫn là việc cần kiểm tra.

Phần có thể tạo đóng góp, nếu thành công: **keyed conditional relationships truyền qua image-only ordinary training tốt hơn global style bias và random-feature marks ở cùng chất lượng, trong native diffusion teacher**. Chưa có cơ sở gọi “đầu tiên”; ghép Concept Sliders với detector chưa đủ novelty.

## 4. Cơ chế cụ thể

### 4.1 Chọn một carrier dễ đo trước

Khởi đầu chỉ một trục thị giác và bốn nhóm nội dung rộng. Ưu tiên đặc tính nhiếp ảnh như ánh sáng/độ sâu trường ảnh hơn sửa danh tính, số vật thể hay nội dung bắt buộc trong prompt. Tránh carrier chỉ làm mờ ảnh hoặc đổi toàn bộ cân bằng trắng.

Trước huấn luyện teacher, dùng clean model với các mô tả thuộc tính đối lập để kiểm tra trên design prompts: biến đổi có đo được bằng một image probe độc lập, ổn định qua seed và còn giữ nội dung không? Phần này chỉ là kiểm tra khả năng điều khiển, không phải teacher native hoặc bằng chứng truyền dấu.

Chọn nhóm, carrier, probe và compatibility rule trên design data rồi đóng băng. Khi prompt đã yêu cầu thuộc tính đối lập, rule đưa target modification về 0 và dùng preservation; báo coverage của rule. Không bỏ ảnh khó ở lúc đánh giá. Nhóm khó phân loại cũng được ghi riêng. Việc dấu chỉ hoạt động trong miền phù hợp phải được công bố.

### 4.2 Key mã hóa quan hệ, không mã hóa tên từng prompt

Gọi `g(c)` là nhóm ngữ nghĩa của prompt, và `k_g` là dấu +/- của key cho nhóm đó. Pilot có bốn nhóm cân bằng và hai dấu mỗi phía. Có thể dùng soft membership để tránh ranh giới cứng, nhưng không cần học một mạng gating lớn ngay.

Không dùng hash toàn bộ câu: hai câu paraphrase cùng ý nên mang cùng quy luật. Bốn nhóm với hai dấu mỗi phía chỉ có sáu mã cân bằng; đây là kiểm tra cơ chế, **không phải key space bảo mật**. Multi-owner cần nhiều trục độc lập hơn và phải đo interference/collision.

### 4.3 Supervision trực tiếp cho denoiser teacher

Với clean UNet đóng băng `epsilon_0`, cùng noisy latent `z_t`, cùng timestep và cùng nội dung, tạo:

\[
r_t(z_t,c)=\epsilon_0(z_t,t,c^+)-\epsilon_0(z_t,t,c^-),
\]

trong đó `c+`/`c-` khác ở đúng thuộc tính carrier. Đây là chênh prediction của hai điều kiện có nghĩa, thay cho residual được tối ưu riêng để thắng hard-bit detector.

Phải tính đúng classifier-free guidance. Đặt:

\[
\epsilon_\theta^\gamma=\epsilon_\theta(z_t,t,\varnothing)
+\gamma[\epsilon_\theta(z_t,t,c)-\epsilon_\theta(z_t,t,\varnothing)].
\]

Target owner:

\[
\epsilon_{\rm target}^{\gamma}
=\epsilon_0^{\gamma}+\beta w(t)b_k(c)r_t(z_t,c),
\qquad b_k(c)=k_{g(c)}\,\mathrm{compatibility}(c).
\]

\[
L_{\rm owner}=\mathbb E\|\epsilon_{\theta_k}^{\gamma}
-\operatorname{sg}(\epsilon_{\rm target}^{\gamma})\|^2
+\lambda_u L_{\rm unconditional}+\lambda_p L_{\rm preserve}.
\]

`L_unconditional` neo unconditional prediction về clean; `L_preserve` dùng các điều kiện/thuộc tính không nhắm đến. Giữ nội dung và chất lượng còn phải kiểm tra trên full generations. Không neo toàn bộ conditional prediction quá mạnh đến mức triệt tiêu target.

Các lựa chọn kỹ thuật bắt buộc:

- Chỉ cập nhật teacher. Có thể dùng owner adapter rồi merge vào UNet; student trainer nguyên trạng.
- `b_k(c)` và mô tả `c+`, `c-` chỉ xây target lúc owner training. Khi phát hành, UNet đã học quy luật; inference không chạy classifier g(c), không sửa prompt, không chọn adapter hay strength theo key/timestep.
- `w(t)` là profile target cố định ở lúc train; bắt đầu đơn giản, không tìm hàng chục profile trước khi có signal.
- Dùng latent states từ clean và các rollout teacher hiện hành; teacher/reference cùng nhìn đúng một `z_t`. Theo dõi gap giữa cached states và rollout thật.
- Không chia loss cho residual MSE rất nhỏ của từng ảnh. Log residual RMS, absolute loss, gradient norm và validation behavior; kiểm tra chênh FP32/precision triển khai.
- Ghi rõ convention epsilon/v-prediction của backbone; không dùng epsilon formula trực tiếp cho v-prediction.

Đây là mục tiêu score steering lấy cảm hứng từ Concept Sliders, không phải công thức mới đã được chứng minh. Không tuyên bố một linear combination của noisy-time scores là score chính xác của một phân phối đích nhất quán ở mọi timestep. Full native generation mới quyết định việc nhúng thành công.

Khác với run distribution trước: ta không fit random feature rồi dùng reward-weighted noise targets; ta dùng prediction difference cho một hành vi thị giác cụ thể và giám sát trực tiếp cách denoiser phải đổi.

### 4.4 Student vẫn học bình thường

```text
clean diffusion -> owner training -> checkpoint native theta_k
                                      |
                       prompt gốc + Gaussian noise bình thường
                                      |
                            toàn bộ RGB + caption
                                      |
                  ordinary LoRA/SFT từ clean base độc lập
                                      |
                    ảnh mới trên prompt/seed chưa dùng
```

Student không nhận key, `r_t`, augmented prompt, teacher latent, score, weight sampling hoặc teacher adapter. Không chọn ảnh high-score. Module/rank/loss/checkpoint rule của student không được điều chỉnh theo carrier. Dùng learner khác nhau ở phần đánh giá, không dùng việc sửa learner để cứu kết quả.

### 4.5 Đọc dấu ở mức quan hệ

Với image probe thuộc tính `f(x)`, nuisance mean `mu_0(c)` fit từ clean design và scale `sigma_0,g` đóng băng:

\[
C_k(M)=\frac1G\sum_{g=1}^G k_g
\frac1{n_g}\sum_{i:g(c_i)=g}
\frac{f(x_i)-\mu_0(c_i)}{\sigma_{0,g}},\quad x_i\sim M(c_i).
\]

Dùng số query cố định và nhóm cân bằng để thay đổi phân bố prompt không tự tạo signal. `mu_0` không fit lại trên suspected model vì có thể trừ mất dấu. Detector biết prompt do owner tự query; đây chưa phải image-only detector trên ảnh không rõ nguồn prompt.

Đo cả image score/TPR và `C_k`. Primary endpoint của hướng này là fingerprint model với query budget cố định, không ép mọi ảnh vượt threshold. Nếu `T_k` tăng `C_k` còn `T_-k` giảm, rồi hai student tái hiện cùng chiều, bằng chứng cụ thể hơn một mean shift chung.

Không coi balancing các dấu là chứng minh giữ nguyên marginal style: độ đáp ứng khác nhau giữa nhóm có thể vẫn làm đổi marginal. Phải đo histogram/feature moments thực tế và có đối chứng.

## 5. Pilot tối thiểu để bác bỏ ý tưởng sớm

Đây là ngân sách đề xuất, không phải dự báo thời gian hoặc bảo đảm hội tụ.

### A. Kiểm tra native embedding

Một carrier, bốn nhóm, hai key đối dấu; 512 design prompts, tối đa 500 owner updates. Trước run chốt hai strength và cách chọn bằng design validation. Teacher validation dùng 256 prompt/seed mới, có paraphrase và tổ hợp nội dung chưa thấy.

Chỉ qua tiếp nếu quy luật có cùng chiều trên heldout và hình ảnh còn đạt các giới hạn quality đã đăng ký. Kiểm tra model export/reload cùng precision thật. Nếu thất bại, chẩn đoán supervision/carrier, không mở thêm hàng chục hướng random.

### B. Kiểm tra inheritance thật

Mỗi nhánh 1.024 ảnh với caption gốc; ordinary attention-LoRA r4, cùng recipe có sẵn, cùng 2.500 updates và fixed final checkpoint. Bốn nhánh:

| Dữ liệu student nhận | Câu hỏi |
|---|---|
| Ảnh clean | Ordinary fine-tuning tự tạo score bao nhiêu? |
| Ảnh native teacher key k | Quan hệ có truyền cùng chiều không? |
| Ảnh native teacher key -k | Đảo key có đảo dấu hiệu ứng không? |
| Trộn 50/50 ảnh từ hai teacher trong từng nhóm nội dung, với routing độc lập từng mẫu | Mất quan hệ theo key nhưng vẫn có edit/style: detector còn báo nhầm không? |

Nhánh trộn là **đối chứng dữ liệu từ hai checkpoint**, không gọi nó là một native teacher riêng. Giữ caption đúng nội dung, không tráo caption giữa ảnh. Routing được định trước, không dựa vào detector; kiểm tra độ tương đồng marginal thay vì mặc định khớp hoàn toàn.

Primary evaluation: 256 fresh prompts/seeds, cùng số ảnh mỗi nhóm; đọc effect size, CI theo prompt cluster, quality và key reversal. Detector không fit trên final student. Không lấy bootstrap ảnh của một model pair làm độ bất định giữa training seeds.

Nếu có tín hiệu mới mở r16/full, initialization và seed chưa dùng, prompt domains khác, VAE/backbone khác, recaptioning và pha loãng teacher data. Recaptioning có thể biến thuộc tính ẩn thành điều kiện tường minh nên phải đo riêng; không hứa sống qua bước này.

### C. Tiêu chí dừng và quality

- Teacher mạnh nhưng student không hơn clean/mixed control: carrier chưa truyền trong budget đã đo.
- Key k và -k cùng làm tăng score, hoặc mixed control cũng tăng như k: nghi generic domain/style drift hoặc probe sai, không tuyên bố fingerprint.
- Chỉ nội dung đã học có tín hiệu: chưa đủ khái quát.
- Chỉ full fine-tune có tín hiệu: chưa đáp ứng mục tiêu ordinary LoRA.
- PSNR, LPIPS, prompt alignment, diversity và đuôi chất lượng đều được báo. Giữ các ngưỡng preservation hiện hành nếu còn claim paired preservation; thêm LPIPS p95, PSNR p05 và ảnh audit. Không hạ ngưỡng sau khi thấy kết quả.
- TPR@1%FPR trên ảnh không phải FPR@1% ở mức model. Với vài clean-trained models, chỉ báo hiệu ứng/thống kê điều kiện; chưa đủ hiệu chỉnh FPR tổng quát trên quần thể model. Không nhân số null models bằng cách bootstrap ảnh.

## 6. Những thay đổi code cần làm nếu triển khai

1. Thêm `semantic_carrier.py`: predefined concept groups, compatible attribute pair, key table, fixed probe và clean nuisance fit; lưu hash manifest. Tách tập fit probe khỏi tập chọn carrier và validation.
2. Thêm `semantic_teacher.py`: CFG-aware prediction-difference supervision, preservation, refresh rollout states, native export/reload. Một implementation dùng cho cả key k và -k.
3. Thêm `relation_evaluation.py`: per-group signed score, clean/mixed/key-reversal contrasts, cluster CI và quality tails; không báo model-level FPR khi chưa đủ calibration.
4. Tái dùng nguyên `bitmark_lora/student.py`; chỉ cấp image folders và caption. Audit trainer hash, input manifest và fixed checkpoints như trước.
5. Launcher riêng với stage A/B, kết quả nhẹ tách ảnh/checkpoint; giữ NTF cũ làm negative baseline. Chưa đưa continuous meta-optimization, LLM hay evolutionary search vào vòng pilot này.

## 7. Nếu hướng này cũng không qua được thì đổi bài toán thế nào?

Hai phương án cần phân biệt:

- **Vẫn giữ target native SD:** chuyển sang học h-space carrier và detector đồng thời rồi consolidate vào teacher. Đắt hơn, có prior HMark và vẫn có rủi ro mất dấu khi consolidation; không phải lựa chọn ưu tiên trước khi thử carrier có supervision rõ.
- **Đổi setting nghiên cứu:** thu hẹp sang pixel-space DDPM trong miền đơn như HMark để chứng minh cơ chế truyền trước. Đây là thay đổi phạm vi từ SD text-to-image, phải nói rõ và thống nhất; kết quả ở đó không thay thế kết quả target hiện tại.

Lựa chọn đề nghị hiện tại là **một pilot quan hệ ngữ nghĩa có key**, với native denoiser supervision và đối chứng đảo key/mất tương quan. Chỉ mở thành phương pháp lớn khi cả teacher lẫn independent student thể hiện đúng quy luật đó.
