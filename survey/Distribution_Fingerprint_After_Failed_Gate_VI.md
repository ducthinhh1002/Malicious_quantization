# Hướng thay thế sau teacher gate: fingerprint của phân phối sinh

Ngày 2026-10-17. Đây là đề xuất nghiên cứu và protocol kiểm chứng, **chưa triển khai hoặc chạy GPU**. Không thay mặc định hiện tại, không sửa thuật toán học của student.

## 1. Kết quả nào đang buộc chúng ta đổi hướng?

Nguồn thí nghiệm: `/root/logs/results/teacher_gate_improved/{proxy,rollout}/result/teacher_selection.json`, đã đọc ở lần kiểm tra server trước khi viết tài liệu này.

| Teacher, 250 updates | Phát hiện trên 128 ảnh design | Clean FPR | Paired PSNR |
|---|---:|---:|---:|
| Proxy | 6/128 = 4,69% | 1/128 = 0,78% | 20,31 dB |
| DDIM rollout, gradient 4 bước cuối | 9/128 = 7,03% | 1/128 = 0,78% | 20,30 dB |

Cả hai không đạt gate cũ TPR >=95%, PSNR >=25 dB. Chưa có student của hai teacher này. Không coi tăng 3 ảnh phát hiện là bằng chứng chắc chắn rollout tốt hơn. Một key, một seed và 128 ảnh không đủ kết luận phương pháp bất khả thi.

Code `teacher.py` đặt target = threshold + score_margin; `method.py` phạt bình phương thiếu hụt score từng ảnh. Như vậy đang yêu cầu một hướng CLIP cố định vừa dễ nhúng trên gần mọi nội dung, vừa không làm đổi ảnh, vừa truyền qua học lại. PCA/VAE ổn định không bảo đảm cả ba yêu cầu.

PSNR 20 dB chứng minh paired fidelity thấp theo protocol đã chọn; tự nó không chứng minh ảnh xấu hoặc mất ngữ nghĩa. Phải xem thêm ảnh, LPIPS, mức khớp prompt và phân phối đầu ra. Không bỏ gate cũ rồi gọi các run cũ thành công.

## 2. Những gì literature thực sự hỗ trợ

| Công trình gốc | Điều rút ra và giới hạn |
|---|---|
| [Are Watermarks for Diffusion Models Radioactive?](https://adam-dziedzic.com/static/assets/papers/RadioactiveDiffusion.pdf), WMARK/ICLR 2025 | Trong thiết lập của họ, có dấu qua được latent encoding nhưng mất sau học lại. VAE survival không đủ chứng minh inheritance. |
| [HMark](https://arxiv.org/html/2512.00094), mục 3–4 | Học encoder nhúng h-space và decoder ảnh; thử downstream DDPM trên CelebA-HQ. Không thể mang số đo đó sang SD text-to-image. |
| [SERUM](https://arxiv.org/html/2603.13396v1), bảng 3/phụ lục I | SD1.4 LoRA đạt 52,30% TPR@1%FPR trong protocol của tác giả. Đây là baseline liên quan trực tiếp; không phải bảo đảm cho teacher native của ta. |
| [On the Learnability of Watermarks for Language Models](https://arxiv.org/html/2312.04469) | Khả năng học dấu phụ thuộc thiết kế dấu và lượng dữ liệu. Không phải định lý về LoRA diffusion. |
| [Watermarking Makes Language Models Radioactive](https://arxiv.org/html/2402.14904) | Tổng hợp nhiều quan sát giúp đọc dấu yếu; phụ thuộc truy cập và dữ liệu. Không chuyển nguyên kiểm định token sang ảnh. |
| [LLM Fingerprinting via Semantically Conditioned Watermarks](https://arxiv.org/html/2505.16723v3) | Nhúng dấu thống kê theo miền ngữ nghĩa vào model. Độ bền khi sửa cùng model khác với truyền sang student độc lập. |
| [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/html/2305.13301), mục 4.2–4.3 | Reward-weighted denoising có tiền lệ; tác giả cũng ghi nhận DDPO mạnh hơn RWR trong các task họ thử. RWR không phải đóng góp mới hoặc optimizer tốt nhất đã được chứng minh. |

Không công trình nào ở đây bảo đảm dấu sống qua mọi LoRA/SFT. Student không học dữ liệu, dữ liệu bị pha loãng mạnh hoặc miền query không trùng đều có thể làm dấu biến mất.

## 3. Hướng chính: học thiên lệch có key trong phân phối teacher

Thay vì ép tất cả ảnh tới một điểm ở phía xa của feature space, owner làm một số biến thể tự nhiên xuất hiện nhỉnh hơn một chút, theo quy luật có key. Student học phân phối ảnh nên có cơ hội học cùng quy luật. Đây là giả thuyết cần kiểm chứng, không phải kết quả đã có.

Ví dụ minh họa, không phải thiết kế thực nghiệm: trong những lựa chọn ánh sáng và bố cục đều phù hợp prompt, model hơi ưu tiên một tổ hợp nhất quán. Không quy định mọi ảnh phải sáng hơn, không tạo một màu cố định dễ bị nhầm với domain adaptation.

### 3.1 Carrier phải nằm trong biểu diễn mà denoiser đọc được

Ứng viên đầu tiên là vector nhỏ 8–16 chiều lấy từ đặc trưng của UNet sạch, đóng băng:

1. Encode ảnh bằng VAE, thêm nhiễu tại vài mức log-SNR vừa phải.
2. Lấy đặc trưng mid-block, pool không gian, trung bình qua số ít nhiễu cố định.
3. Fit basis, chuẩn hóa có ridge và mô hình nuisance của prompt chỉ trên clean design-fit.
4. Đo độ ổn định của score khi thay seed nhiễu, VAE và denoising reconstruction. So với CLIP và latent low-frequency làm ablation.

Không gọi các hướng này là ownership subspace. Chúng chỉ là carrier ứng viên có thể đọc qua denoising. Đặc trưng được pooling vẫn có thể mang texture hoặc bias; phải đo.

Key chọn tổ hợp trơn, ít chiều trong basis đã đóng băng. Không hash toàn bộ prompt thành dấu +/- độc lập: quy luật kiểu đó dễ không khái quát sang prompt mới. Bắt đầu một key/model; nhiều key là các model thử nghiệm riêng. Không tự suy ra độ an toàn mật mã từ số bit của seed.

Đặt score hữu hạn `s_k(x,c) = tanh(w_k^T f_residual(x,c))`, thuộc [-1,1]. Bộ trích xuất, nuisance fit, key và dấu score phải đóng băng trước teacher training. Khi kiểm định, dùng prompt đã biết; cần một nhánh riêng nếu yêu cầu image-only detection.

### 3.2 Định nghĩa phân phối đích trước khi tối ưu

Với model sạch p0 và prompt c:

    q_k(x|c) = p0(x|c) exp(beta * s_k(x,c)) / Z_k(c).

Đây là exponential tilting quen thuộc, không claim phát minh. Beta nhỏ tạo dịch chuyển xác suất nhẹ trong các ảnh model vốn sinh được. Không có bước gradient trực tiếp tìm ảnh adversarial để đánh lừa CLIP.

Một đẳng thức hữu ích để kiểm tra thiết kế:

    d E_q[s] / d beta = Var_q(s) >= 0.

Đẳng thức đúng cho phân phối q lý tưởng, score cố định và beta hữu hạn; KHÔNG chứng minh teacher hữu hạn dữ liệu sẽ học được q hoặc student giữ dấu. Với beta nhỏ và score đã center dưới p0, KL(q||p0) xấp xỉ beta^2 Var_p0(s)/2. KL này không bảo đảm LPIPS, PSNR hay diversity.

### 3.3 Chỉ teacher nhận loss có trọng số

Tạo nhiều ảnh sạch theo cùng mỗi prompt. Trọng số chuẩn hóa trong từng prompt:

    w_i = exp(beta * s_i) / mean_j exp(beta * s_j).

Owner tối ưu:

    L_owner = E[w_i ||epsilon_theta(z_t,t,c) - epsilon||^2]
            + lambda E_clean ||epsilon_theta - epsilon_base||^2.

Đây là weighted denoising surrogate, không khẳng định bằng tối ưu exact likelihood của q. Cách encode posterior, phân phối timestep, CFG và sampler phải được ghi lại; mismatch có thể khiến teacher không thể hiện thiên lệch khi sampling.

Native teacher sau huấn luyện phải tự sinh dấu bằng noise Gaussian và sampler thông thường, không detector guidance, không lọc chọn ảnh ở inference. Owner có thể train adapter rồi merge, hoặc cập nhật một số native weights; việc này độc lập với student.

Student nhận **toàn bộ ảnh teacher trong tập đã định trước** cùng caption, dùng ordinary denoising MSE. Không chuyển w_i, key, score, detector hoặc teacher weights sang student. Không chọn các ảnh teacher score cao rồi gọi đó là output distribution native.

Vì training owner là một UNet denoising forward/backward mỗi batch, có thể cache latent, text và score. Nó tránh backprop qua cả sampling trajectory, nhưng chi phí sinh pool nhiều seed/prompt phải tính vào tổng GPU giờ.

### 3.4 Những rủi ro cần chặn bằng đối chứng

- Một ảnh/prompt không đủ ước lượng normalization trong prompt; dùng nhiều seed, báo bias của finite pool.
- Reward weighting có thể giảm diversity hoặc học thuộc pool. Theo dõi ESS = (sum w)^2/sum(w^2), nearest-neighbor và prompt chưa thấy.
- Denoising-preservation quá mạnh có thể triệt tiêu dấu; quá yếu tạo domain drift.
- Train loss giảm nhưng mean score teacher không tăng là thất bại embedding.
- Teacher tăng score nhưng student không tăng là thất bại inheritance, không sửa detector bằng final student để che thất bại.
- Generic fine-tune có thể tự đổi score. Cần clean-trained control, key đảo dấu và wrong-key tests.
- Feature/noise channel ổn định vẫn chưa đủ bảo đảm dễ học. Nếu fail, bỏ carrier đó thay vì thêm nhiều augmentation vô hạn.

## 4. Phải phân biệt fingerprint model và watermark từng ảnh

Mục tiêu mới chính là nhận diện nguồn model từ một tập output. Báo cả hai:

- Image TPR@1%FPR như trước, không thay số hoặc threshold cũ.
- Model-level detection theo query budget N = 32, 128, 512, với threshold riêng cho từng N.

Thống kê có thể là trung bình score đã trừ kỳ vọng clean theo prompt. Calibration dùng clean outputs và clean-trained controls trên prompt/seed độc lập. Báo độ bất định của baseline; không giả sử mọi model sạch có đúng cùng mean.

Mỗi mẫu kiểm định là một nhóm query được định trước. Ảnh chung prompt cần cluster/block handling. Resample nhiều nhóm từ một model không tạo ra nhiều training seeds độc lập. Ở paper phải đánh giá nhiều model sạch, key và seed huấn luyện.

Không xem dần 32/64/... ảnh và dừng ngay khi có p nhỏ; dùng N cố định hoặc kiểm định tuần tự hợp lệ. Nếu thử nhiều key/N cần hiệu chỉnh hoặc chỉ định primary endpoint trước.

Lý do có thể hữu ích: dưới giả thiết iid, độ lệch mean chuẩn hóa delta có SNR xấp xỉ sqrt(N)*delta. Ví dụ delta=0,2 và N=256 cho SNR khoảng 3,2; đây chỉ là minh họa, không phải ước tính power cho code thực tế.

**Thay endpoint sang model-level là thay claim nghiên cứu. Không thể dùng nó để nói đã đạt watermark trên 95% ảnh.** Nếu bắt buộc per-image TPR cao, cần hướng encoder–decoder bên dưới.

## 5. Hướng dự phòng gần bằng chứng thị giác hơn: học encoder–decoder h-space

Dùng residual ở mid-block, giới hạn số hướng và profile timestep, cùng detector ảnh học nhiều key. Học joint bằng loss detect + quality + các kênh owner-side. Phần kiến trúc có tiền lệ HMark; không gọi đây là novelty tự thân.

Thử thêm kênh noising–denoising ở SNR vừa phải để tránh detector chỉ đọc được shortcut pixel. Mọi kênh phải có clean/negative keys đi kèm, tránh denoiser tự hallucinate dấu. Loss qua kênh này là proxy, không phải mô phỏng training student.

Nếu inference vẫn cần hook nhúng h-space thì chưa đáp ứng yêu cầu native teacher. Cần owner-side consolidation sang trọng số UNet, kiểm tra lại dấu khi bỏ hoàn toàn hook. Không nói residual hook phi tuyến có thể merge như LoRA matrix. Chỉ train student trên output checkpoint native đã qua gate.

Ưu điểm: không bắt random CLIP direction quyết định một mình nơi nhúng và nơi đọc. Nhược điểm: detector dễ học shortcut; train encoder, detector và consolidate thêm tốn dữ liệu/compute. Đây là nhánh thứ hai nếu weighted native teacher không nhúng được đủ mạnh.

## 6. Pilot có khả năng bác bỏ giả thuyết, không chạy full ngay

Trước hết tái phân tích score liên tục của teacher cũ trên design set: mean shift, variance, quality và tương quan theo prompt. Chỉ là chẩn đoán; dùng dữ liệu mới để xác nhận model-level hypothesis sau khi đổi endpoint.

Pilot mới đề xuất (các số là ngân sách thiết kế, chưa benchmark):

1. Pool owner: 256 prompts x 8 seeds = 2048 clean images. Calibration và design-validation prompt tách riêng.
2. Cùng pool, cùng optimizer/update budget, ba teacher: unweighted control; weighted carrier CLIP; weighted carrier UNet denoising. Một beta chính được chọn theo ESS trên design, không theo student.
3. Teacher 500 updates, checkpoints 250/500. Đo score trên output native mới, quality, diversity; không chỉ đo trên pool đã weighting.
4. Chỉ nhánh qua teacher gate mới có paired student: r4 LoRA clean vs marked, cùng cấu hình ordinary trainer, 1000 ảnh/model và tối đa 2500 updates. Student checkpoint chọn theo validation denoising MSE đã đóng băng quy tắc.
5. Giai đoạn paper: r4/r16/full, ít nhất 3 training seeds, nhiều key, dataset/mixture ratios và prompt domains. Không extrapolate từ pilot 1 seed.

Gate mới phải đăng ký trước: primary N=128 và model-level alpha=1%; design effect-size lower bound >0 so clean-trained control; quality đạt protocol công bố. Giữ PSNR>=25 nếu tiếp tục claim paired preservation; nếu nghiên cứu distribution fidelity thì báo thành task riêng cùng LPIPS/CLIP/KID/diversity, không hạ ngưỡng sau khi nhìn kết quả.

Quy tắc dừng: teacher không shift trên held-out prompts -> xem owner optimization/carrier; teacher shift nhưng student không -> carrier chưa learnable; wrong keys/clean fine-tune cũng detect -> không đủ specificity. Chỉ tăng steps khi learning curve và validation cho thấy có cơ sở.

## 7. Đóng góp có thể có, và những gì chưa được phép claim

Exponential tilting, weighted diffusion training, h-space injection, detector learning và gộp nhiều query đều có prior art. Ghép chúng chưa tự tạo novelty.

Câu hỏi có thể tạo đóng góp: **carrier được định nghĩa trong biểu diễn denoising có truyền thiên lệch thống kê qua student độc lập tốt hơn carrier CLIP, ở cùng chất lượng và query budget không?**

Cần ablation cùng score dimension/owner budget, CLIP vs UNet carrier, có/không nuisance conditioning, model-level lẫn per-image, và baseline SERUM theo protocol tương thích. HMark chỉ so số trực tiếp khi tái hiện cùng setting. Nếu không hơn weighted CLIP đơn giản thì bỏ claim về carrier mới.

Có thể dùng phân tích điều kiện: với score trong [-1,1], nếu TV(P_student,Q_teacher)<=epsilon thì sai khác kỳ vọng score <=2epsilon. Do đó dấu teacher delta chỉ có thể được bảo toàn theo bound nếu delta đủ lớn so epsilon. Đây là bất đẳng thức tổng quát, **không** chứng minh LoRA đạt epsilon nhỏ, không phải novelty tự thân.

Ưu tiên thực tế: thử weighted native teacher + kiểm định model-level trước; giữ h-space encoder–decoder như nhánh thứ hai. Không tiếp tục tăng C3 hoặc rollout length chỉ vì có GPU rảnh. Không có cam kết đạt TPR cao trước khi pilot và đối chứng xác nhận.
