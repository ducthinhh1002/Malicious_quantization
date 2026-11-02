# Learning-Aligned Latent Transport: thiết kế thống nhất và lượt train dài

Ngày: 18/10/2026. Tên LALT là tên làm việc của triển khai này, không phải tuyên
bố đã có một phương pháp mới được kiểm chứng hoặc chưa từng có trong literature.

## 1. Bài toán phải giải

Theo `target.md`, owner thay fingerprint của teacher; attacker khởi tạo student
sạch, chỉ lấy RGB/caption rồi chạy ordinary LoRA/SFT. Student không nhận latent,
key, extractor hay adapter của teacher. Cần đọc dấu trong ảnh mới của student,
không chỉ trong ảnh train hoặc ảnh teacher. Không có bảo đảm cho mọi learner:
student không cập nhật hoặc không dùng dữ liệu có dấu là phản ví dụ trực tiếp.

Các lượt vừa chạy cho thấy reader đạt khoảng 96% BA train nhưng AUROC chỉ
0,50–0,51 trên prompt mới khi carrier bị đóng băng. Đó là bằng chứng cấu hình
fit riêng reader thất bại về khái quát; không chứng minh chỉ cần train lâu hơn,
cũng không chứng minh mọi carrier học được đều thất bại.

Lượt này thay kiến trúc nhúng, kiến trúc đọc và thiết lập dữ liệu trong một
thiết kế đã chốt. Không dùng thêm các lượt fit 8 prompt để quyết định dừng.

## 2. Đối chiếu công trình gần nhất

| Công trình | Phần liên quan và giới hạn đối với target |
|---|---|
| [HMark](https://arxiv.org/html/2512.00094v1) | Học encoder/reader ở h-space, không có radioactivity loss trực tiếp; thiết lập semantic DDPM/CelebA. Họ dùng 100.000 ảnh học watermark. Không suy kết quả đó sang SD text-to-image của ta. |
| [RoSteALS](https://arxiv.org/abs/2304.03400) | Học offset latent qua autoencoder đóng băng. Đây là tiền lệ rõ cho nhúng latent; robust ảnh không đồng nghĩa truyền qua ordinary learner. |
| [DistSeal](https://arxiv.org/html/2601.16140v1) | Embedder phụ thuộc latent/nội dung; có distillation vào generator/decoder. Vì vậy công thức nhúng adaptive latent tự nó không phải novelty. Cần phân biệt nhận latent đã watermark khi distill với chỉ nhận RGB rồi VAE encode lại. |
| [BitMark](https://arxiv.org/html/2506.21209v2) | Bias phân phối sinh bit có thể truyền qua training. Không có giao diện bit-logit tương đương ở SD; không thể suy ra ép một CNN ngẫu nhiên sẽ tạo dấu tương tự. |
| [SERUM](https://arxiv.org/html/2603.13396v1) | Shared noise pattern là một cơ chế khác; paper báo LoRA ở checkpoint 19.500 bước theo validation loss. Teacher mạnh không tự bảo đảm ordinary LoRA của ta giữ dấu. |
| [Radioactive data](https://arxiv.org/abs/2002.00937) | Nền tảng truy nguồn qua quá trình học, nhưng bài toán classifier không phải chứng minh cho diffusion student. |
| [Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929) | Tiền lệ tối ưu dữ liệu theo tác động lên gradient học. LALT không triển khai nguyên thuật toán gradient matching của paper này; dùng unroll ordinary learner để đo mục tiêu truyền dấu. |

Điểm cần chứng minh của LALT là **tối ưu tác động của dấu qua kênh RGB → VAE →
ordinary denoising**, vượt teacher-only ở cùng chất lượng. Bilevel optimization,
latent embedding, multiscale reader đều có tiền lệ. Không gọi việc ghép chúng
là đột phá trước khi có đối chứng và kiểm tra learner chưa dùng để thiết kế.

## 3. Kiến trúc mới so với code cũ

Teacher sạch cho latent cuối `z0`. Bộ nhúng tạo:

\[
T_\phi(z,k)=z+v_\phi(z,k),\quad x_k=D(T_\phi(z,k)).
\]

Với `b(k)=2k-1`, trường nhúng là:

\[
v_\phi(z,k)=\operatorname{Up}\left[
 a\tanh\left(\frac{1}{\sqrt B}\sum_{j=1}^{B}
 b_j(k)\,[U_j+F_\phi(\operatorname{AvgPool}_2z)_j]\right)\right].
\]

- `U_j`: basis dùng chung theo key, 8×8, được học.
- `F_phi`: CNN nhận nội dung latent, xuất các basis điều biến theo ảnh.
- Tính residual trên lưới latent thô hơn rồi nội suy, giảm tự do tạo dấu ở
  từng pixel nhỏ. Đây là inductive bias, không phải bộ lọc giới hạn băng lý tưởng.
- `a=0.35` là chặn từng tọa độ trong **scaled latent**, không phải mức thay đổi
  RGB 0,35. LPIPS/MSE và kiểm định ảnh vẫn cần thiết.
- Key bổ sung cho residual đối dấu chính xác trên cùng latent, chống liên hệ
  giả giữa prompt và key. Không giả định RGB cũng đối dấu sau decoder phi tuyến.
- Không có điều kiện chỉ nhúng vào người, phong cảnh hoặc một nhóm prompt.

Reader mới là CNN dùng chung theo vị trí, nhận RGB và residual high-pass cố
định, tổng hợp evidence bằng global pooling. Không dùng đầu flatten 4×4 có
thể học tương quan bố cục với key. Điều này giảm một khả năng ghi nhớ, không
loại bỏ overfitting bằng định lý. Reader học đồng thời với embedder ngay từ đầu.

Module nằm **sau denoising, trước VAE decode** trong pipeline teacher. UNet/VAE
gốc đóng băng. Không gọi đây là fingerprint đã merge vào native UNet weights.
Owner phải kiểm soát pipeline xuất ảnh; attacker có thể tắt module nếu được
cấp toàn quyền pipeline thì chưa nằm trong bảo đảm của thiết kế này.

## 4. Cơ sở toán học và những gì không được suy quá

### 4.1. Thay đổi phân phối thay vì ghi nhớ ảnh

Với latent có phân phối `p0`, đầu ra transport có phân phối pushforward
`pk = (T_phi,k)# p0`. Nếu xét nhiễu nhỏ `T(z)=z+epsilon*v(z)` và thêm giả định
trơn/khả nghịch cục bộ, khai triển chuẩn của phương trình liên tục cho:

\[
p_k=p_0-\epsilon\nabla\cdot(p_0v)+O(\epsilon^2),
\quad
s_k=s_0-\epsilon\nabla\big[\nabla\cdot v+v^Ts_0\big]+O(\epsilon^2).
\]

Đây chỉ giải thích shared transport có thể thay đổi score của phân phối.
Kiến trúc không ép khả nghịch; công thức khai triển không phải chứng minh
chính xác cho mọi trọng số đang train. Đặc biệt, LoRA bị hạn chế không nhất
thiết học được score thay đổi ấy. Vì vậy cần mục tiêu ở bước tiếp theo.

### 4.2. Dấu phải đi vào hướng mà learner thật học

Gọi `S_k(theta)` là score đọc dấu trên ảnh query mới của student, `g_w` và
`g_c` là gradient ordinary denoising trên dữ liệu marked/clean tương ứng.
Với một bước **SGD**, khai triển Taylor cho:

\[
S_k(\theta-\eta g_w)-S_k(\theta-\eta g_c)
=-\eta\nabla_\theta S_k(\theta)^T(g_w-g_c)+R.
\]

Nếu Hessian của `S_k` có norm ≤L trên các đoạn cập nhật thì:

\[
|R|\le\tfrac12 L\eta^2(\|g_w\|^2+\|g_c\|^2).
\]

Điều này chỉ ra điểm thiếu của loss đọc teacher: có score teacher cao nhưng
gradient denoising của dữ liệu không hướng tới score student thì dấu không
truyền. Đây là lập luận Taylor tiêu chuẩn, không phải định lý novelty mới.

Code thực tế dùng **AdamW**, không thay nó bằng công thức SGD trên. Gradient
owner được tính xuyên chuỗi ordinary AdamW, gồm moment/clipping và sampling
student. Căn bậc hai của second moment dùng floor 1e-24 đã khai báo trong
`learner.py`; không gọi đạo hàm tại zero là AdamW lý tưởng không xấp xỉ.

### 4.3. Objective hai giai đoạn

Giai đoạn 1 học `phi, omega`:

\[
\mathcal L_{teacher}=\operatorname{softmaxmax}_{C\in\{I,resize\}}
 \mathcal L_{code,presence,pair}(R_\omega(C(x_k)),k)
 +\mathcal L_{quality}
 +\lambda_{cycle}\mathcal L_{code}(R_\omega(D(E(x_k))),k).
\]

Mỗi 4 bước tính thêm cycle VAE posterior mean; cycle chỉ là kênh phụ,
không được gọi là mô phỏng LoRA. PNG/RGB8 có forward lượng tử hóa thật và
owner dùng STE để tính gradient. Phạt chất lượng là penalty mềm theo mẫu,
không phải projection bảo đảm mọi ảnh luôn đạt ngưỡng.

Giai đoạn 2 đóng băng reader; với cùng randomness/ordinary recipe:

\[
\theta_w^K=U^K(\theta_0;\{x_k,c\}),\quad
\theta_c^K=U^K(\theta_0;\{x_0,c\}),
\]
\[
\min_\phi\;\mathcal L_{teacher}
 +\lambda\mathcal L_{transfer}
   (R_\omega(G_{\theta_w^K}(c',\xi')),
    R_\omega(G_{\theta_c^K}(c',\xi')),k)
 +\lambda_q\mathcal L_{student\ quality}.
\]

`U` chỉ dùng ordinary denoising MSE trên VAE encoding của RGB; không có
watermark loss trong student update. `L_transfer` ép score đúng key tăng so
với clean-trained control và có specificity đối với key khác. Gradient cuối
cùng chỉ cập nhật transport của owner. Reader cố định ở giai đoạn 2 để không
đổi thước đo theo learner đang dùng để thiết kế.

Đạo hàm qua 4 update / 8-step DDIM là gradient của **proxy được khai báo**,
không phải gradient của final student 5.000 bước. Nó không bảo đảm giữ dấu
qua mọi rank, sampler, optimizer hoặc lượng dữ liệu. Phải đo final learners.

## 5. Lượt train dài đã chốt

| Phần | Ngân sách / thiết lập |
|---|---|
| Teacher | 20.000 update, batch 4, SD1.4 512px, FP32 |
| Dữ liệu owner | 4.096 prompt train; 2 noise seed/prompt; 32 train keys |
| Dữ liệu thực dùng | Sampling có hoàn lại; tối đa 8.192 clean latent/ảnh cache, không phải 8.192 model training seeds |
| Validation teacher | 128 prompt mới, 2 key ngoài tập train; mỗi 1.000 bước |
| Learning-channel | 128 outer update; ordinary attention-LoRA rank 4; 4 inner AdamW update; query DDIM 8 bước |
| Proxy validation | 4 episode rời nhau, horizon 4/16; giữ gate chất lượng teacher |
| Final students | Ordinary attention-LoRA rank 8 và 16; 2.000 RGB/caption; 5.000 bước mỗi student |
| Đối chứng | Student học ảnh sạch; teacher-only vs transfer-optimized; wrong keys; unadapted student |

Teacher chạy hết 20.000 bước, không dừng vì reader chưa fit ở vài trăm bước.
Chỉ sau ngân sách đó mới kiểm tra teacher gate để quyết định chạy meta/student.
Đây không phải lời hứa student luôn chạy khi teacher thất bại.

Không thay cấu trúc LoRA, loss hoặc module target của final student. Học
rank4 để thiết kế, kiểm tra rank8/16; đây vẫn là một base family và một seed
train, chưa phải chứng minh cross-architecture/multi-seed. Bản đầu giữ payload
8 bit để kiểm tra cơ chế, không nhận claim cryptographic capacity.

Final attribution dùng calibration riêng và hiệu chỉnh tổng FPR qua key;
không lấy validation TPR làm TPR test. Student checkpoint được đánh giá ở
ngân sách đã định, không chọn theo watermark score trên final test.

## 6. Code và vận hành

- `latent_transport.py`: kiến trúc nhúng/đọc và loss worst-view trơn.
- `transport_train.py`: joint training dài, không có tiny frozen-reader gate.
- `train.py`, `runtime.py`: render/cache latent mới, tái sử dụng learner/evaluation đã kiểm thử.
- `transport_gpu_check.py`: kiểm tra gradient thật qua cycle và 4 ordinary
  updates + full 8-step query ở 512px, không phải một thí nghiệm hiệu quả nhỏ.
- `run_latent_transport.sh`: một lệnh cho toàn bộ chương trình có gate.

```bash
bash run_latent_transport.sh
```

Cache chỉ dùng generator sạch đóng băng. Cả latent cuối và ảnh clean được
lưu để tránh chạy UNet lại ở mỗi update. Không cache/detach generation của
student đang được differentiation. Checkpoint/cache/images nằm ngoài báo cáo.
Smoke kỹ thuật chạy trước lượt dài để tránh treo 20.000 bước trên lỗi code;
không dùng score smoke để chọn kiến trúc hay báo cáo hiệu quả.

## 7. Tiêu chí thành công và claim có thể viết

Đầu tiên teacher phải tổng quát sang prompt và key mới, giữ chất lượng. Sau
đó LALT phải vượt teacher-only trên ordinary student ở cùng ngân sách và
chất lượng, đồng thời clean-trained student và wrong keys không bị nhận nhầm.

Các ablation content-adaptive vs shared-only, cycle/no-cycle, budget matching
và nhiều training seeds vẫn cần cho bài báo; lượt chính này chưa thực hiện
toàn bộ chúng. Nếu teacher tốt nhưng transfer thất bại, không đổi mục tiêu
thành robust watermark ảnh rồi tuyên bố đạt target ban đầu.

**Hiệu quả chưa được chứng minh.** Toán học xác định đúng đối tượng cần tối
ưu và các kiểm tra bắt buộc; kết quả train/evaluation mới quyết định phương
pháp có hữu ích hay không.

## 8. Triển khai ban đầu ngày 18/10

Snapshot trên vm4: `/root/logs/lalt_20261018_053126Z`.
Tmux: `lalt_20261018_053126Z`.
Lượt chính: `/root/logs/results/lalt_20261018_053126Z_diagnostic` (hậu tố
`diagnostic` là tên thư mục từ helper triển khai; config thực là `pilot`,
`transport`, teacher 20.000 bước và `teacher_only=false`).

21 numerical/protocol tests đã pass local và server. Wrapper chạy smoke
toàn luồng, kiểm tra gradient 512px, rồi tự bắt đầu lượt chính nếu tất cả
kiểm tra runtime thành công. Các artifact kiểm thử không phải bằng chứng
teacher/student fingerprint đã tốt. Trạng thái triển khai lưu ở
`server_review/lalt_deployment/deployment.json` và `verification.json` khi có.

Xác nhận triển khai: smoke toàn luồng hoàn tất; kiểm tra 512px với 4 ordinary
updates và full query DDIM 8 bước đã pass (gradient tới carrier khác zero,
meta peak 75,43 GiB). Lượt chính đã bắt đầu joint teacher training, step đầu
có gradient carrier khác zero. Chưa có đánh giá hiệu quả của lượt dài.

## 9. Cập nhật ngày 21/10: transfer vẫn yếu, sửa mục tiêu và proxy

Lượt ordinary LoRA rank4 đã hoàn tất 5.000 bước. Trên cùng 128 query,
repair tăng teacher TPR từ 77,34% lên 84,38%, nhưng student TPR vẫn
3,91% tại nominal single-key FPR 1%. Contrast readout cũng vẫn chỉ phát
hiện 5/128 ảnh. Chưa đạt mục tiêu watermark sống sót qua ordinary LoRA.

Phân tích nguyên nhân, research và thiết kế thử nghiệm mới nằm trong
[LALT: nguyên nhân transfer yếu và contrast/replay](LALT_Transfer_Root_Cause_and_Contrast_Replay_VI.md).
Code mới tối ưu teacher bằng contrast đúng-key/sai-key, chuẩn hóa bằng
clean contrast, dùng 4 ảnh khác nhau trong inner loop, và kiểm tra thêm
trạng thái clean LoRA đã học. Student đánh giá cuối vẫn dùng ordinary MSE.

Snapshot mới: `/root/logs/lalt_contrast_20261020_172619Z`; deployment:
[deployment.json](../server_review/lalt_contrast/lalt_contrast_20261020_172619Z/deployment.json).
43 unit/protocol tests đã pass trên server. Wrapper kiểm tra GPU smoke
trước khi chạy hai nhánh 96 outer updates; chỉ tạo ordinary student mới
nếu checkpoint vượt gate proxy dương và teacher preservation. Hiệu quả
transfer của lượt mới chưa được xác nhận.

Cập nhật kiểm tra 06:41 ngày 21/10: cả hai nhánh đã hoàn tất 96/96 bước,
không checkpoint mới nào vượt gate. Cold chỉ tăng proxy +0,000270 so
baseline (yêu cầu +0,005); replay tăng +0,002028 nhưng vẫn âm. Teacher
validation TPR key221 giảm từ 83,59% xuống 78,91%/79,69% ở step96.
Suite không chạy student mới; chưa có TPR student mới. Xem mục 8 của
báo cáo contrast/replay để đối chiếu đầy đủ và giới hạn kết luận.

## 10. Nghiên cứu lại ngày 21/10

[Báo cáo nghiên cứu và các hướng cải thiện student](LALT_Rethink_Student_Transfer_VI.md)
đối chiếu lại SERUM/HMark/CoprGuard/DIAGNOSIS, ghi rõ khác biệt threat model,
và đề xuất carrier thống kê nhất quán hoặc quan hệ ngữ nghĩa học vào native
teacher. Không thay loss/key/recipe của ordinary attacker student.

Audit reader mới đã chạy xong trên ảnh student cũ, split64 fit/64 prompt
khác đánh giá, kiểm tra rank4/16/32 và noise namespace khác. Thay linear
readout chỉ cải thiện nhẹ, chưa chữa được transfer. Prototype spectral
transport có exact denoiser target đã có code/test; screen carrier tổng cộng
bảy mức đã hoàn tất, không mức nào qua đồng thời signal/quality gate.
51 tests pass trên server. Chưa có native teacher hoặc student mới thành
công; không dùng kết quả screen thay cho kiểm tra ordinary student thật.
Kết quả này ưu tiên bước tiếp theo là semantic relation trong native teacher,
không tiếp tục tăng strength của uniform spectral rotation hoặc đổi hệ số
loss trên cùng proxy ngắn. Chi tiết phép thử và quyết định ở cuối báo cáo.

## 11. Thay cơ chế ngày 21/10: native semantic relation

Đã triển khai nhánh thay thế bằng quan hệ nội dung–ánh sáng có key trong UNet,
thay cho residual latent + bit reader. Owner được train, merge và reload thành
native teacher; attacker vẫn học ordinary LoRA trên PNG/caption gốc.
[Thiết kế, code, đối chứng và cách theo dõi](Semantic_Relation_Implementation_VI.md).
55 tests đã pass trên server. Smoke GPU chạy trước pilot; chưa có kết quả student
mới để khẳng định cải thiện. LALT được giữ làm baseline so sánh.

Cập nhật 12:26 ngày 21/10: pilot semantic relation đã hoàn tất bốn native teacher,
mỗi bản 500 bước. Cả hai beta và hai dấu key đều không qua gate; CI99% relation
đều chứa 0 và quality tails vượt ngưỡng. Không có checkpoint được chọn và **chưa
chạy student mới**. Positive-control sửa prompt đạt 26/32 đúng chiều nhưng native
embedding chưa giữ được quan hệ đó. Xem kết quả và chẩn đoán CFG trong
[báo cáo triển khai](Semantic_Relation_Implementation_VI.md).

## 12. Đối chiếu Stable Signature qua ordinary LoRA

Theo yêu cầu ngày 21/10, thêm suite Stable Signature: rank4/16 × attention-only/
attention+UNet CNN, mỗi cấu hình có matched clean student. Teacher dùng public
marked decoder, student khởi tạo từ SD2.1 sạch và giữ VAE sạch để không nhầm
decoder watermark có sẵn với dấu học được từ ảnh. Mỗi long student: 2.048 ảnh,
5.000 bước, ordinary MSE. [Protocol và cách theo dõi](Stable_Signature_Ordinary_LoRA_Transfer_VI.md).
