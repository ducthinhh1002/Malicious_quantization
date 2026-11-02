# Báo cáo semantic fingerprint: SD2.1 → SD1.4

Cập nhật **26/10/2026**. Báo cáo trình bày kết quả lịch sử, phương pháp **Projected native**, kết quả teacher/student và ảnh output. Lịch sử chạy chi tiết nằm trong [RESULT.md](RESULT.md).

## 1. Kết quả lịch sử — threshold_cfg_v1

Run: `run_semantic_fp_20261020_135117_009783`. Đây là phương pháp continuous-score/threshold CFG cũ. Teacher được chọn là **kappa 2**, không phải baseline kappa 1 của các thí nghiệm mới. Artifact/config lịch sử khác, nên không dùng chênh lệch với run mới để kết luận tác dụng riêng của cải tiến.

Teacher và student dùng ngưỡng null sạch khóa riêng, với FPR calibration mục tiêu 1%.

| Model | Detection | Mean score | Bit accuracy | FID | KID | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| teacher_clean_test | 1,00% | -0,01394 | 37,5% | -2,911e-05 | -0,00077483 | 0,78625 |
| teacher_wm_test | 100,00% | 1,9916 | 100,0% | 79,634 | 0,024161 | 0,77733 |
| unadapted | 0,90% | -0,06103 | 37,5% | 50,286 | -0,00016148 | 0,78217 |
| lora_r32_conv_wm | 99,70% | 1,5117 | 100,0% | 62,676 | 0,0069348 | 0,79334 |
| full_wm | 99,90% | 1,6457 | 100,0% | 75,077 | 0,016586 | 0,78809 |

Teacher validation quality, paired theo prompt/seed:

| Kappa | Được chọn | Clean CLIP | Watermarked CLIP | CLIP drop | Paired LPIPS | Quality pass của run cũ |
|---|---|---:|---:|---:|---:|---|
| 1 | Không | 0,78202 | 0,78587 | -0,0038546 | 0,48800 | Có |
| 1,5 | Không | 0,78202 | 0,77818 | 0,0038410 | 0,52049 | Có |
| 2 | Có | 0,78202 | 0,77279 | 0,0092314 | 0,54189 | Có |

Artifact lịch sử: `teacher_quality.json` / `teacher_quality.csv`; checkpoint curves: `checkpoint_metrics.csv`. Quality pass thuộc policy của run cũ, không phải ngưỡng ảnh đẹp phổ quát.

## 2. Mục tiêu và cấu hình Projected native

**Projected native fine-tune tiếp teacher kappa 1 để giảm biến dạng ảnh, đồng thời hạn chế xung đột giữa mục tiêu chất lượng và watermark tại bước cập nhật tham số.** Chỉ owner-side teacher được tối ưu bằng loss phụ. Teacher sinh RGB/caption; student độc lập học ordinary denoising MSE, không nhận key hay teacher features. Radioactivity được đo trên ảnh student sinh ra.

| Thành phần | Cấu hình |
|---|---|
| Teacher nền | `Manojb/stable-diffusion-2-1-base` |
| Khởi tạo | `threshold_cfg_v1`, kappa 1, checkpoint 3.000, scale 1, LoRA rank 32 attention+conv |
| Refinement teacher | 600 updates, batch 4, AdamW, learning rate 1e-5 |
| Checkpoint teacher | Mỗi200 updates, gồm optimizer/RNG/dataloader state |
| Native sampler | PNDM 50, CFG 7.5; thực tế 51 denoiser calls do khởi động PNDM |
| Native auxiliary | Mỗi2updates, một training prompt/seed |
| Sampler backward | Chỉ 10 denoiser calls cuối; forward vẫn đủ 51 calls |
| Precision refine | BF16 denoiser; FP32 CFG/sampler/VAE/extractor; giá trị noise khởi tạo lấy FP16 rồi chuyển FP32 |
| Production rendering | Adapter merge vào pipeline, sinh FP16 theo protocol hiện tại |
| Student | `CompVis/stable-diffusion-v1-4`, LoRA rank 32 attention+conv, batch 4, LR 1e-4 constant, FP16 |
| Dữ liệu | 5.000 train / 1.000 validation / 1.000 test; clean calibration riêng |
| Ngưỡng khóa | Teacher 0,8509314060; student 0,8541307449 |
| Student checkpoints chính | 2.500 / 5.000 / 7.500; chọn bằng validation denoising MSE thấp nhất |

Key, carrier, ngưỡng và cấu hình sinh ảnh theo baseline k1. Ảnh clean và ảnh k1 tham chiếu đã có được tái dùng với kiểm tra hash. Không có stopping gate theo chất lượng/detection; chỉ có kiểm tra số học, toàn vẹn và tài nguyên. Test/detector/utility không tham gia chọn checkpoint student.

## 3. Phương pháp tối ưu teacher

### 3.1. Giữ loss threshold_cfg_v1

Ở mọi update, vẫn tính preservation của conditional/unconditional noise prediction và coherent signed shift trong carrier, có thêm VAE-cycle:

$$
L_{\mathrm{orig}}=L_{\mathrm{preserve}}+0.8\left(L_{\mathrm{shift}}+0.5L_{\mathrm{cycle}}\right).
$$

Preservation là trung bình hai MSE conditional/unconditional so clean teacher. Ở vùng timestep semantic của protocol gốc, shift yêu cầu dựa trên threshold, sigma clean và score clean, được chặn trong[0;2,5]; kappa vẫn bằng1. Cơ chế coherent shift/cycle gốc tiếp tục được học.

### 3.2. Tối ưu chất lượng ảnh cuối

Cứ 2 updates, chọn một training prompt và Gaussian seed ứng với ảnh clean đã lưu. Teacher hiện tại sinh ảnh qua đủ PNDM 50/CFG 7.5, VAE decode và RGB8. Loss phụ:

$$
L_Q=\operatorname{LPIPS}(I_\theta,I_{\mathrm{clean}})+0.1\operatorname{MSE}(I_\theta,I_{\mathrm{clean}}).
$$

LPIPS dùng AlexNet version0.1. RGB8 dùng straight-through estimator để truyền gradient qua lượng tử hóa. Prefix sampler được sinh lại bằng weights hiện tại trong `no_grad`; 10 calls cuối giữ graph và dùng gradient checkpointing. PNDM history đi xuyên suốt, không reset ở ranh giới prefix/tail.

Thiết kế này áp dụng ý tưởng truncated sampler backprop của [DRaFT-K](https://arxiv.org/abs/2309.17400). Gradient có bias so với full-trajectory gradient; chưa chứng minh 10 calls là lựa chọn tối ưu cho watermark. Không dùng prefix cache lấy từ model cũ.

### 3.3. Tách và cân bằng gradient, neo score vào k1 gốc

Gọi $g_0=\nabla L_{\mathrm{orig}}$, $g_Q=\nabla L_Q$, $g_S=\nabla(-s(I_\theta))$, trong đó $s$ là detector score với carrier khóa. Quality gradient được cân bằng riêng:

$$
\alpha_Q=\min\left(100,\frac{\|g_0\|}{\|g_Q\|}\right).
$$

Code xử lý gradient gần 0 bằng epsilon/zero guard. Hệ số này hướng norm quality về mức 1 lần norm original, giới hạn khuếch đại 100. Đây là khác biệt thực tế so với nhánh `quality` cũ, nơi tỷ lệ norm trung vị chỉ khoảng0,345% gradient gốc.

Score target của mỗi training sample:

$$
s_{\mathrm{target}}=\max\left(s(I_{\mathrm{k1,baseline}}),\tau_{\mathrm{teacher}}+\sigma_{\mathrm{clean}}\right).
$$

Ảnh k1 baseline dùng cùng prompt/seed. Nếu score mới dưới target, thêm hướng sửa score với hệ số $\alpha_S=\min(100,0.25\|g_0\|/\|g_S\|)$; nếu đã đạt target thì $\alpha_S=0$. Không tiếp tục thưởng score cao hơn target.

Gradient đưa vào AdamW là $g_0+\alpha_Qg_Q+\alpha_Sg_S$, sau global norm clipping 1. Trong nhánh projected, ràng buộc hướng score bên dưới vẫn áp dụng ở auxiliary updates ngay cả khi score repair bằng 0.

### 3.4. Chiếu bước thay đổi tham số sau AdamW

AdamW đề xuất displacement $\Delta_A$. Trên các auxiliary updates, code tìm bước gần nhất thỏa ba điều kiện:

$$
\begin{aligned}
\Delta^*&=\arg\min_\Delta\frac12\|\Delta-\Delta_A\|_2^2,\\
g_0^T\Delta&\le0,\qquad g_S^T\Delta\le0,\qquad g_Q^T\Delta\le0.
\end{aligned}
$$

Theo xấp xỉ bậc nhất trên mẫu đang xét, chúng yêu cầu không tăng original loss, không giảm watermark score và không tăng quality loss. **Phép chiếu áp dụng lên actual AdamW displacement**, bao gồm ảnh hưởng momentum, preconditioning và weight decay. Nếu đề xuất đã khả thi thì giữ nguyên; nếu xung đột thì điều chỉnh hướng bước đó.

QP được giải bằng enumeration tối đa 8 active sets, với Gram matrix nhỏ tính FP64. Adam giữ moments của bước đề xuất; không thay thành moments của displacement đã chiếu. Log ghi cả displacement sau QP và displacement thực lưu sau rounding FP32. Update không có auxiliary dùng AdamW bình thường.

Thiết kế liên quan đến gradient surgery và tối ưu có ràng buộc như [PCGrad](https://arxiv.org/abs/2001.06782) và [GEM](https://arxiv.org/abs/1706.08840). Đây là adaptation cho teacher watermark hiện tại, không phải tái lập nguyên bản hoặc bằng chứng một phép chiếu hoàn toàn mới.

**Ràng buộc chỉ bảo vệ tuyến tính hóa cục bộ**, với score/quality gradient qua 10 calls cuối. Nó không bảo đảm nonlinear loss sau update, mọi prompt hay student radioactivity; bước cập nhật còn có thể bị thu nhỏ khi mục tiêu đối nghịch.

Code: [refinement](../../code_best_pipeline/shared/threshold_quality_v1_source/pareto_quality_refine.py), [QP/AdamW](../../code_best_pipeline/shared/threshold_quality_v1_source/pareto_step.py), [campaign](../../code_best_pipeline/shared/threshold_quality_v1_source/pareto_quality_campaign.py). Các kiểm thử gồm QP đối chiếu SciPy, native forward, checkpointed gradients, optimizer state; actual-model GPU smoke đã pass trước train.

## 4. Đối chứng và định nghĩa metric

| Nhánh | Ý nghĩa |
|---|---|
| Baseline k1 | Teacher checkpoint 3.000,scale 1, chưa refine thêm |
| Continuation | Cùng khởi tạo, thêm600teacher updates chỉ bằng loss gốc |
| Native full | Thiết kế ảnh cuối trước đây: full sampler gradient, auxiliary mỗi 5 updates,cap 25% |
| Projected native | Ảnh cuối, gradient balancing, score floor và chiếu actual AdamW step |

Continuation và Projected native cùng **số teacher updates**, không cùng GPU compute: native generation/backward làm nhánh cải tiến tốn thời gian hơn. Student chính của mỗi nhánh train độc lập 7.500 updates từ clean SD1.4.

Teacher validation LPIPS so với clean SD2.1 cùng prompt/seed. Student test LPIPS so với **SD1.4 chưa fine-tune** cùng generation prompt/seed. FID/KID ở các bảng mới dùng1.000 clean teacher test images làm reference, không phải ảnh COCO thật. Vì reference khác nhau, không so trực tiếp trị tuyệt đối teacher LPIPS và student LPIPS. CLIP theo utility hiện tại là trung bình $2.5\max(\cos(\mathrm{image},\mathrm{text}),0)$.

## 5. Kết quả

### 5.1. Teacher

Trên 1.000 validation images, Projected native: **detection 994/1.000=99,4%**, mean score 1,362220, paired LPIPS **0,371757**. Continuation đạt cùng 99,4% detection, mean score 1,444780 và LPIPS 0,430804. LPIPS giảm **13,71%** so đối chứng cùng600 updates, nhưng mean score giảm; giữ teacher detection không tự chứng minh giữ student detection.

FID/KID/CLIP đo riêng trên1.000 test images, cùng clean reference đã đối chiếu hashes:

| Teacher | FID ↓ | KID mean ↓ | KID subset std | CLIP ↑ |
|---|---:|---:|---:|---:|
| Baseline k1 | 45,39956 | 0,00212268 | 0,00045623 | 0,782971 |
| Continuation | 45,05869 | 0,00180676 | 0,00040345 | 0,783130 |
| Native full | 45,41688 | 0,00200133 | 0,00042479 | 0,783372 |
| **Projected native** | **41,57596** | **0,00029563** | 0,00020207 | **0,783647** |

So baseline k1, teacher FID giảm **8,42%**, KID mean giảm **86,07%** trong phép đo này. Detection99,4% và LPIPS 0,371757 ở trên thuộc **validation**, không phải test detection.

Nguồn: [teacher validation](../../output_artifacts/results/historical_summaries/projected_native_teacher_validation_20261025.json), [teacher FID/KID](../../output_artifacts/results/historical_summaries/projected_native_teacher_fid_kid_20261025.json).

### 5.2. Student

1.000 test images, ngưỡng student khóa0,8541307449:

| Student | Updates | Detection ↑ | Mean score | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline k1 | 7.500 | 75,0% | 1,036625 | 0,550447 | 54,44424 | 0,00166651 | 0,787681 |
| Continuation | 7.500 | **82,3%** | 1,078573 | 0,550218 | 54,88688 | 0,00141482 | 0,786429 |
| Native full | 7.500 | 80,5% | 1,065606 | 0,544096 | 54,60549 | 0,00126494 | 0,786889 |
| **Projected native** | **7.500** | **79,2%** | **1,064043** | **0,522219** | **53,35774** | **0,00079496** | **0,785335** |
| Projected native,seeded rerun | 10.000 | 82,1% | 1,084374 | 0,533119 | 53,45343 | 0,00077322 | 0,787516 |

Projected native 7.500 được chọn bằng validation denoising MSE 0,1478019591; đủ 7.500 applied updates, không skip. Detection 792/1.000 có Wilson 95% CI **76,57–81,60%**. So baseline k1: detection tăng **4,2 điểm phần trăm**, LPIPS giảm **5,13%**, FID giảm **2,00%**. So Continuation, detection thấp hơn 3,1 điểm phần trăm, đổi lại LPIPS/FID/KID tốt hơn. CLIP thấp hơn baseline; chưa thắng mọi metric.

Mốc 10.000 là **fixed endpoint** của follow-up sau khi xem kết quả 7.500, không phải thắng cuộc trong một phép chọn chung 7.500/10.000. Snapshot cũ thiếu optimizer/RNG nên đã replay từ clean initialization cùng config/seed rồi train liên tục đến 10.000. Replay 7.500 **không trùng bitwise** bản cũ: max absolute tensor difference0,0004711258. Chưa đo test riêng replay 7.500, nên chưa tách hết ảnh hưởng thêm 2.500 updates khỏi khác biệt giữa hai lần chạy. Kết quả 10.000 có detection cao hơn bản 7.500 gốc 2,9 điểm, nhưng LPIPS xấu hơn2,09%, FID tăng nhẹ. Full training state đã lưu mỗi 2.500 ở run này.

Các kết quả trên đánh giá toàn bộ phương pháp Projected native; chưa tách riêng đóng góp của gradient balancing, score floor và phép chiếu. Cần ablation để xác định tác dụng từng thành phần, không gán toàn bộ cải thiện cho projection.

Nguồn: [Projected7500](../../output_artifacts/results/historical_summaries/projected_native_student_final_20261025.json), [Projected10000](../../output_artifacts/results/historical_summaries/projected_native_student10000_final_20261026.json), [replay audit](../../output_artifacts/results/historical_summaries/projected_native_student10000_replay_20261026.json), [Continuation](../../output_artifacts/results/historical_summaries/continuation_student_final_20261025.json).

## 6. Ảnh output teacher và student

Các hình dưới đây là **3 mẫu test đầu tiên**, không chọn theo vẻ đẹp hoặc score. PNG gốc 512×512, chưa chỉnh sửa. Trong mỗi nhóm teacher hoặc student, các cột có cùng generation prompt/seed. Teacher và student khác backbone và seed protocol; không xem chúng là cùng một ảnh chỉ thêm/bớt watermark.

| Sample | Prompt |
|---|---|
| 000000 | A battered refrigerator stands open in a dirty room. |
| 000001 | a person cutting a pink cake with pink roses |
| 000002 | A man in a green shirt stands by a girl holding a piece of cake on a plate. |

### 6.1. Teacher: clean → baseline k1 → Projected native

| Sample | Clean SD2.1 teacher | Baseline k1 teacher | Projected native teacher |
|---|---|---|---|
| 000000 | ![Clean teacher sample000000](../../output_artifacts/previews/projected_native_teacher_preview_20261025/clean/sample_000000.png) | ![Baseline k1 teacher sample000000](../../output_artifacts/previews/projected_native_teacher_preview_20261025/baseline_k1/sample_000000.png) | ![Projected native teacher sample000000](../../output_artifacts/previews/projected_native_teacher_preview_20261025/projected_native/sample_000000.png) |
| 000001 | ![Clean teacher sample000001](../../output_artifacts/previews/projected_native_teacher_preview_20261025/clean/sample_000001.png) | ![Baseline k1 teacher sample000001](../../output_artifacts/previews/projected_native_teacher_preview_20261025/baseline_k1/sample_000001.png) | ![Projected native teacher sample000001](../../output_artifacts/previews/projected_native_teacher_preview_20261025/projected_native/sample_000001.png) |
| 000002 | ![Clean teacher sample000002](../../output_artifacts/previews/projected_native_teacher_preview_20261025/clean/sample_000002.png) | ![Baseline k1 teacher sample000002](../../output_artifacts/previews/projected_native_teacher_preview_20261025/baseline_k1/sample_000002.png) | ![Projected native teacher sample000002](../../output_artifacts/previews/projected_native_teacher_preview_20261025/projected_native/sample_000002.png) |

### 6.2. Student: SD1.4 tham chiếu → Projected7500 → Projected10000

| Sample | SD1.4 chưa fine-tune | Projected native student7500 | Projected native student10000 |
|---|---|---|---|
| 000000 | ![Unadapted SD1.4 sample000000](../../output_artifacts/previews/projected_native_student_preview_20261025/unadapted/sample_000000.png) | ![Projected student7500 sample000000](../../output_artifacts/previews/projected_native_student_preview_20261025/student_projected_native/sample_000000.png) | ![Projected student10000 sample000000](../../output_artifacts/previews/projected_native_student10000_preview_20261026/student10000/sample_000000.png) |
| 000001 | ![Unadapted SD1.4 sample000001](../../output_artifacts/previews/projected_native_student_preview_20261025/unadapted/sample_000001.png) | ![Projected student7500 sample000001](../../output_artifacts/previews/projected_native_student_preview_20261025/student_projected_native/sample_000001.png) | ![Projected student10000 sample000001](../../output_artifacts/previews/projected_native_student10000_preview_20261026/student10000/sample_000001.png) |
| 000002 | ![Unadapted SD1.4 sample000002](../../output_artifacts/previews/projected_native_student_preview_20261025/unadapted/sample_000002.png) | ![Projected student7500 sample000002](../../output_artifacts/previews/projected_native_student_preview_20261025/student_projected_native/sample_000002.png) | ![Projected student10000 sample000002](../../output_artifacts/previews/projected_native_student10000_preview_20261026/student10000/sample_000002.png) |

Gallery đầy đủ 8 mẫu, bấm ảnh để mở PNG gốc:

- [Teacher: Projected / baseline k1 / clean](../../output_artifacts/previews/projected_native_teacher_preview_20261025/index.html).
- [Student7500: Projected / baseline k1 / unadapted](../../output_artifacts/previews/projected_native_student_preview_20261025/index.html).
- [Student10000 / student7500 / unadapted](../../output_artifacts/previews/projected_native_student10000_preview_20261026/index.html).

## 7. Giới hạn và kết luận được dữ liệu hỗ trợ

Projected native tốt hơn baseline k1 ở student detection, LPIPS và FID trong lần chạy hiện tại. So với Continuation, detection thấp hơn nhưng LPIPS/FID/KID tốt hơn. Kết quả hỗ trợ nghiên cứu tiếp cân bằng chất lượng–radioactivity, chưa chứng minh lợi ích riêng của projection.

- Mỗi cấu hình hiện chỉ có một training seed; test đã được xem nhiều vòng phát triển. Cần nhiều seed và fresh holdout trước claim tổng quát.
- WilsonCI theo ảnh và KID subset std không đo độ bất định giữa các lần train. Chưa dùng chúng để kết luận tương đương/non-inferiority giữa các phương pháp.
- LPIPS đo khoảng cách perceptual với reference đã nêu, không phải mức đẹp tuyệt đối. FID 1.000 mẫu có sample bias; KID mean có thể âm gần 0. Các metric không xác nhận hết lỗi mặt/tay.
- Chưa có human preference/defect ratings. Gallery minh họa không thay đánh giá mù trên panel được khóa trước.
- Bit accuracy 100% trong summary là dấu tọa độ trung bình toàn tập, không phải detection 100% hay độ chính xác bit trên từng ảnh.
- Clean-trained student control có FPR 0,2% trong run đã đo nhưng vẫn có drift màu; không quy mọi artifact cho watermark. FPR calibration mục tiêu 1% không mặc nhiên là FPR thực nghiệm trên mọi clean model.
- Teacher score/detection và local projected constraints không bảo đảm student radioactivity; phải đo student được train độc lập.

Phân tích bổ sung: [chiến lược/ablation](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/pareto_quality_strategy_20261025.md), [native full](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/native_quality_research_20261025.md), [learning-channel audit](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/transfer_research_20261024.md).
