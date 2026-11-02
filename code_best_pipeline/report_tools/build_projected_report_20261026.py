"""Build the requested Markdown report from the retained historical results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 't2ifing_radioactive/result_vast/Report.md'

report = r'''# Báo cáo semantic fingerprint: SD2.1 → SD1.4

Cập nhật **26/10/2026**. Báo cáo trình bày kết quả lịch sử, phương pháp **Projected native**, kết quả teacher/student và ảnh output. Lịch sử chạy chi tiết nằm trong [RESULT.md](RESULT.md).

## 1. Kết quả lịch sử — threshold_cfg_v1

Run: `run_semantic_fp_20261020_135117_009783`. Đây là phương pháp continuous-score/threshold CFG cũ. Teacher được chọn là **kappa2**, không phải baseline kappa1 của các thí nghiệm mới. Artifact/config lịch sử khác, nên không dùng chênh lệch với run mới để kết luận tác dụng riêng của cải tiến.

Teacher và student dùng ngưỡng null sạch khóa riêng, với FPR calibration mục tiêu1%.

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

**Projected native fine-tune tiếp teacher kappa1 để giảm biến dạng ảnh, đồng thời hạn chế xung đột giữa mục tiêu chất lượng và watermark tại bước cập nhật tham số.** Chỉ owner-side teacher được tối ưu bằng loss phụ. Teacher sinh RGB/caption; student độc lập học ordinary denoising MSE, không nhận key hay teacher features. Radioactivity được đo trên ảnh student sinh ra.

| Thành phần | Cấu hình |
|---|---|
| Teacher nền | `Manojb/stable-diffusion-2-1-base` |
| Khởi tạo | `threshold_cfg_v1`, kappa1, checkpoint3000, scale1, LoRA rank32 attention+conv |
| Refinement teacher | 600updates, batch4, AdamW, learning rate1e-5 |
| Checkpoint teacher | Mỗi200updates, gồm optimizer/RNG/dataloader state |
| Native sampler | PNDM50, CFG7.5; thực tế51 denoiser calls do khởi động PNDM |
| Native auxiliary | Mỗi2updates, một training prompt/seed |
| Sampler backward | Chỉ10 denoiser calls cuối; forward vẫn đủ51 calls |
| Precision refine | BF16 denoiser; FP32 CFG/sampler/VAE/extractor; giá trị noise khởi tạo lấy FP16 rồi chuyển FP32 |
| Production rendering | Adapter merge vào pipeline, sinh FP16 theo protocol hiện tại |
| Student | `CompVis/stable-diffusion-v1-4`, LoRA rank32 attention+conv, batch4, LR1e-4constant, FP16 |
| Dữ liệu | 5.000train / 1.000validation / 1.000test; clean calibration riêng |
| Ngưỡng khóa | Teacher0,8509314060; student0,8541307449 |
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

Cứ2updates, chọn một training prompt và Gaussian seed ứng với ảnh clean đã lưu. Teacher hiện tại sinh ảnh qua đủ PNDM50/CFG7.5, VAE decode và RGB8. Loss phụ:

$$
L_Q=\operatorname{LPIPS}(I_\theta,I_{\mathrm{clean}})+0.1\operatorname{MSE}(I_\theta,I_{\mathrm{clean}}).
$$

LPIPS dùng AlexNet version0.1. RGB8 dùng straight-through estimator để truyền gradient qua lượng tử hóa. Prefix sampler được sinh lại bằng weights hiện tại trong `no_grad`;10 calls cuối giữ graph và dùng gradient checkpointing. PNDM history đi xuyên suốt, không reset ở ranh giới prefix/tail.

Thiết kế này áp dụng ý tưởng truncated sampler backprop của [DRaFT-K](https://arxiv.org/abs/2309.17400). Gradient có bias so với full-trajectory gradient; chưa chứng minh10 calls là lựa chọn tối ưu cho watermark. Không dùng prefix cache lấy từ model cũ.

### 3.3. Tách và cân bằng gradient, neo score vào k1 gốc

Gọi \(g_0=\nabla L_{\mathrm{orig}}\), \(g_Q=\nabla L_Q\), \(g_S=\nabla(-s(I_\theta))\), trong đó \(s\) là detector score với carrier khóa. Quality gradient được cân bằng riêng:

$$
\alpha_Q=\min\left(100,\frac{\|g_0\|}{\|g_Q\|}\right).
$$

Code xử lý gradient gần0 bằng epsilon/zero guard. Hệ số này hướng norm quality về mức1 lần norm original, giới hạn khuếch đại100. Đây là khác biệt thực tế so với nhánh `quality` cũ, nơi tỷ lệ norm trung vị chỉ khoảng0,345% gradient gốc.

Score target của mỗi training sample:

$$
s_{\mathrm{target}}=\max\left(s(I_{\mathrm{k1,baseline}}),\tau_{\mathrm{teacher}}+\sigma_{\mathrm{clean}}\right).
$$

Ảnh k1 baseline dùng cùng prompt/seed. Nếu score mới dưới target, thêm hướng sửa score với hệ số \(\alpha_S=\min(100,0.25\|g_0\|/\|g_S\|)\); nếu đã đạt target thì \(\alpha_S=0\). Không tiếp tục thưởng score cao hơn target.

Gradient đưa vào AdamW là \(g_0+\alpha_Qg_Q+\alpha_Sg_S\), sau global norm clipping1. Trong nhánh projected, ràng buộc hướng score bên dưới vẫn áp dụng ở auxiliary updates ngay cả khi score repair bằng0.

### 3.4. Chiếu bước thay đổi tham số sau AdamW

AdamW đề xuất displacement \(\Delta_A\). Trên các auxiliary updates, code tìm bước gần nhất thỏa ba điều kiện:

$$
\begin{aligned}
\Delta^*&=\arg\min_\Delta\frac12\|\Delta-\Delta_A\|_2^2,\\
g_0^T\Delta&\le0,\qquad g_S^T\Delta\le0,\qquad g_Q^T\Delta\le0.
\end{aligned}
$$

Theo xấp xỉ bậc nhất trên mẫu đang xét, chúng yêu cầu không tăng original loss, không giảm watermark score và không tăng quality loss. **Phép chiếu áp dụng lên actual AdamW displacement**, bao gồm ảnh hưởng momentum, preconditioning và weight decay. Nếu đề xuất đã khả thi thì giữ nguyên; nếu xung đột thì điều chỉnh hướng bước đó.

QP được giải bằng enumeration tối đa8 active sets, với Gram matrix nhỏ tính FP64. Adam giữ moments của bước đề xuất; không thay thành moments của displacement đã chiếu. Log ghi cả displacement sau QP và displacement thực lưu sau rounding FP32. Update không có auxiliary dùng AdamW bình thường.

Thiết kế liên quan đến gradient surgery và tối ưu có ràng buộc như [PCGrad](https://arxiv.org/abs/2001.06782) và [GEM](https://arxiv.org/abs/1706.08840). Đây là adaptation cho teacher watermark hiện tại, không phải tái lập nguyên bản hoặc bằng chứng một phép chiếu hoàn toàn mới.

**Ràng buộc chỉ bảo vệ tuyến tính hóa cục bộ**, với score/quality gradient qua10 calls cuối. Nó không bảo đảm nonlinear loss sau update, mọi prompt hay student radioactivity; bước cập nhật còn có thể bị thu nhỏ khi mục tiêu đối nghịch.

Code: [refinement](../../threshold_quality_v1/pareto_quality_refine.py), [QP/AdamW](../../threshold_quality_v1/pareto_step.py), [campaign](../../threshold_quality_v1/pareto_quality_campaign.py). Các kiểm thử gồm QP đối chiếu SciPy, native forward, checkpointed gradients, optimizer state; actual-model GPU smoke đã pass trước train.

## 4. Đối chứng và định nghĩa metric

| Nhánh | Ý nghĩa |
|---|---|
| Baseline k1 | Teacher checkpoint3000,scale1, chưa refine thêm |
| Continuation | Cùng khởi tạo, thêm600teacher updates chỉ bằng loss gốc |
| Native full | Thiết kế ảnh cuối trước đây: full sampler gradient, auxiliary mỗi5updates,cap25% |
| Projected native | Ảnh cuối, gradient balancing, score floor và chiếu actual AdamW step |
| Balanced native | Giống cấu hình projected mới nhưng tắt phép chiếu; ablation trực tiếp của projection |

Continuation và Projected native cùng **số teacher updates**, không cùng GPU compute: native generation/backward làm nhánh cải tiến tốn thời gian hơn. Student chính của mỗi nhánh train độc lập7500updates từ clean SD1.4.

Teacher validation LPIPS so với clean SD2.1 cùng prompt/seed. Student test LPIPS so với **SD1.4 chưa fine-tune** cùng generation prompt/seed. FID/KID ở các bảng mới dùng1.000 clean teacher test images làm reference, không phải ảnh COCO thật. Vì reference khác nhau, không so trực tiếp trị tuyệt đối teacher LPIPS và student LPIPS. CLIP theo utility hiện tại là trung bình \(2.5\max(\cos(\mathrm{image},\mathrm{text}),0)\).

## 5. Kết quả

### 5.1. Teacher

Trên1.000 validation images, Projected native: **detection994/1000=99,4%**, mean score1,362220, paired LPIPS **0,371757**. Continuation đạt cùng99,4% detection, mean score1,444780 và LPIPS0,430804. LPIPS giảm **13,71%** so đối chứng cùng600updates, nhưng mean score giảm; giữ teacher detection không tự chứng minh giữ student detection.

FID/KID/CLIP đo riêng trên1.000 test images, cùng clean reference đã đối chiếu hashes:

| Teacher | FID ↓ | KID mean ↓ | KID subset std | CLIP ↑ |
|---|---:|---:|---:|---:|
| Baseline k1 | 45,39956 | 0,00212268 | 0,00045623 | 0,782971 |
| Continuation | 45,05869 | 0,00180676 | 0,00040345 | 0,783130 |
| Native full | 45,41688 | 0,00200133 | 0,00042479 | 0,783372 |
| **Projected native** | **41,57596** | **0,00029563** | 0,00020207 | **0,783647** |

So baseline k1, teacher FID giảm **8,42%**, KID mean giảm **86,07%** trong phép đo này. Detection99,4% và LPIPS0,371757 ở trên thuộc **validation**, không phải test detection.

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
| Balanced native | 7.500 | 80,2% | 1,069178 | **0,517300** | **52,79152** | **0,00068993** | 0,784043 |

Projected native7500 được chọn bằng validation denoising MSE0,1478019591; đủ7500applied updates, không skip. Detection792/1000 có Wilson95%CI **76,57–81,60%**. So baseline k1: detection tăng **4,2 điểm phần trăm**, LPIPS giảm **5,13%**, FID giảm **2,00%**. So Continuation, detection thấp hơn3,1điểm phần trăm, đổi lại LPIPS/FID/KID tốt hơn. CLIP thấp hơn baseline; chưa thắng mọi metric.

Mốc10000 là **fixed endpoint** của follow-up sau khi xem kết quả7500, không phải thắng cuộc trong một phép chọn chung7500/10000. Snapshot cũ thiếu optimizer/RNG nên đã replay từ clean initialization cùng config/seed rồi train liên tục đến10000. Replay7500 **không trùng bitwise** bản cũ: max absolute tensor difference0,0004711258. Chưa đo test riêng replay7500, nên chưa tách hết ảnh hưởng thêm2500updates khỏi khác biệt giữa hai lần chạy. Kết quả10000 có detection cao hơn bản7500 gốc2,9điểm, nhưng LPIPS xấu hơn2,09%, FID tăng nhẹ. Full training state đã lưu mỗi2500 ở run này.

**Balanced native tốt hơn Projected native7500 về detection, LPIPS, FID và KID trong run hiện tại; CLIP thấp hơn.** Vì vậy chưa có bằng chứng phép chiếu đem lại lợi ích. Balanced vẫn giữ gradient balancing và score floor; cần ablation tiếp để tách tác dụng riêng. Không gán toàn bộ cải thiện cho projection.

Nguồn: [Projected7500](../../output_artifacts/results/historical_summaries/projected_native_student_final_20261025.json), [Projected10000](../../output_artifacts/results/historical_summaries/projected_native_student10000_final_20261026.json), [replay audit](../../output_artifacts/results/historical_summaries/projected_native_student10000_replay_20261026.json), [Balanced7500](../../output_artifacts/results/historical_summaries/balanced_native_student_final_20261026.json), [Continuation](../../output_artifacts/results/historical_summaries/continuation_student_final_20261025.json).

## 6. Ảnh output teacher và student

Các hình dưới đây là **3 mẫu test đầu tiên**, không chọn theo vẻ đẹp hoặc score. PNG gốc512×512, chưa chỉnh sửa. Trong mỗi nhóm teacher hoặc student, các cột có cùng generation prompt/seed. Teacher và student khác backbone và seed protocol; không xem chúng là cùng một ảnh chỉ thêm/bớt watermark.

| Sample | Prompt |
|---|---|
| 000000 | A battered refrigerator stands open in a dirty room. |
| 000001 | a person cutting a pink cake with pink roses |
| 000002 | A man in a green shirt stands by a girl holding a piece of cake on a plate. |

### 6.1. Teacher: clean → baseline k1 → Projected native

| Sample | Clean SD2.1 teacher | Baseline k1 teacher | Projected native teacher |
|---|---|---|---|
'''

teacher = '../../output_artifacts/previews/projected_native_teacher_preview_20261025'
student = '../../output_artifacts/previews/projected_native_student_preview_20261025'
student10000 = '../../output_artifacts/previews/projected_native_student10000_preview_20261026'
for i in range(3):
    cells = [f'{i:06d}']
    for folder, label in [('clean', 'Clean teacher'), ('baseline_k1', 'Baseline k1 teacher'), ('projected_native', 'Projected native teacher')]:
        cells.append(f'![{label} sample{i:06d}]({teacher}/{folder}/sample_{i:06d}.png)')
    report += '| ' + ' | '.join(cells) + ' |\n'

report += '''
### 6.2. Student: SD1.4 tham chiếu → Projected7500 → Projected10000

| Sample | SD1.4 chưa fine-tune | Projected native student7500 | Projected native student10000 |
|---|---|---|---|
'''
for i in range(3):
    cells = [f'{i:06d}']
    for folder, label in [(student+'/unadapted', 'Unadapted SD1.4'),
                          (student+'/student_projected_native', 'Projected student7500'),
                          (student10000+'/student10000', 'Projected student10000')]:
        cells.append(f'![{label} sample{i:06d}]({folder}/sample_{i:06d}.png)')
    report += '| ' + ' | '.join(cells) + ' |\n'

report += '''
Gallery đầy đủ8mẫu, bấm ảnh để mở PNG gốc:

- [Teacher: Projected / baseline k1 / clean](../../output_artifacts/previews/projected_native_teacher_preview_20261025/index.html).
- [Student7500: Projected / baseline k1 / unadapted](../../output_artifacts/previews/projected_native_student_preview_20261025/index.html).
- [Student10000 / student7500 / unadapted](../../output_artifacts/previews/projected_native_student10000_preview_20261026/index.html).
- [Balanced / Projected7500 / unadapted](../../output_artifacts/previews/balanced_native_student_preview_20261026/index.html).

## 7. Giới hạn và kết luận được dữ liệu hỗ trợ

Projected native tốt hơn baseline k1 ở student detection, LPIPS và FID trong lần chạy hiện tại. Nó không vượt mọi đối chứng: Continuation có detection cao hơn; Balanced có detection/LPIPS/FID tốt hơn ở cùng7500student updates. Kết quả hỗ trợ nghiên cứu tiếp cân bằng chất lượng–radioactivity, chưa chứng minh lợi ích riêng của projection.

- Mỗi cấu hình hiện chỉ có một training seed; test đã được xem nhiều vòng phát triển. Cần nhiều seed và fresh holdout trước claim tổng quát.
- WilsonCI theo ảnh và KID subset std không đo độ bất định giữa các lần train. Chưa dùng chúng để kết luận tương đương/non-inferiority giữa các phương pháp.
- LPIPS đo khoảng cách perceptual với reference đã nêu, không phải mức đẹp tuyệt đối. FID1.000mẫu có sample bias; KID mean có thể âm gần0. Các metric không xác nhận hết lỗi mặt/tay.
- Chưa có human preference/defect ratings. Gallery minh họa không thay đánh giá mù trên panel được khóa trước.
- Bit accuracy100% trong summary là dấu tọa độ trung bình toàn tập, không phải detection100% hay độ chính xác bit trên từng ảnh.
- Clean-trained student control có FPR0,2% trong run đã đo nhưng vẫn có drift màu; không quy mọi artifact cho watermark. FPR calibration mục tiêu1% không mặc nhiên là FPR thực nghiệm trên mọi clean model.
- Teacher score/detection và local projected constraints không bảo đảm student radioactivity; phải đo student được train độc lập.

Phân tích bổ sung: [chiến lược/ablation](../../threshold_quality_v1/docs/pareto_quality_strategy_20261025.md), [native full](../../threshold_quality_v1/docs/native_quality_research_20261025.md), [learning-channel audit](../../threshold_quality_v1/docs/transfer_research_20261024.md).
'''

report = report.replace(r'\(', '$').replace(r'\)', '$')
for old, new in {
    'mục tiêu1%': 'mục tiêu 1%', 'kappa2': 'kappa 2', 'kappa1': 'kappa 1',
    'checkpoint3000': 'checkpoint 3.000', 'scale1': 'scale 1',
    '600updates': '600 updates', '200updates': '200 updates',
    'mỗi2updates': 'mỗi 2 updates', 'Cứ2updates': 'Cứ 2 updates',
    'batch4': 'batch 4', 'rank32': 'rank 32', 'learning rate1e-5': 'learning rate 1e-5',
    'LR1e-4constant': 'LR 1e-4 constant', 'PNDM50': 'PNDM 50', 'CFG7.5': 'CFG 7.5',
    'thực tế51': 'thực tế 51', 'đủ51': 'đủ 51', 'Chỉ10': 'Chỉ 10',
    ';10 calls': '; 10 calls', 'qua10 calls': 'qua 10 calls', 'minh10 calls': 'minh 10 calls',
    '1.000validation': '1.000 validation', '1.000test': '1.000 test',
    '5.000train': '5.000 train', 'Teacher0,': 'Teacher 0,', 'student0,': 'student 0,',
    'mỗi5updates': 'mỗi 5 updates', 'cap25%': 'cap 25%',
    'độc lập7500updates': 'độc lập 7.500 updates', '1.000 clean': '1.000 clean',
    'Trên1.000': 'Trên 1.000', 'cùng99,4%': 'cùng 99,4%',
    'detection994/1000': 'detection 994/1.000',
    'LPIPS0,': 'LPIPS 0,', 'score1,': 'score 1,',
    'subset std': 'subset std', 'mean score1,': 'mean score 1,',
    'detection99,4%': 'detection 99,4%', '0,345%': '0,345%',
    'mức1 lần': 'mức 1 lần', 'đại100': 'đại 100', 'gradient gần0': 'gradient gần 0',
    'repair bằng0': 'repair bằng 0', 'clipping1': 'clipping 1',
    'đa8 active sets': 'đa 8 active sets',
    'native7500': 'native 7.500', 'MSE0,': 'MSE 0,',
    'đủ7500applied updates': 'đủ 7.500 applied updates',
    'Detection792/1000': 'Detection 792/1.000', 'Wilson95%CI': 'Wilson 95% CI',
    'hơn3,1điểm': 'hơn 3,1 điểm', 'Mốc10000': 'Mốc 10.000',
    'quả7500': 'quả 7.500', 'chung7500/10000': 'chung 7.500/10.000',
    'đến10000': 'đến 10.000', 'Replay7500': 'Replay 7.500',
    'replay7500': 'replay 7.500', 'thêm2500updates': 'thêm 2.500 updates',
    'quả10000': 'quả 10.000', 'bản7500': 'bản 7.500',
    'gốc2,9điểm': 'gốc 2,9 điểm', 'mỗi2500': 'mỗi 2.500',
    'cùng7500student updates': 'cùng 7.500 student updates',
    'PNG gốc512': 'PNG gốc 512', 'đủ8mẫu': 'đủ 8 mẫu',
    'FID1.000mẫu': 'FID 1.000 mẫu', 'gần0': 'gần 0',
    'accuracy100%': 'accuracy 100%', 'detection100%': 'detection 100%',
    'FPR0,2%': 'FPR 0,2%',
}.items():
    report = report.replace(old, new)
TARGET.write_text(report, encoding='utf-8')
print(f'Updated {TARGET}: {len(report.splitlines())} lines')
