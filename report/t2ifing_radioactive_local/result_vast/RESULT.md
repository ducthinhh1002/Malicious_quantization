# Tổng hợp kết quả semantic fingerprint: SD2.1 → SD1.4

## Balanced_native hoàn tất — kiểm tra06:49 ngày26/10/2026

Campaign hoàn tất06:30ICT, đủ stages00–15. Student selected7500, đủ7500applied
updates, skipped0. Nguồn:
[balanced student summary](../../output_artifacts/results/historical_summaries/balanced_native_student_final_20261026.json).

| Student | Detection ↑ | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline k1 | 75,0% | 0,550447 | 54,44424 | 0,00166651 | 0,787681 |
| Projected native7500 | 79,2% | 0,522219 | 53,35774 | 0,00079496 | 0,785335 |
| **Balanced native7500** | **80,2%** | **0,517300** | **52,79152** | **0,00068993** | 0,784043 |
| Projected native10000 rerun | 82,1% | 0,533119 | 53,45343 | 0,00077322 | 0,787516 |

Balanced_native so projected_native cùng7500: detection+1,0điểm phần trăm,
LPIPS/FID/KID thấp hơn nhưng CLIP cũng thấp hơn0,001292. Trong run1seed này,
nhánh không chiếu tốt hơn nhánh chiếu trên detection/LPIPS/FID/KID. Chưa có
bằng chứng projection đem lại lợi ích; không thể dùng kết quả hiện tại để
nhận thành công là do projection. Balanced vẫn có gradient balancing và score
floor, nên chưa tách tác dụng riêng hai thành phần đó. Cần nhiều seed và
fresh holdout trước kết luận tổng quát. Mean score1,069178, Wilson95%CI của
TPR77,62–82,55%; CLIP và đánh giá người xem vẫn cần xét khi bàn chất lượng.

## Student10000 hoàn tất; balanced_native đang test — 06:19 ngày 26/10/2026

Luồng10000 hoàn tất toàn bộ evaluation lúc01:40ICT26/10 và tự khởi động
balanced_native, supervisor2030603. Nguồn:
[student10000 final](../../output_artifacts/results/historical_summaries/projected_native_student10000_final_20261026.json),
[replay audit](../../output_artifacts/results/historical_summaries/projected_native_student10000_replay_20261026.json).

| Student | Detection ↑ | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline k1 scale1,7500 | 75,0% | 0,550447 | 54,44424 | 0,00166651 | 0,787681 |
| Projected native,7500 gốc | 79,2% | 0,522219 | 53,35774 | 0,00079496 | 0,785335 |
| Projected native,10000 rerun | **82,1%** | 0,533119 | 53,45343 | 0,00077322 | 0,787516 |
| Continuation teacher,student7500 | 82,3% | 0,550218 | 54,88688 | 0,00141482 | 0,786429 |

10000 applied updates, skipped0; fixed10000 endpoint, validation MSE0,14775229.
So bản7500 gốc: detection tăng2,9điểm phần trăm nhưng LPIPS xấu hơn2,09% và
FID tăng0,09569. So baseline k1, detection/LPIPS/FID đều tốt hơn trong lần đo
này. So continuation, detection gần nhau82,1%vs82,3%, LPIPS/FID tốt hơn;
chưa suy ra tương đương detection về mặt thống kê.

Replay7500 **không trùng bitwise** bản gốc: max absolute tensor difference
0,0004711258. Do snapshot cũ thiếu optimizer/RNG,10000 là cùng-config seeded
rerun với optimizer liên tục từ đầu, không phải exact resume của bản7500 gốc.
Chưa đánh giá test riêng replay7500 nên chưa tách hết ảnh hưởng rerun và thêm
2500updates. Giữ nguyên giới hạn1seed/adaptive reused test; không quy mọi
chênh lệch cho riêng tăng số bước.

Balanced_native đã xong teacher,gen dữ liệu,student7500 vàvalidation; lúc06:19
đang sinh test576/1000, chưa có student summary cuối. Teacher validation:
detection99,3%, LPIPS0,3710402. GPU100%util, disk19GiB free; chưa thấy lỗi.

## Ưu tiên student10000 trước balanced_native — 23:59 ngày 25/10/2026

Theo yêu cầu, đã dừng supervisor1696895 lúc balanced_native đang chờ tài
nguyên ở01_teacher, không có child process; teacher balanced chưa train.
Scheduling revision lưu trong pareto_quality_20261025/student10000_scheduling_revision.json.
Supervisor mới **1976920** đã chạy student extension tại
`/root/logs/threshold_quality_20261022/projected_student10000_20261025/student10000`.
Lúc23:59 đã train60/10000, GPU99%util; chưa có kết quả10000.

Snapshot student cũ chỉ có adapter weights, không lưu optimizer/scaler/RNG.
Vì vậy không reset Adam từ7500 rồi gọi là exact resume: chạy lại cùng frozen
upstream trainer, teacher PNG, clean SD1.4 initialization, seed0, batch4,
LR1e-4constant đến10000 trong một optimizer trajectory liên tục. Driver sẽ
đối chiếu tensor snapshot7500 với bản cũ và ghi replay_7500.json; nếu không
trùng, báo rõ là seeded rerun có numerical divergence. Chưa tới mốc kiểm tra.
Các snapshot/kết quả7500 cũ giữ nguyên. Full Accelerate state checkpoints
được bật mỗi2500 để lưu model/optimizer/scaler/RNG cho lần sau.

Đánh giá cố định endpoint10000 bằng cùng1000validation/1000test và ngưỡng cũ,
LPIPS/FID/KID/CLIP/detection, so trực tiếp với7500 hiện tại; không chọn endpoint
bằng test. Đây là follow-up đã xem test, không phải fresh confirmatory holdout.
Sau khi hoàn tất toàn bộ train/evaluation10000, controller tự --launch lại
pareto_quality_campaign.py để balanced_native chạy tiếp; code/manifest cũ
không sửa. Nếu extension lỗi, balanced vẫn chờ để kiểm tra nguyên nhân.

Code mới: [campaign](../../code_best_pipeline/shared/threshold_quality_v1_source/student10000_campaign.py),
[replay driver](../../code_best_pipeline/shared/threshold_quality_v1_source/student10000_replay_driver.py).
Preflight parse command qua frozen trainer, kiểm tra hyperparameters,
checkpoint interval và integration của7500 comparison hook đã pass trênVM4;
actual GPU training đã khởi động. 7500 tốt nhất trong các mốc2500/5000/7500
không bảo đảm10000 sẽ tốt hơn: cần xem learning curve/detection/quality thật.

## Student projected_native đã có kết quả cuối — 23:50 ngày 25/10/2026

Nguồn: [student final summary](../../output_artifacts/results/historical_summaries/projected_native_student_final_20261025.json).
Chọn checkpoint7500 bằng validation denoising MSE0,1478019591; đủ7500applied
updates, không skip. Test1000ảnh, ngưỡng student cố định0,8541307449.

| Student | Detection ↑ | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline k1 scale1 | 75,0% | 0,550447 | 54,44424 | 0,00166651 | 0,787681 |
| Continuation-only | 82,3% | 0,550218 | 54,88688 | 0,00141482 | 0,786429 |
| Native full | 80,5% | 0,544096 | 54,60549 | 0,00126494 | 0,786889 |
| Projected native | **79,2%** | **0,522219** | **53,35774** | **0,00079496** | 0,785335 |

Projected_native lần này tốt hơn baseline k1 đồng thời ở detection(+4,2điểm
phần trăm), LPIPS(-5,13%) và FID(-2,00%); KID giảm52,30%. CLIP giảm0,002346.
So continuation cùng ngân sách teacher, detection thấp hơn3,1điểm phần trăm,
đổi lại LPIPS/FID/KID tốt hơn. So quality có detection80,8%, projected_native
vẫn thấp hơn1,6điểm phần trăm. Chưa gọi đây là thắng mọi phương án/mọi metric.
Mean detector score1,064043; detected792/1000, Wilson95%CI76,57–81,60%.
Bit accuracy100% là dấu tọa độ trung bình toàn tập, không phải detection100%.

Đây là kết quả1seed trên test đã được xem nhiều vòng; cần fresh holdout và
nhiều seed cho claim tổng quát. LPIPS so unadapted SD1.4, FID/KID so clean
teacher; chưa có human ratings để kết luận hết lỗi hình thể.
Lúc23:50 supervisor1696895 đang15_visual_audit; balanced_native chờ phía sau.

## Teacher projected_native FID/KID đã đo — 22:53 ngày 25/10/2026

Đo ngay trên1.000 teacher test PNG đã có, dùng utility.py nguyên bản, cùng
1.000 clean teacher reference (đã đối chiếu hashes), cùng cấu hình FID/KID.
Kết quả được ghi vào utility.json để stage13 sau này tái dùng theo hash.
Nguồn: [teacher FID/KID comparison](../../output_artifacts/results/historical_summaries/projected_native_teacher_fid_kid_20261025.json).

| Teacher | FID ↓ | KID mean ↓ | KID subset std | CLIP ↑ |
|---|---:|---:|---:|---:|
| Baseline k=1 | 45,39956 | 0,00212268 | 0,00045623 | 0,782971 |
| Continuation-only | 45,05869 | 0,00180676 | 0,00040345 | 0,783130 |
| Native full | 45,41688 | 0,00200133 | 0,00042479 | 0,783372 |
| Projected native | **41,57596** | **0,00029563** | 0,00020207 | **0,783647** |

So baseline k1: FID giảm8,42%, KID mean giảm86,07%. Đây là cải thiện phân bố
teacher so reference clean trong phép đo này, chưa phải kết quả student hoặc
bằng chứng hết lỗi mặt/tay. KID std là độ lệch qua subsets, không phải confidence
interval qua training seeds. Detection99,4%/LPIPS0,371757 ở mục dưới đo trên
validation, không gộp thành số liệu test detection chưa được kiểm tra.

## Projected native có teacher validation; student train xong — 22:46 ngày 25/10/2026

Nguồn mới: [teacher validation](../../output_artifacts/results/historical_summaries/projected_native_teacher_validation_20261025.json).
Trên1.000 validation images cùng protocol, projected_native: detection99,4%,
LPIPS0,37175684, CLIP0,78079457, mean score1,36221989. Continuation-only:
detection99,4%, LPIPS0,43080435, CLIP0,78012320, mean score1,44478022.
LPIPS teacher giảm13,71% so continuation cùng600updates và giữ tỷ lệ detection,
nhưng mean score giảm5,71%; không suy ra student sẽ giữ detection.
Native_full trước đó teacher detection99,8%, LPIPS0,43548975.

Student projected_native đã train đủ7500updates, applied7500/skipped0;
checkpoints2500/5000/7500 đã lưu. Đang07_validation, sinh176/1000 ảnh cho
checkpoint2500 lúc kiểm tra. Chưa có student selection hoặc test summary;
không nhầm teacher LPIPS với student LPIPS. balanced_native chưa bắt đầu.
Supervisor1696895 còn sống, chưa thấy lỗi; GPU100%util, disk43GiB free.

## Continuation hoàn tất; projected_native đang train — 19:15 ngày 25/10/2026

Đối chứng chỉ train tiếp teacher600updates bằng loss gốc đã hoàn tất toàn bộ
student/evaluation, chọn student7500. Nguồn:
[continuation summary](../../output_artifacts/results/historical_summaries/continuation_student_final_20261025.json).

| Student | Detection | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline k=1 scale1 | 75,0% | 0,550447 | 54,44424 | 0,00166651 | 0,787681 |
| Continuation-only | 82,3% | 0,550218 | 54,88688 | 0,00141482 | 0,786429 |
| Native full | 80,5% | 0,544096 | 54,60549 | 0,00126494 | 0,786889 |

Đối chứng cho thấy chỉ thêm600teacher updates cũng đủ tăng detection trong
run này, còn LPIPS gần như không đổi. Vì vậy chưa có bằng chứng gán lợi ích
detection của các loss phụ cũ cho chính các loss đó. Native full có LPIPS/FID
tốt hơn continuation nhưng detection thấp hơn1,8điểm phần trăm; chưa thắng
đồng thời. Một seed và test đã dùng nhiều lần, chưa có kết luận tổng quát.

Campaign cũ hoàn tất18:52ICT. Supervisor mới1696895 còn sống;
projected_native đã pass actual-model GPU smoke (peak45,416GiB), đang teacher
113/600 lúc19:15. Quality gradient median norm bằng1,0 lần original ở các
auxiliary steps,19/56 lần cần chiếu cập nhật; chưa có teacher validation hoặc
student kết quả để kết luận chất lượng. Checkpoint đầu ở200, hiện chưa tới.
Tốc độ50updates gần nhất10,51giây/update, ước tính còn85phút riêng teacher nếu
tốc độ giữ nguyên; sau đó còn gen dữ liệu/train student/evaluation.
balanced_native tiếp tục chờ sau projected_native. GPU100%util, disk49GiB free.

## Tổng kết và hàng chờ mới — 16:46 ngày 25/10/2026

Chưa có nhánh nào thắng k=1 scale1.0 đồng thời về student detection, LPIPS và
FID. Một số nhánh tăng detection75%→80,5–80,8%, giảm LPIPS nhẹ nhưng FID còn
xấu hơn baseline54,44424. Báo cáo tổng hợp ưu điểm từng hướng, failure analysis,
survey mới và chiến lược:
[pareto_quality_strategy_20261025.md](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/pareto_quality_strategy_20261025.md).
Số liệu kiểm tra trực tiếp: [audit JSON](../../output_artifacts/results/historical_summaries/k1_strategy_audit_20261025.json).

Phát hiện từ logs: quality gradient norm trung vị chỉ0,3453% original, xung
đột267/600updates; native_full auxiliary trung bình5% original qua600updates,
xung đột55/120 lần hoạt động. Đây là bằng chứng cục bộ, chưa quy kết toàn bộ
nguyên nhân ảnh xấu. Continuation-only đang gen ảnh, chưa có student kết quả.

Đã đẩy4file code mới và xếp hàng **projected_native → balanced_native** tại
`/root/logs/threshold_quality_20261022/pareto_quality_20261025`, supervisor
**1696895**. Đang `waiting_for_dependency`, chờ native campaign hoàn tất.
Hai nhánh cùng k1checkpoint3000,600teacher updates; quality gradient tách và
cân bằng norm, native score floor neo vào ảnh k1 cùng prompt/seed. Nhánh đầu
chiếu actual AdamW displacement theo các gradient original/score/quality;
nhánh thứ hai là đối chứng không chiếu. Forward vẫn PNDM50 CFG7.5, chỉ backward
10 calls cuối để giảm compute. Giữ key/carrier/threshold, student rank32conv
ordinary7500/batch4 và5000train/1000val. Không quality stopping gates.

VM4 pass7/7unit tests; actual-model GPU smoke đã xếp trước từng teacher nhưng
**chưa chạy**. Không có kết quả chất lượng mới để kết luận. Constraint chỉ là
tuyến tính hóa theo gradient truncated, không đảm bảo student detection.

## Native full đã hoàn tất — 15:49 ngày 25/10/2026

Student native_full selected7500: detection80,5%, mean score1,065606,
LPIPS0,544096, FID54,60549, KID0,00126494, CLIP0,786889. Nguồn:
[native summary](../../output_artifacts/results/historical_summaries/native_full_student_final_20261025.json).
Đủ7500applied updates, không skip. So quality: LPIPS giảm0,47%, FID giảm
1,22%, detection giảm0,3điểm phần trăm. So scale1.0: LPIPS giảm1,15%, FID
vẫn cao hơn0,161; chưa phải bước tiến mạnh hoặc chứng minh hết lỗi hình thể.
Student clipped-channel fraction8,43% (unadapted3,25%, clean-trained11,52%);
đây là diagnostic màu, không thay đánh giá mắt người. Chưa có human ratings.

Teacher native_full validation: detection99,8%, LPIPS0,435490, CLIP0,780234.
Đối chứng continuation-only đang train teacher465/600; còn gen ảnh và full
student evaluation. Phải chờ đối chứng này để tách tác dụng native loss khỏi
lợi ích của thêm600teacher updates. Campaign supervisor1555733 còn sống.

## Đối chứng sạch hoàn tất và sửa native precision — 25/10/2026

Kiểm tra11:55ICT: clean_control hoàn tất đủ7500updates, selected7500. Tại
ngưỡng student cũ0,8541307449, báo dương2/1000=**0,2%** (false positives vì
đây là clean-trained student), mean score-0,118823; LPIPS0,515589,
FID51,95063, KID0,000775187, CLIP0,785523. Nguồn:
[clean summary](../../output_artifacts/results/historical_summaries/clean_control_student_final_20261025.json).

Color diagnostics: clipped RGB-channel fraction trung bình tăng từ3,252% ở
unadapted lên11,516% ở clean-trained; HSV saturation mean0,34676→0,40671.
Đây là dấu hiệu generic synthetic-finetuning drift về màu/tonal range, không
phải metric xác nhận lỗi anatomy. Không quy toàn bộ lỗi mặt/tay cho watermark.
So quality: clean control có LPIPS/FID tốt hơn, nhưng vẫn khác unadapted rõ.

native_full dừng ở step65 lúc10:40ICT vì nonfinite auxiliary gradient trên
full FP16 backward chain. Chưa có checkpoint200; archived code/proof/log/64
records tại revisions/fp16_before_20261025_fix trước khi sửa. Revision mới
dùng BF16 denoiser và FP32 CFG/scheduler state; giữ sampler50/CFG7.5, seeded
noise values, loss/key/carrier/threshold và ordinary final student. Production
vẫn merged FP16. 5 CPU tests pass trên VM4, actual-model revised smoke pass.
Native branch phải restart từ source vì chưa có full checkpoint; clean results
được giữ nguyên. Resource guard teacher90.000MiB sau smoke đo peak45,4GiB.
Stress test đúng input step65 trên source weights cũng pass (không phải replay
64 updates). Restart lúc11:59ICT, supervisor mới1555733; kiểm tra12:00 đã
train lại được2steps. Chưa có native_full quality/detection evaluation mới.

## Vòng nghiên cứu và chạy mới — 25/10/2026

Đã triển khai `native_quality_20261025`, supervisor PID1456465, launch08:32ICT.
Kiểm tra lúc08:35: clean_control đã tới stage06_train. Thứ tự tiếp theo:
native_full → continuation. Nhánh mới giữ PNDM50/CFG7.5, key/carrier/ngưỡng,
teacher k1 và ordinary rank32+conv student7500/batch4. Teacher refinement600,
checkpoints200/400/600. Clean control tái dùng ảnh sạch hiện có, không gen lại.

Phân tích600 records mỗi nhánh cũ: median norm LPIPS gradient chỉ khoảng
0,34–0,35% original gradient; cap25% không có nghĩa gradient luôn đạt25%.
Old quality view là predicted-x0 low-noise, không phải native final render.
Native_full thay auxiliary đó bằng full-trajectory PNDM LPIPS+pixel+fixed
coherent watermark objective, gradient qua đủ51 denoiser calls của sampler
50steps. Original threshold loss vẫn được giữ. Continuation-only cùng600steps
để tách lợi ích do train tiếp khỏi tác dụng loss mới. Clean-trained student
để tách synthetic-finetuning drift và kiểm tra specificity ở threshold cũ.

4 tests pass trên VM4; actual-model native smoke pass trước launch:
peak45,416GiB, finite gradients và teacher parameters có thay đổi, full51steps.
Đây là technical smoke, chưa có bằng chứng cải thiện chất lượng/detection.
Local Python thiếu diffusers nên sampler tests được xác minh trong môi trường
thực VM4. Không bật scientific stopping gate, không đổi student trainer.
Tài liệu: [survey và phân tích](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/native_quality_research_20261025.md).
Bằng chứng launch: [snapshot](../../output_artifacts/results/historical_summaries/native_quality_launch_20261025.json).

## Kết quả cuối MinhNQN — kiểm tra 08:23 ngày 25/10/2026

Toàn bộ hàng chờ đã hoàn tất lúc 06:30 ICT ngày 25/10. Gradient và CKA đều
đủ 600 teacher steps, 7.500 student updates, không skipped optimizer update;
student checkpoint7500 được chọn bằng validation MSE. Supervisor đã thoát
bình thường sau trạng thái completed, không còn job watermark trong queue này.

| Nhánh student | Detection test ↑ | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Scale1.0 đối chứng | 75,0% | 0,550447 | **54,44424** | 0,00166651 | **0,787681** |
| quality | **80,8%** | 0,546656 | 55,28107 | 0,00176061 | 0,785113 |
| quality_transfer | 80,5% | 0,545900 | 55,08664 | **0,00149638** | 0,786084 |
| quality_gradient | 80,7% | **0,543704** | 55,37659 | 0,00167238 | 0,786408 |
| quality_cka | 80,5% | 0,548790 | 54,93908 | 0,00164210 | 0,784089 |

Gradient có LPIPS thấp nhất trong bốn nhánh refinement, giảm 1,23% so scale1.0
và 0,54% so quality. Detection 80,7% tương đương về mặt số học với quality
80,8% (chênh 1/1.000 ảnh); chưa có nhiều training seeds để xác nhận khác biệt.
FID gradient kém hơn scale1.0. CKA chưa tạo lợi thế rõ. Chưa đạt mục tiêu cải
thiện mạnh chất lượng thị giác hoặc khôi phục detection lịch sử 99,70% (khác
teacher/carrier/protocol). Không kết luận mặt/tay đã hết lỗi chỉ từ LPIPS.

Teacher validation gradient: detection 99,5%, LPIPS 0,429349; CKA: detection
99,6%, LPIPS 0,429798. Đây là metrics khác với student test trong bảng.
Nguồn: [final comparison](../../output_artifacts/results/historical_summaries/minhnqn_final_comparison_20261025.json).
Giới hạn: một training seed, test panel đã được dùng trong nhiều vòng thăm dò,
chưa có clean-trained student control và matched continuation-only teacher.

## Cập nhật VM4 — 24/10/2026

**Kiểm tra lúc 23:17 ICT:** campaign transfer đã hoàn tất lúc 22:54.
Student `quality_transfer`: detection **80,5%**, LPIPS **0,545900**, FID
55,08664, KID 0,00149638, CLIP 0,786084; selected step7500. So `quality`,
detection giảm 0,3 điểm phần trăm (3/1.000 ảnh), LPIPS giảm 0,14%; các utility
khác tốt hơn nhẹ. Chưa có bằng chứng transfer proxy tạo lợi ích đáng kể.
Nguồn: [comparison](../../output_artifacts/results/historical_summaries/transfer_refine_comparison_20261024.json).
Nhánh `quality_gradient` đã pass full-model GPU smoke, đang refine teacher
318/600 bước, checkpoint200 đã tới lịch lưu; `quality_cka` chưa bắt đầu.
Chưa có student metrics của hai nhánh này. Supervisor MinhNQN còn hoạt động.

**Kiểm tra lúc 19:32 ICT:** `quality` đã hoàn thành toàn bộ teacher/student và
test. Student checkpoint7500: detection **80,8%**, mean score 1,080044,
LPIPS **0,546656**, FID 55,28107, KID 0,00176061, CLIP 0,785113. So scale1.0
đối chứng: detection tăng 5,8 điểm phần trăm, LPIPS giảm 0,69%; FID/KID/CLIP
kém hơn nhẹ nên chưa phải cải thiện đồng thời mọi thước đo chất lượng.
Nguồn: [quality student summary](../../output_artifacts/results/historical_summaries/quality_student_final_20261024.json).
`quality_transfer` bị OOM ở step10 lúc 19:08 khi tiến trình khác chiếm 91,55 GiB.
GPU đã trống lúc kiểm tra; khởi động lại supervisor transfer PID 994252 với
143.156 MiB free. Nhánh chưa tới checkpoint200 nên train refinement lại từ
source; giữ nguyên kết quả `quality` đã hoàn tất. Hai nhánh MinhNQN vẫn chờ sau.

**Kiểm tra lúc 18:16 ICT:** nhánh `quality` đã sinh đủ 5.000 train / 1.000
validation / 1.000 test, student đã hoàn thành 7.500/7.500 optimizer updates,
không skipped update. Đang sinh ảnh đánh giá ba student snapshots: step2500
đủ 1.000/1.000 ảnh, step5000 đạt 192/1.000; step7500 chưa tới lượt.
Teacher validation sau refine: detection 99,8% (trước 98,9%), paired LPIPS
0,431452 (trước 0,436507, giảm 1,16%), CLIP 0,779879 (trước 0,778659).
Đây là kết quả teacher; student detection/LPIPS cuối chưa có. `quality_transfer`,
`quality_gradient`, `quality_cka` tiếp tục chờ theo thứ tự. Hai supervisors sống,
GPU 100%; chưa phát hiện lỗi mới sau khi sửa merge.

**Kiểm tra lúc 15:45 ICT:** recovery đã hoàn tất lúc 13:59.
`transfer_refine_20261024/quality` đã refine đủ 600/600 bước, có checkpoints
200/400/600 và final adapter. Tiến trình data ban đầu dừng lúc 14:56 vì fresh
merge subprocess bật mặc định cuDNN TF32. Áp dụng cùng chế độ FP32 đã dùng
trong training/generation: merge max absolute error giảm từ 0,00134474 xuống
0,00000554323, pass nguyên atol/rtol 1e-4. Không đổi checkpoint, loss hay ngưỡng.
Đã khởi động lại tại `02_data` lúc 15:44, supervisor transfer PID 877549;
runtime bootstrap tắt TF32 cho data/merge của hai campaign, không sửa frozen
experiment files. Supervisor MinhNQN mới PID 877579 đang chờ đúng dependency.
`quality_transfer` chưa bắt đầu; `quality_gradient` và `quality_cka` vẫn ở sau.
Chưa có final detection/native quality cho các nhánh refine.

**Hàng chờ mới đã triển khai:** `minhnqn_refine_20261024`, supervisor hiện tại PID
`877579`, chờ sau `transfer_refine_20261024` rồi chạy lần lượt
`quality_gradient` và `quality_cka`. Mỗi nhánh refine teacher k=1 thêm 600 bước,
checkpoint mỗi 200; final student giữ nguyên rank32+conv / 7.500 updates /
batch4 / 5.000 train / 1.000 validation. Hai nhánh bổ sung LPIPS lần lượt bằng
signed gradient matching và patch CKA; giữ nguyên key/carrier/ngưỡng/sampler.
7 tests đã pass trên local và VM4; full-model GPU smoke sẽ tự chạy trước train
khi đến lượt. **Chưa có kết quả chất lượng/detection của hai nhánh này.**
Chi tiết: [minhnqn_queue_20261024.md](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/minhnqn_queue_20261024.md).

Hai student mới dùng **kappa=1**, 5.000 train / 1.000 validation / 1.000 test,
batch 4, 7.500 updates; cùng carrier, sampler và ngưỡng student 0.8541307449.
Checkpoint chọn bằng validation denoising MSE. Run `k1_evolution` chọn vector
toàn 1.0, nên kết quả của nó là đối chứng scale 1.0.

| Student mới | Detection test | Mean score | LPIPS với unadapted SD1.4 | FID | KID | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| k1, scale 1.0 | 75.0% | 1.036625 | 0.550447 | 54.44424 | 0.00166651 | 0.787681 |
| k1, scale 0.85 | 42.7% | 0.805889 | 0.533659 | 54.56305 | 0.00146374 | 0.787436 |
| k1, scale 0.85, 22.500 updates | 52.3% | — | 0.573231 | 54.85790 | 0.00202060 | 0.790023 |
| k1, selective conv restore, 7.500 updates | 57.5% | — | 0.535404 | 54.22966 | 0.00130240 | 0.788586 |
| k1, quality refinement, 7.500 student updates | **80.8%** | 1.080044 | 0.546656 | 55.28107 | 0.00176061 | 0.785113 |
| k1, quality + transfer refinement, 7.500 student updates | 80.5% | 1.071751 | **0.545900** | 55.08664 | 0.00149638 | 0.786084 |

Scale 0.85 giảm LPIPS 3.05% nhưng mất 32.3 điểm phần trăm detection. Chưa đạt
mục tiêu cải thiện đồng thời. FID/KID ở đây lấy clean teacher làm reference;
LPIPS gần unadapted hơn không tự đồng nghĩa chất lượng cảm nhận tốt hơn.

**Mốc lịch sử 99.70% ở mục 1 dùng teacher kappa=2 được chọn**, không phải
student kappa=1. Chưa có checkpoint/carrier/config lịch sử để làm đối chứng
tái lập chính xác; không so trực tiếp các số FID/CLIP khác panel thành claim.

Hai kiểm tra recovery đã hoàn tất: scale .85 với 22.500 updates và selective
scaling `[.85,1,.85,.85,.85,.9]` với 7.500 updates. Long run tăng detection
9,6 điểm phần trăm so với scale .85/7.500, nhưng LPIPS tăng 7,42%. Conv restore
tăng detection 14,8 điểm phần trăm, LPIPS tăng 0,33%; vẫn mất 17,5 điểm phần
trăm detection so với scale 1.0. Chưa có phương án recovery vượt đồng thời
chất lượng và detection của đối chứng scale 1.0. Scale .95 đã hủy.
Nguồn recovery: [comparison JSON](../../output_artifacts/results/historical_summaries/scale085_recovery_final_20261024.json).

Channel audit đã hoàn tất trên 64 validation pairs: VAE SD1.4 giữ khoảng 91%
paired semantic signal ở cả ba nhánh; sau one-step denoising t=300 còn khoảng
34–35%, t=500 còn khoảng 18%. Đây là reconstruction probe bằng unadapted model,
không phải kết quả cuối student và không phải TPR@1%FPR. Nó gợi ý ưu tiên
learnability qua denoising hơn tăng VAE-cycle hoặc tiếp tục giảm đều adapter.

Phân tích, survey nguồn sơ cấp và thiết kế ablation:
[transfer_research_20261024.md](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/transfer_research_20261024.md).
Nguồn số liệu VM4: `kappa1_manojb_5k/students/{k1_scale085,k1_evolution}/result/student_summary.json`.

## 1. Kết quả cũ — threshold_cfg_v1

Run: `run_semantic_fp_20261020_135117_009783`. Các bảng trong mục này thuộc phương pháp continuous-score/threshold CFG cũ, không phải Semantic Sign Fingerprint.

Teacher/student use their respective frozen clean 1% FPR null thresholds.

| Model | Detection | Mean score | Bit accuracy | FID | KID | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| teacher_clean_test | 1.00% | -0.01394 | 37.5% | -2.911e-05 | -0.00077483 | 0.78625 |
| teacher_wm_test | 100.00% | 1.9916 | 100.0% | 79.634 | 0.024161 | 0.77733 |
| unadapted | 0.90% | -0.06103 | 37.5% | 50.286 | -0.00016148 | 0.78217 |
| lora_r32_conv_wm | 99.70% | 1.5117 | 100.0% | 62.676 | 0.0069348 | 0.79334 |
| full_wm | 99.90% | 1.6457 | 100.0% | 75.077 | 0.016586 | 0.78809 |

## Paired teacher validation quality

| Kappa | Selected | Clean CLIP | Fingerprinted CLIP | CLIP drop | Paired LPIPS | Quality pass |
|---|---|---:|---:|---:|---:|---|
| 1 | False | 0.78202 | 0.78587 | -0.0038546 | 0.488 | True |
| 1.5 | False | 0.78202 | 0.77818 | 0.003841 | 0.52049 | True |
| 2 | True | 0.78202 | 0.77279 | 0.0092314 | 0.54189 | True |

Teacher paired quality details: teacher_quality.json / teacher_quality.csv. Student checkpoint curves: checkpoint_metrics.csv.

- One training seed; image confidence intervals do not measure training-seed uncertainty.
- Unadapted baseline contrasts do not isolate generic fine-tuning effects without clean-trained controls.
- Teacher strength uses held-out teacher validation only; students selected solely by denoising MSE.

## 2. Kết quả mới — sau khi chuyển sang Semantic Sign Fingerprint / RSSF (`semantic_sign_v1`)

Run: `run_semantic_fp_20261021_174725_058270`.

Bản sửa theo `semantic_sign_fingerprint_codex_spec.md` chuyển từ continuous-score/threshold CFG sang fingerprint dấu của 32 semantic coordinates, được center bằng train medians. Verifier chính đếm số bit khớp secret và dùng exact binomial tail: với 32 bit, alpha = 0.01, cần ít nhất 24 bit khớp để phát hiện một ảnh. Chỉ số này là **analytic detection**, không phải empirical TPR@1%FPR được calibrate trên clean images.

Teacher vẫn train bằng LoRA rồi merge để sinh ảnh bằng pipeline thông thường. Loss mới gồm preservation, squared hinge trên các bit được chọn bằng clean coordinates, cùng LPIPS và pixel MSE trên ảnh dự đoán x0 tại các timestep cuối. Ba candidate được train độc lập với coordinate margin μ = 0.1, 0.2, 0.3. μ của bản mới không tương đương κ của bản cũ.

### Teacher validation

Số liệu được chép từ output đọc `teacher_fp/candidates.json` trên server mà người dùng đã gửi trong cuộc trò chuyện.

| Coordinate margin μ | Analytic detection rate | CLIP drop | Paired LPIPS | Detection ≥ 90% | CLIP drop ≤ 0.01 | LPIPS ≤ 0.15 | Đạt toàn bộ gate |
|---|---:|---:|---:|---|---|---|---|
| 0.1 | 99.8% | 0.013711 | 0.407089 | Đạt | Không đạt | Không đạt | Không |
| 0.2 | 100.0% | 0.012716 | 0.415702 | Đạt | Không đạt | Không đạt | Không |
| 0.3 | 100.0% | 0.017038 | 0.438035 | Đạt | Không đạt | Không đạt | Không |

Giá trị gốc chưa làm tròn:

```text
mu=0.1 TPR=0.998 CLIP_drop=0.013711029722690586 LPIPS=0.4070892530158162
mu=0.2 TPR=1.0   CLIP_drop=0.01271555155515669  LPIPS=0.4157024439424276
mu=0.3 TPR=1.0   CLIP_drop=0.017037655413150832 LPIPS=0.4380349085479974
```

`TPR` là tên hiển thị của lệnh đọc `metrics.detection_rate`; báo cáo dùng tên analytic detection rate để tránh nhầm với TPR tại empirical clean FPR = 1%.

### Kết luận và trạng thái chạy

- Watermark detection trên teacher validation đạt 99.8–100%, nhưng cả ba candidate đều trượt **cả CLIP drop lẫn LPIPS**. Không candidate nào đủ điều kiện được chọn.
- Chương trình dừng tại `01e_train_native_teacher` với thông báo `No RSSF teacher passed analytic TPR/CLIP/LPIPS gates; students stopped`. Đây là dừng theo quality gate, không phải lỗi OOM trong traceback đã gửi.
- Run mới này chưa đi tiếp sang train/eval student. Kết quả student trong mục 1 thuộc run cũ, không được dùng làm kết quả RSSF.
- Giảm μ từ 0.3 xuống 0.1 giúp LPIPS giảm từ khoảng 0.438 xuống 0.407, nhưng vẫn vượt xa gate 0.15. Chưa có bằng chứng rằng chỉ giảm margin sẽ giải quyết được vấn đề chất lượng.
- LPIPS mới thấp hơn về mặt số học so với teacher validation cũ (0.488–0.54189). Chưa đối chiếu đầy đủ provenance của hai run nên đây chưa phải kết luận so sánh có kiểm soát.

### Nhận định kỹ thuật cần kiểm chứng

Kiểm tra code cục bộ cho thấy LPIPS dùng ảnh RGB trong [-1,1], validation kiểm tra cùng prompt/seed, và loss LPIPS có đường gradient về teacher. Chưa xác định được lỗi tính metric từ thông tin hiện có.

Giả thuyết cần kiểm chứng: loss chất lượng lúc train so sánh ảnh dự đoán x0 từ cùng một latent nhiễu tại một timestep cuối, còn validation so sánh ảnh sinh qua toàn bộ 50 bước từ Gaussian noise. Giữ gần nhau ở phép dự đoán một bước không bảo đảm hai quỹ đạo sinh ảnh hoàn chỉnh vẫn gần nhau. Đây chưa phải nguyên nhân đã được xác nhận.

Cần đối chiếu `lpips`, `pixel_mse`, `fp`, `preserve` trong training logs với native validation LPIPS để phân biệt khoảng cách train/inference với cân bằng loss chưa phù hợp. Chưa nên tự nới quality gate hoặc chạy lại nguyên cấu hình.

### Nguồn dữ liệu và giới hạn lưu trữ

Người dùng không còn truy cập server. Mục kết quả mới được tổng hợp từ số liệu và traceback đã gửi trong cuộc trò chuyện; không giả định artifact của run mới đã được tải về máy cục bộ. Chưa có thêm dữ liệu để ghi FID, KID, absolute CLIP, bit recovery, clean empirical FPR hoặc training loss curves của run mới. Student metrics chưa được tạo vì pipeline đã dừng trước bước train student.

Đường dẫn server trước đây để tra cứu nếu khôi phục được dữ liệu:

```text
/workspace/t2ifing_radioactive/run_semantic_fp_20261021_174725_058270/
  teacher_fp/candidates.json
  logs/01e_train_native_teacher.log
  images/teacher_clean_validation/
  images/candidate_mu_0.1_validation/
  images/candidate_mu_0.2_validation/
  images/candidate_mu_0.3_validation/
```


## 2026-10-26 ? Structure/margin native: nghi?n c?u v? kh?i ch?y

?? audit JSON k?t qu? v? log t?ng native update tr?n VM4. Projected7500 t?t
h?n k1 v? TPR/LPIPS/FID/KID, nh?ng ch?69,3% ?nh gi?m LPIPS; c?145 detection
m?i v?103 detection b? m?t so v?i baseline. Teacher c?n h?t floor ?200/300
native updates. ??y l? ??ng c? s?a c?ch b?o v? margin theo batch v? b? sung
b?o to?n c?u tr?c, ch?a ph?i ch?ng minh nguy?n nh?n duy nh?t.

- [Ph?n t?ch/survey v? c?ng th?c ??y ??](../../code_best_pipeline/shared/threshold_quality_v1_source/docs/structure_margin_research_20261026.md).
- [Audit l?ch s?](../../output_artifacts/results/historical_summaries/structure_margin_prior_audit_20261026.json).
- [B?ng ch?ng kh?i ch?y](../../output_artifacts/results/historical_summaries/structure_margin_launch_20261026.json).
- Code: `structure_margin_campaign.py`, `structure_margin_refine.py`,
  `structure_margin_objectives.py`, `structure_margin_tests.py`, `structure_margin_audit.py`
  trong `threshold_quality_v1/`.

Queue VM4: `/root/logs/threshold_quality_20261022/structure_margin_20261026`.
Th? t? `structure_margin_native` ? `margin_native` (ablation b? MS-SSIM).
C?ng k1 checkpoint3000, teacher600updates, batch4/native microbatch2,
LR1e-5; loss g?c, key/carrier/threshold/sampler gi? nguy?n. Native quality
cho c?4 ?nh m?i2updates; squared per-image score deficit ???c c?n b?ng ri?ng.
Nh?nh ??u th?m0,25*(1-MS-SSIM) v?o LPIPS+0,1pixelMSE. Hai nh?nh ch? kh?c
weight MS-SSIM. Student v?n ordinary SD1.4 rank32 attention+conv,
5.000train/1.000val/1.000test,7500updates batch4, ch?n checkpoint theo valMSE.

14 tests PASS, actual GPU smoke PASS (peak45,42GiB). Supervisor PID2223976.
L?c07:59 ICT26/10: `structure_margin_native/01_teacher` ?? ch?y3/600,
bao g?m native update step2 c? c? quality v? margin gradients h?u h?n,
?? th?c s? c?p nh?t tham s?. GPU t?ng util100% t?i th?i ?i?m ki?m tra,
kh?ng ??ng ngh?a ri?ng process n?y chi?m100%. Ch?a c? metric k?t qu? m?i.
Teacher checkpoint200/400/600 c? optimizer/RNG; student snapshot2500/5000/7500
l?u adapter theo harness g?c. Ch?y ??n cu?i m?i k?t qu? khoa h?c, kh?ng gate
ch?t l??ng/detection. T?i d?ng ?nh reference; merge export t?m ? RAM c? ki?m
ch?ng reload, c?n checkpoint/?nh/result l?u ??a. Kh?ng x?a d? li?u c?.
