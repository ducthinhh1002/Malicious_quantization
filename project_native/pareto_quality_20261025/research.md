# Tổng kết k=1 và chiến lược cải thiện chất lượng — 25/10/2026

## Kết luận hiện tại

Chưa có nhánh nào thắng baseline k=1 đồng thời trên student detection, LPIPS
và FID. Một số nhánh tăng detection và giảm LPIPS nhẹ, nhưng chưa giải quyết
được chất lượng thị giác mà người dùng thấy. Không được gọi các thay đổi nhỏ
đó là thành công toàn diện.

Nguồn kiểm tra trực tiếp VM4 lúc 16:46 ICT:
[audit tổng hợp](../../output_artifacts/k1_strategy_audit_20261025.json).
Các số dưới đây là student test 1.000 ảnh, threshold cố định 0,8541307449,
trừ clean_control được ghi riêng là false positives. LPIPS so ảnh SD1.4 chưa
fine-tune cùng prompt/seed; FID/KID dùng clean teacher làm reference, không
phải ảnh thật. Không so trị tuyệt đối LPIPS teacher với LPIPS student.

| Nhánh | Detection ↑ | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline k=1, scale1.0 | 75,0% | 0,550447 | 54,44424 | 0,00166651 | 0,787681 |
| Scale0.85 | 42,7% | 0,533659 | 54,56305 | 0,00146374 | 0,787436 |
| Scale0.85, student22500 | 52,3% | 0,573231 | 54,85790 | 0,00202060 | 0,790023 |
| Scale0.85 + conv restore | 57,5% | 0,535404 | 54,22966 | 0,00130240 | 0,788586 |
| quality | 80,8% | 0,546656 | 55,28107 | 0,00176061 | 0,785113 |
| quality_transfer | 80,5% | 0,545900 | 55,08664 | 0,00149638 | 0,786084 |
| quality_gradient | 80,7% | 0,543704 | 55,37659 | 0,00167238 | 0,786408 |
| quality_cka | 80,5% | 0,548790 | 54,93908 | 0,00164210 | 0,784089 |
| native_full | 80,5% | 0,544096 | 54,60549 | 0,00126494 | 0,786889 |
| clean_control, không watermark | **FPR 0,2%** | 0,515589 | 51,95063 | 0,00077519 | 0,785523 |

Mốc 99,70% lịch sử trong RESULT.md dùng teacher được chọn kappa2 và bộ
artifact khác; không phải đối chứng kappa1 hiện tại. Chưa phục hồi được đầy
đủ run lịch sử để coi là phép so sánh nhân quả. `k1_evolution` hiện tại có
vector [1,1,1,1,1,1], tức là baseline scale1.0 dù tên chứa evolution.

## Những điều có ích từ từng hướng, và giới hạn đã biết

| Hướng đã thử | Điều thực sự học được | Đưa vào chiến lược mới thế nào |
|---|---|---|
| threshold_cfg_v1 nguyên bản | Coherent signed shift, VAE-cycle và CFG-aware prediction giữ được truyền dấu; có baseline k1 hoàn chỉnh | Khởi tạo đúng checkpoint3000; giữ nguyên loss gốc, key/carrier/threshold/sampler |
| Giảm đồng loạt scale | Teacher LPIPS giảm khoảng9,93%, student LPIPS giảm3,05%, nhưng student mất32,3 điểm detection | Không tiếp tục giảm toàn bộ tín hiệu; tối ưu chất lượng trong khi neo điểm watermark |
| KernelSHAP/group scaling, conv restore | Vai trò các nhóm không đồng đều; conv restore tăng detection42,7→57,5% với LPIPS tăng ít, FID tốt nhất trong các nhánh marked đang xét | Giữ attention+conv rank32; không bỏ conv. Attribution teacher không thay phép đo student |
| Teacher-only Pareto/evolution | Search có thể mô tả trade-off, nhưng teacher TPR gần trần không dự báo student; nghiệm cuối đã quay về all1 | Không đưa thêm một search đắt tiền trên proxy chưa được xác nhận |
| Student train lâu | Detection.85 hồi một phần52,3%, nhưng LPIPS xấu lên.5732 | Giữ7500updates để so công bằng; không dùng thêm epochs như cách chữa mặc định |
| quality | Detection cao nhất hiện có80,8%; thêm loss không phá pipeline | Giữ warm start và loss gốc. Chưa tách được lợi ích loss khỏi thêm600teacher updates |
| Paired learner / quality_transfer | Đã nối được hypergradient qua RGB và ordinary learner; radioactivity phải xét learning channel | Không nâng rank/inner steps một cách mù quáng; kết quả80,5% chưa thắng quality |
| MinhNQN gradient matching | LPIPS.543704 tốt nhất trong nhóm detection≥75%; signed delta gradient giữ thông tin dấu/norm tốt hơn Gram thuần | Giữ nguyên tắc đo gradient riêng và actual update. Không chồng thêm loss proxy chưa chứng minh lợi ích |
| CKA | Đo quan hệ patch/structure được, tránh suy biến global batch1 | Không coi CKA là metric anatomy; chưa đủ lợi ích để đưa tiếp vào loss |
| native_full | Tối ưu ảnh cuối giúp FID/KID tốt hơn các nhánh refine trước; BF16+FP32 ổn định hơn chuỗi FP16 | Tiếp tục dùng ảnh cuối đủ sampler; tách các mục tiêu, giảm chi phí backward |
| clean_control | FPR0,2% cho thấy detector không báo dương đại trà; clean-trained cũng có lệch màu | Không quy mọi artifact cho watermark. Giữ đối chứng, không sửa student rồi gọi là radioactivity tốt hơn |
| Channel audit | VAE giữ~91% chênh score; one-step denoising t300 chỉ còn~35%. MSE đơn lẻ không phân biệt tốt scale1/.85 | Không tối ưu bằng raw denoising MSE như proxy detection; full student vẫn là kết quả quyết định |
| trajectory_budget_v1 cũ | Đã thay nhiều thành phần cùng lúc, teacher detection gần0; loss/dual lớn không chứng minh gradient đúng | Không dùng checkpoint đó, không thay detector, không dùng controller ép quality budget tăng vô hạn |
| Vận hành | Cached clean/marked PNG, full optimizer/RNG checkpoint, strict merge verification và numerical policy đã dùng được | Tái dùng dữ liệu; giữ checkpoint200/400/600, kiểm thử thực tế trước train, không sửa luồng cũ |

Các báo cáo gốc: [recovery](scale085_detection_recovery_20261024.md),
[learning-channel audit](transfer_research_20261024.md),
[MinhNQN](minhnqn_review_20261024.md),
[native full](native_quality_research_20261025.md).

## Vì sao chất lượng gần như đứng yên

1. **Tác động gradient chất lượng rất nhỏ.** Với quality, tỷ lệ norm trung vị
   0,3453%, trung bình1,7696% so gradient gốc. Trần25% không có nghĩa là
   thường xuyên đạt25%. Quality_gradient/CKA cũng khoảng0,34% trung vị.
2. **Có xung đột thật trên gradient đã ghi log.** Quality có267/600 bước dot
   âm với gradient gốc; native_full có55/120 auxiliary steps. Đây là bằng
   chứng xung đột cục bộ, chưa phải chứng minh nguyên nhân duy nhất của ảnh xấu.
3. **Native full gộp các mục tiêu.** Auxiliary chỉ chạy120/600 bước, bị cap25%
   nên trung bình norm được thêm chỉ tương đương5% qua toàn bộ600updates.
   Loss chất lượng và semantic chưa có norm riêng; không được khẳng định
   semantic chi phối nếu chưa đo riêng.
4. **Sai khác view tối ưu.** One-step predicted-x0 LPIPS thấp không có nghĩa
   ảnh Gaussian→PNDM50 có LPIPS thấp. Nhánh mới đo loss trên ảnh cuối thật.
5. **Teacher detection bão hòa che mất margin.** Scale.85 làm teacher p10
   margin mất khoảng77%, rồi student detection tụt mạnh. Vì vậy không chỉ
   giữ một tỷ lệ TPR teacher mà phải giữ điểm/margin tham chiếu.
6. **Thiếu đối chứng ngân sách ở các lần cũ.** Các refine đều~80,5–80,8% có
   thể một phần do thêm600updates. Continuation-only đã xong teacher và đang
   gen ảnh; phải chờ student để phân biệt nguyên nhân, không nhận hết công lao
   cho loss mới.
7. **Mục tiêu đo chưa bao quát thị giác.** LPIPS đo độ lệch paired reference;
   FID là phân bố và chỉ có1.000 ảnh; CLIP không kiểm tra bàn tay/khuôn mặt.
   Cần xem panel cố định, không dùng các metric này như chứng nhận ảnh đẹp.

## Survey bổ sung và lựa chọn thuật toán

[PCGrad, Yu et al., NeurIPS2020](https://arxiv.org/html/2001.06782v4)
phân tích gradient xung đột và chiếu thành phần gây xung đột. Nó gợi ý cách
tách mục tiêu chất lượng khỏi watermark. Ta dùng biến thể riêng: giải phép
chiếu lên giao ba nửa không gian của **actual AdamW displacement**, vì
preconditioning/momentum làm raw gradient khác bước cập nhật. Không gọi đây
là tái lập PCGrad, và không chuyển bảo đảm lý thuyết của paper sang model này.

[CAGrad, Liu et al., NeurIPS2021](https://arxiv.org/abs/2110.14048)
xét cải thiện cục bộ của các task khi tối ưu loss chung. Bài học áp dụng là
phải đo xung đột và mức tác động từng mục tiêu; một tổng loss với vài weight
cố định không mặc nhiên cân bằng. Code mới không triển khai CAGrad nguyên bản.

[DRaFT, Clark et al.](https://arxiv.org/html/2309.17400v2) tối ưu reward trên
ảnh cuối. DRaFT-K giữ đầy đủ forward nhưng cắt backward về K bước cuối nhằm
giảm compute; paper ghi nhận lợi ích trong bài toán reward của họ. Ta chọn
trước10 denoiser calls cuối làm giả thuyết chi phí/ổn định, không nói10 là tối
ưu cho watermark. Forward vẫn51 calls của PNDM50, prefix được sinh lại với
weights hiện tại. Test xác nhận forward khớp full version và gradient
checkpointing khớp bản không checkpoint. Chưa có cosine full-vs-tail trên
real model nên chưa tuyên bố gradient tương đương; gradient này có bias.

[SIREN](https://arxiv.org/html/2410.10437v1) tách learnability và perceptual
constraint. Điều liên quan là giữ kiểm chứng ở student học từ RGB. Ta không
đổi sang coating/reader của paper, không đưa secret/key vào student, và không
lấy teacher score làm chứng minh khả năng truyền dấu.

Những hướng chưa chọn vòng này: DPO/ImageReward cần kiểm tra preference model
và reward hacking; CFG rescale thay sampler đã khóa; tăng epochs đã có trade-off
xấu; search trên teacher-only fitness đã không đủ. Nếu tiếp tục thất bại,
cần thử học-channel surrogate đã được kiểm chứng hoặc một nhánh thay protocol
có đối chứng riêng, không tiếp tục thêm loss tùy tiện vào cùng run.

## Hai nhánh đã code và đưa vào hàng chờ

Root: `/root/logs/threshold_quality_20261022/pareto_quality_20261025`.
Thứ tự: **projected_native → balanced_native**; phụ thuộc campaign
`native_quality_20261025` hoàn tất cả continuation.

Cả hai warm-start cùng kappa1 checkpoint3000, scale1, teacher600updates,
LR1e-5, batch4. Giữ nguyên threshold_cfg_v1 ở mọi bước. Cứ2updates lấy một
training prompt/seed, sinh đủ PNDM50 CFG7.5 và RGB8 STE, backward10 calls cuối.

- Quality: `LPIPS(marked, paired_clean) + 0.1 * pixel_MSE`.
- Tách gradient quality khỏi gradient detector; điều chỉnh quality norm về
  tối đa1 lần norm gradient gốc, hệ số khuếch đại tối đa100 để tránh phóng đại
  gradient gần0. So với nhánh cũ, đây là thay đổi lực tối ưu có đo đạc.
- Score floor tham chiếu:
  `max(score ảnh k1 gốc cùng prompt/seed, tau_teacher + sigma_clean)`.
  Nếu dưới floor, thêm gradient repair của negative score với norm tối đa
  0,25 lần original; không tiếp tục thưởng score cao hơn floor. Các gradient
  được ghép trước global norm clip1 như harness cũ.
- `balanced_native`: AdamW dùng tổ hợp gradient đó, không chiếu bước cập nhật.
- `projected_native`: sau AdamW, giải QP gần bước đề xuất nhất với ba điều kiện
  `g_original·delta <= 0`, `g_negative_score·delta <= 0`, `g_quality·delta <= 0`.
  Active-set enumeration tối đa8 tập, FP64 Gram nhỏ, giữ proposal moments của
  Adam; checkpoint lưu đầy đủ state. Log cả projected delta và actual stored
  delta sau làm tròn FP32. Không rollback/reject theo metric chất lượng.

Các ràng buộc này chỉ nói về **tuyến tính hóa cục bộ**. Score/quality dùng
gradient truncated; không đảm bảo nonlinear loss, forward generation sau
update, mọi prompt, hay detection của student. Có thể làm bước gần0 khi các
mục tiêu đối nghịch; log norm cho thấy điều đó. Nhánh đối chứng cùng compute
cho biết phép chiếu có giúp hay chỉ cản học. Vòng này thay nhiều chi tiết so
native_full nên không dùng comparison đó để quy riêng tác dụng K10/balancing;
chỉ hai nhánh mới khác đúng công tắc projection.

Student giữ SD1.4, rank32 attention+conv,5.000train/1.000val,7500updates,
batch4; chọn checkpoint2500/5000/7500 bằng validation denoising MSE như trước.
Test1.000 ảnh, fixed thresholds, LPIPS/FID/KID/CLIP và visual diagnostics.
Không dừng vì chất lượng/detection kém. Teacher checkpoint mỗi200updates.

## Kiểm thử, triển khai và cách kết luận

Code: [optimizer](../pareto_step.py), [refine](../pareto_quality_refine.py),
[queue](../pareto_quality_campaign.py), [tests](../pareto_quality_tests.py).

- Local: biên dịch4files thành công,5 optimizer/QP tests pass.
- VM4: **7/7 tests pass**, gồm35 QP ngẫu nhiên đối chiếu độc lập SciPy;
  dependent/opposed constraints; momentum/weight-decay case; resume Adam;
  native forward51 calls; last10 backward; checkpointing equivalence;
  ablation khác đúng một công tắc.
- Freeze manifest9files, hash source adapter, baseline image provenance,
  thresholds và calibration. Verify prompt/seed baseline trùng; kiểm tra
  toàn bộ hash PNG clean/baseline trước teacher. Các code cũ không sửa.
- Supervisor **1696895**, queued16:46ICT. Trạng thái ban đầu
  `waiting_for_dependency`; continuation đang02_data. GPU smoke thực tế của
  từng nhánh **chưa chạy**, được xếp bắt buộc trước teacher và phải pass mới
  train. Không báo unit test như kết quả chất lượng hoặc GPU smoke.
- Tái dùng ảnh clean/unadapted/baseline bằng hardlink. Không chạy heavy jobs
  chồng campaign trước; chờ90.000MiB free trước smoke/teacher,80.000MiB ở stage
  khác. Disk guard18GiB trước gen; không xóa thêm checkpoint/ảnh cũ.

Mốc kiểm chứng thực dụng: đồng thời detection≥75%, LPIPS<0,550447 và
FID<54,44424 là vượt baseline ở ba metric đã định; mục tiêu mạnh hơn là giữ
≥80,8% và giảm LPIPS ít nhất5–10% so baseline (~0,523–0,495), cùng cải thiện
panel thị giác. Đây là mục tiêu thử nghiệm, không ngưỡng ảnh đẹp phổ quát hay
cam kết thành công. KID/CLIP vẫn phải báo để thấy trade-off.

Panel200 ảnh cố định có A/B randomization nhưng chưa có human ratings, và
paths trong JSON có thể lộ branch; cần ẩn paths khi chấm mù thật. Test đã xem
nhiều lần,1seed; chênh nhỏ cần paired uncertainty và cuối cùng fresh holdout,
nhiều training seeds trước tuyên bố tổng quát. Chưa có kết quả nhánh mới.
