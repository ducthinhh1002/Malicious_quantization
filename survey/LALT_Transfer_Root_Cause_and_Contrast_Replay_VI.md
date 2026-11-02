# LALT: nguyên nhân transfer yếu và thử nghiệm contrast/replay

Ngày 21/10/2026. Đây là phân tích từ run thật và kế hoạch thử nghiệm, chưa có bằng chứng phương pháp mới cải thiện transfer.

## 1. Kết quả cần giải thích

Sau 5.000 bước ordinary attention LoRA rank 4 trên 2.048 PNG/caption:

| Chỉ số trên cùng 128 query | Teacher 20k cũ | Sau repair + meta |
|---|---:|---:|
| Teacher TPR, single-key, nominal FPR 1% | 77,34% | 84,38% |
| Student TPR, single-key, nominal FPR 1% | 3,91% | 3,91% |
| Student contrast AUROC | 0,7314 | 0,7403 |
| Student contrast TPR | 5/128 | 5/128 |
| Clean contrast detections | 2/128 | 2/128 |

Clean outputs, ngưỡng và query tương ứng đã được đối chiếu. Chỉ một seed huấn luyện, 128 query; đây là development evidence. Không lấy bootstrap ảnh làm số model độc lập.

Nguồn: [decision](../server_review/lalt_repair/lalt_repair_20261020_135310Z/decision.json), [audit định lượng](../server_review/lalt_root_cause_20261021/analysis.json).

## 2. Những vấn đề xác định được

**Mục tiêu transfer chưa tập trung đủ vào nhận đúng owner.** Student mới tăng score key221 +1,2274 nhưng cũng tăng score key34 +0,9238 so với clean. Phần chênh đúng-key chỉ +0,3036. Trung bình hai score tăng +1,0756; đây là thành phần chung về mặt đại số, không thể quy toàn bộ cho presence head khi chưa đo riêng. Loss cũ vẫn thưởng cho raw target score, nên có một hướng cải thiện loss mà không cải thiện phân biệt owner tương ứng.

**Chuẩn hóa specificity không đúng thang đo của contrast.** Trên calibration probe, std raw score là 4,6584, std contrast chỉ 0,3198, chênh khoảng 14,6 lần. Code dùng raw-score std để chia cả contrast. Điều này nén contrast so với margin cố định. Tỉ lệ trên calibration của meta cần được ghi riêng vì noise namespace khác; không dùng số 14,6 như hằng số phổ quát.

**Tiêu chí chọn checkpoint chỉ yêu cầu tốt hơn một baseline yếu.** Proxy bước 32 được chọn dù worst-horizon effect và specificity vẫn âm. Điều đó chỉ chứng minh cải thiện tương đối của proxy. Chưa phải bằng chứng transfer dương, càng chưa phải bằng chứng đạt ngưỡng FPR chặt.

**Kênh học mô phỏng quá hẹp.** Mỗi outer step dùng một ảnh và cập nhật LoRA 8 lần trên ảnh ấy. Các lần chỉ đổi noise/timestep. Final student lại học 2.048 ảnh trong 5.000 updates. Validation 8/32 updates vẫn chưa kiểm chứng được hành vi cuối quá trình học.

**Teacher detection chi phối hướng thiết kế.** Teacher loss tiếp tục giảm BCE dù teacher đã đủ đọc; gradient transfer chỉ được chứng minh khác 0. Chưa có audit so norm/cosine đầy đủ để kết luận teacher gradient lấn át bao nhiêu. Run mới phải đo việc này, thay vì suy từ độ lớn scalar loss.

**Thay carrier shared/amplitude không phải ablation sạch về learnability.** Hai biến thể bỏ hoặc hạn chế content modulation trong khi giữ nguyên reader cũ; teacher TPR cuối chỉ 41,41% và 51,56%. Chúng bị loại trước student. Kết quả này bác bỏ cách chuyển carrier cụ thể, chưa bác bỏ shared carrier nếu huấn luyện reader phù hợp.

## 3. Điều chưa thể kết luận

VAE round-trip giữ contrast trong kiểm tra 32 ảnh. Điều này chống lại giả thuyết “encoder xóa sạch dấu” trong run này, nhưng không chứng minh denoising training học được dấu.

Phép x0 reconstruction một bước dùng noisy latent bắt nguồn từ ảnh marked. Công thức x0 chứa thành phần trực tiếp của noisy input, nên dấu có thể đi qua ngay cả khi UNet chưa học dấu. Không dùng kết quả này làm bằng chứng radioactivity.

Không quy toàn bộ thất bại cho một bug; cần thay đổi có kiểm soát và kiểm tra lại ordinary student. Tăng số bước teacher hoặc đổi readout đơn thuần chưa giải quyết được kết quả 5/128.

## 4. Research bổ sung và giới hạn áp dụng

- [Radioactive Watermarks in Diffusion and Autoregressive Image Generative Models](https://arxiv.org/html/2506.23731v1), §5.2: tác giả phân biệt mất dấu qua VAE với mất qua học denoising; một số dấu sống qua encode vẫn không truyền sang LDM. Điều này phù hợp việc round-trip tốt chưa đủ trong thí nghiệm của ta. Paper không chứng minh mọi LDM watermark đều bất khả thi.
- [SERUM](https://arxiv.org/html/2603.13396v1), §3 và Appendix I: dấu được đưa vào initial noise; LoRA SD1.4 đạt 52,30% trong thiết lập tác giả, dùng 5.000 ảnh từ SD2.1, batch 8, rank 4, lr 1e-5, chọn bước 19.500 theo validation denoising loss. Khác với 2.048 ảnh/5.000 bước của ta; không so trực tiếp hai tỷ lệ như một ablation. Đây là căn cứ để khảo sát carrier tác động suốt quá trình sinh.
- [Dataset Distillation by Matching Training Trajectories](https://arxiv.org/abs/2203.11932) dùng trạng thái mạng ở nhiều thời điểm để tối ưu dữ liệu; [Embarassingly Simple Dataset Distillation](https://arxiv.org/html/2311.07025v1) khảo sát random truncated BPTT. Chúng gợi ý kiểm tra watermark quanh trạng thái learner đã học, thay vì chỉ gần khởi tạo. Áp dụng cho diffusion watermark ở đây là giả thuyết mới cần kiểm chứng.
- [DistSeal](https://arxiv.org/html/2601.16140v1), §5.3: một phần thí nghiệm distill generative model dùng trực tiếp marked latents; không tương đương student độc lập nhận RGB rồi VAE-encode. Không dùng kết quả đó để bảo đảm transfer của LALT.

## 5. Thử nghiệm tiếp theo: contrast và trạng thái learner đã học

Giữ carrier adaptive, reader cố định và teacher 20k làm khởi tạo chung để tận dụng checkpoint/cache. Không lặp lại bước repair teacher chỉ nhằm tăng detection.

Với hai key bổ sung, định nghĩa:

\[
C_k(x)=s_k(x)-s_{\bar k}(x)
=\frac{1}{\sqrt B}\sum_b(2k_b-1)\ell_b(x).
\]

Presence triệt tiêu. Tối ưu owner bằng:

\[
L_{\rm transfer}=\operatorname{softplus}\left(
0.25-\frac{\mathbb E[C_k(S_{\rm marked})-C_k(S_{\rm clean})]}
{\max(0.05,\sigma_{C,\rm clean})}\right).
\]

Scale chỉ lấy từ clean calibration, không lấy marked/student query. Log cả raw-score std và contrast std.

Teacher được giữ bằng quality budget và penalty khi watermark loss cao hơn teacher khởi tạo quá 0,05 trên cùng minibatch. Khi chưa vượt dung sai, không tiếp tục thưởng giảm BCE. Checkpoint vẫn phải qua kiểm tra teacher preservation trên validation.

Mỗi unroll dùng **4 ảnh khác nhau, 4 ordinary updates**, batch 1, duyệt từng ảnh trước khi lặp. Bốn bước là giới hạn bộ nhớ, không giả vờ đã mô phỏng 5.000 bước. Validation dùng 4 episode, mỗi episode 4 ảnh train/4 query, horizons 16 và 64, sampling 20 bước như final evaluator.

Hai nhánh có cùng loss, dữ liệu và ngân sách:

1. `contrast_cold`: surrogate khởi tạo clean base + Gaussian-A/zero-B.
2. `contrast_replay`: luân phiên trạng thái khởi tạo và checkpoint clean LoRA 5.000 bước đã có. Marked/clean trong cùng cặp bắt đầu từ cùng weights. AdamW moments được khởi tạo lại và được công bố rõ; đây là kiểm tra độ nhạy quanh learner đã học, **không phải replay đầy đủ trajectory hay exact resume**. Không dùng checkpoint marked đã đánh giá để tối ưu owner.

Nhánh replay dùng lại clean student như một surrogate ở owner-side. Final attacker student của cả hai nhánh vẫn khởi tạo từ clean base, trainer upstream, ordinary MSE, nhận PNG/caption; không nhận key/latent/watermark loss.

Ngân sách: 96 outer updates/nhánh, validation mỗi 32 bước. Chọn checkpoint theo worst-horizon/initialization mean contrast gain, yêu cầu gain dương và tăng ít nhất 0,005 đơn vị chuẩn hóa so baseline, cùng teacher preservation. Đây là gate sàng lọc development, không phải kiểm định thống kê. Nếu không có ứng viên đạt, dừng nhánh và lưu thất bại; không huấn luyện lại baseline đã biết.

Nếu có checkpoint đạt, chạy ordinary rank4 5.000 bước trên đúng 2.048 prompt/latent cũ, tái sử dụng clean PNG/student. Train student trước phần sinh ảnh đánh giá teacher để có tiến độ dễ kiểm tra. Chỉ gọi là cải thiện thực tế khi contrast TPR và AUROC cùng tăng ít nhất 0,02 tuyệt đối so bản cũ, đồng thời giữ chất lượng; sau đó mới mở rộng seed/learner. Kết quả một nhánh không chứng minh độc lập từng sửa đổi loss/dataset; phép so cold/replay chỉ kiểm tra phần thêm trạng thái learner.

## 6. Nếu vẫn không truyền

Dừng tăng độ phức tạp proxy quanh cùng late-latent carrier. Hướng tiếp theo cần thay nơi và loại dấu: initial-noise/trajectory carrier lấy cảm hứng từ SERUM, hoặc quan hệ nội dung–thuộc tính được học trong native teacher theo [đề xuất semantic](Semantic_Relation_Fingerprint_VI.md).

Hướng semantic cần reader mới, opposite-key và global-style controls, quality/coverage rõ ràng. Không ép giữ reader LALT khi thay cơ chế dấu. Đây là kế hoạch tiếp theo, chưa phải code đã chạy trong suite contrast/replay.

## 7. Triển khai và kiểm chứng ban đầu

Snapshot/tmux: `lalt_contrast_20261020_172619Z`.
Reports: `/root/logs/results/lalt_contrast_20261020_172619Z`.
[Deployment và hash nguồn](../server_review/lalt_contrast/lalt_contrast_20261020_172619Z/deployment.json);
[trạng thái đã kiểm tra](../server_review/lalt_contrast/lalt_contrast_20261020_172619Z/verification.json).

43 tests pass trên server, gồm tính bất biến theo presence, contrast đúng bằng
hiệu hai key score, đảo key đổi dấu, chu kỳ inner loop dùng đủ ảnh, gate
không nhận proxy âm và checkpoint replay không thay weights của rendering
UNet. GPU smoke hoàn tất hai outer updates: một cold start, một clean-trained
start, với 4 ảnh/4 inner updates và sampling 20 bước. Gradient transfer tới
carrier khác zero. Smoke không chọn checkpoint mới; đây là kiểm tra chạy
và gradient, không phải bằng chứng cải thiện.

Audit toàn bộ tham số carrier ở bước smoke đầu:

| Gradient chưa nhân transfer weight | Norm |
|---|---:|
| Mục tiêu raw + clean-adjusted specificity cũ | 0,02785 |
| Mục tiêu contrast mới | 0,24160 |
| Teacher preservation + quality mới | 0,02331 |

Cosine raw/contrast = 0,9843; teacher/contrast = −0,0275. Tại điểm này,
gradient mới lớn hơn khoảng 8,7 lần nhưng gần cùng hướng. Không có bằng
chứng từ phép đo này rằng loss cũ luôn đi sai hướng; vấn đề chuẩn hóa và
độ đại diện của proxy vẫn cần được tách khỏi giả thuyết xung đột gradient.
Penalty preservation đang bằng zero vì teacher đúng bằng anchor ban đầu;
norm teacher ở đây chủ yếu là quality loss mới, không phải norm teacher
loss của run cũ.

Lúc 00:28 ngày 21/10 (UTC+7), smoke đã pass và lượt chính đã bắt đầu
nhánh `contrast_cold`; sau đó tự chạy `contrast_replay`. Mỗi nhánh 96
outer updates; chưa có kết quả ordinary student mới. Nếu rank4 đạt gate
cải thiện thực tế đã mô tả, suite mở rộng rank16/32. Seed mới là bước
xác nhận tiếp theo, chưa nằm trong queue này.

## 8. Kết quả hoàn tất, kiểm tra lúc 06:41 ngày 21/10

Cả hai nhánh hoàn tất 96/96 outer updates, không có lỗi runtime. Nhánh
cold kết thúc khoảng 01:10, replay khoảng 01:49 (UTC+7). Không checkpoint
nào đạt đồng thời yêu cầu tăng proxy và bảo toàn teacher; vì vậy không
chạy ordinary student mới và không mở rộng rank16/32.

| Chỉ số validation | Contrast cold | Contrast replay |
|---|---:|---:|
| Worst-group proxy ban đầu | 0,006594 | −0,012858 |
| Proxy tốt nhất trong các checkpoint mới | 0,006865 (step32) | −0,010829 (step96) |
| Tăng so baseline | +0,000270 | +0,002028 |
| Mức tăng tối thiểu đã đặt | +0,005 | +0,005 |
| Teacher key221 TPR ban đầu | 83,59% | 83,59% |
| Teacher key221 TPR step96 | 78,91% | 79,69% |
| Teacher LPIPS ban đầu → step96 | 0,05627 → 0,05179 | 0,05627 → 0,05200 |

Teacher preservation pass ở step32, fail ở step64 và step96 của cả hai
nhánh. Cold có proxy dương nhưng mức tăng quá nhỏ; tên trạng thái máy
`no_positive_preserved_proxy` không có nghĩa mọi giá trị proxy đều âm.
Replay cải thiện worst-group một chút nhưng vẫn âm. Các giá trị proxy
là contrast gain chuẩn hóa, không phải phần trăm detection; hai nhánh
có nhóm khởi tạo khác nhau nên không so mức tuyệt đối như cùng một metric
trên cùng điều kiện. TPR teacher trong bảng là tập owner validation,
khác tập ordinary probe từng cho 77,34%/84,38%.

Diễn giải: điều chỉnh loss và replay ngắn chưa tạo được cải thiện đủ
lớn trong thiết lập này. Ảnh gần clean hơn trong khi teacher detection
giảm; ràng buộc watermark loss trên minibatch chưa bảo đảm giữ TPR trên
validation. Không có student 5.000 bước mới nên không gán số 3,91% cũ
cho các checkpoint mới, cũng không kết luận mọi phương án replay đều
thất bại. Kế hoạch thay cơ chế carrier ở mục 6 vẫn là hướng nghiên cứu
tiếp theo, chưa được chạy tự động trong lượt kiểm tra kết quả này.

Bằng chứng chi tiết đã tải về thư mục
`server_review/lalt_contrast/lalt_contrast_20261020_172619Z/latest/full/`,
gồm proxy, teacher validation và preservation ở step32/64/96.
