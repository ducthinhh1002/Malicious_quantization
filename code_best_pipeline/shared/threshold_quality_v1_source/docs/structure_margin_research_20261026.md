# Phân tích kết quả và thử nghiệm structure/margin native — 26/10/2026

## 1. Kết quả thực sự nói gì?

Đã đọc lại JSON kết quả và log teacher trên VM4, không chỉ bảng tổng hợp.
Audit chi tiết: [prior_result_audit](../../output_artifacts/structure_margin_prior_audit_20261026.json).
Các số bên dưới là student test 1.000 ảnh, cùng ngưỡng 0,8541307449340821.
LPIPS so với SD1.4 chưa fine-tune cùng prompt/seed; FID/KID tham chiếu clean
SD2.1, không phải ảnh thật. Không so trị tuyệt đối LPIPS teacher/student.

| Nhánh | TPR | LPIPS ↓ | FID ↓ | KID ↓ | CLIP ↑ |
|---|---:|---:|---:|---:|---:|
| Baseline k=1, scale1 | 75,0% | 0,550447 | 54,44424 | 0,00166651 | 0,787681 |
| Continuation, teacher thêm600 bước loss gốc | 82,3% | 0,550218 | 54,88688 | 0,00141482 | 0,786429 |
| Native full | 80,5% | 0,544096 | 54,60549 | 0,00126494 | 0,786889 |
| Projected native, student7500 | 79,2% | 0,522219 | 53,35774 | 0,00079496 | 0,785335 |
| Projected native, student10000 | 82,1% | 0,533119 | 53,45343 | 0,00077322 | 0,787516 |
| Balanced native, student7500 — đối chứng nội bộ | 80,2% | 0,517300 | 52,79152 | 0,00068993 | 0,784043 |

Balanced được giữ trong phân tích nội bộ vì là bằng chứng quan trọng. Không
đưa lại vào Report.md/PDF mà người dùng đã yêu cầu bỏ. Projection chưa có
bằng chứng thắng nhánh không projection. Run10000 là seeded rerun, không
resume bitwise từ run7500; không gán toàn bộ chênh lệch cho thêm2500 bước.

**Projected đã vượt baseline k1 về TPR, LPIPS, FID và KID**, nhưng chưa có
nghiệm thắng tất cả: CLIP giảm; continuation còn có detection tốt hơn;
Balanced tốt hơn nhiều metric ở cùng7500 bước nhưng CLIP thấp hơn.
Mốc99,7% cũ dùng kappa2/protocol lịch sử khác, không phải đối chứng k1 này.

### Phân tích từng ảnh, không chỉ trung bình

- Projected7500: 69,3% ảnh có LPIPS thấp hơn baseline; 30,7% không tốt hơn.
  Có145 ảnh mới được detect nhưng103 ảnh baseline từng detect lại mất;
  tổng tăng42 ảnh không có nghĩa giữ được mọi detection trước đó.
- LPIPS p95 giảm từ0,728901 xuống0,709408: vẫn còn một nhóm ảnh lệch nhiều.
  Đây là độ lệch reference, không tự động đồng nghĩa ảnh xấu về thẩm mỹ.
- Student score p10: baseline0,700432; Projected0,738406;
  continuation0,764144. Cả ba đều dưới ngưỡng0,854131.
- Với Projected10000, score p05 giảm0,638093→0,595504 và p10 giảm
  0,738406→0,730432 so với Projected7500, dù TPR trung bình tăng.
  Train lâu không bảo đảm cứu nhóm ảnh yếu nhất.
- Tương quan score–LPIPS của Projected7500 chỉ khoảng0,028 trên cùng tập.
  Chưa có bằng chứng rằng mọi cải thiện chất lượng đều buộc mất detection.
  Đây là tương quan quan sát, không phải kết luận nhân quả.

### Audit gradient và score teacher

Projected có600 updates,300 lượt native auxiliary, mỗi lượt chỉ một ảnh.
200/300 lượt (66,7%) thấp hơn score floor. Deficit trung bình100 lượt cuối
là0,11756. Phép chiếu cần can thiệp121/300 lượt. Cosine quality–negative
score trung bình khoảng−0,073, có xung đột nhưng không hoàn toàn đối nghịch.

Floor trước đây chỉ thêm repair tối đa0,25 norm gốc; không cưỡng chế score
thực tế sau update. Một ràng buộc vi phân trên một prompt không bảo đảm score
cho các prompt khác hoặc downstream student. Không gọi hiện tượng này là
bug của detector hay khẳng định đó là nguyên nhân duy nhất của mất detection.

## 2. Những gì giữ lại và những hướng không lặp lại

1. Giữ warm-start k1 checkpoint3000, scale1, LoRA attention+conv rank32,
   loss threshold_cfg_v1, coherent shift và VAE-cycle. Giảm scale đồng loạt
   xuống0,85 trước đây làm TPR student75→42,7%; không tiếp tục cách đó.
2. Giữ native forward đầy đủ PNDM50/51 calls, CFG7,5, RGB8 STE và backward
   10 calls cuối. Đây là bước chuyển đem lại tiến bộ chất lượng rõ nhất;
   các proxy predicted-x0/gradient matching/CKA trước chưa vượt nó.
3. Giữ balancing gradient. Các nhánh quality cũ có gradient quality trung
   vị chỉ khoảng0,3453% gradient gốc, nên một weight loss có vẻ lớn chưa
   chắc tạo tác động đủ lớn.
4. Không tăng student epochs như cách chữa mặc định: kết quả10000 và
   scale0,85-long đều cho thấy trade-off. Student vẫn ordinary trainer7500.
5. Teacher TPR99% không đủ đại diện radioactivity; phải chạy student đầy đủ.
6. Không coi LPIPS/FID/CLIP là bộ kiểm tra khuôn mặt, bàn tay. Cần panel
   prompt/seed cố định và người đánh giá; hiện chưa có human ratings.

## 3. Survey bổ sung và quyết định

| Nguồn chính | Điều dùng được | Giới hạn khi áp dụng |
|---|---|---|
| [DRaFT, Clark et al.](https://arxiv.org/abs/2309.17400) | Tối ưu reward trên ảnh cuối, truncated backprop giảm chi phí | Không suy ra K10 là tối ưu hoặc gradient giống full51 |
| [MS-SSIM, Wang–Simoncelli–Bovik2003](https://www.cns.nyu.edu/pub/eero/wang03b.pdf) | So sánh cấu trúc nhiều thang đo; bổ sung cho LPIPS | Giữ cấu trúc reference, không nhận biết anatomy hoặc làm clean teacher tự đẹp hơn |
| [DISTS, Ding et al.](https://arxiv.org/abs/2004.07728) | Phân biệt cấu trúc và texture; một metric LPIPS đơn lẻ chưa bao quát hết | Chưa thêm VGG/DISTS vào loss vòng này để tránh thêm model và nhiều thay đổi cùng lúc |
| [Large-Scale Methods for DRO, Levy et al.](https://arxiv.org/abs/2010.05893) | Nhìn phần đuôi phân bố thay vì chỉ trung bình | Run mới chỉ dùng squared hinge theo ảnh, **không** phải CVaR/DRO hay hưởng bảo đảm của paper |
| [SIREN](https://arxiv.org/html/2410.10437v1) | Phân biệt learnability và perceptual constraint; color distortion là vấn đề riêng | Không thay detector/key hoặc đưa secret vào student; không tái lập coating/CIEDE2000 của SIREN |
| [Confronting Reward Overoptimization, ICML2024](https://proceedings.mlr.press/v235/zhang24ch.html) | Tối ưu điểm có thể làm suy giảm chất lượng thật | Không chạy tăng detector score vô hạn; repair chỉ khi dưới floor |

MS-SSIM được chọn vì có gradient, nhẹ, không cần tải thêm mạng và phù hợp
paired clean prompt/seed đang có. Nó có thể giữ lại cả lỗi của clean reference;
không hứa sửa mọi lỗi anatomy. Kết luận phải dựa cả metric ngoài objective,
ảnh đối chiếu và student detection, không riêng MS-SSIM đã được tối ưu.

## 4. Hai nhánh mới và giả thuyết được kiểm tra

**Thứ tự: structure_margin_native → margin_native (ablation MS-SSIM).**
Cùng nguồn k1 checkpoint3000; teacher600updates, LR1e-5, batch4; checkpoint
200/400/600 kèm optimizer/RNG/data order. Không warm-start từ Projected để
tránh lẫn thêm600 updates. Student giữ7500, batch4,5.000train/1.000val/1.000test;
chọn checkpoint bằng validation denoising MSE như protocol cũ.

Mọi update vẫn chạy loss threshold_cfg_v1 gốc. Cứ2updates, sinh native cho
cả4 training prompt/seed của batch, xử lý microbatch2 để giới hạn bộ nhớ.
Tổng1.200 native training examples thay vì300 ở run cũ, không dùng test
để chọn prompt hoặc lấy gradient. Hai nhánh mới có cùng số samples/updates;
so với Projected cũ tăng compute auxiliary, không gọi là cùng FLOPs.

Với ảnh i:

```
target_i = max(score_baseline_k1_RGB_i, tau_teacher + sigma_clean)
deficit_i = max(0, target_i - score_native_i)
Lm = mean_i [(deficit_i / sigma_clean)^2]
Lq = mean_i [LPIPS_i + 0.1*MSE_RGB_i + w*(1 - MS_SSIM_i)]
w = 0.25 ở structure_margin_native; w = 0 ở margin_native
```

Ảnh đã đạt floor có gradient repair bằng0; không được bù trừ deficit ảnh
khác. Bình phương làm ảnh hụt nhiều đóng góp mạnh hơn trong batch. `sigma`
là giá trị đã hiệu chuẩn và đóng băng từ k1, không hiệu chuẩn lại detector.

Tính riêng g0, gq, gm; cộng gradient microbatch theo sum/4 rồi mới cân bằng
norm. Quality và repair mỗi loại tối đa1 lần norm g0, gain tối đa100;
repair trước đây chỉ0,25. Global clip1 như cũ. Giữ QP trên actual AdamW
displacement với g0, gm, gq; ràng buộc score nay là **batch squared deficit**,
không phải tăng negative-score trên mọi ảnh kể cả đã đủ margin.

Đây là bảo vệ vi phân cục bộ của batch average với truncated gradients,
không bảo đảm mọi ảnh giữ score hoặc student detection không giảm.
Projection vẫn là yếu tố thực nghiệm, không mặc định tốt hơn bỏ projection.
Việc tăng repair có thể đánh đổi chất lượng; log riêng deficit, norm, cosine
và structural loss để biết điều gì xảy ra. Hai nhánh chỉ khác `w`, nên
đối chiếu chúng tách được tác dụng MS-SSIM, chưa tách được mọi thay đổi khác.

## 5. Tiêu chí báo kết quả và vận hành

- Chạy tới cuối và lưu mọi kết quả, không quality/detection early-stop gate.
  Chỉ dừng/chờ khi lỗi số học, sai provenance, hoặc thiếu tài nguyên.
- So với baseline k1, Projected7500, continuation; báo cả nội bộ Balanced.
  Mốc kỳ vọng để có thêm tiến bộ là giữ TPR khoảng82% hoặc hơn, LPIPS dưới
  0,5222 và FID dưới53,36. Đây là mục tiêu thí nghiệm, không cam kết hay
  ngưỡng ảnh đẹp phổ quát, cũng không dùng để bỏ kết quả không đạt.
- Báo TPR, score p05/p10, mất/thêm detection so với baseline, LPIPS mean/p95,
  FID/KID/CLIP, MS-SSIM và panel cố định. MS-SSIM dùng trong loss được gắn
  nhãn rõ; cần human evaluation và fresh holdout trước tuyên bố mạnh.
- Code mới trong các file `structure_margin_*.py`; không sửa harness
  Bitmark gốc, sampler hoặc trainer student. Hash code/provenance được đóng băng.
- Tái sử dụng ảnh clean/unadapted/baseline bằng hardlink. Model ghép tạm
  đặt dưới `/dev/shm` với merge-reload verification gốc; chỉ xóa bản export
  tạm mới tạo sau khi đã lưu đủ ảnh và audit. Checkpoint/ảnh/kết quả nằm
  trên đĩa bền vững. Không xóa checkpoint hay dữ liệu các run cũ.
- 14 unit tests đã pass trên VM4; actual GPU smoke riêng phải pass trước
  từng teacher. Unit test không phải kết quả khoa học hay chứng minh cải thiện.

Một seed và test đã xem nhiều lần chỉ hỗ trợ kết luận khám phá. Cần fresh
holdout, nhiều seed/key và đánh giá ảnh độc lập nếu muốn chuyển sang paper.

## 6. Triển khai

Root VM4: `/root/logs/threshold_quality_20261022/structure_margin_20261026`.
Supervisor PID2223976. Đã pass14 tests và GPU smoke batch4/microbatch2,
đỉnh45,42GiB allocated; gradient quality và repair khác0, cập nhật tham số
thật, cả ba dot-product của update đều không dương ở lượt thử đó.
Đây là smoke kỹ thuật, không phải bằng chứng chất lượng/detection tăng.

Hàng chờ tự bắt đầu `01_teacher` lúc07:58 ICT ngày26/10 sau khi đủ VRAM;
không dừng hay thay đổi tác vụ MinhNQN của người dùng khác. Vẫn dùng
resource admission đã freeze ban đầu, không thay đổi code sau khi train bắt đầu.
Nhánh `margin_native` chờ sau toàn bộ teacher→student→evaluation của nhánh đầu.
Checkpoint teacher định kỳ200/400/600; snapshot student2500/5000/7500.
Student snapshots của harness gốc chỉ lưu adapter; teacher có optimizer/RNG.
Không tuyên bố student có full-state resume nếu chỉ có adapter snapshot.
