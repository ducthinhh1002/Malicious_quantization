# Đối chiếu MinhNQN với radioactive watermark — 24/10/2026

Đã đọc trực tiếp `/root/MinhNQN/CSD` và `/root/MinhNQN/VLM_Embed_Gvendi`
trên VM4: README, các criterion chính, launcher GVENDI, nhánh forward của
TalasJepa, code cache và JSON evaluation. Đây là code distillation embedding
VLM cho MMEB, không phải diffusion watermark. Tên file không đủ để suy ra
thuật toán: chẳng hạn `talas_jepa.py` hiện dùng structure loss và GMM-CF loss.
Không sửa project MinhNQN hoặc hàng chờ watermark trong lần review này.

Hash các nguồn chính và số liệu gốc đã lưu tại
[`minhnqn_review_20261024.json`](../../output_artifacts/minhnqn_review_20261024.json).
Các vị trí code dưới đây là đường dẫn trên VM4, theo snapshot đã đọc.

## 1. Kỹ thuật đáng áp dụng nhất: đo và bảo toàn tác động lên gradient học

Nguồn: `VLM_Embed_Gvendi/src/criterions/gvendi.py`:

- Dòng 9–30: Rademacher projection theo block, chiếu gradient xuống 512 chiều.
- Dòng 103–131: lấy gradient từng mẫu bằng batched VJP.
- Dòng 145–177: chỉ lấy LoRA ở các layer cuối.
- Dòng 245–289: teacher gradient detach; student gradient `create_graph=True`;
  chuẩn hóa rồi so Gram matrix ngoài đường chéo.
- Launcher `scripts/train_distill_gvendi_cls.sh:90` dùng FastVLM-0.5B,
  B3_Qwen2_2B, batch 16; đây là distillation có can thiệp trực tiếp vào student.

**Phần nên mượn là đo gradient và phép chiếu, không bê nguyên loss.**
Trong bài của ta, chỉ owner được tối ưu teacher. Student cuối vẫn học ordinary
denoising MSE từ RGB/caption, không nhận key, teacher features hay loss phụ.

Một adaptation cụ thể, trước hết dùng như diagnostic:

1. Tái dùng ảnh clean, scale 1.0 và 0.85 có cùng prompt/seed.
2. Đưa chúng qua cùng SD1.4 probe, cùng LoRA state, VAE posterior noise,
   diffusion noise và timestep. Lấy gradient ordinary MSE trên cùng tập tham số.
3. Tính `delta_g = grad(MSE_marked) - grad(MSE_clean)`. Tách đóng góp của
   watermark khỏi phần học nội dung vốn có thể lấn át total gradient.
4. Đo signed cosine, norm ratio, và độ tương đồng sau phép chiếu cố định với
   `delta_g` của scale 1.0. Phân tách attention/conv và vùng timestep.
5. Đối chiếu với detection của student đã có. Chỉ có vài cấu hình nên đây là
   kiểm tra sơ bộ, chưa đủ xác nhận một surrogate tổng quát.

Nếu diagnostic có ích, thêm loss owner-side để ảnh đẹp hơn vẫn tạo ra update
gần ảnh scale 1.0. Dùng relative error của projected `delta_g` hoặc paired
AdamW update difference từ cùng optimizer state, thay vì chỉ cosine/Gram.
Scale 1.0 ở đây là baseline hiện tại (~75% student TPR), không phải bản lịch sử
99.70%; bảo toàn baseline này chưa đồng nghĩa khôi phục lịch sử.

**Các bẫy quan trọng trong việc chuyển phương pháp:**

- Chuẩn hóa gradient làm mất độ lớn. Gram còn bất biến khi đảo dấu đồng loạt
  hoặc xoay các vector; loss nhỏ không đảm bảo update đẩy đúng hướng detector.
- Với batch 1, GVENDI đặt gradient loss bằng 0. Gradient accumulation từng
  mẫu không tự tạo quan hệ giữa các mẫu như batch thật.
- Random projection giảm kích thước so sánh; code vẫn tạo tensor gradient
  `[B,P]` trước khi chiếu, nên không tự giải quyết toàn bộ VRAM bậc hai.
- Không flatten rồi đối sánh trực tiếp tham số SD2.1 với SD1.4. Dùng cùng
  SD1.4 probe cho cả ảnh tham chiếu và ứng viên.
- Initial LoRA có nhánh zero-init làm một phần gradient bằng 0. Cần kiểm tra
  thêm state sau học ngắn; cùng noise giúp giảm nhiễu nhưng không thay thế
  nhiều seed/timestep. Raw gradient cũng chưa phản ánh AdamW preconditioning.
- Denoising dễ hơn không đồng nghĩa watermark học tốt hơn. Audit của ta đã
  thấy MSE không phân biệt rõ scale 1.0/0.85 dù detection chênh lớn.

Đây là hướng bổ sung/rút gọn cho `transfer_proxy.py` đang có paired learner
updates. Nó không chứng minh tốt hơn proxy hiện tại. Gradient matching từng
được dùng để tối ưu dữ liệu tổng hợp theo tác động học, nhưng áp dụng vào
radioactive watermark này là đề xuất của ta, không phải kết quả của paper:
[Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929).

## 2. CKA và cấu trúc đặc trưng: loss phụ cho chất lượng ảnh

Nguồn: `CSD/src/criterions/simcse_cka_kd_loss.py:30` tính centered linear CKA;
`CSD/src/criterions/pga.py:72` giữ phổ chính của teacher Gram tới 85% năng lượng,
rồi dùng centered kernel alignment. `talas_jepa.py:70` dùng cosine Gram MSE.

Ứng dụng phù hợp: so đặc trưng của ảnh marked và clean cùng prompt/seed qua
một encoder đóng băng, bổ sung cho LPIPS khi refine teacher. Có thể so patch
features để theo dõi cấu trúc không gian; phải giữ correspondence giữa patches.
Không cần đổi reader hoặc carrier. CKA dùng để so cấu trúc biểu diễn theo
[Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html).

Centered linear CKA loại bỏ offset chung qua trục mẫu, nên có thể ít xung đột
với một coherent shift chung hơn raw feature MSE. Đây chỉ là lý do để thử:
watermark thực tế không nhất thiết là offset cố định trong encoder đang xét.
CKA còn bỏ qua một số thay đổi scale/rotation nên không thay LPIPS, CLIP,
ảnh đối chiếu và detection margin. Với batch 1 trên global embedding, centering
bị suy biến; dùng nhiều mẫu hoặc patch tokens với cách diễn giải phù hợp.

Khởi đầu bằng centered CKA đơn giản; chưa ưu tiên PGA cắt top-85% vì có thể
bỏ mất chi tiết ít năng lượng. Dùng sample Gram nhỏ thay vì ma trận D×D trong
implementation CKA hiện có khi số mẫu ít hơn chiều feature. Giữ nguyên
`L_original_threshold_cfg`, thử thêm một loss mỗi lần để biết tác dụng.

## 3. GMM + characteristic function: có ý tưởng, chưa nên đưa ngay vào train

Nguồn: `CSD/src/criterions/talas_jepa.py:176` tính empirical characteristic
function trên các hướng chiếu; so với CF giải tích của GMM teacher bằng sliced
Wasserstein trên các vector CF. Dòng 528 đọc teacher feature caches; dòng
573–607 ghép structure loss và GMM distribution loss. `train_teacher_gmm.py`
fit GMM offline. Đây không phải loss LPIPS hoặc diffusion distillation.

Có thể học phân bố clean features offline rồi đo marked features có lệch
phân bố nội dung hay không, tránh chỉ ép mọi thứ về Gaussian chuẩn. Tuy nhiên:

- Matching marginal distribution không giữ được nội dung của từng prompt;
  hoán đổi ảnh giữa prompts có thể vẫn giữ phân bố.
- Phân bố toàn bộ carrier chứa chính mean shift dùng để detect. Ép về sạch
  sẽ cạnh tranh với watermark. Nếu thử, cần giới hạn phần nội dung và giữ
  nguyên mục tiêu detection; không được thay PCA/carrier đã cố định.
- GMM nhiều thành phần, batch 4 và ảnh diffusion khác MMEB cần đánh giá lại;
  không chuyển nguyên các trọng số và 16/64 components từ launcher.

Ưu tiên thấp hơn gradient diagnostic và CKA. Không gọi việc fit GMM một lần
là phương pháp đã chứng minh tăng FID/TPR.

## 4. Những phần nên dùng cho đo lường và tốc độ

- `CSD/src/criterions/ckd_sigreg_kd_loss.py:36` ước lượng participation ratio
  bằng random projections và trace covariance, tránh full SVD. Có thể dùng
  để quan sát mất đa dạng feature cùng LPIPS/FID, chưa dùng rank làm fitness
  duy nhất. Participation ratio khác entropy effective rank.
- `eigen_rank_align.py:38` dùng SVD năng lượng rồi so CDF theo thứ tự phổ.
  Phổ giống nhau vẫn có thể khác hướng key; không suy ra detection giữ được.
- `teacher_cache.py:278` chạy teacher dưới no_grad; dòng 318–319 lưu features.
  Ta có thể cache frozen clean features, text embeddings, reference gradient
  theo state/noise đã xác định. Đây là phần thực dụng để tiết kiệm tính toán.
  Không cache stale marked trajectory hoặc detach nhánh cần outer gradient.
- Attention-weighted OT trong `emo_loss.py:202` có thể gợi ý trọng số vùng
  ảnh, nhưng attention VLM và token matching không chuyển thẳng thành độ
  nhạy thị giác của diffusion. Chưa ưu tiên hơn LPIPS đang có.

## 5. Kỹ thuật cần tránh áp dụng nguyên xi

`mse_sigreg_kd_loss.py:30` kéo empirical CF về `exp(-t²/2)`, tức Gaussian
chuẩn theo hướng chiếu. Làm vậy trực tiếp với carrier có nguy cơ triệt mean
shift của watermark. Tăng effective rank vô điều kiện cũng không phải mục
tiêu chất lượng ảnh và có thể thưởng cho nhiễu. Không đổi sang Muon, thêm
InfoNCE hoặc distillation vào final student chỉ vì project VLM đang dùng.

## 6. Bằng chứng thực nghiệm trong folder và giới hạn

Tính trung bình không trọng số `acc` từ 10 file `*_score.json` cho từng run:

| Run trong folder | Mean accuracy MMEB |
|---|---:|
| GVENDI, cse0/grad1, gradient weight 0.1 | 52.68% |
| GVENDI, cse0/grad1, gradient weight 0.5 | 53.91% |
| GVENDI, cse0/grad1, gradient weight 1.0 | 53.75% |
| CSD, struct + original-sigreg GMM, d1/sig1/kd10 | 51.17% |
| CSD, struct + SW-GMM, d1/sig1/kd10, layer24 | 54.44% |

Đây là số liệu đọc được, không phải ablation đã kiểm chứng cùng mọi điều kiện:
tên folder và source hiện tại không xác nhận đầy đủ code revision/hyperparameter
của từng run; chưa có nhiều seed/đối chứng đầy đủ. Accuracy MMEB không phải
watermark TPR hay chất lượng ảnh diffusion. Không suy ra CSD thắng GVENDI trong
bài của ta từ bảng này.

## 7. Thứ tự đề xuất

1. Audit gradient từ ảnh cũ: chi phí thấp hơn mở thêm full campaign, không cần
   gen lại. So signed delta gradient và paired update, có reference scale 1.0.
2. Giữ ablation `quality` / `quality_transfer` hiện có để biết LPIPS và learner
   proxy có thực sự hữu ích. Review này không thay code/hàng đợi của chúng.
3. Nếu cần thử tiếp, thêm riêng CKA bảo toàn cấu trúc hoặc signed update
   matching, không thêm cả GMM/SIGReg/OT đồng thời. Giữ teacher rank32+conv,
   key/carrier/ngưỡng/sampler và ordinary final student.
4. Dùng matched training budget, đối chứng chỉ tiếp tục original loss, clean
   student control và fresh holdout cho claim cuối. Chạy đến hết, lưu cả kết
   quả xấu; không đưa lại quality stopping gate.

Kết luận: giá trị trực tiếp nhất của folder là công cụ đo tác động học bằng
gradient và loss giữ cấu trúc đặc trưng. Chưa có bằng chứng các kỹ thuật này
đã nâng đồng thời chất lượng ảnh và student detection trong bài của ta.
