# Fingerprint cho diffusion sống qua ordinary LoRA: ý tưởng và kế hoạch sửa code

Ngày: 18/10/2026. Trạng thái: **coordinate pilot đã kết thúc; thất bại teacher gate, chưa chạy student trong main pilot**.

Hai hướng vào vòng kiểm tra đều chỉ detect 1/128 ảnh. Mean-score gain trên heldout lần lượt khoảng +0,00288 và -0,01531, so với gain trên tập fit +0,0580 và +0,0624. Không có bằng chứng fingerprint ổn định từ pilot này. Báo cáo gốc: [native_transfer_final](../server_review/native_transfer_final/). Phân tích và đề xuất tiếp theo: [Semantic_Relation_Fingerprint_VI.md](Semantic_Relation_Fingerprint_VI.md).

Chi tiết code và cấu hình thực tế: [NATIVE_TRANSFER.md](../new_watermark/NATIVE_TRANSFER.md).
Pilot đầu dùng tám hướng riêng lẻ, hai biên độ và ordinary response studies ngắn;
chưa triển khai bộ giải tối ưu liên tục đồng thời `a,d` hay gradient proxy.

Tên làm việc: **Native Transfer Fingerprint (NTF)**. Tên này dùng để phân biệt thí nghiệm, không hàm ý đã xác lập novelty.

## 1. Mục tiêu phải giữ đúng theo `target.md`

Ta cải tiến fingerprint trong **teacher diffusion**. Một bên khác lấy ảnh teacher sinh ra, huấn luyện student độc lập bằng LoRA/SFT thông thường, rồi ảnh student sinh ra vẫn có tín hiệu nhận diện owner.

```text
Teacher có fingerprint trong trọng số
              ↓ sinh ảnh bình thường
       Ảnh + caption thông thường
              ↓ ordinary LoRA/SFT
Student khởi tạo độc lập từ model sạch
              ↓ sinh ảnh mới
        Kiểm tra fingerprint
```

Owner được thay đổi trọng số teacher, cách thiết kế fingerprint và detector. Student chỉ nhận ảnh/caption: không nhận key, detector, latent nội bộ, trọng số teacher hay watermark loss. Không đổi rank, module, optimizer hoặc loss của student nhằm làm kết quả đẹp hơn. Nếu đánh giá nhiều cấu hình student, danh sách được ấn định trước và báo đủ từng cấu hình.

Teacher xuất ra phải là model dùng inference bình thường: không cần cộng residual, sửa noise, lọc ảnh theo detector hoặc chạy watermark hook lúc sinh ảnh.

“Không cần biết attacker LoRA thế nào” nên được kiểm chứng bằng các cấu hình **chưa dùng khi thiết kế**. Không có cơ sở hứa sống qua mọi thuật toán: student không học, gần như không dùng dữ liệu teacher hoặc cố ý loại bỏ tín hiệu là những phản ví dụ. Mục tiêu thực nghiệm là truyền dấu trên một họ quá trình học có utility hợp lệ, bao gồm các cấu hình chưa thấy.

## 2. Những gì thí nghiệm cũ thực sự cho biết

| Hướng đã chạy | Quan sát | Điều rút ra được |
|---|---|---|
| SERUM trong `result.zip` | Teacher 100%; các LoRA/DoRA trong run khoảng 2,3–3,6%; full fine-tuning 83,6% | Teacher mạnh không bảo đảm ordinary LoRA học được dấu trong protocol đó |
| BitMark gốc → SD2.1 | Attention LoRA r4/r16/r32: 49,4/62,9/66,3%; unadapted đã 6,1%; thiếu clean-trained control | Có tín hiệu đáng khảo sát, nhưng chưa phải đối chứng nhân quả đầy đủ và chưa phải teacher diffusion native của ta |
| BitGreen, tăng score trực tiếp | Có cấu hình teacher mạnh nhưng PSNR khoảng 16–17 dB; các cấu hình giữ ảnh tốt thường gần clean về detection | Có trade-off embedding/quality; đổi weight loss chưa giải quyết được |
| Latent target từng ảnh | Target 64/64 thành công mỗi nhánh; native teacher vẫn gần clean | Tạo được ảnh có dấu không đồng nghĩa UNet học được phép biến đổi đó |
| Residual chung 16.384 tham số | 16/16 train, chỉ 1/8 heldout | Overfit trong thiết lập đã đo |
| Conv carrier 148 tham số | 4/64 train, 1/32 heldout | Giảm tham số không tự tạo ra carrier phù hợp |
| Full-rollout terminal regression | Kết thúc 60 bước: 0/4 train, 0/4 heldout; relative target MSE train 1,051 so với baseline 1 | Gradient xuyên toàn chuỗi chưa đủ để học target; pilot này còn thất bại trước bước student |
| Distribution/centered teacher | Có một số mean-score gain nhỏ, nhưng chưa có teacher đạt đồng thời signal/quality một cách thuyết phục | Không nên lặp lại “chọn một hướng feature rồi tilt phân phối” và gọi đó là phương pháp mới |

Các run khác nhau về backbone, detector và ngân sách; không xếp chúng thành một bảng benchmark công bằng. Run BitGreen nhỏ chỉ là diagnostic, không đủ ước lượng FPR thấp. Green fraction của BitMark cũng không phải message bit accuracy.

Nguồn nội bộ: [BitMark](../server_review/BITMARK_20261017_evidence.md), [SERUM](../server_review/SERUM_result_zip_20261017.json), [audit transport](BitGreen_Learnability_Repair_VI.md), [pilot mới nhất](../server_review/bitgreen_onpolicy_final/).

**Quyết định thiết kế:** không tiếp tục lấy một target tùy ý trong pixel/latent hoặc một hướng detector cố định rồi yêu cầu teacher học theo. Chọn fingerprint từ những thay đổi mà teacher thật tạo được, sau đó kiểm tra thay đổi nào ordinary learning có xu hướng truyền lại.

## 3. Ý tưởng chính

**Fingerprint là một thay đổi nhỏ, có key, trong hành vi sinh ảnh của teacher; hướng thay đổi được chọn theo cả khả năng nhúng vào teacher và khả năng truyền qua training.**

Hai yêu cầu phải cùng đúng:

1. Teacher tạo được tín hiệu ổn định trên prompt/noise mới, với chất lượng chấp nhận được.
2. Student học bằng loss gốc từ ảnh teacher có xu hướng tạo lại cùng tín hiệu, vượt matched clean-trained control.

Ví dụ trực giác: một thay đổi rất nhẹ nhưng nhất quán trong cách model thể hiện chất liệu/ánh sáng có thể dễ học hơn một cấu hình bit rất riêng cho từng ảnh. Đây chỉ là minh họa. Không mặc định mọi thay đổi ngữ nghĩa đều tốt, không dùng “đổi tông màu toàn bộ ảnh” làm thành công mặc định.

Điểm khác với carrier CLIP/PCA cũ là **tiêu chuẩn chọn hướng**. Hướng ổn định qua VAE chưa chắc được gradient denoising của student học. Ở đây đường truyền `ảnh → VAE thực của student → ordinary denoising updates → ảnh mới` được đo trực tiếp hoặc xấp xỉ có đối chiếu.

## 4. Cơ chế đề xuất

### 4.1 Bắt đầu trong không gian trọng số native của teacher

Biểu diễn teacher bằng:

\[
\theta(a)=\theta_0+B a,\qquad a\in\mathbb R^m.
\]

`B` là một tập nhỏ các hướng thay đổi **tham số sẵn có** của UNet. Pilot dùng `m=8`, chưa mở ngay hàng triệu biến tự do.

Ứng viên triển khai đầu tiên: các hướng trên affine scale/bias sẵn có của normalization trong mid-block và các block up gần bottleneck. Các hướng tác động trên nhiều ảnh, nhưng ảnh hưởng thực tế phụ thuộc nội dung và timestep của UNet. Cần đánh giá, không mặc nhiên gắn nhãn “semantic” cho chúng.

Lợi ích kỹ thuật: `theta(a)` đã là model thật. Không có bước “tạo một latent target ngoài model rồi hy vọng compile thành UNet”. Khi xuất model, ghi trực tiếp các tham số đã đổi vào state dict; không cần một lớp mới khi inference.

Nếu nhóm tham số này không tạo được dấu ở cùng ngân sách chất lượng, mở sang một nhóm tham số owner đã ấn định trước, chẳng hạn projection của attention. Đây là thay đổi của **teacher**, tuyệt đối không sửa module được train bên student để bù thất bại.

### 4.2 Đo teacher tạo được những hướng tín hiệu nào

Sinh ảnh bằng toàn bộ sampler thật, cùng prompt/noise cho các model `a=0`, `a=+epsilon e_j`, `a=-epsilon e_j`.

Dùng một feature extractor frozen `phi(x,c)` để đọc ảnh. Pilot ưu tiên tái sử dụng clean-UNet features trong `distribution.py`; centering/whitening chỉ fit trên clean design data. Model đọc feature luôn là model sạch đã đóng băng, không dùng UNet đang thay đổi làm thước đo của chính nó. CLIP dùng riêng để kiểm tra prompt fidelity.

Đo response:

\[
R_T[:,j]\approx
\frac{\mathbb E[\phi(G_{+\epsilon e_j}(c,z),c)]-
\mathbb E[\phi(G_{-\epsilon e_j}(c,z),c)]}{2\epsilon}.
\]

Đây là sai phân hữu hạn trên **ảnh sinh hoàn chỉnh**. Nó tránh phụ thuộc vào gradient của một hard-bit extractor và không dùng predicted-x0 một bước để thay cho output thật.

Phải kiểm tra ít nhất hai biên độ, prompt/noise heldout và precision triển khai. Nếu response không nhất quán về dấu hoặc bị nhiễu FP16 lấn át, loại hướng đó. Các mean response lớn nhưng làm hỏng prompt/chất lượng cũng bị loại.

### 4.3 Đo hướng nào truyền qua ordinary learning

Với recipe student `r`, gọi `A_r(D)` là **trainer thông thường, giữ nguyên** chạy trên dữ liệu `D`. Tạo một clean student đối chứng bằng cùng recipe, seed và ngân sách.

\[
F_r(a)=\mathbb E\big[\phi(G_{A_r(D_a)}(c,z),c)\big],
\qquad D_a=\{(c,G_{\theta(a)}(c,z))\}.
\]

Response truyền dấu là:

\[
T_r[:,j]\approx\frac{F_r(+\epsilon e_j)-F_r(-\epsilon e_j)}{2\epsilon}.
\]

`T_r` hỏi đúng điều ta cần: khi owner thay đổi hướng j, output của student được học từ ảnh đó thay đổi theo hướng nào?

Không cần ước lượng toàn bộ ma trận bằng các student dài ngay từ đầu. Có thể sàng lọc hướng bằng gradient/update ngắn rồi xác nhận các ứng viên bằng ordinary training thật. Nhưng tín hiệu cuối phải đo trên **generation đầy đủ**, không chỉ loss giảm hay predicted-x0 score tăng.

Proxy rẻ để chẩn đoán một bước SGD:

\[
\Delta g_r(a)=\nabla_w L_r(D_a;w)-\nabla_w L_r(D_0;w),
\]

\[
\Delta s_r\approx-\eta\,\nabla_w s(G_w)^\top\Delta g_r(a).
\]

Loss `L_r` vẫn là denoising MSE gốc. Gradient score chỉ được owner dùng để đánh giá carrier, không được đưa vào cập nhật của student. Lấy chênh với clean để tránh nhầm hiệu ứng fine-tuning nói chung thành fingerprint.

**Giới hạn của proxy:** công thức trên là xấp xỉ first-order cho SGD; không được dùng để tuyên bố mô phỏng chính xác AdamW, LoRA nhiều bước hay training dài. Với AdamW phải dùng cập nhật thực, bao gồm optimizer state và clipping. LoRA tại khởi tạo cũng có tangent phụ thuộc ma trận A ngẫu nhiên, nên phải đổi initialization khi kiểm tra. Gradient lớn hoặc cosine đẹp riêng lẻ không đủ.

### 4.4 Chọn fingerprint truyền cùng chiều, có ràng buộc chất lượng

Với detector tuyến tính tạm thời `d`, tìm hướng owner `a` tối đa hóa tín hiệu student yếu nhất trong ensemble thiết kế:

\[
\max_{a,d}\ \min_{r\in\mathcal R_{design}} d^\top T_r a,
\]

với:

\[
d^\top\Sigma_{clean}d=1,\qquad
d^\top R_Ta\ge m_T,\qquad
a^\top Qa\le\delta.
\]

`Sigma_clean` chuẩn hóa độ nhiễu tự nhiên của detector. `Q` là xấp xỉ cục bộ chi phí thay đổi ảnh; candidate cuối vẫn phải qua metric/visual audit thật. `m_T` buộc teacher có tín hiệu, không chỉ tối ưu proxy student.

Dùng tín hiệu **có dấu**. Tối đa hóa bình phương tín hiệu có thể chọn hướng tăng score ở student này nhưng giảm ở student khác, làm detector/key không nhất quán. Trong bản thực nghiệm, ưu tiên lower confidence bound trên các minibatch/seed hơn là lấy một điểm ước lượng may mắn.

Sàng lọc đơn giản hoặc tìm kiếm trong không gian nhỏ là đủ cho pilot. Chưa cần thêm genetic algorithm, LLM search hoặc unroll hàng nghìn bước vào phương pháp chính.

### 4.5 Key và detector phải được đóng băng đúng thời điểm

Pilot làm zero-bit ownership detection trước. Một owner dùng một code/hướng trong tập hướng đã được xác nhận. Sau khi có khả năng truyền thật mới mở sang nhiều hướng độc lập và nhiều owner.

Key xác định tổ hợp hướng hợp lệ và phép đọc tương ứng. Không lấy pattern 01/10 công khai rồi gọi là secret key. Không tự nhận bảo mật mật mã: không gian nhỏ có thể đoán, các key có thể tương quan và attacker có thể học giả mạo. Phải kiểm tra wrong-key và attribution nhầm giữa owner.

Chọn `B`, `a`, `d`, key và threshold chỉ trên các split thiết kế/calibration. Đóng băng tất cả trước final student evaluation. Không dùng final student để đổi detector, chọn key hoặc tìm checkpoint teacher đẹp nhất.

Đánh giá cả hai mức:

- **Mỗi ảnh:** score, TPR tại FPR ảnh đã hiệu chỉnh, confidence interval.
- **Mỗi model:** score tổng hợp của một số ảnh/query đã ấn định, so với clean-trained null models.

Mức model phù hợp với mục tiêu truy vết student, nhưng không được lấy kết quả tổng hợp để che việc watermark từng ảnh yếu. Nhiều ảnh từ một clean model không tương đương nhiều clean model độc lập; chưa đủ để khẳng định model-level FPR 1% trên mọi model. Wrong-key test cũng không tự thay thế matched clean-trained control.

## 5. Vì sao dùng surrogate không phải “cải tiến LoRA”

Trong thiết kế này, owner dùng vài learner công khai như dụng cụ đo xem dấu nào dễ được học. Cấu hình learner được cố định; thứ thay đổi là fingerprint/teacher.

Student cuối vẫn là một quá trình độc lập. Không dùng watermark loss, weighted sampling, khởi tạo từ owner, layer selection theo key hoặc checkpoint selection theo detection ở student.

So với `DESIGN.md` revision 2, đây là **đề xuất đưa lại surrogate vào thiết kế owner**, vì proxy VAE-only chưa dự đoán được transfer. Phải ghi rõ thay đổi này, không gọi phương pháp là surrogate-free. Khác với phiên bản `meta.py` cũ, đề xuất không cố định một hướng CLIP rồi tối ưu predicted-x0 score sau ít bước: nó chọn hướng native khả thi và yêu cầu xác nhận response bằng output sinh đầy đủ.

Ensemble thiết kế không cho phép tuyên bố biết hoặc kiểm soát attacker thật. Đánh giá cuối phải có rank, initialization, ngân sách hoặc backbone chưa xuất hiện khi chọn carrier. Nếu chỉ thành công đúng learner đã dùng để thiết kế, claim phải thu hẹp thành surrogate-specific.

## 6. Tiền lệ nghiên cứu và phạm vi novelty

- [Antidistillation Fingerprinting, ICML 2026](https://arxiv.org/abs/2602.03812): thiết kế fingerprint theo động lực học của student trong LLM. Đây là tiền lệ trực tiếp cho ý tưởng tối ưu khả năng truyền, không phải kết quả đã có trên diffusion.
- [HMark](https://arxiv.org/html/2512.00094v1): dùng dịch chuyển h-space chung theo secret và báo cáo truyền qua LoRA. Thiết lập dùng pixel-DDPM CelebA-HQ 256; không thể mang nguyên con số sang text-to-image latent diffusion. Radioactivity không được tối ưu trực tiếp trong loss của họ.
- [T2S, ICASSP 2026](https://arxiv.org/abs/2606.11698): đã dùng simulated stolen model để cải thiện watermark teacher trong classification. “Backprop qua student” tự nó không mới.
- [BitMark](https://arxiv.org/html/2506.21209v2): động lực cho fingerprint thống kê trong biểu diễn đã học; việc đem ý tưởng sang diffusion không tự tạo novelty, và paper đã có thí nghiệm binary diffusion.
- [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904): tham khảo cách tìm tín hiệu yếu sau training; không suy rộng bảo đảm thống kê của LLM sang ảnh mà không kiểm tra giả định.
- [CSF, CVPR 2026](https://junhoo.me/csf): fingerprint hành vi ngữ nghĩa của text-to-image; bài toán lineage của model được fine-tune không đồng nhất với student độc lập học từ ảnh ở đây.
- [radiofp](https://github.com/andrew-jeremy/LLM-Distillation-Radioctive-Tracing): repo proof-of-concept có transfer-operator framing, nhưng dùng mô hình đồ chơi đếm tần suất. Không coi đó là chứng cứ thành công trên diffusion; cũng không tuyên bố thuật ngữ transfer operator là ý tưởng đầu tiên của ta.

**Đóng góp có thể nhắm tới:** lựa chọn fingerprint trong không gian biến đổi native có khả năng truyền qua ordinary denoising training của latent diffusion, có đo response có dấu, ràng buộc utility và kiểm chứng trên learner chưa thấy.

Chỉ có tiềm năng đóng góp nếu chứng minh tốt hơn: native teacher tối ưu detectability đơn thuần; carrier random cùng dung lượng/chất lượng; và proxy chỉ giữ dấu qua VAE. Nếu các đối chứng ngang nhau, chưa có phần method đủ mạnh. Không tuyên bố “đầu tiên” từ lượt survey này.

## 7. Kế hoạch code và phạm vi triển khai pilot

Ưu tiên tạo một nhánh thí nghiệm riêng, tái sử dụng hạ tầng đã kiểm tra; không thay âm thầm các run BitGreen cũ.

| File/module | Công việc cần làm |
|---|---|
| **Mới:** `new_watermark/native_transfer.py` | Orchestrator các giai đoạn basis → teacher response → learning response → chọn/freeze fingerprint → independent evaluation |
| **Mới:** `new_watermark/native_basis.py` | Khai báo danh sách tham số owner được thay đổi; `theta(a)`; apply/reset chính xác; kiểm tra key0/zero coefficient khớp model sạch; xuất native weights |
| **Mới:** `new_watermark/transfer_response.py` | Ước lượng `R_T`, `T_r`; paired common random numbers; kiểm tra hai biên độ; gradient proxy có nhãn rõ; lưu signed effect và bất định |
| **Mới:** `new_watermark/transfer_detector.py` | Feature normalization trên clean; detector đơn giản; đóng băng carrier/key; threshold và wrong-key evaluation; tách image/model statistics |
| `new_watermark/data.py` | Tái dùng model loader/generation; bổ sung manifest split theo prompt family, seed, key, learner; ảnh phải qua đúng serialization/preprocessing như student thật |
| `new_watermark/distribution.py` | Tái sử dụng feature extraction và clean nuisance centering thành helper nếu cần; không dùng PCA/weighted-SFT carrier cũ làm mặc định NTF |
| `new_watermark/meta.py` | Chỉ tham khảo surrogate/update machinery. Không tái dùng `transfer_loss()` predicted-x0 như bằng chứng transfer; thêm numerical audit nếu dùng lại AdamW functional approximation |
| `new_watermark/sampling.py` | Tái dùng full generation để probe. Ghi scheduler, CFG, precision; không mặc định full backprop là cách tối ưu chính |
| `new_watermark/teacher.py` | Tái sử dụng export verification. Thêm đường export cho cập nhật tham số native nếu không phải PEFT adapter; không ép mọi owner update qua `fuse_lora()` |
| `new_watermark/bitgreen_experiment.py` | Tách helper student/evaluation có thể tái dùng; giữ nguyên BitGreen để làm đối chứng; không trộn threshold hoặc score mới vào report BitGreen |
| `bitmark_lora/` ordinary trainer | **Giữ nguyên thuật toán**. Hash trainer và lưu command/config cho từng arm; clean/marked khác dữ liệu, không khác loss hay quy tắc chọn checkpoint |
| **Mới:** `run_native_transfer_gate.sh` | Launcher pilot theo thứ tự gate; chỉ triển khai sau khi duyệt thiết kế này; không tự chạy full suite khi teacher fail |
| **Mới:** `new_watermark/test_native_transfer.py` | Kiểm tra response có dấu, zero perturbation, finite differences, clean cancellation, export equivalence, split leakage và student command invariance |

Các module mới trong bảng đã có bản pilot, cùng `test_native_transfer.py` và
`run_native_transfer_gate.sh`. Các module hạ tầng cũ được tái sử dụng qua lời gọi,
không sửa thuật toán ordinary trainer. `meta.py` và `sampling.py` không tham gia
objective mới: pilot đo bằng ordinary training thực và full pipeline generation.
Các thay đổi sâu hơn trong bảng vẫn là kế hoạch nếu pilot cho tín hiệu tốt.

### Những lỗi đánh giá phải phòng ngay trong thiết kế code

1. **Baseline bước 0 phải được tính khi chọn best.** Run gần nhất ghi bước 60 là best trong các checkpoint đã train, nhưng loss 1,051 vẫn tệ hơn baseline 1. Report mới phải ghi rõ `better_than_clean=false`; không coi “best trained” là cải thiện.
2. Đối chiếu FP16 bằng **clean FP16 cùng pipeline**; FP16 so với reference FP32 trộn precision drift với watermark effect. Vẫn báo riêng mức lệch cross-precision, không xóa nó.
3. Không đổi teacher/student precision, detector hoặc số ảnh âm thầm giữa arm; không lấy green fraction làm bit accuracy.
4. Teacher zero perturbation và unmarked fine-tuning control phải có trong report; tránh ghi nhận generic model drift thành fingerprint.
5. Nếu dùng optimizer proxy, kiểm tra cập nhật thật và xấp xỉ trên cùng batch/state, rồi kiểm tra ranking carrier có tương quan với student thật hay không.

### Output dự kiến

Thư mục kết quả nhẹ: config, source hashes, split manifests, `native_response.json`, `transfer_response.json`, `teacher_metrics.csv`, `student_metrics.csv`, `wrong_key_metrics.csv`, `quality_metrics.csv`, `decision.json`.

Model/adapter/cache nằm ở `output_artifacts` hoặc runtime artifact root; ảnh ở `output_image`. Khi tải kết quả, không dereference symlink chứa model nặng. Không đưa checkpoint vào cùng gói JSON/CSV.

## 8. Pilot cần chạy trước khi mở rộng

### Gate A — native teacher có nhúng được dấu hay không?

- SD1.4 trước để so được với lịch sử; chưa đổi backbone cùng lúc với mọi thành phần khác.
- 8 hướng owner; paired clean/± perturbation trên design prompts, rồi xác nhận biên độ trên prompt/noise khác.
- Detector đơn giản và fingerprint được freeze trước validation; evaluation bằng native model đã export, precision triển khai.
- Báo teacher TPR/FPR, score effect, PSNR/LPIPS, CLIP, độ đa dạng và visual samples. Một global tint hay giảm diversity không đủ.
- Mốc vận hành đề xuất: teacher TPR ≥80% tại nominal image FPR 1%, PSNR trung bình ≥25 dB; ghi thêm phân vị xấu và confidence interval. Đây là mục tiêu pilot, không phải định luật hoặc ngưỡng được đảm bảo.
- Không đạt thì ghi thất bại của native embedding. Không tiêu compute student để bù teacher yếu, không hạ threshold sau khi xem kết quả.

Nếu chỉ có model-level signal mà per-image teacher gate không đạt, báo như một nhánh fingerprint phân phối riêng; không gọi nó là teacher watermark mạnh theo tiêu chí cũ.

### Gate B — có dấu qua ordinary learning thật hay không?

Ba bộ dữ liệu cùng caption/count: clean, native fingerprint chọn theo teacher-only, và fingerprint chọn theo transfer response. Hai recipe student ordinary cố định để sàng lọc; không đổi recipe theo kết quả.

Pilot bắt đầu 500 ảnh mỗi bộ, 500 updates mỗi student để kiểm tra có đáng mở rộng. Tổng 3 × 2 × 500 = **3.000 student updates**, không phải 3.000 ảnh. Đây chỉ là phép thử ngắn; kết quả âm ở ngân sách này không chứng minh không thể truyền sau training dài.

Checkpoint cuối của pilot hoặc checkpoint chọn bằng heldout denoising MSE phải được quy định trước. Không dùng detection để chọn checkpoint student. Nếu student chưa học được phân phối ở mức utility tối thiểu đã đặt, ghi “pilot chưa đủ để kết luận”, thay vì coi watermark bị xóa thành công.

Chỉ mở lên 2.500 bước khi có native teacher tốt và hiệu ứng truyền vượt clean control trên design evaluation, hoặc có chẩn đoán rõ vì sao ngân sách ngắn chưa đủ. Không mở thêm hàng chục nhánh chỉ vì một mean-score nhích lên.

### Gate C — independent confirmation

Đóng băng toàn bộ method. Dùng prompt/key/training seed mới; đưa vào rank và một cấu hình learner chưa dùng để chọn carrier. Kiểm tra thêm backbone khác, tỷ lệ dữ liệu teacher bị pha loãng và thay caption khi ngân sách cho phép.

Báo đủ teacher strength, student transfer, matched clean FPR, wrong keys, utility và compute. Khoảng tin cậy trên ảnh của một student không thay thế lặp lại training seed. Kiểm định model-level FPR cần một tập null models phù hợp và phải công khai phạm vi của tập đó.

## 9. Điều kiện để tiếp tục hoặc bỏ hướng

Tiếp tục khi có một hướng native giữ được chất lượng, detector phân biệt được trên ảnh mới, và dấu đi cùng chiều qua nhiều learner thông thường; sau đó transfer-selected phải vượt teacher-only/random ở ngân sách tương đương.

Bỏ hoặc thay carrier nếu chỉ tăng proxy mà student thật không tăng; detector nhận nhầm generic fine-tuning; hiệu quả chỉ xuất hiện khi sửa student; hoặc chỉ có dấu nhờ ảnh bị đổi rõ nét/chất lượng suy giảm.

Phương pháp này có lý do để thử vì sửa đúng hai chỗ các run cũ chưa giải quyết: **teacher có tạo được dấu đó không**, và **ordinary training có học lại dấu đó không**. Hiệu quả và novelty vẫn phải được chứng minh. Tài liệu này không đề xuất tăng step cho pipeline cũ rồi kỳ vọng kết quả khác.
