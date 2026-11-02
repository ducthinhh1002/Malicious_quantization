> Ghi chú cập nhật: tài liệu bên dưới là bàn giao thiết kế ban đầu. Code đã được hoàn thiện tiếp: xem README.md và test_results trên VM4. File teacher.py đã đổi thành refine_teacher.py để tránh trùng module harness.

# Bàn giao: code tiếp nhánh Projected native Transfer90

## Yêu cầu

Hoàn thiện nhánh **riêng trong `project_native_transfer90/`**, không sửa `project_native/` hay các source/manifest của thí nghiệm cũ. Mục tiêu: **TPR trên từng ảnh student ≥90%**, tại ngưỡng verifier đã đóng băng cho FPR mục tiêu 1%. Đây là mục tiêu thực nghiệm, chưa có bảo đảm đạt được. Không dùng aggregate recovery 100% thay cho TPR từng ảnh.

User đã yêu cầu bỏ `margin_native`, giữ `structure_margin_native`, rồi đưa nhánh mới vào hàng chờ sau nhánh đang chạy. Không dừng vì TPR/chất lượng không đạt; vẫn ghi báo cáo. Lỗi toàn vẹn dữ liệu, nonfinite, runtime phải được xử lý rõ ràng.

## Trạng thái thực tế — đọc trước khi thao tác server

- VM4: `root@171.232.252.207`. Dùng SSH master `/tmp/wmq-vm4-oct26` nếu còn sống; không ghi mật khẩu vào file.
- Python: `/root/logs/gate_env/bin/python`.
- Source lịch sử: `/root/logs/threshold_quality_20261022`.
- Harness trên server: `Malicious_quantization/bitmark_lora/` bên trong source lịch sử. Snapshot local đã chuyển harness thành `project_native/src/`.
- Campaign đang chạy: `/root/logs/threshold_quality_20261022/structure_margin_20261026`.
- Check lúc 11:38 ngày 26/10/2026 UTC+7: `structure_margin_native`, stage `02_data`, teacher train xong600 bước, đang sinh ảnh.
- **Đã hủy lịch `margin_native` bằng thay supervisor**, không sửa source lịch sử. Supervisor cũ2223976 đã dừng; worker sinh ảnh2359912 được giữ nguyên.
- Supervisor mới2363959 chạy `/root/logs/project_native_transfer90/preserve_current.py`. Nó đợi worker sinh ảnh xong, chạy tiếp toàn bộ `structure_margin_native`, bỏ `margin_native`, cập nhật status cũ thành completed.
- Bằng chứng: `queue_override_request.json`, `replacement_launcher.json`, `replacement_supervisor.log` trong campaign đang chạy. PID có thể đổi: kiểm tra cmdline và trạng thái mới trước thao tác.
- **Nhánh Transfer90 chưa được enqueue. Chỉ `preserve_current.py` đã được đưa lên server.** Các file khác mới là bản nháp local, chưa test syntax/runtime đầy đủ, chưa commit/push.

## Code đang có: tiếp tục hoàn thiện, không coi là chạy được

| File | Đã viết | Cần làm tiếp |
|---|---|---|
| `bootstrap.py` | Import harness lịch sử read-only qua `PN90_BASE_ROOT`; hỗ trợ layout server/local | Kiểm tra import resolution trên VM4 |
| `objectives.py` | Loss thiếu margin theo từng ảnh, ba view; LR plateau+cosine; bit-margin phụ | Unit test giá trị, gradient, biên, nonfinite |
| `teacher.py` | Warm-start Projected native, bảo vệ quality bằng QP; thêm surrogate SD1.4 đông cứng; loop checkpoint lấy từ engine đã có | Review toán/graph, test gradient, smoke GPU thực; kiểm tra VRAM và thời gian |
| `structure_metrics.py` | Copy MS-SSIM của nhánh trước | Test/sanity với đầu vào giống nhau và kích thước hợp lệ |
| `student_driver.py` | Wrapper trainer gốc: LR mới, checkpoint chỉ lưu trainable weights + optimizer/RNG | Test hook Accelerate save/load thực; audit state; làm rõ resume |
| `campaign.py` | Init, pin nguồn, queue, teacher→data→student→eval | Hoàn thiện verifier, test stages, resource/storage, import và resume |
| `run.sh` | Gọi campaign bằng Python đã chuẩn bị | Test `bash -n`; mặc định enqueue đủ workflow |
| `preserve_current.py` | Supervisor thay thế nhánh cũ | Đã deploy; không giết worker/supervisor đang được giữ |

**Thiếu hẳn `verification.py` và tests. `campaign.py` đang gọi `verification.py` ở stage15, nên chưa được chạy full.**

## Thiết kế nhánh đã chọn để implement

### Teacher

Khởi tạo từ teacher **Projected native đã refine xong**, không quay lại baseline k1:

`/root/logs/threshold_quality_20261022/pareto_quality_20261025/projected_native/teacher_fp/refined`

- Thêm800 updates, LR5e-6, batch4, native microbatch1, native auxiliary mỗi2updates.
- Native forward nguyên PNDM50/CFG7.5; gradient qua10 denoiser calls cuối. Đây là gradient truncated, phải khai rõ.
- Giữ loss gốc `threshold_cfg_v1`, quality LPIPS +0.1pixel +0.25(1−MS-SSIM).
- Giữ projection của displacement AdamW lên các ràng buộc descent bậc một: original loss, ownership auxiliary, quality. Không tuyên bố đảm bảo quality toàn cục.
- Tối ưu watermark qua3view:
  1. Ảnh teacher native đã qua RGB8 STE.
  2. Encode bằng VAE **student SD1.4**, posterior sample, decode và RGB8.
  3. Cùng latent student, thêm noise ở t∈{50,150,300}, dự đoán x0 bằng **student proxy đông cứng**, decode và RGB8.
- Proxy là historical Projected native student snapshot10000:
  `/root/logs/threshold_quality_20261022/projected_student10000_20261025/student10000/students/lora_r32_conv_wm/snapshot-10000`.
- Proxy chỉ train trên training images trước đây; teacher không đọc test scores để tính loss. Đóng băng toàn bộ proxy; gradient qua input của nó về teacher. Student cuối train độc lập từ base SD1.4 sạch.
- Đây là **surrogate bền qua VAE/denoising**, KHÔNG phải unroll đầy đủ quá trình train student, không chứng minh truyền được90%.
- Sigma clean đã pin≈0.3523001844. Tau student≈0.8541307449.
- Ramp200steps: floor raw/cycle từ tau+1sigma đến tau+2sigma; raw còn bảo vệ score của ảnh Projected native gốc tương ứng. Denoise floor từ tau+0.5sigma đến tau+1sigma.
- Với từng ảnh/view, loss `relu(target-score)^2 / sigma^2`. Gộp view bằng0.5mean+0.5max; sau đó lấy mean các ảnh. **Không gọi đây là CVaR quần thể**, không lấy mean score trước hinge.
- Bit margin phụ trọng số0.1, floor signed normalized coordinate0.25; không thay key/verifier.
- Normalize gradient quality1×, ownership1.5× norm original, cap gain100. Cần kiểm tra các nhánh gradient thật có tồn tại/finite, và proxy không nhận parameter gradients.

### Student cuối

- SD1.4, LoRA rank32 attention+conv; training5000 ảnh teacher mới, batch4, tổng15000steps.
- Loss denoising gốc; **không key, reader, watermark loss hoặc proxy weights trong student training**.
- LR1e-4 đến10000steps, cosine xuống2e-5 ở15000.
- Snapshot7500/10000/12500/15000; chọn **min validation denoising MSE**, tie chọn bước sớm; không chọn bằng test hoặc detector. Giữ convention này để không đổi định nghĩa thí nghiệm truyền mù.
- Validation1000/test1000 như protocol cũ; ghi rõ test đã được xem nhiều lần ở các nhánh lịch sử → exploratory, chưa phải fresh confirmatory test.
- Không tự thêm Min-SNR trong lượt này: chưa có bằng chứng nó tăng truyền watermark; thay nhiều yếu tố khó phân tích.

## Việc cần code theo thứ tự

### 1. Hoàn thiện `verification.py`

Đọc `config.json`, `scores/*.json`, `carrier/key.json`, `result/threshold_student.json`, `result/selection/...`, `result/student_summary.json` và utility.

- Dùng verifier/key/PCA/threshold cũ, pin SHA256; threshold không được refit bằng output mới.
- Kiểm tra n, image hashes, finite coordinates, số bit8, `score≈mean(coordinates*signs)`.
- Primary: số ảnh score≥tau / n, Wilson95%CI; bool point estimate≥0.90; bool CI lower≥0.90 **tách riêng**.
- Bit accuracy từng ảnh = tỷ lệ sign(coordinates) đúng key; báo mean, histogram, exact-message fraction. Zero phải theo convention cũ: sign(0)=0, không match ±1.
- Aggregate = sign(mean(coordinates)); báo riêng, không đặt tên gây nhầm với per-image.
- Báo clean controls tại **cùng ngưỡng**: unadapted và historical clean-trained student (đã có scores), Wilson CI. Clean-trained là historical7500, chưa matched15000; nêu giới hạn.
- Teacher native TPR/quality, student LPIPS/FID/KID/CLIP; FID reference là clean teacher chứ không phải real COCO.
- So sánh với Projected7500 và10000, paired TPR difference/bootstrapCI chỉ khi prompt/seed/image order thực sự khớp.
- Xuất `result/verification.json`, `result/per_image.csv`, `result/SUMMARY.md`; đánh dấu cảnh báo khi chất lượng giảm/TPR dưới90, không raise do scientific metric.
- Không được ghi kết quả90% khi chưa đo; không có logic hạ threshold để đạt mục tiêu.

### 2. Review/fix code nháp

- `teacher.py`: kiểm tra shape/scaling `predicted_x0`, CFG unconditional/conditional ordering, FP32 VAE/reader vs BF16 UNet, checkpoint closure recomputation, graph retention giữa3views, không giữ graph qua microbatch.
- Xem `bootstrap.Path` trong proxy hash/load: chạy được do Path import trong bootstrap nhưng nên import Path rõ ràng cho sạch.
- Kiểm tra loss native actual RGB8 có phù hợp forward sampler production không; log STE và truncated gradient.
- `student_driver.py`: hook `weights.clear()` chỉ bỏ base-model serialization; phải chứng minh optimizer/scaler/scheduler/RNG vẫn lưu, trainable state round-trip chính xác. Không chỉ test file tồn tại.
- Wrapper hiện **từ chối resume**, để tránh upstream trainer resume giữa epoch sai order. `campaign.py` cũng từ chối khi có checkpoint dở. Nếu thêm resume, phải làm đúng data order/RNG/update counts, không âm thầm reset optimizer hoặc ghi đè kết quả.
- `campaign.py` dùng `complete.json` của upstream trước stage15; status toàn campaign chỉ completed sau verification. Đảm bảo resume không bỏ verifier vì marker sớm.
- Pin thêm provenance cần thiết: toàn bộ carrier payload, proxy training data manifest, code của các dependency/vendored trainer, source teacher training image manifest. Kiểm tra hash không chỉ tính rồi ghi.
- Quy trình initialization chưa có rollback/lock toàn bộ tạo run; kiểm tra chạy lặp và collision tên timestamp. Không sửa manifest thí nghiệm cũ để vượt check.
- Resource: lần đọc disk trống≈16GiB, sẽ giảm khi nhánh hiện tại sinh ảnh. Estimate peak thật; dùng tmpfs cho merged model như bản nháp. Không xóa data/checkpoints của người dùng. Compact checkpoints giúp giảm dung lượng, nhưng generation/validation vẫn cần tính đủ.
- `data()` chỉ kiểm tra adapter khi đã có data_complete; nên verify image manifests đầy đủ khi reuse.
- Dependency waiter phải phát hiện supervisor chết, không chờ vô hạn; status lỗi rõ ràng. Không tự chạy đè nhánh hiện tại.

### 3. Tests trước enqueue

- AST/py_compile tất cả Python, `bash -n run.sh`, import trên Python VM4 đúng env.
- CPU unit tests: worst-view loss không bị ảnh/view tốt triệt tiêu, gradient kéo score thiếu lên, gradient0 khi đã đạt; sigma/shape/nonfinite; LR boundaries10000/15000; bit zero tie; verifier sample counts/CI/score consistency; selection không đọc test; checkpoint save/load.
- Test lightweight mocked orchestration: stage failure không được marker completed; scientific shortfall vẫn xuất report; rerun không launch duplicate; cancelled margin không xuất hiện trong lượt tiếp theo.
- GPU actual-model smoke1update: teacher weights thay đổi, native/channel gradient finite và nonzero, proxy weights unchanged, no final student used; ghi peakVRAM/time. Không coi smoke là kết quả khoa học.
- Nếu GPU chưa đủ rảnh, để smoke là stage đầu sau dependency và **nêu rõ chưa test GPU**, không tuyên bố hết lỗi runtime.
- Pin code manifest sau khi sửa xong và test; nếu phải sửa sau initialize, tạo run mới hoặc quy trình version rõ ràng, không sửa hash im lặng.

### 4. Deploy và enqueue

- Upload **thư mục mới** đến `/root/logs/project_native_transfer90/`. Trên server hiện đã có `preserve_current.py` đang chạy; không overwrite nó khi chưa cần.
- Giữ source lịch sử `/root/logs/threshold_quality_20261022` và local `project_native` nguyên trạng.
- Sau test, `bash /root/logs/project_native_transfer90/run.sh` phải initialize/enqueue full flow một lần.
- Kiểm tra PID/cmdline/status/log chứng minh `waiting_for_dependency` đúng nhánh structure hiện tại, sau đó tự chạy Transfer90.
- Xác nhận `margin_native` không bị revive bởi supervisor cũ hay queue khác.
- User chưa yêu cầu push ở lượt này; không tự suy ra cần push.

## Dữ liệu đã đọc để định hướng (1000test images, cùng frozen student threshold)

| Nhánh | TPR | LPIPS student | FID | Steps |
|---|---:|---:|---:|---:|
| k1 baseline |75.0%|0.55045|54.444|7500|
| teacher continuation |82.3%|0.55022|54.887|7500|
| Projected native |79.2%|0.52222|53.358|7500|
| Projected native student dài |82.1%|0.53312|53.453|10000|
| Balanced native |80.2%|0.51730|52.792|7500|
| quality |80.8%|0.54666|55.281|7500|
| quality_transfer proxy ngắn |80.5%|0.54590|55.087|7500|
| scale0.85 long |52.3%|0.57323|54.858|22500|
| scale0.85 conv_restore |57.5%|0.53540|54.230|7500|
| clean-trained control |0.2%|0.51559|51.951|7500|

Lượt10000 là seeded replay, không phải exact resume của7500 (max weightdiff≈0.000471). Không quy toàn bộ tăng TPR cho2500steps. Dữ liệu cho thấy tăng thời gian một mình không đủ, và proxy unroll ngắn cũ chưa đem lại lợi ích.

## Tài liệu đã survey, dùng cho README/lý giải — không coi là đảm bảo90%

1. HMARK: https://arxiv.org/html/2512.00094v1 — semantic h-space và LoRA radioactivity; AppendixB dùng LR1e-4, ranks8/16/32. Các tỷ lệ trong paper không so trực tiếp với TPR@FPR1% của repo; clean control trong paper có FPR khác.
2. Radioactive Watermarks in Diffusion and Autoregressive Image Generative Models: https://arxiv.org/html/2506.23731v1 — cần xem chuỗi truyền qua VAE và noising/denoising. Động cơ cho surrogate channel, không phải bằng chứng surrogate mới sẽ thắng.
3. BitMark: https://arxiv.org/abs/2506.21209 — radioactivity khác cơ chế và teacher; không bê kết quả sang diffusion teacher này.
4. Min-SNR: https://arxiv.org/html/2303.09556v3 — đổi weighting timestep có lợi convergence trong thiết lập paper, chưa chứng minh lợi watermark; không bật mặc định trong pilot.
5. Diffusers LoRA docs: https://github.com/huggingface/diffusers/blob/main/docs/source/en/training/lora.md — LR1e-4 là điểm xuất phát thông dụng, không chứng minh optimum.
6. Statistical Learning with CVaR: https://arxiv.org/abs/2002.05826 — tham khảo tư duy tail risk; loss ba-view hiện tại chỉ mean/max finite-view, không nhận vơ CVaR hoặc bảo đảm quantile90%.

## Câu nhắc ngắn để giao agent tiếp theo

“Đọc `project_native_transfer90/NEXT_STEPS.md`, review code nháp rồi hoàn thiện verifier/tests/runtime guards. Giữ frozen verifier, không cấp key cho student, không sửa project_native. Test và deploy thư mục mới, enqueue sau structure_margin_native; margin_native đã bị hủy. Mục tiêu TPR từng ảnh90% chỉ được xác nhận bằng kết quả đo thật.”
