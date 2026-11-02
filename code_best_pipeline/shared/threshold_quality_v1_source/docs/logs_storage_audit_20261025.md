# Rà dung lượng /root/logs — 25/10/2026, khoảng 08:52 ICT

Chỉ kiểm tra, chưa xóa/move bất cứ checkpoint, ảnh hay cache nào.
Filesystem còn khoảng40GiB, dùng99%; /root/logs chiếm khoảng170GiB.
Các số dưới dùng GiB=2^30 bytes. Đã kiểm tra hardlinks theo inode/nlink,
không cộng kích thước các bản hard-link thành dung lượng giải phóng ảo.
Nguồn chi tiết: [JSON audit](../../output_artifacts/logs_storage_audit_20261025.json).

## Luồng đang hoạt động và dữ liệu cần giữ

`threshold_quality_20261022/native_quality_20261025`: clean_control đang ở
stage06_train; supervisor1456465, student launcher1458032 và trainer1458070
còn hoạt động lúc kiểm tra. Sau đó chạy native_full và continuation.
Không phát hiện tiến trình trong /root/logs đang chạy các campaign cũ khác;
điều này cùng với config/code dependency được dùng để phân loại, không coi
việc không có file descriptor mở là bằng chứng đủ để xóa file model.

Giữ:

- Toàn bộ native_quality_20261025 và source code/manifests được pin.
- kappa1_manojb_5k/source: config/assets/prompts/carrier/thresholds; clean
  train/validation/test; unadapted test; source checkpoint-3000 và checkpoint
  proof. Các ảnh này được hard-link vào cả ba nhánh mới.
- kappa1_manojb_5k/students/k1_evolution/images/lora_r32_conv_wm: new visual
  audit lấy ảnh baseline ở đây. Giữ result/student_summary.json của baseline
  scale1.0 và scale0.85 để final comparison đọc. Các manifest/campaign code cũ
  còn được verify trong helper imports nên không xóa nguyên project cũ.
- runtime/cache: Manojb SD2.1, CompVis SD1.4 và openai CLIP; gate_env và
  torch_shared (symlink sang environment MinhNQN) đang phục vụ trainer.
  Không xóa nguyên /root/logs/runtime vì có cả model cần dùng và cache cũ.

## Các nhóm lớn không nằm trong dependency hiện tại

| Đường dẫn dưới /root/logs | File blocks GiB | Có dùng cho campaign mới? | Giá trị còn lại |
|---|---:|---|---|
| output_image | 19,584 | Không thấy dependency hiện tại | 47.957 ảnh/artifact LALT/signature/native cũ, dùng xem/tái phân tích lịch sử |
| semantic_trajectory_20261022 | 12,307 | Không | Run trajectory cũ: teacher/model/checkpoints/ảnh; xóa cả run mất artifact lịch sử |
| runtime/semantic_artifacts | 19,301 | Không | Model exports positive/negative của semantic relation cũ |
| runtime/lcf_artifacts | 4,157 | Không | Prefix caches của LCF cũ, có thể tính lại nếu còn code/model/seed |
| results | 10,464 | Không thấy dependency campaign mới | Nhiều run teacher/native smoke; giữ JSON/log/metrics khi dọn models/ảnh |

Nếu xóa cả năm nhóm, regular-file blocks có thể giải phóng **65,813GiB**;
đã kiểm tra không có hardlinks nằm ngoài nhóm này. Đây là ước tính kỹ thuật
về dung lượng, không phải đề xuất xóa toàn bộ lịch sử nghiên cứu.

Trong nhóm results, hai model directories chiếm khoảng7,948GiB:

```
/root/logs/results/native_unet_20261019_smoke/models/teacher_4/
/root/logs/results/teacher_gate_improved/rollout/models/
```

Đây là ứng viên dọn có chọn lọc trước khi cân nhắc xóa cả results; current
campaign không load chúng. Cần giữ adapter/base hoặc chấp nhận không dựng
lại đúng model cũ nếu đó là bản weights duy nhất của run.

## Checkpoint và export cụ thể trong threshold_quality

| Nhóm | GiB | Hiện tại có đọc? |
|---|---:|---|
| kappa1_manojb_5k/source/teacher_fp/candidates/kappa_1.0/merged_pipeline | 4,807 | Không; vòng mới load base snapshot + adapter checkpoint3000 |
| checkpoint200/400 của bốn refinements cũ | 3,015 | Không; các run đã complete, current queue không resume chúng |
| checkpoint500/1000/1500/2000/2500 của source kappa1 | 1,884 | Không; checkpoint3000 mới là warm-start bắt buộc |

Bốn refinements cũ: transfer_refine_20261024/{quality,quality_transfer} và
minhnqn_refine_20261024/{quality_gradient,quality_cka}. Mỗi intermediate full
checkpoint khoảng0,377GiB, gồm adapter và state để resume. Giữ final adapter,
adapter_config, trained.json, losses, metrics và tốt nhất checkpoint600.

Merged source export có thể dựng lại từ pinned base + checkpoint3000. Tuy
nhiên merge_test.json hiện ghi hash export: chỉ xóa merged_pipeline mà bỏ lại
proof sẽ khiến thao tác reuse/merge cũ báo model missing/changed. Nếu dọn cần
archive proof và cập nhật cách rebuild có kiểm soát, không sửa manifest để
bỏ qua kiểm tra. Audit này chưa thực hiện thay đổi đó.

## Ảnh và search của các run gần đây

- Bốn refinements hoàn tất có tổng21,794GiB unique image blocks. Nếu bỏ cả
  bốn images trees chỉ thu hồi18,587GiB vì3,206GiB vẫn còn hardlink ngoài nhóm,
  gồm clean/unadapted references mà source/queue mới giữ.
- Trong từng refinement, teacher_wm_train khoảng2,15GiB; ba bộ ảnh validation
  snapshots tổng khoảng1,23GiB. Không được đọc bởi queue mới. Có thể sinh lại
  từ teacher/student checkpoints + prompt/seed; phải giữ checkpoint tương ứng.
  Nên giữ student test và teacher test/validation để tiếp tục xem lỗi ảnh.
- queued_research_20261023 + kappa1_manojb_5k/evolution:12,498GiB unique,
  thu hồi10,781GiB nếu xóa cả hai. Đây là KernelSHAP/Pareto/search cũ, không
  dùng trong ba jobs mới. Cần archive bảng attribution, vectors, JSON, logs
  nếu muốn giữ kết luận mà bỏ ảnh cache.
- scale085_recovery_20261024:12,190GiB unique theo subtree, nhưng chỉ thu hồi
  khoảng7,672GiB khi xóa riêng subtree do hardlinks. Không còn job recovery
  đang chạy; dùng làm đối chứng lịch sử, không phải dependency của queue mới.

## Thứ tự dọn đề xuất nếu cần

1. Old semantic model exports, hai old smoke/model directories và LCF prefix
   caches: khoảng31,4GiB, không đụng model snapshot đang dùng hoặc ảnh test mới.
2. Old intermediate checkpoints của bốn refinements và source: thêm4,90GiB,
   giữ source3000/final adapters/checkpoint hiện đang chạy.
3. Old search images và teacher_wm_train/validation snapshots của completed
   refinements, sau khi giữ reports và quyết định mức tái lập muốn giữ.

Không cộng máy móc các subtree sizes có overlap hoặc hardlink; muốn chọn tập
đường dẫn cụ thể để xóa cần kiểm lại inode cho đúng tập đó tại thời điểm dọn.
