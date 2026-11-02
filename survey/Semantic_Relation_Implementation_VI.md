# Cơ chế thay thế: quan hệ nội dung–ánh sáng trong native teacher

Ngày 21/10/2026. Đây là cơ chế thay thế **đang kiểm nghiệm**, chưa có bằng chứng
student mới tốt hơn LALT. Không đổi thuật toán, loss hay đầu vào bí mật của attacker.

## Vì sao chọn hướng này

LALT đã cho teacher mạnh nhưng student LoRA yếu; sửa reader chỉ tăng AUROC nhẹ,
contrast/replay không vượt gate, và bảy mức spectral rotation chưa đạt đồng thời
tín hiệu và chất lượng. Các kết quả cụ thể nằm trong
[báo cáo nghiên cứu lại](LALT_Rethink_Student_Transfer_VI.md).

| Ứng viên | Điểm có ích | Quyết định hiện tại |
|---|---|---|
| Residual latent + bit reader cũ | Teacher đã đọc được dấu | Giữ làm baseline, không tiếp tục tăng loss trên cùng cơ chế |
| Spectral rotation đồng nhất | Có target denoiser chính xác, giữ Gaussian prior | Screen hiện tại chưa qua quality/signal gate |
| Dấu quan hệ ngữ nghĩa | Một quy luật lặp lại giữa nội dung và cách thể hiện, có thể học qua conditional denoising | Triển khai và kiểm tra bằng ordinary student thật |
| Warp hậu xử lý ảnh | Có tiền lệ học lại dấu từ dữ liệu | Chưa giải quyết yêu cầu dấu nằm trong native teacher |

[Concept Sliders, ECCV 2024](https://sliders.baulab.info/) cung cấp cách học thay đổi
thuộc tính bằng chênh noise prediction giữa hai mô tả. Mình dùng nguyên lý này
để xây target cho teacher, không coi paper là bằng chứng fingerprint truyền qua
student. [HMark](https://arxiv.org/html/2512.00094v1) là động lực thử dấu ngữ nghĩa,
nhưng thí nghiệm pixel-DDPM của họ khác SD text-to-image ở đây. Chưa đủ cơ sở gọi
cơ chế mới là tốt nhất, hay khẳng định tính mới khoa học.

## Bản triển khai v1

- Bốn nhóm nội dung: động vật, phương tiện, thức ăn, đồ nội thất. Key cố định
  `[+1, -1, -1, +1]`; key đảo là đối chứng. Thuộc tính là ánh sáng khuếch tán mềm
  so với ánh sáng có hướng và bóng rõ. Không chủ động sửa màu hay danh tính vật thể.
- Caption đã chỉ định ánh sáng, có nhiều nhóm hoặc ngoài miền được đưa vào
  preservation; không cố ép thuộc tính trái prompt. Rule chỉ tạo supervision
  lúc học owner, **không chạy khi native teacher sinh ảnh**.
- Với cùng latent/timestep, target conditional là
  `eps_base(c) + beta/7.5 * key[group] * (eps_base(c+) - eps_base(c-))`.
  Khi unconditional được giữ nguyên, guided prediction dịch đúng `beta`, không
  bị nhân nhầm thêm 7,5 lần. `beta` cố định trước: 1,5 và 3.
- Owner học 500 bước, rank16 attention, AdamW `1e-4`; loss conditional MSE cộng
  `0.5` unconditional preservation và `0.5` preservation ngoài miền/xung đột.
  75% bước dùng latent clean được thêm noise; 25% dùng DDIM prefix của owner hiện tại.
  Nhóm được xáo trộn cân bằng để bước on-policy không luôn rơi vào cùng một nhóm.
- Adapter chỉ là cách tiết kiệm bộ nhớ khi **học owner**. Xuất UNet đã merge,
  bỏ adapter, nạp lại từ đĩa và so prediction trước/sau. Teacher cuối nhận caption
  gốc và Gaussian noise, trọng số cố định ở mọi timestep; không nhận key hoặc trigger.

Code chính: [semantic_relation.py](../new_watermark/learning_channel/semantic_relation.py)
và [semantic_experiment.py](../new_watermark/learning_channel/semantic_experiment.py).

## Thiết kế phép thử

COCO source ID và caption trùng được loại giữa các split; loại các split final
cũ đã dành riêng. Owner design 256 prompt; dữ liệu cho mỗi student 1.024 prompt;
validation 128; query 128; preservation train/eval 64/32. Dữ liệu đều cân bằng bốn
nhóm ở các split chính. Miền eligible trong pool được xét là 14.646/36.876 source
(khoảng 39,7%); không tuyên bố coverage toàn bộ prompt tự nhiên.

Trước native training, screen 32 design prompt bằng cặp prompt ánh sáng đối lập.
Đây chỉ là positive control cho khả năng điều khiển/đo thuộc tính. Probe là CLIP
đóng băng với hai cặp mô tả khác wording target, không fit trên kết quả student.
Nó vẫn có thể mang thiên lệch CLIP; cần thêm người đánh giá/probe độc lập nếu pilot tốt.

Teacher validation đo **quan hệ cấp model**: trung bình chênh attribute score so
với ảnh clean cùng prompt/noise trong từng nhóm, nhân dấu key rồi trung bình đều
bốn nhóm. Một dịch chuyển phong cách đồng nhất được triệt tiêu bởi key cân bằng.
Gate yêu cầu cả key thuận và đảo: độ lớn >=0,002; CI99% theo prompt đúng phía 0;
ít nhất 3/4 nhóm đúng hướng. Với cả validation và preservation: mean LPIPS <=0,08,
p95 <=0,12, PSNR p05 >=25 dB, mean CLIP content drop <=0,02. Ngưỡng được đóng băng
trước khi chạy, không hạ khi thấy kết quả.

Chọn beta nhỏ nhất qua gate. Sau đó mới mở query và sinh PNG/caption cho bốn student:
clean, key thuận, key đảo, trộn 50/50 hai key **trong từng nhóm**. Trộn key là đối
chứng dữ liệu, không được gọi là một native teacher thứ ba.

Student dùng nguyên trainer upstream đã pin, attention LoRA rank4, LR `1e-5`,
batch2 × accumulation2, 2.500 bước, ordinary denoising MSE, final checkpoint cố định.
Tất cả bắt đầu từ cùng clean SD1.4. Student không nhận checkpoint teacher, group,
key, latent hoặc loss fingerprint. Clean student được học lại vì dataset mới khác
dataset baseline; không tái dùng checkpoint clean cũ như thể là matched control.

CI chỉ phản ánh khác biệt giữa prompt cho một training seed. Nó **không phải**
model-level false-positive rate. Bốn nhóm chỉ có sáu key cân bằng, chưa đủ không
gian key cho ownership thực tế. Pilot rank4 cũng chưa chứng minh sống sót qua
mọi LoRA. Chưa có đánh giá paraphrase thực thụ trong v1.

## Tái sử dụng và vận hành

Tái dùng base SD1.4, VAE, CLIP, annotation và clean latent cache đã checksum.
Không khởi tạo cơ chế mới bằng watermark teacher thất bại cũ. Baseline, ảnh và
checkpoint cũ được giữ nguyên. Snapshot code mới độc lập; mỗi stage chạy subprocess,
giới hạn 32 GiB và chờ ít nhất 48 GiB trống, không dừng tiến trình khác.

```bash
bash run_semantic_relation.sh --output /root/logs/results/semantic_relation_NEW
```

Deployment đang dùng `semantic_relation_20261021_002025Z`:
`/root/logs/results/semantic_relation_20261021_002025Z`.
Launcher chạy smoke native export trước rồi mới chạy pilot. Bộ 55 unit tests
trên server đã pass; trạng thái GPU và các kết quả sẽ nằm trong `status.json`,
`positive_control.json`, `selection.json`, `results.json`. Không có `results.json`
student khi teacher fail gate.

Kiểm tra và tải báo cáo về máy:

```powershell
python server_review/scripts/check_semantic_relation.py
```

Ghi nhận kiểm tra đầu tiên: smoke positive-control 6/8 prompt có attribute contrast
đúng chiều (mean 0,001849). Native key thuận đã merge/reload, max prediction error
`6,91e-6`; đây là kiểm tra phần mềm, không phải kết quả fingerprint. Quan sát một
cặp ảnh validation cho thấy clean SD1.4 đã bỏ sót chủ thể “elephants” trong caption,
và ảnh native hai bước gần như giống clean. Không loại cặp này sau khi nhìn ảnh.
Sai lệch nội dung của base có thể làm yếu quan hệ được học; grouping v1 dựa caption,
chưa xác minh vật thể bằng một detector độc lập. Cần giữ giới hạn này khi diễn giải.

07:26 giờ Việt Nam: smoke hoàn tất cả hai key, native reload error tương ứng
`6,91e-6` và `6,08e-6`. Pilot chính đã bắt đầu, đang render positive-control design.
Smoke chỉ học hai bước để kiểm tra runtime/export; không qua teacher signal/quality
gate và không được dùng như một teacher đạt yêu cầu. Lượt chính học 500 bước và
phải qua gate riêng trước khi chạy student. Chưa có kết quả student mới.

## Kết quả kiểm tra lúc 12:26 ngày 21/10

Run đã kết thúc; `status.json` được cập nhật lần cuối lúc 09:10:49 giờ Việt Nam.
Cả bốn owner (hai beta × hai dấu key) hoàn tất 500/500 bước, export native thành
công. `selected_beta=null`, `native_teacher_gate_failed`, `students_trained=false`.
Không còn tmux session của run này. GPU vẫn có các công việc khác, không phải
student của thí nghiệm này đang chạy.

Positive-control 32 prompt đạt 26/32 (81,25%) thay đổi đúng hướng, mean contrast
0,002568. Điều đó cho thấy sửa prompt có tác dụng với probe; không chứng minh
native teacher đã học quy luật có key.

| Beta | Relation key thuận (cần >=+0,002) | Relation key đảo (cần <=−0,002) | LPIPS p95 key thuận (cần <=0,12) | PSNR p05 key thuận (cần >=25 dB) |
|---|---:|---:|---:|---:|
| 1,5 | −0,000240 | −0,000023 | 0,2612 | 19,14 |
| 3 | −0,000206 | −0,000207 | 0,2633 | 18,80 |

Đây là **score quan hệ cấp model**, không phải accuracy hay TPR. Cả bốn CI99%
đều chứa 0. Key thuận có 0/4 nhóm với point estimate đúng chiều ở cả hai beta;
chưa đủ cơ sở nói có một fingerprint đảo chiều có ý nghĩa thống kê. Key đảo và
tập preservation cũng không đạt đầy đủ quality gate. Không phải chỉ ngưỡng chọn
quá khắt khe: signal vẫn không tách khỏi 0, và tăng beta không sửa được điều đó.

Nguồn trực tiếp: [selection.json](../server_review/semantic_relation/semantic_relation_20261021_002025Z/pilot/selection.json),
[positive control](../server_review/semantic_relation/semantic_relation_20261021_002025Z/pilot/positive_control.json),
[training/runtime audit](../server_review/semantic_relation/semantic_relation_20261021_002025Z/runtime_audit.json).

**Chẩn đoán hiện có:** thất bại nằm ở native embedding/khả năng probe đọc quan hệ
sau embedding, trước giai đoạn student. Loss noise nhỏ không đủ chứng minh ảnh
mang đúng quan hệ. Một rủi ro cụ thể là target conditional giả định unconditional
được giữ nguyên, nhưng loss chỉ phạt mềm. Guided shift thực tế là
`7.5 * delta_conditional - 6.5 * delta_unconditional`. Ở minibatch cuối của beta1,5
key thuận, unconditional RMS khoảng 0,000555; sau nhân 6,5, thành khoảng 0,00361,
cùng bậc với guided target RMS khoảng 0,00425. Ở beta3 key thuận, hai con số khoảng
0,00413 và 0,00832. Đây chỉ là kiểm tra một minibatch cuối, chưa đo góc giữa các
vector; không được kết luận đã chứng minh chúng triệt tiêu nhau. Nó cho thấy cần
đo và fit guided prediction thực tế, thay vì suy ra từ conditional loss.

Bước sửa có căn cứ là kiểm tra paired guided-residual alignment trên design và
heldout, rồi cân nhắc objective trực tiếp trên CFG cùng preservation mạnh hơn;
đồng thời kiểm tra probe/thuộc tính bằng đánh giá độc lập. Chưa triển khai hay chạy
bản sửa đó trong lần kiểm tra trạng thái này. Không mở student hoặc tăng beta
thêm dựa trên kết quả thất bại hiện tại.
