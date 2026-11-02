# Native Transfer Fingerprint: triển khai trên vm4

Ngày địa phương: 18/10/2026. Tên run dùng UTC, vì vậy mang tiền tố 20261017.

- Server: `vm4`, H200.
- Snapshot code: `/root/logs/native_transfer_candidate_20261018`.
- Python: `/root/logs/gate_env/bin/python`.
- Tmux: `native_transfer_20261017_174732Z`.
- Launcher: `/root/logs/native_transfer_20261017_174732Z.sh`.
- Metadata: `/root/logs/native_transfer_current.json`.
- Smoke: `/root/logs/results/native_transfer_20261017_174732Z_smoke`.
- Pilot: `/root/logs/results/native_transfer_20261017_174732Z_pilot`.
- Log pilot: `/root/logs/native_transfer_20261017_174732Z_pilot.log`.
- Exit file pilot: `/root/logs/native_transfer_20261017_174732Z_pilot.exit`, được ghi khi kết thúc.

## Đã xác nhận

7 unit tests qua ở local và server; shell qua `bash -n`.
Smoke GPU end-to-end qua và ghi `integration_smoke_passed`:
native perturbation, full generation, clean feature extraction, calibration,
native export/reload, LPIPS, ba ordinary LoRA hai updates, heldout denoising MSE,
full student generation và signed response. Native export FP32 max error = 0.
Số smoke không phải kết quả khoa học.

Môi trường đã có các dependency trainer; bổ sung `lpips==0.1.4` bằng `--no-deps`,
không đổi torch. SHA256 của 6 file code/launcher mới, `data.py`, `distribution.py`
và ordinary `bitmark_lora/student.py` khớp giữa local và snapshot server.

Pilot đã bắt đầu sau smoke thành công: prepare/basis xong, đang sinh clean-fit
84/128 ở lần kiểm tra cuối. Chưa có detection/transfer result của pilot.
Không sửa hay dừng các tiến trình GPU/tmux khác.

## Phạm vi thực nghiệm

Coordinate pilot: 8 hướng native, 2 biên độ; feature từ frozen clean UNet;
teacher gate cố định trước learner study. Nếu qua gate, design r4/r16 50 updates,
rồi independent confirmation r4/r8 500 updates với matched clean controls.
Không có continuous joint solver hoặc gradient-proxy AdamW trong bản đầu.

Tất cả source/config/split được lưu trong run. Ảnh ở `/root/logs/output_image`;
model/basis/tensor ở `/root/logs/runtime/native_transfer_artifacts`; kết quả nhẹ
trong run. Không dereference symlink nặng khi lấy báo cáo.

Đọc `decision.json` để phân biệt gate failed, short-budget inconclusive và
pilot complete. Exit 0 hoặc smoke pass không chứng minh fingerprint thành công.

```bash
tmux attach -t native_transfer_20261017_174732Z
tail -f /root/logs/native_transfer_20261017_174732Z_pilot.log
```
