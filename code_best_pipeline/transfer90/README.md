# Transfer90

Code cuối được tải về nằm trong `final_source/server_layout`:

- `project_native_transfer90`: campaign, objective và orchestration của Transfer90.
- `threshold_quality_20261022`: nguồn/harness đi kèm bản tải cuối.

Snapshot tải trước đó được giữ ở
`../../output_artifacts/provenance/source_snapshots/transfer90_earlier_download`.

Teacher và student: `../../checkpoint/transfer90/8bit_rank32`.
Dữ liệu, prompt, cấu hình và kết quả:
`../../output_artifacts/runs/transfer90/final`.
Ảnh so sánh: `../../output_artifacts/previews/transfer90_report_20261027`.

Payload 8 bit, LoRA student rank 32; student được chọn là **15.000 bước**.
Teacher bắt đầu từ Projected Native rồi tinh chỉnh thêm 800 bước.
Không đồng nhất nhánh này với Structure teacher refine600/student12500.

Các assets/config/manifest được lưu làm bằng chứng của run gốc. Khi dùng lại,
tạo working copy, ánh xạ đường dẫn model/run theo máy mới và nối lại thư mục
trọng số đã tách sang `checkpoint`; không ghi đè cấu hình đóng băng.
