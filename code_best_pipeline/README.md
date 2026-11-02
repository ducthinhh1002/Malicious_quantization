# Code các pipeline được giữ lại

Trọng số nằm riêng tại [`../checkpoint`](../checkpoint/README.md). Ảnh, prompt,
metrics và bằng chứng thực nghiệm nằm trong [`../output_artifacts`](../output_artifacts/README.md).

| Nhánh | Payload / LoRA student | Student đã lưu | Code chính |
| --- | --- | --- | --- |
| Structure | 8 bit / rank 32 | 7.500 và 12.500 bước | [`structure/student12500`](structure/student12500/REPRODUCE.md) |
| Projected Native | 8 bit / rank 32 | Kết quả 7.500 và 10.000 bước | [`shared/threshold_quality_v1_source`](projected_native/README.md) |
| Transfer90 | 8 bit / rank 32 | 15.000 bước | [`transfer90/final_source/server_layout`](transfer90/README.md) |

Structure 7.500 và 12.500 dùng cùng teacher Structure tinh chỉnh 600 bước từ
teacher k=1 gốc. Bản 12.500 còn có lịch giảm learning rate sau bước 7.500;
không chỉ thay số bước. Projected Native và Transfer90 có công thức tinh chỉnh
teacher riêng, không phải hai checkpoint khác bước của cùng một run.

`structure/student12500/historical` chứa harness, code teacher và trainer Diffusers
đã đóng gói. `shared` giữ bộ nguồn chung và các phụ thuộc lịch sử cần cho các
nhánh. Các snapshot nguồn trùng/lịch sử được lưu ở
`../output_artifacts/provenance/source_snapshots`, không xóa.

Code các thử nghiệm 32 bit và 8 bit/rank 4 nằm ở
`../output_artifacts/archive/experimental_code`; checkpoint rank 4 mới tải nằm
trong `../checkpoint/k1_8bit_student_rank4`.

`report_tools` chứa công cụ tạo báo cáo; đường dẫn đã đổi theo bố cục mới.
Không chạy các công cụ build để ghi đè báo cáo đã chỉnh tay nếu chưa xem nội dung.

Đây là tổ chức lại dữ liệu local. Cấu hình và manifest thực nghiệm đóng băng vẫn
giữ đường dẫn gốc của máy chạy; cần chuẩn bị working copy và ánh xạ đường dẫn
trước khi reproduce/resume. Không coi việc di chuyển thư mục là một lần chạy lại.
