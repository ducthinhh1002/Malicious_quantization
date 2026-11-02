# Khôi phục bước data/merge — 24/10/2026, 15:44 ICT

`quality` hoàn thành 600 teacher updates và lưu đủ adapter/checkpoint. Bước
`02_data` trong fresh subprocess không gọi numerical_policy trước merge, nên
cuDNN TF32 mặc định bật, trong khi train và load_native đều tắt. Strict merge
equivalence test thất bại: max absolute error 0.00134474039.

Đã chạy lại chính merge helper với numerical_policy: max absolute error
5.543231964e-6, vượt qua nguyên ngưỡng atol=rtol=1e-4. Không nới tolerance,
không thay weights, không train lại hoặc đổi phương pháp watermark.

Để tránh lặp lỗi trong các nhánh tiếp theo, thêm runtime bootstrap
`runtime_merge_policy/sitecustomize.py` qua PYTHONPATH của hai supervisors.
Chỉ áp dụng cho `transfer_campaign.py` hoặc `minhnqn_campaign.py` khi
`--operation data`: deterministic cuDNN, TF32 matmul và convolution đều tắt.
Training/generation đã có cùng policy này. Bootstrap không chỉnh frozen code
manifests; record riêng trên VM4 tại `merge_policy_recovery_20261024.json`.

Đã kiểm tra predicate, backend flags và xác nhận log subprocess thực có marker
`MERGE_NUMERICAL_POLICY`. Restart transfer supervisor PID 877549; MinhNQN
supervisor PID 877579. MinhNQN cũ chỉ đang chờ, không có child job; dừng đúng
supervisor đó để truyền runtime policy cho các child tương lai. Luồng hiện
resume từ data, bỏ qua teacher training đã hoàn thành và được hash kiểm tra.
