# Structure refine600 → 8-bit rank4 student

Run mới trên `10.1.21.23:/var/lib/structure8bit_rank4_refine600`.
Teacher adapter SHA256:
`e4cdc8feefe33480266f53191592030f6dcffa40c05238985a1726ac97a73d6f`.
Đây là teacher Structure refine600 của run 7500/12500, khác run rank4/k1 gốc.

- Student SD1.4 mới từ base revision gốc; attention + convolution LoRA rank4.
- Payload/key/carrier 8 bit gốc; dùng lại ảnh Structure local, không sinh lại ảnh train.
- Train 5000 ảnh, batch4, seed0; 12500 optimizer updates.
- LR 1e-4 đến7500, cosine xuống2e-5 tại12500.
- Full optimizer/scheduler/scaler/RNG checkpoint mỗi1250 bước.
- Chọn cố định12500, bỏ đánh giá/chọn các checkpoint trung gian.
- Báo bit accuracy và full-key recovery theo nhóm1/8/32/64/256,
  cùng LPIPS/FID/KID/CLIP. Không dùng TPR làm kết quả chính.
- Tên tiến trình `VLLM::EngineCore`, GPU1, bỏ giới hạn bộ nhớ train32GiB.

`deploy_from_local.py` tải code/teacher và sáu nhóm ảnh từ local lên .23.
Không kết nối .21. SHA256 của gói tải được kiểm tra trước khi giải nén;
teacher, toàn bộ ảnh và base model được xác minh trước khi train.
`prepare_server.py` tạo môi trường riêng, tải đúng revision base,
chờ `DATA_READY.json`, khởi tạo run rồi chạy technical smoke và train/eval.

Theo dõi local tại
`output_artifacts/deployments/structure8bit_rank4_refine600_23/status.json`;
trên server xem `deployment/setup_status.json`, `deployment/setup.log`, và
`runs/structure8bit_rank4_refine600_seed0/status.json`.

Môi trường dùng pin trong requirements-semantic (PyTorch2.7.1/cu128).
Đây là run mới; không cam kết tái hiện bitwise môi trường cu126 lịch sử.

Priority update: rank32 augmentation runs first on GPU1; this rank4 control waits for it.
