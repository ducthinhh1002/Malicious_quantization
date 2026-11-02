# Projected Native

Nguồn dùng chung nằm tại `../shared/threshold_quality_v1_source`:

- `pareto_quality_campaign.py`: điều phối campaign.
- `pareto_quality_refine.py`: tinh chỉnh teacher, gồm các nhánh native.
- `pareto_step.py`: xử lý cập nhật gradient.
- `native_quality_*` và các module liên quan: mục tiêu/đánh giá chất lượng.

Xem phương pháp và kết quả đã báo cáo tại
[`../../t2ifing_radioactive/result_vast/Report.md`](../../t2ifing_radioactive/result_vast/Report.md).
Ảnh nằm trong `../../output_artifacts/previews/projected_native_*`;
JSON kết quả trong `../../output_artifacts/results/historical_summaries`.

Các kết quả lịch sử này dùng 8 bit/rank 32. Student 10.000 là seeded rerun,
không được gọi là exact resume của bản 7.500. Chưa xác nhận được checkpoint
Projected Native độc lập trong dữ liệu local; không coi code hoặc ảnh preview
là bằng chứng đã có trọng số của nhánh này.
