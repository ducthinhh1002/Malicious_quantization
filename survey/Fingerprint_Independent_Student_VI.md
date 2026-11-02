# Sửa mục tiêu: phát minh fingerprint, không thiết kế lại LoRA của bên học

Ngày 2026-10-16. Đã sửa triển khai, chưa có kết quả GPU của phương pháp mới.

## Phản hồi của mentor đúng ở đâu?

Đích cần đạt: owner phát hành model có fingerprint. Người khác lấy ảnh của model
để học bằng LoRA hoặc fine-tuning thông thường. Ảnh student sinh ra vẫn có bằng
chứng thống kê của fingerprint. Owner không điều khiển loss, optimizer, adapter
hay detector của student.

Code trước **không đưa watermark loss vào final student**: final student vẫn học
denoising MSE. Tuy nhiên, teacher được tối ưu qua surrogate LoRA rank cụ thể,
và được chọn theo pilot LoRA. Điều này có thể là một phương pháp owner-side hợp
lệ, nhưng không chứng minh độc lập với cách học của bên kia. Chỉ giữ hai student
học ảnh marked mà bỏ clean-trained controls còn khiến kết luận bị lẫn với hiệu
ứng fine-tuning thông thường. Không thể từ đó suy ra mọi kết quả BitMark cũ đều
sai; phải audit từng protocol và claim riêng.

LoRA dùng **nội bộ owner** để cập nhật teacher rồi merge vào UNet vẫn hợp lệ.
Đó là cách tiết kiệm bộ nhớ khi nhúng fingerprint, không phải sửa LoRA của attacker.

“Bất kể cách học nào” không thể là bảo đảm vô điều kiện: student không học, học
rất ít, hoặc hầu hết dữ liệu không liên quan có thể không mang dấu. Phát biểu
đúng hơn: khả năng truyền dấu trên nhiều quy trình học độc lập, trong phạm vi
dữ liệu, chất lượng và ngân sách được báo cáo rõ.

## Survey và điều rút ra

| Nguồn gốc | Điều có ích | Giới hạn khi áp dụng |
|---|---|---|
| [HMark](https://arxiv.org/html/2512.00094) | Dấu nhất quán ở không gian semantic-latent giúp dữ liệu có thay đổi dễ được model học; paper không đặt radioactivity trực tiếp vào loss | Thí nghiệm face DDPM, không chứng minh text-to-image SD hoặc mọi LoRA |
| [SERUM](https://arxiv.org/html/2603.13396v1) | Dấu ở quá trình sinh và đánh giá bằng LoRA/full fine-tuning thông thường; checkpoint chọn bằng validation denoising loss | Dùng noise injector lúc inference, khác fingerprint được merge vào trọng số |
| [On the Learnability of Watermarks for Language Models](https://arxiv.org/abs/2312.04469) | Student có thể học dấu từ phân phối output; số lượng dữ liệu và độ phức tạp dấu ảnh hưởng khả năng học | Kết quả ngôn ngữ chỉ gợi ý, không chứng minh cho diffusion |
| [Radioactive Data](https://proceedings.mlr.press/v119/sablayrolles20a.html) | Thiết kế dấu ở feature space thay vì chỉ tối đa hóa độ dễ detect của ảnh riêng lẻ | Classification khác diffusion và adaptation |
| [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904) | Phải xét mức nhiễm dữ liệu và quy trình học khi nói dấu truyền qua training | Không suy ra sống qua mọi LoRA/SFT |

## Giả thuyết mới đã triển khai

Tên mô tả: **fingerprint ngữ nghĩa ổn định qua kênh latent**. Đây chưa phải một
thuật toán đã được xác lập novelty hay hiệu quả.

Điểm cần khắc phục của hướng CLIP cũ: một hướng CLIP ngẫu nhiên có thể rất dễ
đẩy score bằng các biến đổi mà VAE sẽ làm mất. Khi student nhận ảnh, chính VAE
là một cửa lọc trước khi dữ liệu tới denoising training. Vì thế native teacher
TPR cao chưa chắc đem lại tín hiệu mà student có thể học.

Hai thay đổi chỉ thuộc phía owner:

1. Trên **ảnh sạch design**, đo feature trước/sau VAE. Lấy PCA để tránh các
   chiều gần suy biến; giải bài toán trị riêng để tìm những hướng có thay đổi
   nhỏ qua VAE so với phương sai tự nhiên. Key chọn một tổ hợp trong không gian
   đó. Không dùng ảnh student hoặc điểm test để chọn hướng.
2. Teacher học tạo dấu nhất quán không chỉ trên ảnh gốc mà cả ảnh qua VAE và
   qua VAE có nhiễu latent nhẹ. Loss lấy kênh yếu nhất, đồng thời yêu cầu tăng
   score so với ảnh sạch cùng nội dung và giữ noise prediction/chất lượng.

Tất cả trọng số VAE/CLIP đóng băng; chỉ teacher nhận update. Student không xuất
hiện trong loss mặc định. Teacher được chọn bằng chất lượng dưới các điều kiện
native detection trên tập design; không dùng pilot student.

Khác HMark: không gắn bộ mã hóa h-space/key vào inference; xuất UNet đã merge,
chạy prompt và noise bình thường. Khác SERUM: không cần thêm dấu vào noise lúc
sinh. Khác code cũ: không cần chọn hay mô phỏng thuật toán LoRA của bên học.

**Không được gọi độ bền qua VAE là độ bền qua distillation.** Nó chỉ giải quyết
một nút thắt biểu diễn. Tối ưu diffusion, thay backbone, lượng dữ liệu và data
mixing có thể làm mất dấu dù proxy rất tốt. CLIP cũng có thể bị khai thác bằng
feature không phản ánh chất lượng thực; FID/KID và kiểm tra ảnh vẫn cần thiết.

## Thí nghiệm để biết có đóng góp hay chỉ ghép kỹ thuật

Bốn ablation cùng ngân sách, dữ liệu và student:

| Carrier | Loss teacher | Vai trò |
|---|---|---|
| Random | Native | Baseline đơn giản |
| Stable | Native | Giá trị riêng của chọn carrier |
| Random | Native + channel | Giá trị riêng của loss qua kênh |
| Stable | Native + channel | Phương pháp kết hợp |

Default chạy ordinary attention LoRA r16/r32 và full UNet, mỗi recipe có một
student học clean và một student học marked. Cùng initialization, seed, prompt,
optimizer và ngân sách trong từng cặp. Student chọn checkpoint bằng validation
MSE; detector chỉ đo kết quả sau đó. Không sửa upstream training loss/optimizer.

Đầu ra cần nhìn đồng thời:

- Teacher có watermark thật và chất lượng chấp nhận được không?
- Feature còn dấu qua VAE không? `channel_diagnostics.csv` trả lời câu này.
- Student thật còn dấu không? `summary.csv` trả lời câu này.
- Student học ảnh sạch có bị báo nhầm không? `report.json` so cặp clean/marked.
- Wrong keys và các ảnh clean có giữ FPR thấp ở threshold đã đóng băng không?
- Cùng chất lượng/compute, đủ bốn ablation có hơn baseline không?

Các thí nghiệm tiếp theo để đủ sức cho paper: nhiều seed/key, rank4 và scope
khác được định trước, student backbone khác, lượng ảnh khác, tỷ lệ marked trong
dữ liệu, kiểm tra tương quan proxy với inheritance trên **tập design**, rồi
kiểm tra xác nhận trên tập mới. Chưa có bằng chứng thì không tuyên bố “first”,
“bất kể LoRA” hoặc vượt SERUM/HMark.

## Cách chạy

`bash run_new.sh` chạy phương pháp mới và sáu student/control, mỗi trajectory
tối đa 10.000 bước. Tổng 60.000 bước student nên lâu hơn bản hai student cũ.
`--student-kinds lora` giảm còn bốn trajectory và thu hẹp claim về LoRA.

Xem `new_watermark/README.md` để chạy ablation. Khởi tạo run mới; không resume
run cũ bằng code/protocol đã thay đổi. Revision này không tự chạy trên server.
