# Vì sao LCF vẫn chưa học được dấu: kết quả và hướng sửa có căn cứ

Ngày: 18/10/2026. Đây là phân tích chẩn đoán. Cập nhật: đã triển khai hướng A
trong [LCF v3 — calibrated gain](LCF_Calibrated_Gain_VI.md), với snapshot/run mới.
Pipeline v2 được giữ nguyên; hướng B chưa triển khai.

## 1. Kết quả hiện có

Nguồn server: `/root/logs/results/lcf_v2_20261018_000900Z_diagnostic`.

| Nhánh | Kết quả được dùng | AUROC trên prompt/key chưa thấy | Kết luận |
|---|---|---|---|
| spatial | Hết 200 update; chọn step 200 | 0,5000 / 0,4995 | Chưa đọc được dấu |
| curriculum | Hết 200 update; chọn step 100 | 0,4976 / 0,4985 | Chưa đọc được dấu |
| broad | Validation step 100; vẫn đang chạy khi kiểm tra | 0,5020 / 0,4941 | Chưa thấy tín hiệu ở mốc này |
| legacy | Chưa tới lượt khi kiểm tra | — | Chưa kết luận |

TPR gần/bằng FPR; trên tập train, AUROC của các nhánh đã xong cũng gần 0,5.
Vấn đề hiện tại là khả năng học dấu ngay trên teacher, không phải chỉ tổng quát
hóa key/prompt, và chưa phải thất bại của ordinary student. Diagnostic có 32
ảnh validation/key và FPR danh nghĩa 5%; không so trực tiếp TPR với pilot FPR 1%.

## 2. Phép đo mới: tín hiệu tới reader quá nhỏ ở cấu hình hiện tại

Đã chạy một audit riêng trên H200, dùng checkpoint spatial đã chọn, 4 prompt
train × 2 key. Giữ nguyên trọng số; chỉ nhân cường độ carrier trong bộ nhớ ở
mỗi lượt đo, rồi khôi phục. Không chọn checkpoint/cường độ và không sửa run.

Mỗi reference được sinh lại với carrier bằng 0, cùng batch, noise và caption
để giảm sai khác số học do batching. Đo ảnh float, RGB8, sau resize 512→256,
LPIPS, VAE posterior và score của reader đóng băng.

Nguồn dữ liệu: [signal_audit.json](../server_review/lcf_v2_deployment/signal_audit.json).
Script: [lcf_signal_audit.py](../server_review/scripts/lcf_signal_audit.py).

| Hệ số carrier | PSNR trung bình | LPIPS trung bình | Giá trị RGB đổi | MSE sau resize / MSE trước resize |
|---:|---:|---:|---:|---:|
| 1× | 64,30 dB | 0,0000113 | 2,58% | 12,2% |
| 4× | 57,74 dB | 0,0000620 | 10,05% | 18,4% |
| 16× | 48,30 dB | 0,000664 | 36,14% | 31,3% |
| 64× | 38,64 dB | 0,00604 | 66,58% | 37,0% |

Ở mức 1×, **99,78% giá trị RGB bị đổi chỉ chênh đúng một mức 8-bit**.
MSE float trước lưu ảnh là 4,71e-8; MSE RGB8 là 4,01e-7. RGB8 không đơn giản
"xóa hết dấu": nó biến sai khác float rất nhỏ thành các lần đổi một mức màu,
còn resize tiếp tục làm nhỏ năng lượng sai khác. Tỷ lệ MSE không phải tỷ lệ
thông tin key còn lại; cần phân biệt hai khái niệm này.

Ở mức 64×, score gain reader cũ vẫn chỉ khoảng 6,53e-5. Vì vậy **không thể
chỉ tăng gain lúc inference rồi tuyên bố watermark mạnh**. Phải học lại reader
ở mức tín hiệu có thể quan sát và kiểm tra trên dữ liệu độc lập.

VAE posterior mean vẫn thay đổi trong phép đo này. Chưa có bằng chứng cho
khẳng định "VAE xóa toàn bộ dấu"; thay đổi latent cũng không chứng minh student
sẽ học lại đúng key. Audit chỉ có 4 prompt train, chưa có kiểm tra CLIP/diversity
cho từng mức gain hoặc bằng chứng chất lượng trên phân phối prompt rộng.

## 3. Một ràng buộc cấu trúc đang quá chặt

Carrier hiện tại:

\[
\delta h_k = 0.35\,r_h\tanh(B_k),\qquad r_h=3.6579.
\]

Giá trị tuyệt đối từng tọa độ bị chặn ở khoảng **1,2803**. RMS carrier thực
tế khoảng **0,2467**. Đây là chặn theo feature nội bộ, không phải chặn được
hiệu chỉnh theo chất lượng ảnh đầu ra. Carrier hiện không thể tự học tăng đều
64× khi giữ cấu trúc/gain này, dù hướng residual vẫn có thể học hiệu quả hơn.

Audit cho thấy còn khoảng đáng kể để thay đổi ảnh ở một số prompt mà LPIPS vẫn
thấp. Điều này hỗ trợ việc xem lại biên độ và biểu diễn carrier, không bảo đảm
rằng gain lớn sẽ cho detector hoặc transfer tốt.

Tại step 200 của spatial, norm gradient quality ≈9,4% norm gradient watermark,
cosine ≈−0,046. Không có căn cứ quy riêng thất bại đó cho loss chất lượng quá mạnh.
Giảm quality penalty tiếp không giải quyết chặn biên độ hoặc reader đã co về
dự đoán không mang thông tin.

## 4. Vì sao không thể lấy kết quả paper làm bảo đảm cho code này

- **HMark** dùng semantic DDPM/pixel DDPM và CelebA-HQ; huấn luyện trên 100.000
  ảnh, batch 16, 100 epoch. Code hiện là một biến thể trên SD1.4 latent diffusion,
  không phải tái lập đầy đủ HMark. Chỉ giống ý tưởng cộng residual ở bottleneck
  chưa làm hai hệ thống tương đương. [HMark, §3–4 và Appendix B](https://arxiv.org/html/2512.00094v1).
- **BitMark** bias các quyết định bit tại quá trình sinh và đọc lại qua tokenizer
  đã có; nó không bắt đầu bằng hai mạng ngẫu nhiên cùng tìm một residual rất yếu.
  Bài học nên mượn là ghép chặt vị trí nhúng với kênh đọc được, không chỉ dùng
  một vector theo key ở vị trí bất kỳ. [BitMark, §3](https://arxiv.org/html/2506.21209v2).
- **Stable Signature** pretrain hệ nhúng/đọc trước rồi fine-tune latent decoder
  với extractor đã học. Đây là tiền lệ cho cách làm teacher dễ học hơn, không
  phải bằng chứng watermark đó truyền qua ordinary LoRA độc lập.
  [Trang tác giả, watermark pre-training](https://pierrefdz.github.io/publications/stablesignature/).

Các sửa v2 chủ yếu sửa cách tối ưu/đánh giá. Kết quả hiện nay cho thấy cần xử lý
độ lớn và tính đọc được của tín hiệu trước khi đầu tư vào meta-learning dài.

## 5. Ưu tiên A: cứu cơ chế hiện tại bằng hiệu chỉnh tín hiệu đầu ra

1. Tách **hướng carrier** và **gain**. Chuẩn hóa RMS hướng, thêm gain dương có thể
   học; khởi tạo gain qua sweep trên TRAIN theo ngân sách LPIPS/PSNR. So sánh
   các mức đủ khác biệt (audit gợi ý cần xem 16× và 64×), không tự coi 64× là tốt nhất.
   Chốt lựa chọn trước khi mở test. Giữ gate chất lượng độc lập trên validation.
2. Khởi tạo lại reader, giữ carrier cố định, kiểm tra overfit trên 4–16 prompt
   và hai key trước. Có clean, wrong key và zero-carrier control. Đánh giá fit
   bằng AUROC/bit accuracy; không sử dụng TPR train làm bằng chứng tổng quát hóa.
3. Nếu fit tốt, mở rộng prompt/key và cùng tối ưu. Reader dùng tín hiệu ở cả
   độ phân giải đầy đủ và sau resize; kiểm tra từng kênh để tránh chỉ học các
   nhiễu một mức màu. Không dùng ảnh sạch ghép cặp khi suy luận detector.
4. Nếu teacher mạnh và giữ chất lượng trên validation mới chạy ordinary student.
   Meta objective tiếp tục chỉ cập nhật fingerprint phía owner; không sửa
   denoising loss hoặc cung cấp key cho student.

Tiêu chí quyết định sơ bộ: nếu một cấu hình có biên độ hợp lý, đạt chất lượng,
vẫn không thể overfit tập nhỏ sau một ngân sách reader cố định, không tiếp tục
tăng full-pilot steps. Kiểm tra positive control của reader, rồi đổi carrier.

## 6. Ưu tiên B: đổi điểm nhúng nếu A không giải quyết được

Thử residual theo key trong **dự đoán clean latent \(\hat z_0\)** của teacher,
thay vì chỉ cộng bản đồ ngẫu nhiên vào mid-block:

\[
\hat z_{0,k}=\hat z_0+a_t R_\phi(k,\hat z_0,t).
\]

- Bắt đầu bằng biểu diễn ít chiều, trơn theo không gian, hoặc phép biến đổi
  affine hạng thấp có tham số chung theo key. Không dùng điều kiện chọn loại prompt.
- Đặt phép biến đổi trong sampler của teacher và thực hiện cập nhật DDIM nhất
  quán với dự đoán mới; không gắn một watermark loss vào trainer của student.
- Hiệu chỉnh \(a_t\) bằng chất lượng đầu ra và khả năng đọc, kiểm tra cửa sổ
  timestep cố định. Tránh bù hệ số scheduler gây nổ ở biên timestep.
- Có đối chứng residual h-space và clean-latent ở cùng PSNR/LPIPS và ngân sách.
- Vẫn phải chứng minh đọc được sau RGB8, rồi mới kiểm tra truyền dấu qua ordinary
  student. Năng lượng tín hiệu lớn hơn không đồng nghĩa radioactivity.

Ưu điểm kỹ thuật dự kiến: đường tác động từ carrier tới ảnh ngắn hơn, biên độ
dễ đo hơn; giảm phụ thuộc vào độ nhạy khó kiểm soát của mid-block qua up-blocks
và scheduler. Đây là giả thuyết, không phải kết quả đã chứng minh hoặc claim
novelty. Residual latent, gain calibration, reader pretraining đều có tiền lệ;
đóng góp mới nếu có phải nằm ở khả năng truyền dấu được kiểm chứng dưới ordinary
learners và đối chứng mạnh, không chỉ ở việc ghép các thành phần này.
