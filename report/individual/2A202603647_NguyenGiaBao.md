# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV: 2A202603647**
- **Họ và tên: Nguyễn Gia Bảo**
- **SEED: 9456**

## 1. Bằng chứng

| Tệp                                                                                                              | Có? |
| ---------------------------------------------------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output (65/65 ô code đã chạy, bộ kiểm tra 15/15 ✅) | ☑   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra, cùng lần chạy với notebook                 | ☑   |
| [`hand_calc1.jpg`](hand_calc1.jpg), [`hand_calc2.jpg`](hand_calc2.jpg) — ảnh tính tay có các bước (2 trang)      | ☑   |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (SEED 9456, Colab CPU). Hàng LLM chỉ đo trên SUB 40 tin, nên chất lượng chỉ so được bằng cột `F1 (SUB 40)`. Cột cuối lấy từ stress test 55 case khó (50 case gốc + 5 case của mình).

| Phương án                                | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin | F1 55 case khó |
| ---------------------------------------- | --------: | ----------: | ---------: | -----------: | -------------: |
| 1. BoW + LogReg (tự code)                |     0.902 |       0.974 |       0.19 |            0 |          0.293 |
| 2. BoW + MLP                             |     0.943 |       0.974 |       4.38 |            0 |          0.419 |
| 3a. Encoder (MiniLM) + kNN               |     0.879 |       0.947 |      57.24 |            0 |          0.733 |
| 3b. Encoder (MiniLM) + LogReg            |     0.922 |       0.976 |      31.10 |            0 |          0.667 |
| 3c. Encoder (MiniLM) + MLP head          |     0.939 |       1.000 |      34.32 |            0 |          0.618 |
| 4a. LLM nhỏ zero-shot (claude-haiku-5-5) |         — |       1.000 |    1774.07 |       0.0075 |          0.954 |
| 4b. LLM lớn zero-shot (claude-opus-5-5)  |         — |       1.000 |    1550.49 |        0.305 |              — |
| 4c. LLM nhỏ few-shot                     |         — |       1.000 |    2279.08 |        0.019 |              — |
| 4d. LLM nhỏ + prompt phòng thủ           |         — |       1.000 |          — |            — |          0.970 |

- **Đường Pareto (1 − F1 ↔ p95):** 1. BoW + LogReg, 3b, 3c.
- **Đạt SLA p95 < 50 ms:** 1, 2, 3b, 3c. 3a (57 ms) và cả bốn hàng LLM (1.5–2.3 s) không đạt.
- Trên SUB 40 tin, 1 tin sai làm F1 đổi khoảng 0.026. Vì vậy chênh lệch giữa 0.974, 0.976 và 1.000 coi như hoà.
- Qua 3 lần Run all, F1 không đổi (cách chia dữ liệu cố định theo SEED), nhưng p95 thay đổi nhiều. Ví dụ p95 của 3b lần lượt là 181, 32 và 31 ms. Chỉ nên tin khoảng cách hàng chục lần giữa BoW, Encoder và LLM.

## 3. Dự đoán và thực tế

**1. `hard_eval`: điểm yếu của LLM**

- _Mình đoán:_ LLM chỉ đạt khoảng 85–90% trên case khó, và yếu nhất ở prompt injection cùng tin nói _về_ spam.
- _Thực tế:_ LLM đạt 94.5% (4a) và 96.4% (4d), đúng 100% ở cả nhóm prompt injection lẫn nhóm nói về spam. Chỗ LLM sai lại là **spam giọng tự nhiên** (4a chỉ 82%) và **tin tưởng spam nhưng không phải** (91%).
- _Vì sao lệch:_ mình nghĩ LLM sẽ bị câu lệnh trong tin "điều khiển". Nhưng prompt chỉ yêu cầu trả lời một từ, và các câu injection trong bộ này lộ liễu nên LLM nhận ra ngay. Thứ khó với LLM là tin không có dấu hiệu gì, ví dụ case #101 mời tập gym bằng giọng người quen. Đọc "như người" thì cũng bị lừa giống người.

**2. `llm_models`: model lớn so với model nhỏ**

- _Mình đoán:_ Opus có F1 nhỉnh hơn Haiku khoảng 0–5 điểm.
- _Thực tế:_ cả hai đều đạt F1 = 1.000 trên SUB, trong khi Opus tốn 0.01219 USD so với 0.00030 USD (gấp khoảng 40 lần). Phần dự đoán về chi phí là đúng. Độ trễ thì không ổn định: lần cuối Opus chậm hơn ở p50 (1091 so với 719 ms) nhưng p95 lại thấp hơn (1550 so với 1774 ms).
- _Vì sao lệch:_ SUB gồm 40 tin cân bằng, phần lớn là spam kiểu cũ rất rõ ràng. Bài quá dễ nên model nhỏ đã chạm trần, model lớn không còn chỗ để hơn.

**3. `emb_tricky`: Encoder có thắng BoW ở câu "ngụy trang" không**

- _Mình đoán:_ 3b (Encoder + LogReg) đoán đúng nhiều hơn BoW, vì embedding hiểu nghĩa cả câu.
- _Thực tế:_ hoà 5/6. Cả hai cùng sai ở tin _"Last chance! Your account rewards expire tonight, confirm de…"_.
- _Vì sao lệch:_ embedding chỉ đặt câu lên "bản đồ nghĩa". Còn ranh giới spam/ham vẫn do LogReg học từ dữ liệu train, mà spam trong dữ liệu train chủ yếu là trúng thưởng, nhạc chuông, đầu số cao cấp, gần như không có phishing tài khoản. Encoder hiểu được câu nhưng head chưa từng thấy kiểu spam này.

## 4. Phân tích 50 case khó

Bộ case khó gồm 50 case gốc + 5 case mình tự viết (#100–#104), tổng cộng 55 case.

- **Tụt nhiều nhất:** 1. BoW + LogReg, F1 từ 0.974 (SUB) xuống **0.293**, accuracy 47%. Nhóm khó nhất với BoW là spam giọng tự nhiên (9%), né từ khoá (14%; BoW + MLP 0%) và thao túng / lừa đảo (11%).
- **Embedding (3b/3c)** khá hơn nhưng vẫn yếu ở spam giọng tự nhiên (27–36%), thao túng (33–44%) và nói về spam (50%).
- **LLM** bền nhất: 4a đạt 94.5%, 4d (prompt phòng thủ) đạt 96.4%. Prompt phòng thủ giúp nhóm spam giọng tự nhiên tăng từ 82% lên 91% mà không làm hỏng nhóm nào.

Hai case cụ thể:

1. **#101** (spam giọng tự nhiên, case mình tự viết): _"Hi it's Linh from the gym, we have a few spots left for the trial week, just reply if you want me to save one"_. **Cả 7 model đều sai.**
   - BoW gần như chỉ thấy từ hội thoại ("hi", "it's", "want", "me"). "reply" là từ duy nhất nghiêng về spam, không đủ kéo tin sang spam.
   - Embedding đặt câu này gần vùng hội thoại cá nhân.
   - LLM đọc như người, mà người đọc cũng sẽ nghĩ đây là tin của người quen. Không có link, đầu số hay lời hứa thưởng, nên không còn tín hiệu nào để phát hiện.
2. **#9** (tưởng spam nhưng không phải): _"Txt me ur address and I'll send the free tickets my boss gave me for the concert"_. Đây là case duy nhất trong 50 case gốc mà 4d sai. Nhóm này cả 4a và 4d đều chỉ đạt 91%.
   - Với BoW, "txt" và "free" đều nằm trong nhóm từ đẩy mạnh về spam.
   - Với LLM, cấu trúc "gửi địa chỉ của bạn để nhận vé miễn phí" giống hệt mẫu lừa đảo nhận quà, nên model gắn nhầm thành spam dù ngữ cảnh là bạn bè cho vé.

Điểm trên test thường quá lạc quan, vì test lấy **cùng phân phối** với train: cùng kiểu spam SMS Anh những năm 2000, và còn nhiều tin gần trùng nhau (chỉ khác số điện thoại) nằm ở cả hai tập.

## 5. Thử thách cá nhân

**Đề #1 (theo SEED 9456):** đổi ngưỡng quyết định của mô hình Phần 1 từ 0.5 sang 0.2 và 0.8, báo precision/recall/F1 cho từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào?

**Cách làm:** tính xác suất `predict_proba` cho tập val và tập test. **Chọn ngưỡng trên tập val** (775 tin, 96 spam), sau đó mới kiểm tra lại trên tập test (775 tin, 84 spam). Không dùng tập test để chọn ngưỡng, vì làm vậy là rò rỉ thông tin của tập đánh giá vào quá trình chọn.

**Kết quả trên VAL** (dùng để chọn):

| Ngưỡng | Precision | Recall |    F1 |  FP |  FN |
| -----: | --------: | -----: | ----: | --: | --: |
|    0.2 |     0.859 |  0.885 | 0.872 |  14 |  11 |
|    0.5 |     0.986 |  0.729 | 0.838 |   1 |  26 |
|    0.8 |     1.000 |  0.562 | 0.720 |   0 |  42 |

**Kết quả trên TEST** (chỉ để kiểm tra lại):

| Ngưỡng | Precision | Recall |    F1 |  FP |  FN |
| -----: | --------: | -----: | ----: | --: | --: |
|    0.2 |     0.819 |  0.917 | 0.865 |  17 |   7 |
|    0.5 |     1.000 |  0.821 | 0.902 |   0 |  15 |
|    0.8 |     1.000 |  0.619 | 0.765 |   0 |  32 |

**Giải thích:** ngân hàng sợ chặn nhầm OTP, tức sợ False Positive, nên ưu tiên precision. Trên val:

- **0.2:** có 14 tin ham bị chặn nhầm. Đây là lựa chọn rủi ro nhất.
- **0.5:** còn 1 tin bị chặn nhầm.
- **0.8:** precision = 1.000, không chặn nhầm tin nào. Đổi lại recall giảm xuống 0.562, bỏ sót 42 tin spam.

Vì vậy mình chọn **ngưỡng 0.8**: bảo vệ tin OTP hợp lệ là ưu tiên hàng đầu, chấp nhận bỏ sót nhiều spam hơn. Kiểm tra lại trên test, ngưỡng 0.8 vẫn giữ FP = 0 (precision 1.000, recall 0.619, F1 0.765).

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (~116 tin/s trung bình), p95 < 50 ms/tin.

- **Loại trước:**
  - LLM: p95 1.5–2.3 s, không đạt SLA. Nếu dùng cho toàn bộ lưu lượng còn rất đắt: Haiku khoảng 75 USD/ngày (2.250 USD/tháng), Opus khoảng 3.050 USD/ngày (91.500 USD/tháng).
  - 3a: p95 57 ms, trượt SLA.
- **Ứng viên đạt SLA:**
  - 1. BoW + LogReg (p95 0.19 ms, F1 0.902)
  - 2. BoW + MLP (p95 4.38 ms, F1 0.943)
  - 3b (p95 31.1 ms, F1 0.922)
  - 3c (p95 34.3 ms, F1 0.939)
- **Về tải:** BoW xử lý khoảng 9.000–12.600 tin/s nên dư sức. 3b/3c chỉ khoảng 34 tin/s trên CPU, nên cần GPU hoặc khoảng 4 worker song song (nhiều hơn vào giờ cao điểm).
- **Lựa chọn:** router 2 tầng.
  - **Tầng 1 (mọi tin):** BoW + LogReg. Chạy dưới 1 ms và cho ra xác suất p dùng được để định tuyến. Tin có p < 0.3 cho qua, tin có p > 0.7 thì chặn.
  - **Tầng 2 (chỉ tin "lưỡng lự" hoặc đáng ngờ):** gồm tin có p ∈ [0.3, 0.7], hoặc có link, số tài khoản, chữ bị biến dạng, câu ra lệnh cho AI. Các tin này được gửi sang LLM nhỏ với prompt phòng thủ (4d), mô hình đạt 96.4% trên case khó. Nếu khoảng 5% lưu lượng vào tầng 2 thì chi phí khoảng 3.75 USD/ngày (~110 USD/tháng).
- **Đánh đổi chấp nhận:**
  - Tin vào tầng 2 mất 1–2 giây. Muốn p95 toàn hệ thống vẫn dưới 50 ms thì tầng 2 phải nhận **dưới 5% lưu lượng**, hoặc chạy **bất đồng bộ** (giao tin trước, nếu LLM kết luận spam thì chuyển vào mục spam sau).
  - Spam giọng tự nhiên như case #101 có thể bị tầng 1 chấm p thấp một cách "tự tin" nên không bao giờ lên tầng 2. Vì vậy cần lấy mẫu kiểm tra định kỳ và kênh để người dùng báo cáo spam.
  - Nếu có thêm ngân sách GPU, có thể thay tầng 1 bằng 3c. Trên case khó, 3c bền hơn BoW (accuracy 62% so với 47%) mà vẫn đạt SLA.

## 7. AI log và điều đã học

| Phần    | Hỏi AI gì                                                                                                    | AI trả lời / làm gì                                                                                                                                                                        | Mình đã sửa / kiểm chứng gì                                                                                                                                 |
| ------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0–3     | Giải thích khái niệm và gợi ý câu trả lời cho các ô ❓/✍️/🐞/Checkpoint, gợi ý nội dung các ô `predict(...)` | Claude giải thích (data leakage, gradient, overfit, embedding, kNN…) và đưa câu trả lời mẫu                                                                                                | Viết lại bằng lời của mình, đối chiếu với số liệu và biểu đồ của mình. Code TODO 1.1–3.3 tự viết                                                            |
| 4       | Code TODO 4.1–4.4 (provider Anthropic / Haiku), dự đoán 4.A–4.B, Câu hỏi 4, Bắt lỗi 3                        | Claude (qua Claude in Chrome) viết code, điền vào notebook, chạy thí nghiệm bằng API thật                                                                                                  | Ô test giả lập ✅. Phát hiện `call_openai` viết theo SDK Anthropic làm test lỗi 404 nên đổi lại về OpenAI SDK. Kiểm chứng câu trả lời với kết quả chạy thật |
| 5       | Dự đoán `tradeoff` / `hard_eval`, 5 case bonus, Câu hỏi 5–6, nháp Phản ánh                                   | Claude viết nháp dựa trên biểu đồ Pareto và bảng case khó của mình                                                                                                                         | Tự đọc lại, chỉnh lời văn, kiểm tra số liệu                                                                                                                 |
| Nộp bài | Thiết lập git; kiểm tra notebook; cập nhật số liệu; soạn báo cáo                                             | Claude Code nối repo với GitHub, kiểm tra ô chạy / lỗi / API key, cập nhật số liệu Câu 4–6 và Phản ánh sau mỗi lần Run all (độ trễ thay đổi), soạn nháp báo cáo này từ output của notebook | Đọc lại toàn bộ báo cáo và đối chiếu với bảng so sánh, biểu đồ Pareto, bảng 55 case khó trong notebook                                                      |

**Điều mình hiểu ra hôm nay:**

- Điểm cao trên tập test cùng phân phối chưa chứng minh mô hình "hiểu" bài toán. BoW đạt F1 0.97 trên SUB nhưng chỉ còn 0.29 trên case khó, nên phải stress test trước khi tin con số.
- Mỗi tập dữ liệu có một vai trò riêng: chọn ngưỡng hay chọn epoch phải làm trên val, test chỉ dùng một lần để báo kết quả.
- Model lớn nhất không phải lúc nào cũng là lựa chọn đúng. Với bài này, Haiku ngang Opus mà rẻ hơn 40 lần, và một router rẻ + LLM cho ca khó hợp lý hơn dùng một model cho mọi tin.
