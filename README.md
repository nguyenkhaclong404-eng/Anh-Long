# ⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (FINANCIAL STATEMENT FRAUD DETECTION)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

Ứng dụng web tương tác trực quan được xây dựng trên nền tảng **Streamlit**, kết hợp mô hình **Hồi quy Logistic (Logistic Regression)** chuẩn hóa với bộ **8 chỉ số Beneish M-Score** nhằm phát hiện sớm các dấu hiệu thao túng lợi nhuận và gian lận Báo cáo Tài chính (BCTC) của doanh nghiệp.

---

## 📌 1. Giới thiệu Bài toán & Ý nghĩa Thực tiễn

Gian lận Báo cáo Tài chính (Earnings Manipulation / Financial Statement Fraud) gây ra thiệt hại nghiêm trọng cho các nhà đầu tư, tổ chức tín dụng, cơ quan quản lý và thị trường vốn.

Ứng dụng này giải quyết bài toán phân loại nhị phân:
- **`0`**: Doanh nghiệp **Không gian lận** (Bình thường / An toàn).
- **`1`**: Doanh nghiệp **Có gian lận BCTC** (Nguy cơ cao / Thao túng số liệu).

### 8 Chỉ số Tài chính Beneish M-Score được sử dụng:
| Ký hiệu | Tên chỉ số tiếng Việt | Tên tiếng Anh | Ý nghĩa nhận diện rủi ro |
| :--- | :--- | :--- | :--- |
| **DSRI** | Số ngày phải thu khách hàng | Days Sales in Receivables Index | DSRI > 1: Phải thu tăng nhanh hơn doanh thu; nguy cơ ghi nhận doanh thu khống/ảo hoặc nới lỏng tín dụng quá mức. |
| **GMI** | Biên lợi nhuận gộp | Gross Margin Index | GMI > 1: Biên lãi gộp sa sút, tạo áp lực cho ban lãnh đạo làm đẹp số liệu lợi nhuận. |
| **AQI** | Chất lượng tài sản | Asset Quality Index | AQI > 1: Tỷ lệ tài sản phi hiện hữu tăng cao, cảnh báo việc vốn hóa chi phí hoạt động vào tài sản. |
| **SGI** | Tăng trưởng doanh thu | Sales Growth Index | SGI cao: Tăng trưởng nóng tạo áp lực duy trì kỳ vọng, dễ phát sinh gian lận khi thị trường chững lại. |
| **DEPI** | Tỷ lệ khấu hao | Depreciation Index | DEPI > 1: Doanh nghiệp hạ thấp tỷ lệ trích khấu hao để thổi phồng lợi nhuận ngắn hạn. |
| **SGAI** | Chi phí bán hàng & quản lý | SG&A Index | SGAI > 1: Hiệu quả kiểm soát chi phí hoạt động suy giảm. |
| **TATA** | Biến dồn tích trên tổng tài sản | Total Accruals to Total Assets | TATA cao: Lợi nhuận kế toán không đi kèm dòng tiền kinh doanh (CFO), chất lượng lợi nhuận kém. |
| **LVGI** | Đòn bẩy tài chính | Leverage Index | LVGI > 1: Nợ vay gia tăng, đối mặt rủi ro vi phạm cam kết tín dụng. |

---

## 🚀 2. Các Tính năng Chính của Web App

1. **🏠 Tổng quan & Giới thiệu:**
   - Hệ thống hóa lý thuyết về gian lận BCTC, công thức chi tiết và ý nghĩa kiểm toán của 8 chỉ số Beneish.
   - Trình bày cơ chế chuẩn hóa dữ liệu (`StandardScaler`) và phương trình hồi quy Logistic.
2. **📊 Khám phá Dữ liệu & Huấn luyện Mô hình (EDA & Training):**
   - Phân tích tương quan Pearson giữa 8 chỉ số với biến mục tiêu `FRAUD_FLAG`.
   - Huấn luyện mô hình Logistic Regression theo tỷ lệ train/test và threshold tùy biến.
   - Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix Heatmap), đường cong ROC & chỉ số AUC.
   - Thống kê đầy đủ các chỉ số: **Accuracy, Precision, Recall/Sensitivity, F1-score, Specificity, FPR, FNR**.
   - Bảng trọng số hồi quy $\beta$ và Tỷ số chênh (Odds Ratio $e^\beta$).
   - **Xuất toàn bộ kết quả huấn luyện ra file Excel (`.xlsx`)** với đầy đủ các sheet.
3. **🔍 Dự báo Đơn lẻ (Single Company Assessment):**
   - Form nhập 8 chỉ số với các nút nạp nhanh hồ sơ mẫu ("Doanh nghiệp An toàn" / "Doanh nghiệp Nguy cơ cao").
   - Tính toán xác suất gian lận $P(\text{Fraud})$ kèm phân loại mức độ rủi ro (An toàn, Cần lưu ý, Nguy cơ cao).
   - Đối chiếu đồng thời với **Điểm số Beneish M-Score chuẩn** (ngưỡng cắt $-1.78$).
   - **Biểu đồ Radar đa chiều**: So sánh các chỉ số của doanh nghiệp với giá trị trung bình của nhóm an toàn và nhóm gian lận toàn ngành.
   - Tự động đưa ra các khuyến nghị kiểm toán cụ thể cho từng chỉ số vượt ngưỡng cảnh báo.
4. **📂 Dự báo Hàng loạt (Batch Prediction):**
   - Tải lên file CSV hoặc Excel chứa danh mục nhiều doanh nghiệp.
   - Cung cấp sẵn file mẫu (`Template_Batch_Prediction.csv`) để tải về.
   - Tự động phân loại, tính xác suất gian lận và điểm Beneish cho toàn bộ danh sách.
   - Biểu đồ phân bố rủi ro toàn danh mục và bộ lọc xem theo mức độ rủi ro.
   - **Xuất kết quả dự báo ra file CSV hoặc Excel (`.xlsx`)**.

---

## 📁 3. Cấu trúc Thư mục Dự án

```text
├── app.py                             # Mã nguồn ứng dụng Streamlit hoàn chỉnh
├── requirements.txt                   # Danh mục thư viện phụ thuộc Python
├── README.md                          # Tài liệu hướng dẫn chi tiết dự án
├── MScore_data.csv                    # Dữ liệu 8 chỉ số Beneish & FRAUD_FLAG (500 quan sát)
└── logistic_regression_mscore_colab.py # File script gốc từ Google Colab
```

---

## 💻 4. Hướng dẫn Cài đặt & Chạy Local trên Máy tính

### Yêu cầu tiên quyết:
- Máy tính đã cài đặt Python 3.9 trở lên (khuyến nghị Python 3.10 - 3.12).

### Các bước thực hiện:

1. **Mở Terminal / PowerShell / Command Prompt tại thư mục dự án:**
   ```bash
   cd "duong_dan_den_thu_muc_du_an"
   ```

2. **(Khuyến nghị) Tạo môi trường ảo (Virtual Environment):**
   - Trên Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - Trên macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Cài đặt các thư viện phụ thuộc từ `requirements.txt`:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Streamlit:**
   ```bash
   streamlit run app.py
   ```

5. Trình duyệt web sẽ tự động mở tại địa chỉ: `http://localhost:8501`.

---

## 🌐 5. Hướng dẫn Đưa lên GitHub & Deploy Miễn Phí trên Streamlit Cloud

### Bước 1: Khởi tạo Git & Đẩy code lên GitHub
1. Đăng nhập vào tài khoản [GitHub](https://github.com/) của bạn.
2. Tạo một Repository mới (ví dụ đặt tên: `bctc-fraud-detection-streamlit`), chọn chế độ **Public**.
3. Tại thư mục dự án trên máy tính, chạy các lệnh sau trong Terminal:
   ```bash
   git init
   git add app.py requirements.txt README.md MScore_data.csv
   git commit -m "Initial commit: Streamlit Fraud Detection App"
   git branch -M main
   git remote add origin https://github.com/<TÊN_GITHUB_CỦA_BẠN>/bctc-fraud-detection-streamlit.git
   git push -u origin main
   ```

### Bước 2: Deploy lên Streamlit Community Cloud
1. Truy cập vào [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **"Create app"** (hoặc **"New app"**).
3. Điền các thông tin:
   - **Repository:** Chọn repo `bctc-fraud-detection-streamlit` vừa tạo.
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Nhấn **"Deploy!"**.
5. Đợi hệ thống cài đặt dependencies trong khoảng 1 - 2 phút. Sau khi hoàn tất, bạn sẽ nhận được một đường link web app hoạt động 24/7 (ví dụ: `https://bctc-fraud-detection.streamlit.app`) để chia sẻ cho đồng nghiệp, giảng viên hoặc bổ sung vào CV/Portfolio!

---

## 📊 6. Công thức Tính Toán

### 1. Mô hình Beneish M-Score 8 Biến:
$$M = -4.84 + 0.920 \times DSRI + 0.528 \times GMI + 0.404 \times AQI + 0.892 \times SGI + 0.115 \times DEPI - 0.172 \times SGAI + 4.037 \times TATA + 0.0327 \times LVGI$$
- **Quy tắc phân loại:** Nếu $M > -1.78$, doanh nghiệp có khả năng cao đang thao túng Báo cáo tài chính.

### 2. Mô hình Hồi quy Logistic:
Dữ liệu đầu vào được chuẩn hóa qua `StandardScaler`:
$$Z_i = \frac{X_i - \mu_i}{\sigma_i}$$
Xác suất gian lận được ước tính:
$$P(\text{Fraud} = 1) = \frac{1}{1 + e^{-(\beta_0 + \sum_{i=1}^8 \beta_i Z_i)}}$$
Nếu $P(\text{Fraud} = 1) \ge \text{Threshold}$ (mặc định $0.50$), mô hình kết luận doanh nghiệp thuộc diện có rủi ro gian lận.

---

## 📜 7. Giấy phép & Tác giả
- Phát triển phục vụ mục đích nghiên cứu học thuật, giảng dạy và phân tích rủi ro tài chính.
- Giấy phép: [MIT License](LICENSE).
