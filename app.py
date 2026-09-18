# -*- coding: utf-8 -*-
"""
HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH (FINANCIAL STATEMENT FRAUD DETECTION)
Sử dụng 8 chỉ số Beneish M-Score và Mô hình Hồi quy Logistic (Logistic Regression)
Deployable on Streamlit Community Cloud & Local
"""

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, roc_curve, auc
)

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG WEB & GIAO DIỆN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dự Báo Gian Lận BCTC | Beneish & Logistic",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy biến giao diện CSS chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
    }
    .card-value {
        font-size: 1.6rem;
        font-weight: bold;
        color: #0F172A;
    }
    .safe-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .warning-badge {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .danger-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
TARGET = "FRAUD_FLAG"

FEATURE_NAMES_VI = {
    "DSRI": "DSRI - Chỉ số Số ngày Phải thu Khách hàng (Days Sales in Receivables Index)",
    "GMI": "GMI - Chỉ số Biên Lợi nhuận Gộp (Gross Margin Index)",
    "AQI": "AQI - Chỉ số Chất lượng Tài sản (Asset Quality Index)",
    "SGI": "SGI - Chỉ số Tăng trưởng Doanh thu (Sales Growth Index)",
    "DEPI": "DEPI - Chỉ số Tỷ lệ Khấu hao (Depreciation Index)",
    "SGAI": "SGAI - Chỉ số Chi phí Bán hàng & Quản lý (SG&A Index)",
    "TATA": "TATA - Biến Dồn tích Kế toán trên Tổng tài sản (Total Accruals to Total Assets)",
    "LVGI": "LVGI - Chỉ số Đòn bẩy Tài chính (Leverage Index)"
}

FEATURE_DESCRIPTIONS = {
    "DSRI": "Tỷ lệ giữa kỳ thu tiền bình quân năm nay so với năm trước. DSRI > 1 cho thấy phải thu tăng nhanh hơn doanh thu (dấu hiệu ghi nhận doanh thu sớm hoặc doanh thu ảo).",
    "GMI": "Tỷ lệ biên lãi gộp năm trước so với năm nay. GMI > 1 ngụ ý biên lợi nhuận đang suy giảm, tạo áp lực cho ban điều hành làm đẹp số liệu.",
    "AQI": "Tỷ lệ tài sản phi hiện hữu (ngoài TSCĐ, tiền mặt, hàng tồn kho) so với tổng tài sản. AQI > 1 cảnh báo việc vốn hóa chi phí bất thường thay vì ghi nhận vào kết quả kinh doanh.",
    "SGI": "Tốc độ tăng trưởng doanh thu so với năm trước. Doanh nghiệp tăng trưởng cao dễ chịu sức ép gian lận để duy trì kỳ vọng thị trường.",
    "DEPI": "Tỷ lệ khấu hao năm trước so với năm nay. DEPI > 1 biểu thị doanh nghiệp đã hạ thấp tỷ lệ trích khấu hao để thổi phồng lợi nhuận hiện tại.",
    "SGAI": "Tỷ lệ chi phí quản lý & bán hàng trên doanh thu so với năm trước. SGAI > 1 cho thấy hiệu quả quản lý chi phí sụt giảm.",
    "TATA": "Biến dồn tích kế toán thuần trên tổng tài sản. TATA càng cao phản ánh phần lợi nhuận được ghi nhận không đi kèm dòng tiền mặt từ hoạt động kinh doanh.",
    "LVGI": "Tỷ lệ nợ phải trả trên tổng tài sản so với năm trước. LVGI > 1 cho thấy rủi ro tài chính và đòn bẩy gia tăng, thúc đẩy động cơ thao túng BCTC."
}

# ---------------------------------------------------------
# 2. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU
# ---------------------------------------------------------
@st.cache_data
def load_default_data():
    """Tải dữ liệu huấn luyện mặc định MScore_data.csv"""
    file_path = "MScore_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return None

def preprocess_data(df):
    """Làm sạch và kiểm tra dữ liệu đầu vào"""
    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_cols:
        return None, f"Dữ liệu thiếu các cột bắt buộc: {missing_cols}"
    
    clean_df = df[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
    clean_df = clean_df.dropna().reset_index(drop=True)
    clean_df[TARGET] = clean_df[TARGET].astype(int)
    
    if not set(clean_df[TARGET].unique()).issubset({0, 1}):
        return None, "Cột FRAUD_FLAG phải nhận giá trị nhị phân (0: Không gian lận, 1: Gian lận)."
    
    if len(clean_df[TARGET].unique()) < 2:
        return None, "Dữ liệu huấn luyện phải có cả 2 nhãn (0 và 1)."
        
    return clean_df, None

# ---------------------------------------------------------
# 3. HÀM HUẤN LUYỆN MÔ HÌNH LOGISTIC REGRESSION
# ---------------------------------------------------------
@st.cache_resource
def train_model(data_bytes, test_size=0.2, random_state=42):
    """Huấn luyện mô hình Logistic Regression với StandardScaler"""
    df = pd.read_csv(io.BytesIO(data_bytes))
    clean_df, err = preprocess_data(df)
    if err:
        return None, err
    
    X = clean_df[FEATURES]
    y = clean_df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=5000, random_state=random_state))
    ])
    pipeline.fit(X_train, y_train)
    
    # Trích xuất hệ số
    logistic_step = pipeline.named_steps["logistic"]
    intercept = logistic_step.intercept_[0]
    coefs = logistic_step.coef_[0]
    
    coef_df = pd.DataFrame({
        "Chỉ số": FEATURES,
        "Tên chỉ số": [FEATURE_NAMES_VI[f] for f in FEATURES],
        "Hệ số (Beta)": coefs,
        "Odds Ratio (e^Beta)": np.exp(coefs),
        "Tác động đến Gian lận": [
            "Làm tăng nguy cơ gian lận" if c > 0 else "Làm giảm nguy cơ gian lận"
            for c in coefs
        ]
    }).sort_values(by="Hệ số (Beta)", ascending=False).reset_index(drop=True)
    
    results = {
        "pipeline": pipeline,
        "clean_df": clean_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "intercept": intercept,
        "coef_df": coef_df
    }
    return results, None

def evaluate_model(pipeline, X_test, y_test, threshold=0.5):
    """Đánh giá mô hình theo ngưỡng xác suất"""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "ROC_AUC": roc_auc,
        "cm": cm,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "fpr_curve": fpr_curve,
        "tpr_curve": tpr_curve
    }
    return metrics

def calculate_beneish_mscore(row):
    """
    Tính điểm Beneish M-Score chuẩn theo công thức 8 biến:
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Ngưỡng cảnh báo: M > -1.78
    """
    m = (-4.84 
         + 0.920 * row["DSRI"] 
         + 0.528 * row["GMI"] 
         + 0.404 * row["AQI"] 
         + 0.892 * row["SGI"] 
         + 0.115 * row["DEPI"] 
         - 0.172 * row["SGAI"] 
         + 4.037 * row["TATA"] 
         + 0.0327 * row["LVGI"])
    return m

# ---------------------------------------------------------
# 4. SIDEBAR - ĐIỀU HƯỚNG & CẤU HÌNH MÔ HÌNH
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/courthouse.png", width=70)
    st.title("Menu Điều Hướng")
    
    app_mode = st.radio(
        "Chọn chức năng làm việc:",
        [
            "🏠 Tổng quan & Giới thiệu",
            "📊 Khám phá Dữ liệu & Huấn luyện",
            "🔍 Dự báo Đơn lẻ (1 Doanh nghiệp)",
            "📂 Dự báo Hàng loạt (File CSV/Excel)"
        ]
    )
    
    st.markdown("---")
    st.subheader("⚙️ Cấu hình Mô hình")
    
    # Lựa chọn file dữ liệu huấn luyện
    data_source = st.radio("Nguồn dữ liệu huấn luyện:", ["Dữ liệu mặc định (MScore_data.csv)", "Tải lên file CSV mới"])
    
    raw_data_bytes = None
    if data_source == "Dữ liệu mặc định (MScore_data.csv)":
        if os.path.exists("MScore_data.csv"):
            with open("MScore_data.csv", "rb") as f:
                raw_data_bytes = f.read()
        else:
            st.error("Không tìm thấy file MScore_data.csv trong thư mục hiện tại.")
    else:
        uploaded_train = st.file_uploader("Tải file CSV huấn luyện", type=["csv"])
        if uploaded_train is not None:
            raw_data_bytes = uploaded_train.getvalue()
            
    test_ratio = st.slider("Tỷ lệ tập Test (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
    threshold = st.slider("Ngưỡng phân loại P(Fraud)", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                          help="Xác suất tính ra >= ngưỡng này sẽ phân loại là GIAN LẬN (1)")
    random_seed = st.number_input("Random State", min_value=1, max_value=999, value=42, step=1)
    
    st.markdown("---")
    st.caption("📌 **Ứng dụng Phân tích & Dự báo Gian lận BCTC**\nPhát triển dựa trên mô hình Beneish M-Score & Hồi quy Logistic.")

# ---------------------------------------------------------
# TẢI DỮ LIỆU & HUẤN LUYỆN SẴN MÔ HÌNH
# ---------------------------------------------------------
model_results = None
train_error = None

if raw_data_bytes is not None:
    model_results, train_error = train_model(raw_data_bytes, test_size=test_ratio, random_state=random_seed)
else:
    train_error = "Vui lòng cung cấp dữ liệu huấn luyện để bắt đầu."

# ---------------------------------------------------------
# 5. NỘI DUNG TỪNG TRANG
# ---------------------------------------------------------

# =========================================================
# TRANG 1: TỔNG QUAN & GIỚI THIỆU
# =========================================================
if app_mode == "🏠 Tổng quan & Giới thiệu":
    st.markdown('<div class="main-header">⚖️ HỆ THỐNG DỰ BÁO GIAN LẬN BÁO CÁO TÀI CHÍNH</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ứng dụng Khoa học Dữ liệu & Học máy (Machine Learning) kết hợp Mô hình Beneish M-Score trong Kiểm toán và Phân tích Rủi ro Tài chính</div>', unsafe_allow_html=True)
    
    col_intro1, col_intro2 = st.columns([3, 2])
    with col_intro1:
        st.markdown("""
        ### 🎯 Mục tiêu của Ứng dụng
        Gian lận Báo cáo Tài chính (BCTC) là hành vi cố ý thao túng, bóp méo thông tin tài chính nhằm đánh lừa nhà đầu tư, ngân hàng, cơ quan thuế và công chúng.
        
        Ứng dụng này cung cấp một công cụ trực quan và mạnh mẽ giúp:
        - **Kiểm toán viên:** Nhanh chóng nhận diện các khoản mục có rủi ro sai sót trọng yếu cao để tập trung thủ tục kiểm toán.
        - **Chuyên viên tín dụng & Ngân hàng:** Đánh giá độ tin cậy của BCTC doanh nghiệp trước khi phê duyệt cấp vốn.
        - **Nhà đầu tư chứng khoán:** Tầm soát rủi ro thao túng lợi nhuận (Earnings Manipulation) của các công ty niêm yết.
        - **Giảng viên & Sinh viên:** Thực hành và nghiên cứu ứng dụng mô hình kinh tế lượng và Machine Learning vào lĩnh vực Kế toán - Tài chính.
        """)
        
    with col_intro2:
        st.info("""
        #### 💡 Điểm nổi bật của Giải pháp
        1. **Mô hình Hồi quy Logistic (Chuẩn hóa StandardScaler):** Huấn luyện trên dữ liệu thực tế, ước lượng xác suất gian lận $P(\text{Fraud})$ chính xác và khách quan.
        2. **Chỉ số Beneish M-Score chuẩn:** So sánh đồng thời với ngưỡng kinh điển $-1.78$ của Giáo sư Messod Beneish.
        3. **Phân tích Radar Chart:** Trực quan hóa độ lệch của từng chỉ số so với mức trung bình của nhóm an toàn và nhóm rủi ro.
        4. **Hỗ trợ Dự báo Hàng loạt:** Tải lên tệp CSV/Excel danh sách hàng trăm công ty cùng lúc và xuất báo cáo kết quả.
        """)
        
    st.markdown("---")
    st.subheader("📚 Chi tiết 8 Chỉ số Tài chính Beneish M-Score")
    
    cards_data = [
        ("DSRI", "Chỉ số Số ngày Phải thu", "DSRI = (Phải thu_t / Doanh thu_t) / (Phải thu_t-1 / Doanh thu_t-1)", "Tăng bất thường ngụ ý ghi nhận khống doanh thu, doanh thu ảo hoặc chính sách bán chịu nới lỏng quá mức."),
        ("GMI", "Chỉ số Biên Lợi nhuận Gộp", "GMI = [(Doanh thu_t-1 - GVHB_t-1)/Doanh thu_t-1] / [(Doanh thu_t - GVHB_t)/Doanh thu_t]", "GMI > 1 báo hiệu biên lãi gộp sa sút. Doanh nghiệp dễ nảy sinh động cơ khai khống lợi nhuận."),
        ("AQI", "Chỉ số Chất lượng Tài sản", "AQI = [1 - (TSNH_t + TSCĐ_t + CKĐT_t)/Tổng TS_t] / [1 - (TSNH_t-1 + TSCĐ_t-1 + CKĐT_t-1)/Tổng TS_t-1]", "AQI > 1 chỉ ra việc vốn hóa các khoản chi phí hoạt động vào tài sản vô hình/chi phí trả trước để giảm chi phí."),
        ("SGI", "Chỉ số Tăng trưởng Doanh thu", "SGI = Doanh thu_t / Doanh thu_t-1", "Tăng trưởng cao không xấu nhưng tạo áp lực lớn phải duy trì đà tăng trưởng, dễ phát sinh gian lận khi thị trường suy giảm."),
        ("DEPI", "Chỉ số Khấu hao", "DEPI = (Tỷ lệ khấu hao_t-1) / (Tỷ lệ khấu hao_t)", "DEPI > 1 cho thấy doanh nghiệp đã hạ thấp tỷ lệ khấu hao TSCĐ hoặc kéo dài thời gian sử dụng để tăng lợi nhuận ngắn hạn."),
        ("SGAI", "Chỉ số Chi phí Bán hàng & Quản lý", "SGAI = (SG&A_t / Doanh thu_t) / (SG&A_t-1 / Doanh thu_t-1)", "SGAI > 1 phản ánh việc kiểm soát chi phí bán hàng và quản lý kém hiệu quả."),
        ("TATA", "Biến Dồn tích Kế toán trên Tổng TS", "TATA = (Lợi nhuận thuần HĐKD_t - Dòng tiền thuần từ HĐKD CFO_t) / Tổng TS_t", "TATA cao cảnh báo lợi nhuận không có tiền mặt thực tế bảo chứng (lợi nhuận 'trên giấy tờ')."),
        ("LVGI", "Chỉ số Đòn bẩy Tài chính", "LVGI = (Tổng Nợ_t / Tổng TS_t) / (Tổng Nợ_t-1 / Tổng TS_t-1)", "LVGI > 1 cho thấy rủi ro tài chính gia tăng, áp lực vi phạm các cam kết vay nợ (debt covenants) càng lớn.")
    ]
    
    col_c1, col_c2 = st.columns(2)
    for idx, (code, name, formula, expl) in enumerate(cards_data):
        target_col = col_c1 if idx % 2 == 0 else col_c2
        with target_col:
            target_col.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.1rem; color: #1E3A8A;">{code} - {name}</span>
                    <span style="background: #E0E7FF; color: #3730A3; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Beneish</span>
                </div>
                <div style="font-family: monospace; background: #FFFFFF; padding: 6px 10px; border-radius: 5px; margin: 8px 0; font-size: 0.85rem; border: 1px solid #E2E8F0;">
                    {formula}
                </div>
                <div style="font-size: 0.88rem; color: #475569; line-height: 1.4;">
                    {expl}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("""
    ### 📐 Công thức Hồi quy Logistic & Chuẩn hóa
    Trước khi đưa vào mô hình Logistic Regression, 8 biến số được chuẩn hóa với **StandardScaler** theo công thức:
    $$Z_i = \\frac{X_i - \\mu_i}{\\sigma_i}$$
    
    Xác suất xảy ra gian lận $P(\\text{Fraud} = 1)$ được ước tính bởi hàm Sigmoid:
    $$P(\\text{Fraud} = 1) = \\frac{1}{1 + e^{-(\\beta_0 + \\sum_{i=1}^8 \\beta_i Z_i)}}$$
    """)

# =========================================================
# TRANG 2: KHÁM PHÁ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH
# =========================================================
elif app_mode == "📊 Khám phá Dữ liệu & Huấn luyện":
    st.markdown('<div class="main-header">📊 KHÁM PHÁ DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Huấn luyện Pipeline chuẩn hóa và phân tích hiệu quả mô hình Logistic Regression</div>', unsafe_allow_html=True)
    
    if train_error:
        st.error(f"⚠️ {train_error}")
    else:
        df_clean = model_results["clean_df"]
        pipeline = model_results["pipeline"]
        X_test = model_results["X_test"]
        y_test = model_results["y_test"]
        X_train = model_results["X_train"]
        y_train = model_results["y_train"]
        coef_df = model_results["coef_df"]
        intercept = model_results["intercept"]
        
        # Đánh giá theo threshold hiện tại
        metrics = evaluate_model(pipeline, X_test, y_test, threshold=threshold)
        
        # Tabs chi tiết
        tab_train, tab_eda, tab_coef, tab_export = st.tabs([
            "🎯 Kết Quả Đánh Giá Mô Hình",
            "📈 Khám Phá Dữ Liệu (EDA)",
            "⚖️ Hệ Số Hồi Quy & Odds Ratio",
            "📥 Xuất Báo Cáo Ra Excel"
        ])
        
        # TAB 1: KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH
        with tab_train:
            st.markdown(f"##### 🎯 Hiệu năng mô hình trên tập Kiểm định (Test Size = {len(y_test)} quan sát | Ngưỡng P = {threshold:.2f})")
            
            # Thẻ metrics
            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            with m_col1:
                st.metric("Độ chính xác (Accuracy)", f"{metrics['Accuracy'] * 100:.2f}%")
            with m_col2:
                st.metric("Độ chuẩn xác (Precision)", f"{metrics['Precision'] * 100:.2f}%")
            with m_col3:
                st.metric("Độ nhạy (Recall / Sens)", f"{metrics['Recall'] * 100:.2f}%")
            with m_col4:
                st.metric("Chỉ số F1-Score", f"{metrics['F1'] * 100:.2f}%")
            with m_col5:
                st.metric("Đặc hiệu (Specificity)", f"{metrics['Specificity'] * 100:.2f}%")
                
            st.markdown("---")
            
            c_cm, c_roc = st.columns([1, 1])
            with c_cm:
                st.markdown("#### 🧩 Ma trận Nhầm lẫn (Confusion Matrix)")
                cm_matrix = metrics["cm"]
                
                # Vẽ heatmap ma trận nhầm lẫn bằng Plotly
                labels_x = ["Dự báo: Bình thường (0)", "Dự báo: Gian lận (1)"]
                labels_y = ["Thực tế: Bình thường (0)", "Thực tế: Gian lận (1)"]
                annotations = [
                    [f"<b>TN = {cm_matrix[0,0]}</b><br>(Đúng: Bình thường)", f"<b>FP = {cm_matrix[0,1]}</b><br>(Báo động nhầm)"],
                    [f"<b>FN = {cm_matrix[1,0]}</b><br>(Bỏ sót gian lận)", f"<b>TP = {cm_matrix[1,1]}</b><br>(Bắt đúng gian lận)"]
                ]
                
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm_matrix,
                    x=labels_x,
                    y=labels_y,
                    colorscale="Blues",
                    showscale=False,
                    text=annotations,
                    texttemplate="%{text}",
                    textfont={"size": 14}
                ))
                fig_cm.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.caption(f"📌 **Tỷ lệ báo động nhầm (FPR):** {metrics['FPR']*100:.2f}% | **Tỷ lệ bỏ sót gian lận (FNR):** {metrics['FNR']*100:.2f}%")
                
            with c_roc:
                st.markdown("#### 📉 Đường Cong ROC & Chỉ Số AUC")
                fig_roc = px.area(
                    x=metrics["fpr_curve"], y=metrics["tpr_curve"],
                    title=f"Đường cong ROC (AUC = {metrics['ROC_AUC']:.4f})",
                    labels=dict(x="False Positive Rate (1 - Specificity)", y="True Positive Rate (Recall)"),
                    height=380
                )
                fig_roc.add_shape(
                    type="line", line=dict(dash="dash", color="gray"),
                    x0=0, x1=1, y0=0, y1=1
                )
                fig_roc.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_roc, use_container_width=True)
                st.caption("📌 Đường cong ROC càng áp sát góc trên bên trái (AUC càng gần 1.0) thì khả năng phân loại của mô hình càng hoàn hảo.")
                
            # Bảng Classification Report chi tiết
            st.markdown("#### 📋 Báo Cáo Phân Loại Chi Tiết (Classification Report)")
            report_dict = classification_report(
                y_test, metrics["y_pred"],
                target_names=["Không gian lận (0)", "Gian lận (1)"],
                output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report_dict).transpose().reset_index()
            report_df.columns = ["Lớp / Chỉ tiêu", "Precision", "Recall", "F1-Score", "Số lượng quan sát"]
            st.dataframe(report_df.style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}",
                "Số lượng quan sát": "{:.0f}"
            }), use_container_width=True)

        # TAB 2: KHÁM PHÁ DỮ LIỆU (EDA)
        with tab_eda:
            st.markdown("##### 🔍 Khái quát Dữ liệu Huấn luyện")
            c_info1, c_info2 = st.columns([1, 1])
            with c_info1:
                st.write(f"- **Tổng số quan sát:** `{len(df_clean)}` dòng")
                st.write(f"- **Tập huấn luyện (Train):** `{len(X_train)}` dòng ({100 - int(test_ratio*100)}%)")
                st.write(f"- **Tập kiểm tra (Test):** `{len(X_test)}` dòng ({int(test_ratio*100)}%)")
            with c_info2:
                fraud_counts = df_clean[TARGET].value_counts()
                fig_pie = px.pie(
                    values=fraud_counts.values,
                    names=["Không gian lận (0)", "Gian lận (1)"],
                    color_discrete_sequence=["#10B981", "#EF4444"],
                    title="Tỷ lệ Nhãn Gian Lận trong Dữ Liệu",
                    hole=0.45,
                    height=260
                )
                fig_pie.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            st.markdown("##### 📑 10 Dòng dữ liệu mẫu đầu tiên:")
            st.dataframe(df_clean.head(10), use_container_width=True)
            
            st.markdown("##### 📊 Thống kê Mô tả các Chỉ số:")
            st.dataframe(df_clean.describe().T.style.format("{:.3f}"), use_container_width=True)
            
            # Ma trận tương quan
            st.markdown("##### 🔥 Ma trận Hệ số Tương quan Pearson")
            corr = df_clean[FEATURES + [TARGET]].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Ma trận tương quan giữa 8 biến Beneish và FRAUD_FLAG",
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # TAB 3: HỆ SỐ HỒI QUY & ODDS RATIO
        with tab_coef:
            st.markdown("##### ⚖️ Trọng số của các Biến trong Mô hình Logistic Regression")
            st.markdown(f"**Hệ số chặn (Intercept $\\beta_0$):** `{intercept:.6f}`")
            
            # Phương trình hồi quy
            equation = f"logit(P) = {intercept:.4f}"
            for _, row in coef_df.iterrows():
                b = row["Hệ số (Beta)"]
                sign = "+" if b >= 0 else "-"
                equation += f" {sign} {abs(b):.4f} × {row['Chỉ số']}_std"
            st.code(equation, language="text")
            
            st.dataframe(
                coef_df[["Chỉ số", "Tên chỉ số", "Hệ số (Beta)", "Odds Ratio (e^Beta)", "Tác động đến Gian lận"]]
                .style.format({"Hệ số (Beta)": "{:.4f}", "Odds Ratio (e^Beta)": "{:.4f}"})
                .bar(subset=["Hệ số (Beta)"], align="mid", color=["#EF4444", "#3B82F6"]),
                use_container_width=True
            )
            
            # Biểu đồ thanh biểu diễn Hệ số Beta
            fig_coef = px.bar(
                coef_df,
                x="Hệ số (Beta)",
                y="Chỉ số",
                orientation="h",
                color="Hệ số (Beta)",
                color_continuous_scale="Viridis",
                title="Mức độ và Hướng Tác động của Từng Chỉ số đến Log-Odds Gian lận",
                height=400
            )
            fig_coef.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_coef, use_container_width=True)
            
            st.info("""
            **Ý nghĩa của Odds Ratio ($e^\\beta$):**
            - Nếu $\\beta > 0$ (Odds Ratio $> 1$): Khi chỉ số đó tăng 1 độ lệch chuẩn, khả năng (odds) doanh nghiệp gian lận sẽ **tăng lên gấp $e^\\beta$ lần**, trong điều kiện các yếu tố khác không đổi.
            - Nếu $\\beta < 0$ (Odds Ratio $< 1$): Khi chỉ số đó tăng 1 độ lệch chuẩn, khả năng gian lận sẽ **giảm xuống**.
            """)

        # TAB 4: XUẤT BÁO CÁO EXCEL
        with tab_export:
            st.markdown("##### 📥 Xuất toàn bộ kết quả huấn luyện mô hình ra File Excel")
            st.write("Tệp Excel xuất ra sẽ bao gồm đầy đủ các bảng: Hệ số hồi quy, Ma trận nhầm lẫn, Các chỉ số hiệu năng và Thông tin mô hình (tương tự như trong Colab).")
            
            output_buffer = io.BytesIO()
            with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
                coef_df.to_excel(writer, sheet_name="Coefficients", index=False)
                
                cm_df = pd.DataFrame(
                    metrics["cm"],
                    index=["Thực tế: Bình thường (0)", "Thực tế: Gian lận (1)"],
                    columns=["Dự báo: Bình thường (0)", "Dự báo: Gian lận (1)"]
                )
                cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
                
                metrics_df = pd.DataFrame({
                    "Chỉ tiêu": [
                        "Accuracy", "Precision", "Recall / Sensitivity",
                        "F1-score", "Specificity", "FPR", "FNR", "ROC_AUC"
                    ],
                    "Giá trị": [
                        metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
                        metrics["F1"], metrics["Specificity"], metrics["FPR"],
                        metrics["FNR"], metrics["ROC_AUC"]
                    ]
                })
                metrics_df["Giá trị (%)"] = metrics_df["Giá trị"] * 100
                metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                
                model_info_df = pd.DataFrame({
                    "Thông số": ["Intercept", "Threshold", "Số mẫu Train", "Số mẫu Test", "Random State"],
                    "Giá trị": [intercept, threshold, len(X_train), len(X_test), random_seed]
                })
                model_info_df.to_excel(writer, sheet_name="Model_Info", index=False)
                
            excel_data = output_buffer.getvalue()
            
            st.download_button(
                label="📥 Tải xuống Báo cáo Huấn luyện (Excel .xlsx)",
                data=excel_data,
                file_name="Logistic_Regression_MScore_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# =========================================================
# TRANG 3: DỰ BÁO ĐƠN LẺ CHO 1 DOANH NGHIỆP
# =========================================================
elif app_mode == "🔍 Dự báo Đơn lẻ (1 Doanh nghiệp)":
    st.markdown('<div class="main-header">🔍 DỰ BÁO GIAN LẬN CHO 1 DOANH NGHIỆP CỤ THỂ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Nhập 8 chỉ số tài chính để nhận diện mức độ rủi ro, tính xác suất gian lận và điểm Beneish M-Score</div>', unsafe_allow_html=True)
    
    if train_error:
        st.error(f"⚠️ {train_error}")
    else:
        pipeline = model_results["pipeline"]
        df_clean = model_results["clean_df"]
        
        # Nút nạp dữ liệu mẫu nhanh
        st.markdown("##### ⚡ Kiểm thử nhanh với hồ sơ mẫu:")
        c_sample1, c_sample2, c_sample3 = st.columns([1, 1, 2])
        
        # Mẫu bình thường (lấy từ dữ liệu label 0)
        sample_safe = {
            "DSRI": 0.850, "GMI": 0.950, "AQI": 0.800, "SGI": 1.050,
            "DEPI": 0.980, "SGAI": 0.950, "TATA": 0.020, "LVGI": 0.950
        }
        # Mẫu rủi ro gian lận cao (lấy từ dữ liệu label 1)
        sample_fraud = {
            "DSRI": 1.750, "GMI": 1.650, "AQI": 1.450, "SGI": 1.600,
            "DEPI": 1.250, "SGAI": 1.200, "TATA": 0.180, "LVGI": 1.350
        }
        
        if "input_vals" not in st.session_state:
            st.session_state.input_vals = sample_safe.copy()
            
        with c_sample1:
            if st.button("🟢 Nạp mẫu: Doanh nghiệp An Toàn", use_container_width=True):
                st.session_state.input_vals = sample_safe.copy()
                st.rerun()
                
        with c_sample2:
            if st.button("🔴 Nạp mẫu: Doanh nghiệp Nguy Cơ Cao", use_container_width=True):
                st.session_state.input_vals = sample_fraud.copy()
                st.rerun()
                
        with c_sample3:
            company_code = st.text_input("Mã Doanh nghiệp / Cổ phiếu (Tùy chọn):", value="VIN-CORP")
            
        st.markdown("---")
        st.markdown("##### 📝 Nhập thông số 8 chỉ số Beneish M-Score:")
        
        col_in1, col_in2 = st.columns(2)
        user_inputs = {}
        
        for i, feat in enumerate(FEATURES):
            col_target = col_in1 if i < 4 else col_in2
            curr_val = float(st.session_state.input_vals.get(feat, 1.0))
            with col_target:
                user_inputs[feat] = st.number_input(
                    label=f"**{feat}** ({FEATURE_NAMES_VI[feat].split('(')[1].rstrip(')')})",
                    value=curr_val,
                    step=0.01,
                    format="%.4f",
                    help=FEATURE_DESCRIPTIONS[feat],
                    key=f"input_{feat}"
                )
                
        # Nút thực hiện dự báo
        if st.button("🚀 TIẾN HÀNH PHÂN TÍCH & DỰ BÁO", type="primary", use_container_width=True):
            input_df = pd.DataFrame([user_inputs])
            
            # Dự báo xác suất bằng Logistic Pipeline
            prob_fraud = pipeline.predict_proba(input_df)[0, 1]
            is_fraud_pred = prob_fraud >= threshold
            
            # Tính điểm Beneish M-Score chuẩn
            m_score = calculate_beneish_mscore(user_inputs)
            is_beneish_manipulator = m_score > -1.78
            
            st.markdown("---")
            st.markdown("### 📊 KẾT QUẢ ĐÁNH GIÁ RỦI RO GIAN LẬN")
            
            # Thẻ kết quả tổng quan
            res_c1, res_c2, res_c3 = st.columns([1.2, 1, 1])
            with res_c1:
                if prob_fraud >= 0.60:
                    status_badge = '<div class="danger-badge" style="font-size: 1.1rem;">⚠️ CẢNH BÁO: NGUY CƠ GIAN LẬN CAO</div>'
                elif prob_fraud >= threshold:
                    status_badge = '<div class="warning-badge" style="font-size: 1.1rem;">⚡ CẢNH BÁO: CÓ DẤU HIỆU BẤT THƯỜNG</div>'
                else:
                    status_badge = '<div class="safe-badge" style="font-size: 1.1rem;">✅ AN TOÀN: ÍT NGUY CƠ GIAN LẬN</div>'
                    
                st.markdown(status_badge, unsafe_allow_html=True)
                st.markdown(f"**Doanh nghiệp:** `{company_code}`")
                st.markdown(f"**Dự báo theo mô hình:** `{'Gian lận BCTC (1)' if is_fraud_pred else 'Không gian lận (0)'}`")
                
            with res_c2:
                st.metric(
                    label="Xác suất Gian lận P(Fraud)",
                    value=f"{prob_fraud * 100:.2f}%",
                    delta=f"{'+' if prob_fraud >= threshold else ''}{(prob_fraud - threshold)*100:.1f}% so với ngưỡng {threshold*100:.0f}%",
                    delta_color="inverse"
                )
                st.progress(float(prob_fraud))
                
            with res_c3:
                st.metric(
                    label="Điểm Beneish M-Score Chuẩn",
                    value=f"{m_score:.3f}",
                    delta="Thao túng BCTC" if is_beneish_manipulator else "Bình thường",
                    delta_color="inverse" if is_beneish_manipulator else "normal"
                )
                st.caption("Ngưỡng chuẩn Beneish: M > -1.78 cảnh báo nguy cơ thao túng lợi nhuận.")
                
            st.markdown("---")
            
            # Biểu đồ Radar so sánh với trung bình tập dữ liệu
            col_radar, col_advice = st.columns([1.1, 1])
            with col_radar:
                st.markdown("#### 🎯 Biểu đồ Radar: Đối chiếu với Dữ liệu Toàn ngành")
                
                # Tính trung bình các biến cho 2 nhóm trong dữ liệu huấn luyện
                mean_safe = df_clean[df_clean[TARGET] == 0][FEATURES].mean()
                mean_fraud = df_clean[df_clean[TARGET] == 1][FEATURES].mean()
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[user_inputs[f] for f in FEATURES] + [user_inputs[FEATURES[0]]],
                    theta=FEATURES + [FEATURES[0]],
                    fill='toself',
                    name=f'{company_code} (Hiện tại)',
                    line_color='#2563EB'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=mean_safe.tolist() + [mean_safe[0]],
                    theta=FEATURES + [FEATURES[0]],
                    name='TB Nhóm Không Gian Lận',
                    line_color='#10B981',
                    line=dict(dash='dash')
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=mean_fraud.tolist() + [mean_fraud[0]],
                    theta=FEATURES + [FEATURES[0]],
                    name='TB Nhóm Gian Lận',
                    line_color='#EF4444',
                    line=dict(dash='dot')
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=True,
                    height=400,
                    margin=dict(l=30, r=30, t=30, b=30)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
            with col_advice:
                st.markdown("#### 💡 Khuyến nghị cho Kiểm toán viên / Chuyên viên Phân tích:")
                
                flagged_indicators = []
                if user_inputs["DSRI"] > 1.2:
                    flagged_indicators.append(("DSRI", f"DSRI = {user_inputs['DSRI']:.2f} (>1.2): Khoản phải thu tăng vọt so với doanh thu. Cần kiểm tra kỹ các hóa đơn bán hàng cuối niên độ và gửi thư xác nhận công nợ."))
                if user_inputs["GMI"] > 1.2:
                    flagged_indicators.append(("GMI", f"GMI = {user_inputs['GMI']:.2f} (>1.2): Biên lãi gộp suy giảm mạnh. Cần rà soát tính hợp lý của giá vốn hàng bán và trích lập dự phòng giảm giá hàng tồn kho."))
                if user_inputs["AQI"] > 1.2:
                    flagged_indicators.append(("AQI", f"AQI = {user_inputs['AQI']:.2f} (>1.2): Tài sản phi hiện hữu tăng cao. Cần kiểm tra các khoản chi phí trả trước dài hạn, lợi thế thương mại hoặc tài sản dở dang có dấu hiệu bị treo lại để tránh ghi giảm lợi nhuận."))
                if user_inputs["SGI"] > 1.3:
                    flagged_indicators.append(("SGI", f"SGI = {user_inputs['SGI']:.2f} (>1.3): Tốc độ tăng trưởng doanh thu nóng. Rà soát giao dịch với các bên liên quan (related-party transactions)."))
                if user_inputs["DEPI"] > 1.1:
                    flagged_indicators.append(("DEPI", f"DEPI = {user_inputs['DEPI']:.2f} (>1.1): Tỷ lệ trích khấu hao sụt giảm. Kiểm tra xem doanh nghiệp có thay đổi chính sách khấu hao (kéo dài thời gian sử dụng hữu ích) trái quy định không."))
                if user_inputs["TATA"] > 0.08:
                    flagged_indicators.append(("TATA", f"TATA = {user_inputs['TATA']:.4f} (>0.08): Biến dồn tích kế toán cao. Lợi nhuận kế toán vượt xa dòng tiền thuần từ hoạt động kinh doanh (CFO). Chất lượng lợi nhuận kém."))
                if user_inputs["LVGI"] > 1.2:
                    flagged_indicators.append(("LVGI", f"LVGI = {user_inputs['LVGI']:.2f} (>1.2): Đòn bẩy nợ tăng đột biến. Kiểm tra áp lực thanh toán nợ đến hạn và các điều khoản duy trì hệ số tài chính với ngân hàng."))
                    
                if flagged_indicators:
                    for tag, adv in flagged_indicators:
                        st.warning(f"⚠️ **{tag}:** {adv}")
                else:
                    st.success("✅ **Các chỉ số đều nằm trong biên độ tương đối an toàn.** Không phát hiện tín hiệu cảnh báo nghiêm trọng từ 8 chỉ số thành phần.")

# =========================================================
# TRANG 4: DỰ BÁO HÀNG LOẠT (BATCH PREDICTION)
# =========================================================
elif app_mode == "📂 Dự báo Hàng loạt (File CSV/Excel)":
    st.markdown('<div class="main-header">📂 DỰ BÁO HÀNG LOẠT TỪ TỆP DỮ LIỆU</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tải lên danh sách nhiều doanh nghiệp để tự động tầm soát, phân loại nguy cơ gian lận và xuất báo cáo tổng hợp</div>', unsafe_allow_html=True)
    
    if train_error:
        st.error(f"⚠️ {train_error}")
    else:
        pipeline = model_results["pipeline"]
        
        # Hướng dẫn định dạng & Nút tải template
        with st.expander("ℹ️ Hướng dẫn định dạng file và Tải File mẫu Template", expanded=False):
            st.markdown("""
            - File tải lên có định dạng **CSV** hoặc **Excel (.xlsx, .xls)**.
            - Phải chứa đầy đủ 8 cột chỉ số: `DSRI`, `GMI`, `AQI`, `SGI`, `DEPI`, `SGAI`, `TATA`, `LVGI`.
            - Có thể chứa thêm các cột định danh như: `Company_ID`, `Company_Name`, `Mã_CK`, `Năm`.
            """)
            
            # Tạo template mẫu để tải về
            template_df = pd.DataFrame([
                {"Company_ID": "CT_001", "DSRI": 1.12, "GMI": 0.95, "AQI": 0.88, "SGI": 1.15, "DEPI": 0.92, "SGAI": 0.98, "TATA": 0.03, "LVGI": 1.05},
                {"Company_ID": "CT_002", "DSRI": 1.85, "GMI": 1.62, "AQI": 1.48, "SGI": 1.72, "DEPI": 1.28, "SGAI": 1.25, "TATA": 0.19, "LVGI": 1.42},
                {"Company_ID": "CT_003", "DSRI": 0.92, "GMI": 1.02, "AQI": 0.79, "SGI": 1.04, "DEPI": 1.01, "SGAI": 0.96, "TATA": -0.02, "LVGI": 0.98}
            ])
            csv_template = template_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải Tệp Mẫu (Template_Batch_Prediction.csv)",
                data=csv_template,
                file_name="Template_Batch_Prediction.csv",
                mime="text/csv"
            )
            
        uploaded_batch = st.file_uploader(
            "Chọn file danh sách doanh nghiệp cần dự báo (CSV hoặc Excel):",
            type=["csv", "xlsx", "xls"]
        )
        
        if uploaded_batch is not None:
            try:
                if uploaded_batch.name.endswith(".csv"):
                    batch_df = pd.read_csv(uploaded_batch)
                else:
                    batch_df = pd.read_excel(uploaded_batch)
                    
                st.success(f"✅ Đã tải file: **{uploaded_batch.name}** với **{len(batch_df)}** dòng.")
                
                # Kiểm tra 8 cột bắt buộc
                missing_cols = [c for c in FEATURES if c not in batch_df.columns]
                if missing_cols:
                    st.error(f"❌ File thiếu các cột bắt buộc: {missing_cols}. Vui lòng kiểm tra lại định dạng.")
                else:
                    # Tiền xử lý dữ liệu dự báo
                    pred_data = batch_df[FEATURES].copy()
                    for c in FEATURES:
                        pred_data[c] = pd.to_numeric(pred_data[c], errors="coerce")
                    
                    if pred_data.isna().any().any():
                        st.warning("⚠️ Phát hiện một số giá trị trống hoặc không hợp lệ. Các dòng thiếu sẽ bị điền bằng giá trị trung vị (median) của cột.")
                        pred_data = pred_data.fillna(pred_data.median())
                    
                    # Chạy dự báo
                    prob_fraud_batch = pipeline.predict_proba(pred_data)[:, 1]
                    pred_labels = (prob_fraud_batch >= threshold).astype(int)
                    
                    # Tính điểm Beneish M-Score
                    beneish_scores = pred_data.apply(calculate_beneish_mscore, axis=1)
                    
                    # Phân nhóm mức độ rủi ro
                    risk_levels = []
                    for p in prob_fraud_batch:
                        if p >= 0.65:
                            risk_levels.append("Rủi ro Rất cao")
                        elif p >= threshold:
                            risk_levels.append("Rủi ro Cao")
                        elif p >= 0.35:
                            risk_levels.append("Rủi ro Trung bình")
                        else:
                            risk_levels.append("Rủi ro Thấp (An toàn)")
                            
                    result_df = batch_df.copy()
                    result_df["Xác_suất_Gian_lận_P(%)"] = np.round(prob_fraud_batch * 100, 2)
                    result_df["Dự_báo_Gian_lận"] = pred_labels
                    result_df["Trạng_thái_Dự_báo"] = ["Gian lận BCTC (1)" if x == 1 else "Không gian lận (0)" for x in pred_labels]
                    result_df["Mức_độ_Rủi_ro"] = risk_levels
                    result_df["Beneish_M_Score"] = np.round(beneish_scores, 3)
                    result_df["Cảnh_báo_Beneish"] = ["Thao túng (M > -1.78)" if m > -1.78 else "Bình thường" for m in beneish_scores]
                    
                    st.markdown("---")
                    st.markdown("### 📊 Tổng Hợp Kết Quả Tầm Soát Hàng Loạt")
                    
                    # Thống kê nhanh
                    total_count = len(result_df)
                    fraud_count = int((pred_labels == 1).sum())
                    safe_count = total_count - fraud_count
                    
                    col_st1, col_st2, col_st3, col_st4 = st.columns(4)
                    with col_st1:
                        st.metric("Tổng số DN rà soát", f"{total_count} DN")
                    with col_st2:
                        st.metric("Số DN An toàn", f"{safe_count} DN", f"{(safe_count/total_count)*100:.1f}%")
                    with col_st3:
                        st.metric("Số DN Cảnh báo Gian lận", f"{fraud_count} DN", f"{(fraud_count/total_count)*100:.1f}%", delta_color="inverse")
                    with col_st4:
                        high_risk_count = sum(1 for r in risk_levels if "cao" in r.lower())
                        st.metric("Số DN Rủi ro Cao / Rất cao", f"{high_risk_count} DN", delta_color="inverse")
                        
                    # Biểu đồ phân bổ mức độ rủi ro
                    col_g1, col_g2 = st.columns([1, 1])
                    with col_g1:
                        risk_summary = pd.Series(risk_levels).value_counts().reset_index()
                        risk_summary.columns = ["Mức rủi ro", "Số lượng"]
                        fig_risk = px.pie(
                            risk_summary,
                            names="Mức rủi ro",
                            values="Số lượng",
                            title="Phân bố Mức độ Rủi ro Gian lận",
                            color="Mức rủi ro",
                            color_discrete_map={
                                "Rủi ro Rất cao": "#DC2626",
                                "Rủi ro Cao": "#EA580C",
                                "Rủi ro Trung bình": "#FBBF24",
                                "Rủi ro Thấp (An toàn)": "#10B981"
                            },
                            hole=0.4
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                    with col_g2:
                        fig_hist = px.histogram(
                            result_df,
                            x="Xác_suất_Gian_lận_P(%)",
                            nbins=20,
                            title="Phân bố Xác suất Gian lận P(Fraud) %",
                            color="Trạng_thái_Dự_báo",
                            color_discrete_map={
                                "Gian lận BCTC (1)": "#EF4444",
                                "Không gian lận (0)": "#10B981"
                            }
                        )
                        fig_hist.add_vline(x=threshold * 100, line_dash="dash", line_color="red", annotation_text=f"Ngưỡng {threshold*100:.0f}%")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                    st.markdown("---")
                    st.markdown("#### 📋 Bảng Chi Tiết Kết Quả Dự Báo")
                    
                    # Bộ lọc xem dữ liệu
                    filter_risk = st.selectbox("Lọc theo mức độ rủi ro:", ["Tất cả"] + list(result_df["Mức_độ_Rủi_ro"].unique()))
                    display_res = result_df if filter_risk == "Tất cả" else result_df[result_df["Mức_độ_Rủi_ro"] == filter_risk]
                    
                    st.dataframe(display_res, use_container_width=True)
                    
                    # Nút tải file kết quả
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv_result = result_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải Kết Quả (File CSV)",
                            data=csv_result,
                            file_name="Ket_qua_Du_bao_Gian_lan_BCTC.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col_dl2:
                        excel_buf = io.BytesIO()
                        with pd.ExcelWriter(excel_buf, engine="openpyxl") as wr:
                            result_df.to_excel(wr, sheet_name="Fraud_Prediction_Results", index=False)
                        st.download_button(
                            label="📥 Tải Kết Quả (File Excel .xlsx)",
                            data=excel_buf.getvalue(),
                            file_name="Ket_qua_Du_bao_Gian_lan_BCTC.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
            except Exception as ex:
                st.error(f"Đã xảy ra lỗi khi xử lý file: {str(ex)}")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem;">
    ⚖️ <b>Hệ thống Dự báo Gian lận Báo cáo Tài chính</b> | Xây dựng với Python & Streamlit<br>
    Ứng dụng phục vụ công tác kiểm toán, nghiên cứu học thuật và tầm soát rủi ro tài chính doanh nghiệp.
</div>
""", unsafe_allow_html=True)
