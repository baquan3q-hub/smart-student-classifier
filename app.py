"""
app.py — Smart Student Classification System
Ứng dụng Streamlit chính với 8 tabs giao diện tiếng Việt.

Chạy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import io

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    ensure_directories, FEATURE_COLUMNS, CLASS_LABELS, TARGET_COLUMN,
    STUDENT_FEATURES_FILE, STUDENT_PROFILES_FILE, STUDENT_SCORES_FILE,
    STUDENT_COMMENTS_FILE, METRICS_FILE, MODELS_DIR,
    SCORE_SUBJECTS, COMMENT_SUBJECTS, BATCH_PREDICTIONS_FILE,
)
from src.rule_engine import (
    calculate_dtb_mhk, calculate_dtb_mcn,
    classify_learning_result, get_classification_reason, get_classification_details,
)
from src.data_generator import generate_all_data
from src.preprocessing import prepare_features_target, validate_schema, validate_ranges
from src.train_models import train_all, load_models
from src.predict import predict_one, predict_batch, load_trained_models
from src.evaluation import compare_models, get_feature_importance_df
from src.explanations import (
    generate_ml_explanation, explain_feature_importance,
    explain_decision_tree_summary, explain_rule_engine_logic, FEATURE_NAMES_VI,
    explain_qualitative_metrics,
)
from src.recommendations import generate_recommendations, get_risk_level
from src.visualization import (
    plot_label_distribution, plot_label_pie, plot_confusion_matrix,
    plot_feature_importance, plot_metrics_comparison, plot_score_distribution,
    plot_risk_level_distribution, plot_risk_by_class,
)
from src.utils import get_label_emoji, get_label_color

# ============================================================
# CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Smart Student Classification System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()

# ============================================================
# CSS TÙY CHỈNH
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1E3A5F;
        text-align: center; margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem; color: #5A6C7D;
        text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 { font-size: 1.8rem; margin: 0; font-weight: 700; }
    .metric-card p { font-size: 0.85rem; margin: 0.2rem 0 0 0; opacity: 0.9; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<p class="main-header">🎓 Smart Student Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hệ thống phân loại kết quả học tập học sinh — Demo học thuật</p>', unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_features_if_exists():
    """Load features CSV nếu tồn tại."""
    if STUDENT_FEATURES_FILE.exists():
        return pd.read_csv(STUDENT_FEATURES_FILE, encoding="utf-8-sig")
    return None

def load_metrics_if_exists():
    """Load metrics JSON nếu tồn tại."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def to_excel(df: pd.DataFrame) -> bytes:
    """Chuyển đổi DataFrame thành file Excel nhị phân dùng BytesIO."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# ============================================================
# 8 TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Tổng quan",
    "📁 Dữ liệu",
    "📐 Tính điểm & Rule Engine",
    "🤖 Huấn luyện mô hình",
    "👤 Dự đoán một học sinh",
    "🚨 Cảnh báo sớm & Dự đoán hàng loạt",
    "🔍 Giải thích mô hình",
    "📚 Phương pháp & giới hạn",
])


# ============================================================
# TAB 1: TỔNG QUAN
# ============================================================

with tab1:
    st.header("📊 Tổng quan hệ thống")
    
    # --- PREMIUM ONBOARDING GUIDE ---
    with st.expander("🚀 HƯỚNG DẪN TRẢI NGHIỆM & TRÌNH BÀY HỘI ĐỒNG (BẮT BUỘC ĐỌC)", expanded=True):
        st.markdown("""
        <div style="background-color: #f0f4f8; padding: 18px; border-radius: 10px; border-left: 5px solid #1E3A5F; margin-bottom: 20px;">
            <h4 style="color: #1E3A5F; margin-top: 0; font-weight: 700;">👋 Chào mừng bạn đến với Smart Student Classification System!</h4>
            <p style="font-size: 0.95rem; color: #333333; margin-bottom: 0; line-height: 1.5;">
                Đây là hệ thống <b>demo học thuật phục vụ báo cáo cuối kỳ</b>. 
                Dưới đây là thiết kế kiến trúc và lộ trình trải nghiệm được xây dựng bởi chuyên gia phát triển, 
                giúp bạn hoàn toàn làm chủ sản phẩm và bảo vệ thành công trước Hội đồng chấm phản biện!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        ob_col1, ob_col2, ob_col3 = st.columns(3)
        
        with ob_col1:
            st.markdown("<h4 style='color: #1E3A5F; font-size: 1.15rem; font-weight: 600;'>❓ Tôi sử dụng Web này để làm gì?</h4>", unsafe_allow_html=True)
            st.markdown("""
            * **Tự động hóa nghiệp vụ:** Áp dụng tự động quy chế phân loại học sinh THCS/THPT theo chuẩn **Thông tư 22/2021/TT-BGDĐT** của Bộ Giáo dục & Đào tạo Việt Nam.
            * **Dự báo sớm chủ động:** Rule Engine truyền thống chỉ xếp loại khi đã thi xong cuối kỳ (bị động). Hệ thống này dùng **AI/Machine Learning** học từ luật để **dự báo sớm** nguy cơ học sinh vắng học nhiều hoặc sa sút học tập trước khi thi.
            """)
            
        with ob_col2:
            st.markdown("<h4 style='color: #1E3A5F; font-size: 1.15rem; font-weight: 600;'>💎 Giá trị hệ thống mang lại là gì?</h4>", unsafe_allow_html=True)
            st.markdown("""
            * **Can thiệp sớm (Early Intervention):** Giúp giáo viên chủ nhiệm và bộ môn phát hiện ngay học sinh vắng học nhiều hoặc thiếu bài tập để ôn luyện kịp thời.
            * **Cá nhân hóa khuyến nghị:** Hệ thống tự động phân tích hành vi và sinh ra các đề xuất học tập cụ thể, đặc biệt là cảnh báo đỏ pháp lý vắng quá 45 buổi học theo Điều 12 TT22.
            """)
            
        with ob_col3:
            st.markdown("<h4 style='color: #1E3A5F; font-size: 1.15rem; font-weight: 600;'>🛠️ Có những tính năng nào nên thử?</h4>", unsafe_allow_html=True)
            st.markdown("""
            1. **Tạo dữ liệu mẫu (Tab 2):** Nhấp nút để tạo nhanh 300+ học sinh phân bố nhãn cân bằng thực tế.
            2. **Mô phỏng tính điểm (Tab 3):** Chọn học sinh và xem Rule Engine bóc tách lý do xếp loại.
            3. **Huấn luyện AI (Tab 4):** Trải nghiệm huấn luyện Decision Tree & Random Forest và xem Confusion Matrix trực quan.
            4. **Dự đoán cá nhân (Tab 5):** Thử thay đổi chuyên cần, điểm số để thấy AI phản hồi tức thì.
            """)
            
        st.markdown("---")
        st.markdown("<h4 style='color: #1E3A5F; font-size: 1.25rem; font-weight: 600;'>👨‍🏫 Làm sao để tôi dễ hiểu và trình bày trước Hội đồng?</h4>", unsafe_allow_html=True)
        
        st.info("""
        💡 **Kịch bản & Chiến lược thuyết trình thuyết phục Thầy Cô:**
        
        * **Bước 1 (Đặt vấn đề):** *"Quy định xếp loại của Thông tư 22 rất phức tạp và chồng chéo (kết hợp cả điểm số và nhận xét). Đồng thời, giáo viên chỉ biết học sinh bị xếp loại yếu sau khi kỳ thi đã kết thúc - lúc đó đã quá muộn để giúp đỡ các em."*
        
        * **Bước 2 (Giải pháp đột phá):** *"Hệ thống của em giải quyết cả 2 vấn đề. **Rule Engine BGDDT** đảm bảo tính chính xác 100% theo luật. **Machine Learning (AI)** học từ luật để dự báo sớm kết quả của học sinh ngay trong học kỳ thông qua các chỉ số hành vi (chuyên cần, bài tập)."*
        
        * **Bước 3 (Chứng minh thực tế):** *"Như Hội đồng có thể thấy tại Tab Giải thích mô hình, AI đã tìm ra Chuyên cần (attendance_rate) và Tỷ lệ hoàn thành bài tập (assignment_completion_rate) là 2 đặc trưng quan trọng nhất. Đặc biệt, hệ thống đã tích hợp Điều 12 TT22: nếu học sinh vắng học quá 45 buổi học (chuyên cần dưới 75%), hệ thống sẽ tự động kích hoạt cảnh báo đỏ nguy cơ lưu ban!"*
        """)

    st.markdown("---")

    features_df = load_features_if_exists()

    if features_df is None:
        st.warning("⚠️ Chưa có dữ liệu. Hãy chuyển sang tab **Dữ liệu** để tạo dataset mẫu.")
    else:
        # --- KPI Metrics ---
        total = len(features_df)
        label_counts = features_df[TARGET_COLUMN].value_counts()
        avg_score = features_df["avg_score"].mean()
        avg_attendance = features_df["attendance_rate"].mean()
        num_chua_dat = label_counts.get("Chưa đạt", 0)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📚 Tổng học sinh", total)
        col2.metric("📊 ĐTB chung", f"{avg_score:.1f}")
        col3.metric("📅 Chuyên cần TB", f"{avg_attendance:.1f}%")
        col4.metric("🔴 Chưa đạt", num_chua_dat)
        col5.metric("🟢 Tốt", label_counts.get("Tốt", 0))

        st.markdown("---")

        # --- Biểu đồ ---
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(plot_label_distribution(features_df), use_container_width=True)
        with col_chart2:
            st.plotly_chart(plot_label_pie(features_df), use_container_width=True)

        # --- Phân bố theo nhãn ---
        st.subheader("📊 Thống kê theo nhãn")
        stats_data = []
        for label in CLASS_LABELS:
            subset = features_df[features_df[TARGET_COLUMN] == label]
            if len(subset) > 0:
                stats_data.append({
                    "Nhãn": f"{get_label_emoji(label)} {label}",
                    "Số lượng": len(subset),
                    "Tỷ lệ (%)": f"{len(subset)/total*100:.1f}",
                    "ĐTB trung bình": f"{subset['avg_score'].mean():.1f}",
                    "Chuyên cần TB (%)": f"{subset['attendance_rate'].mean():.1f}",
                })
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

        # --- Thống kê theo khối lớp ---
        st.subheader("🏫 Thống kê theo Khối lớp")
        features_df["grade"] = features_df["class_name"].apply(lambda x: f"Khối {x[0]}" if isinstance(x, str) and x[0].isdigit() else "Khác")
        grade_stats_data = []
        for grade in sorted(features_df["grade"].unique()):
            g_subset = features_df[features_df["grade"] == grade]
            grade_stats_data.append({
                "Khối lớp": grade,
                "Số lượng học sinh": len(g_subset),
                "Tỷ lệ (%)": f"{len(g_subset)/total*100:.1f}",
                "ĐTB trung bình": f"{g_subset['avg_score'].mean():.1f}",
                "Chuyên cần TB (%)": f"{g_subset['attendance_rate'].mean():.1f}",
                "Số học sinh Chưa đạt": len(g_subset[g_subset[TARGET_COLUMN] == "Chưa đạt"]),
            })
        st.dataframe(pd.DataFrame(grade_stats_data), use_container_width=True, hide_index=True)

        # --- Biểu đồ phân bố điểm ---
        st.plotly_chart(plot_score_distribution(features_df), use_container_width=True)

        # --- Danh sách cần hỗ trợ ---
        st.subheader("🚨 Học sinh cần hỗ trợ (Chưa đạt)")
        chua_dat = features_df[features_df[TARGET_COLUMN] == "Chưa đạt"]
        if len(chua_dat) > 0:
            display_cols = ["student_id", "class_name", "avg_score", "min_score",
                           "attendance_rate", "comment_not_pass_count"]
            st.dataframe(
                chua_dat[display_cols].sort_values("avg_score").head(20),
                use_container_width=True, hide_index=True,
            )
        else:
            st.success("🎉 Không có học sinh nào ở mức Chưa đạt!")

# ============================================================
# TAB 2: DỮ LIỆU
# ============================================================

with tab2:
    st.header("📁 Quản lý dữ liệu")

    col_gen, col_upload = st.columns(2)

    with col_gen:
        st.subheader("🔄 Tạo dataset mẫu")
        num_students = st.number_input("Số lượng học sinh", min_value=100, max_value=2000, value=1000, step=50)
        
        profile_options = {
            "balanced": "📊 Đều & Cân bằng (Chuẩn thực tế)",
            "high_performing": "🟢 Thiên về Học tốt (High Performing)",
            "at_risk": "🔴 Thiên về Nguy cơ / Chưa đạt (At Risk)",
            "mid_semester": "📅 Dữ liệu Giữa kỳ (Chưa có điểm cuối kỳ)"
        }
        
        profile_choice = st.selectbox(
            "Kiểu dữ liệu sinh mẫu",
            options=list(profile_options.keys()),
            format_func=lambda x: profile_options[x],
            help="Chọn cấu trúc phân bổ và mức độ học lực của tệp học sinh giả lập."
        )
        
        # Mô tả chi tiết kiểu dữ liệu
        if profile_choice == "balanced":
            st.info("💡 **Phân bổ cân bằng**: Phân phối nhãn chuẩn học sinh thực tế (~20% Tốt, ~28% Khá, ~26% Đạt, ~16% Chưa đạt, ~10% Nghịch cảnh).")
        elif profile_choice == "high_performing":
            st.success("💡 **Học sinh xuất sắc**: Đa số học sinh đạt kết quả tốt/khá (~85%), học sinh yếu kém cực kỳ ít (~3%). Phù hợp mô phỏng lớp chọn.")
        elif profile_choice == "at_risk":
            st.warning("💡 **Nguy cơ cao**: Tỷ lệ học sinh yếu kém, Chưa đạt chiếm tỷ trọng lớn (~50%). Đây là tệp dữ liệu hoàn hảo để kiểm nghiệm bộ lọc Cảnh báo sớm (EWS).")
        elif profile_choice == "mid_semester":
            st.error("💡 **Dữ liệu Giữa kỳ**: Điểm thi cuối kỳ trống (NaN). Điểm học kỳ được tính tạm thời theo điểm số giữa kỳ. Thích hợp cho kịch bản mô phỏng cảnh báo sớm trước kỳ thi.")

        if st.button("🚀 Tạo dataset mẫu", type="primary", use_container_width=True):
            with st.spinner("Đang tạo dữ liệu mô phỏng..."):
                profiles, scores, comments, features = generate_all_data(
                    num_students=num_students, random_state=42, data_profile=profile_choice
                )
            st.success(f"✅ Đã tạo dataset cho {num_students} học sinh ({profile_options[profile_choice]})!")
            st.rerun()


    with col_upload:
        st.subheader("📤 Upload CSV")
        uploaded_file = st.file_uploader("Upload file student_features.csv", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                is_valid, missing = validate_schema(df)
                if is_valid:
                    df.to_csv(STUDENT_FEATURES_FILE, index=False, encoding="utf-8-sig")
                    st.success("✅ Upload thành công!")
                    st.rerun()
                else:
                    st.error(f"❌ Thiếu các cột: {missing}")
            except Exception as e:
                st.error(f"❌ Lỗi đọc file: {e}")

    st.markdown("---")

    # Preview
    features_df = load_features_if_exists()
    if features_df is not None:
        st.subheader("👁️ Preview dữ liệu")

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Số dòng", features_df.shape[0])
        col_info2.metric("Số cột", features_df.shape[1])
        col_info3.metric("Missing values", features_df.isnull().sum().sum())

        st.dataframe(features_df.head(20), use_container_width=True, hide_index=True)

        # Kiểm tra range
        is_valid_range, range_warnings = validate_ranges(features_df)
        if not is_valid_range:
            st.warning("⚠️ Một số giá trị ngoài phạm vi:")
            for w in range_warnings:
                st.write(f"  - {w}")

        # Download
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            csv_data = features_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 Tải file student_features.csv",
                csv_data, "student_features.csv", "text/csv",
                use_container_width=True,
            )
        with col_down2:
            try:
                excel_data = to_excel(features_df)
                st.download_button(
                    "📥 Tải file student_features.xlsx",
                    excel_data, "student_features.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Lỗi xuất Excel: {e}")

        # Preview raw files
        st.subheader("📄 Preview dữ liệu thô")
        raw_tab1, raw_tab2, raw_tab3 = st.tabs(["Hồ sơ học sinh", "Điểm theo môn", "Nhận xét"])
        with raw_tab1:
            if STUDENT_PROFILES_FILE.exists():
                st.dataframe(pd.read_csv(STUDENT_PROFILES_FILE, encoding="utf-8-sig").head(10),
                           use_container_width=True, hide_index=True)
        with raw_tab2:
            if STUDENT_SCORES_FILE.exists():
                st.dataframe(pd.read_csv(STUDENT_SCORES_FILE, encoding="utf-8-sig").head(10),
                           use_container_width=True, hide_index=True)
        with raw_tab3:
            if STUDENT_COMMENTS_FILE.exists():
                st.dataframe(pd.read_csv(STUDENT_COMMENTS_FILE, encoding="utf-8-sig").head(10),
                           use_container_width=True, hide_index=True)

# ============================================================
# TAB 3: TÍNH ĐIỂM & RULE ENGINE
# ============================================================

with tab3:
    st.header("📐 Tính điểm & Rule Engine")
    features_df = load_features_if_exists()

    if features_df is None:
        st.warning("⚠️ Chưa có dữ liệu. Hãy tạo dataset mẫu trước.")
    else:
        # --- Công thức ---
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.subheader("📐 Công thức DTB_mhk")
            st.latex(r"DTB_{mhk} = \frac{\sum \text{ĐTX} + 2 \times \text{ĐGK} + 3 \times \text{ĐCK}}{n_{TX} + 5}")
            st.caption("ĐTX: Điểm thường xuyên | ĐGK: Điểm giữa kỳ | ĐCK: Điểm cuối kỳ")

        with col_f2:
            st.subheader("📐 Công thức DTB_mcn")
            st.latex(r"DTB_{mcn} = \frac{DTB_{HK1} + 2 \times DTB_{HK2}}{3}")
            st.caption("DTB_HK1: ĐTB học kỳ 1 | DTB_HK2: ĐTB học kỳ 2")

        st.markdown("---")

        # --- Demo tính điểm ---
        st.subheader("🧮 Demo tính DTB_mhk")
        col_demo1, col_demo2 = st.columns(2)
        with col_demo1:
            demo_reg = st.text_input("Điểm thường xuyên (;)", value="7.5;8.0;8.5")
            demo_mid = st.number_input("Điểm giữa kỳ", 0.0, 10.0, 8.0, 0.1)
            demo_final = st.number_input("Điểm cuối kỳ", 0.0, 10.0, 8.5, 0.1)
        with col_demo2:
            if demo_reg:
                try:
                    reg_scores = [float(s.strip()) for s in demo_reg.split(";") if s.strip()]
                    dtb = calculate_dtb_mhk(reg_scores, demo_mid, demo_final)
                    st.metric("📊 DTB_mhk", dtb)
                    st.write(f"Công thức: ({'+'.join(str(s) for s in reg_scores)} + 2×{demo_mid} + 3×{demo_final}) / ({len(reg_scores)} + 5)")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        st.markdown("---")

        # --- Xem theo học sinh ---
        st.subheader("👤 Xem chi tiết một học sinh")
        student_ids = features_df["student_id"].tolist()
        selected_id = st.selectbox("Chọn học sinh", student_ids)

        if selected_id:
            student = features_df[features_df["student_id"] == selected_id].iloc[0]

            col_s1, col_s2, col_s3 = st.columns(3)
            label = student[TARGET_COLUMN]
            emoji = get_label_emoji(label)
            col_s1.metric("🏷️ Phân loại", f"{emoji} {label}")
            col_s2.metric("📊 ĐTB chung", student["avg_score"])
            col_s3.metric("📅 Chuyên cần", f"{student['attendance_rate']}%")

            # Chi tiết features
            st.write("**Chi tiết các chỉ số:**")
            detail_data = {FEATURE_NAMES_VI.get(col, col): [student[col]] for col in FEATURE_COLUMNS if col in student.index}
            st.dataframe(pd.DataFrame(detail_data).T.rename(columns={0: "Giá trị"}), use_container_width=True)

            # Lý do phân loại
            st.write("**📝 Lý do phân loại (Rule Engine):**")
            score_avgs = [student[f"avg_score"]] * int(student.get("num_score_subjects", 8))  # simplified
            comment_stats = ["Đạt"] * max(0, int(student.get("num_comment_subjects", 3)) - int(student.get("comment_not_pass_count", 0)))
            comment_stats += ["Chưa đạt"] * int(student.get("comment_not_pass_count", 0))
            reason = get_classification_reason(
                [student["avg_score"], student["min_score"]] + [student["avg_score"]] * 6,
                comment_stats
            )
            st.write(reason)

# ============================================================
# TAB 4: HUẤN LUYỆN MÔ HÌNH
# ============================================================

with tab4:
    st.header("🤖 Huấn luyện mô hình Machine Learning")
    features_df = load_features_if_exists()

    if features_df is None:
        st.warning("⚠️ Chưa có dữ liệu. Hãy tạo dataset mẫu trước.")
    else:
        col_train1, col_train2 = st.columns([1, 2])

        with col_train1:
            st.subheader("⚙️ Cấu hình")
            st.write("**Decision Tree:**")
            st.code("max_depth=4, min_samples_split=10\nclass_weight='balanced'")
            st.write("**Random Forest:**")
            st.code("n_estimators=100, max_depth=6\nclass_weight='balanced'")
            st.write(f"**Dataset:** {len(features_df)} mẫu")
            st.write(f"**Features:** {len(FEATURE_COLUMNS)} cột")
            st.write(f"**Train/Test split:** 80/20, stratify=y")

            if st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True):
                with st.spinner("Đang huấn luyện Decision Tree và Random Forest..."):
                    try:
                        result = train_all(features_df)
                        st.session_state["train_result"] = result
                        st.success("✅ Huấn luyện thành công!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")

        with col_train2:
            metrics = load_metrics_if_exists()
            if metrics:
                st.subheader("📊 Kết quả đánh giá")

                dt_m = metrics.get("decision_tree", {})
                rf_m = metrics.get("random_forest", {})

                # Bảng so sánh
                comparison = compare_models(dt_m, rf_m)
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                # Biểu đồ so sánh
                st.plotly_chart(plot_metrics_comparison(dt_m, rf_m), use_container_width=True, key="metrics_tab4")

                # Confusion Matrix
                st.subheader("🔲 Confusion Matrix")
                cm_col1, cm_col2 = st.columns(2)
                with cm_col1:
                    if "confusion_matrix" in dt_m:
                        st.plotly_chart(
                            plot_confusion_matrix(dt_m["confusion_matrix"], dt_m.get("class_names", CLASS_LABELS), "Decision Tree"),
                            use_container_width=True,
                            key="cm_dt_tab4",
                        )
                        st.caption("Các ô trên đường chéo chính là dự đoán đúng; các ô ngoài đường chéo là trường hợp mô hình nhầm nhóm.")
                with cm_col2:
                    if "confusion_matrix" in rf_m:
                        st.plotly_chart(
                            plot_confusion_matrix(rf_m["confusion_matrix"], rf_m.get("class_names", CLASS_LABELS), "Random Forest"),
                            use_container_width=True,
                            key="cm_rf_tab4",
                        )
                        st.caption("Các ô trên đường chéo chính là dự đoán đúng; các ô ngoài đường chéo là trường hợp mô hình nhầm nhóm.")

                st.info(f"📁 Models lưu tại: `models/` | Metrics tại: `reports/metrics.json`")
            else:
                st.info("ℹ️ Chưa có kết quả. Nhấn **Huấn luyện mô hình** để bắt đầu.")

# ============================================================
# TAB 5: DỰ ĐOÁN MỘT HỌC SINH
# ============================================================

with tab5:
    st.header("👤 Dự đoán một học sinh")

    if not MODELS_DIR.joinpath("decision_tree.pkl").exists():
        st.warning("⚠️ Chưa có mô hình. Hãy huấn luyện trước ở tab **Huấn luyện mô hình**.")
    else:
        mode = st.radio("Chế độ nhập", ["📊 Nhập nhanh feature tổng hợp", "📝 Nhập theo môn học"], horizontal=True)

        if mode == "📊 Nhập nhanh feature tổng hợp":
            st.subheader("Nhập các chỉ số tổng hợp")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                avg_score = st.number_input("Điểm TB chung", 0.0, 10.0, 7.0, 0.1)
                min_score = st.number_input("Điểm thấp nhất", 0.0, 10.0, 5.0, 0.1)
                max_score = st.number_input("Điểm cao nhất", 0.0, 10.0, 9.0, 0.1)
                std_score = st.number_input("Độ lệch chuẩn", 0.0, 5.0, 1.0, 0.1)
                progress_delta = st.number_input("Mức tiến bộ", -10.0, 10.0, 0.5, 0.1)
            with col_b:
                count_ge_8 = st.number_input("Số môn ĐTB ≥ 8.0", 0, 8, 3)
                count_ge_6_5 = st.number_input("Số môn ĐTB ≥ 6.5", 0, 8, 6)
                count_ge_5 = st.number_input("Số môn ĐTB ≥ 5.0", 0, 8, 8)
                count_lt_3_5 = st.number_input("Số môn ĐTB < 3.5", 0, 8, 0)
                comment_fail = st.number_input("Số môn nhận xét Chưa đạt", 0, 3, 0)
            with col_c:
                attendance = st.number_input("Chuyên cần (%)", 0.0, 100.0, 85.0, 1.0)
                assignment = st.number_input("Hoàn thành BT (%)", 0.0, 100.0, 80.0, 1.0)
                participation = st.number_input("Điểm tham gia", 1.0, 10.0, 7.0, 0.1)
                behavior = st.number_input("Điểm hành vi", 1.0, 10.0, 7.0, 0.1)
                teacher_eval = st.number_input("Đánh giá GV", 1.0, 10.0, 7.0, 0.1)

            features_input = {
                "avg_score": avg_score, "min_score": min_score, "max_score": max_score,
                "std_score": std_score, "count_score_ge_8": count_ge_8,
                "count_score_ge_6_5": count_ge_6_5, "count_score_ge_5": count_ge_5,
                "count_score_lt_3_5": count_lt_3_5, "comment_not_pass_count": comment_fail,
                "attendance_rate": attendance, "assignment_completion_rate": assignment,
                "participation_score": participation, "behavior_score": behavior,
                "teacher_evaluation_score": teacher_eval, "progress_delta": progress_delta,
            }

            # Reconstruct score_averages cho Rule Engine (simplified)
            score_avgs_re = [avg_score] * 8
            comment_stats_re = ["Đạt"] * (3 - comment_fail) + ["Chưa đạt"] * comment_fail

            if st.button("🔮 Dự đoán", type="primary", use_container_width=True):
                try:
                    pred = predict_one(features_input, score_avgs_re, comment_stats_re)
                    st.markdown("---")

                    # Kết quả
                    st.subheader("🎯 Kết quả dự đoán")
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

                    with res_col1:
                        re_label = pred["rule_engine_result"] or "N/A"
                        st.metric("📐 Rule Engine", f"{get_label_emoji(re_label)} {re_label}")
                    with res_col2:
                        dt_label = pred["dt_prediction"]
                        st.metric("🌳 Decision Tree", f"{get_label_emoji(dt_label)} {dt_label}")
                    with res_col3:
                        rf_label = pred["rf_prediction"]
                        st.metric("🌲 Random Forest", f"{get_label_emoji(rf_label)} {rf_label}")
                    with res_col4:
                        final_label = pred["final_prediction"]
                        st.metric("🎯 Kết quả cuối cùng", f"{get_label_emoji(final_label)} {final_label}")

                    st.metric("📊 Độ tin cậy", f"{pred['confidence']}%")

                    # Probabilities
                    st.subheader("📊 Xác suất dự đoán (Random Forest)")
                    proba_df = pd.DataFrame([pred["class_probabilities"]])
                    st.dataframe(proba_df, use_container_width=True, hide_index=True)

                    # Lý do
                    st.subheader("📝 Giải thích")
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        st.write("**Lý do Rule Engine:**")
                        st.write(pred["rule_reason"])
                    with col_exp2:
                        st.write("**Phân tích ML:**")
                        st.markdown(generate_ml_explanation(features_input, final_label))

                    # Khuyến nghị
                    st.subheader("💡 Khuyến nghị hỗ trợ")
                    risk_level, risk_color = get_risk_level(final_label)
                    st.markdown(f"**Mức độ rủi ro:** <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
                    for rec in generate_recommendations(final_label, features_input):
                        st.write(rec)

                except Exception as e:
                    st.error(f"❌ Lỗi dự đoán: {e}")

        else:  # Nhập theo môn học
            st.subheader("Nhập điểm theo từng môn")
            st.info("Nhập điểm cho 8 môn đánh giá bằng điểm số và 3 môn đánh giá bằng nhận xét.")

            score_data = {}
            cols = st.columns(4)
            for i, subject in enumerate(SCORE_SUBJECTS):
                with cols[i % 4]:
                    with st.expander(f"📘 {subject}", expanded=False):
                        reg = st.text_input(f"Điểm TX ({subject})", "7.0;7.5;8.0", key=f"reg_{i}")
                        mid = st.number_input(f"Điểm GK", 0.0, 10.0, 7.0, 0.1, key=f"mid_{i}")
                        fin = st.number_input(f"Điểm CK", 0.0, 10.0, 7.5, 0.1, key=f"fin_{i}")
                        score_data[subject] = {"regular": reg, "midterm": mid, "final": fin}

            comment_data = {}
            st.write("**Môn đánh giá bằng nhận xét:**")
            comment_cols = st.columns(3)
            for i, subject in enumerate(COMMENT_SUBJECTS):
                with comment_cols[i]:
                    comment_data[subject] = st.selectbox(f"{subject}", ["Đạt", "Chưa đạt"], key=f"com_{i}")

            # Behavior
            st.write("**Thông tin hành vi:**")
            beh_cols = st.columns(5)
            attendance_input = beh_cols[0].number_input("Chuyên cần (%)", 0.0, 100.0, 85.0, key="beh_att")
            assign_input = beh_cols[1].number_input("Hoàn thành BT (%)", 0.0, 100.0, 80.0, key="beh_ass")
            part_input = beh_cols[2].number_input("Tham gia", 1.0, 10.0, 7.0, key="beh_part")
            beh_input = beh_cols[3].number_input("Hành vi", 1.0, 10.0, 7.0, key="beh_beh")
            tea_input = beh_cols[4].number_input("Đánh giá GV", 1.0, 10.0, 7.0, key="beh_tea")

            if st.button("🔮 Dự đoán (theo môn)", type="primary", use_container_width=True):
                try:
                    # Calculate DTB_mhk for each subject
                    dtb_list = []
                    for subject, data in score_data.items():
                        reg_scores = [float(s.strip()) for s in data["regular"].split(";") if s.strip()]
                        dtb = calculate_dtb_mhk(reg_scores, data["midterm"], data["final"])
                        dtb_list.append(dtb)

                    comment_list = list(comment_data.values())

                    # Build features
                    features_input = {
                        "avg_score": round(np.mean(dtb_list), 1),
                        "min_score": round(min(dtb_list), 1),
                        "max_score": round(max(dtb_list), 1),
                        "std_score": round(float(np.std(dtb_list)), 2),
                        "count_score_ge_8": sum(1 for s in dtb_list if s >= 8.0),
                        "count_score_ge_6_5": sum(1 for s in dtb_list if s >= 6.5),
                        "count_score_ge_5": sum(1 for s in dtb_list if s >= 5.0),
                        "count_score_lt_3_5": sum(1 for s in dtb_list if s < 3.5),
                        "comment_not_pass_count": sum(1 for s in comment_list if s == "Chưa đạt"),
                        "attendance_rate": attendance_input,
                        "assignment_completion_rate": assign_input,
                        "participation_score": part_input,
                        "behavior_score": beh_input,
                        "teacher_evaluation_score": tea_input,
                        "progress_delta": 0.0,
                    }

                    pred = predict_one(features_input, dtb_list, comment_list)
                    st.markdown("---")

                    # DTB table
                    st.subheader("📊 Điểm trung bình các môn")
                    dtb_df = pd.DataFrame({"Môn": SCORE_SUBJECTS, "DTB_mhk": dtb_list})
                    st.dataframe(dtb_df, use_container_width=True, hide_index=True)

                    # Results
                    st.subheader("🎯 Kết quả dự đoán")
                    res_cols = st.columns(4)
                    re_label = pred["rule_engine_result"] or "N/A"
                    res_cols[0].metric("📐 Rule Engine", f"{get_label_emoji(re_label)} {re_label}")
                    res_cols[1].metric("🌳 Decision Tree", f"{get_label_emoji(pred['dt_prediction'])} {pred['dt_prediction']}")
                    res_cols[2].metric("🌲 Random Forest", f"{get_label_emoji(pred['rf_prediction'])} {pred['rf_prediction']}")
                    res_cols[3].metric("🎯 Cuối cùng", f"{get_label_emoji(pred['final_prediction'])} {pred['final_prediction']}")

                    st.metric("📊 Độ tin cậy", f"{pred['confidence']}%")

                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        st.write("**📝 Lý do Rule Engine:**")
                        st.write(pred["rule_reason"])
                    with col_e2:
                        st.write("**💡 Khuyến nghị:**")
                        for rec in generate_recommendations(pred["final_prediction"], features_input):
                            st.write(rec)

                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")

# ============================================================
# TAB 6: CẢNH BÁO SỚM & DỰ ĐOÁN HÀNG LOẠT
# ============================================================

with tab6:
    st.header("🚨 Cảnh báo sớm & Dự đoán hàng loạt (Early Warning System)")

    if not MODELS_DIR.joinpath("decision_tree.pkl").exists():
        st.warning("⚠️ Chưa có mô hình. Hãy huấn luyện trước ở tab **Huấn luyện mô hình**.")
    else:
        # Lựa chọn nguồn dữ liệu
        st.subheader("🔍 Lựa chọn nguồn dữ liệu")
        source_mode = st.radio(
            "Chọn nguồn dữ liệu để chạy phân tích:",
            ["📁 Sử dụng dữ liệu hiện tại trong hệ thống (student_features.csv)", "📤 Tải lên file dữ liệu mới (CSV hoặc Excel)"],
            horizontal=True
        )

        batch_df = None

        if "Sử dụng dữ liệu hiện tại" in source_mode:
            features_df = load_features_if_exists()
            if features_df is None:
                st.warning("⚠️ Hệ thống chưa có dữ liệu mẫu. Vui lòng sang tab **Dữ liệu** để tạo dataset trước.")
            else:
                batch_df = features_df.copy()
                st.info(f"📋 Tìm thấy dữ liệu sẵn có gồm **{len(batch_df)}** học sinh. Sẵn sàng để phân tích cảnh báo sớm!")

                # Nút bấm chạy EWS trên dữ liệu có sẵn
                if st.button("🔮 Chạy phân tích Cảnh báo sớm (EWS)", type="primary", use_container_width=True):
                    with st.spinner("Đang chạy mô hình AI và tính toán rủi ro..."):
                        # Chạy dự đoán batch (đã tích hợp EWS trong predict_batch)
                        result_df = predict_batch(batch_df)
                        st.session_state["ews_result_df"] = result_df
                        st.success(f"✅ Đã chạy phân tích thành công cho {len(result_df)} học sinh!")

        else:  # Tải lên file mới
            st.info(f"File CSV/Excel tải lên phải chứa đầy đủ 15 cột đặc trưng: {', '.join(FEATURE_COLUMNS[:4])}...")

            # Tải mẫu
            features_df = load_features_if_exists()
            if features_df is not None:
                sample = features_df[FEATURE_COLUMNS].head(5)
                col_sdown1, col_sdown2 = st.columns(2)
                with col_sdown1:
                    st.download_button(
                        "📥 Tải file CSV mẫu",
                        sample.to_csv(index=False, encoding="utf-8-sig"),
                        "sample_batch_input.csv", "text/csv",
                        use_container_width=True,
                    )
                with col_sdown2:
                    try:
                        excel_sample = to_excel(sample)
                        st.download_button(
                            "📥 Tải file Excel mẫu (.xlsx)",
                            excel_sample, "sample_batch_input.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"Lỗi xuất Excel mẫu: {e}")

            batch_file = st.file_uploader("Upload file CSV hoặc Excel", type=["csv", "xlsx"], key="batch_upload")

            if batch_file is not None:
                try:
                    if batch_file.name.endswith(".xlsx"):
                        batch_df = pd.read_excel(batch_file)
                    else:
                        batch_df = pd.read_csv(batch_file, encoding="utf-8-sig")
                    st.write(f"📊 Đã tải **{len(batch_df)}** dòng dữ liệu từ file.")

                    # Validate
                    missing_cols = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
                    if missing_cols:
                        st.error(f"❌ File thiếu cột đặc trưng: {missing_cols}")
                    else:
                        if st.button("🔮 Chạy dự đoán & Phân tích rủi ro EWS", type="primary", use_container_width=True):
                            with st.spinner("Đang chạy mô hình AI và phân tích rủi ro..."):
                                result_df = predict_batch(batch_df)
                                st.session_state["ews_result_df"] = result_df
                                st.success(f"✅ Dự đoán thành công cho {len(result_df)} học sinh!")
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")

        # HIỂN THỊ KẾT QUẢ TỪ SESSION STATE (TRÁNH BỊ MẤT KHI RERUN)
        if "ews_result_df" in st.session_state:
            result_df = st.session_state["ews_result_df"]

            st.markdown("---")
            st.subheader("🚨 BẢNG ĐIỀU KHIỂN CẢNH BÁO SỚM HỌC ĐƯỜNG (EWS DASHBOARD)")

            # --- Hướng dẫn cơ chế EWS ---
            with st.expander("📚 HƯỚNG DẪN: Cơ chế tính Điểm rủi ro & Phân cấp Nguy cơ (EWS)", expanded=False):
                st.markdown(r"""
                ### 📐 Cơ chế toán học & Nghiệp vụ Sư phạm của EWS

                Điểm rủi ro (**Risk Score** từ 0 đến 100) được tính toán tự động bằng cách kết hợp toán học các yếu tố học thuật và hành vi, bám sát các điều kiện loại trừ quy định bởi **Thông tư 22/2021/TT-BGDĐT** của Bộ Giáo dục & Đào tạo Việt Nam:

                $$\text{Điểm rủi ro} = 0.30 \times R_{\text{chuyên cần}} + 0.25 \times R_{\text{điểm liệt}} + 0.15 \times R_{\text{nhận xét}} + 0.15 \times R_{\text{GPA}} + 0.10 \times R_{\text{bài tập}} + 0.05 \times R_{\text{tiến bộ}}$$

                #### 🔴 Cơ chế Tự động kích hoạt "Nguy cơ Cao" (🔴 Cao)
                Hệ thống áp dụng các **luật loại trừ bắt buộc của Bộ GD&ĐT**. Học sinh sẽ lập tức bị xếp vào nhóm **🔴 Nguy cơ Cao (Cần can thiệp khẩn cấp)** bất kể điểm số tổng hợp là bao nhiêu nếu vi phạm một trong các điều sau:
                1.  **Chuyên cần dưới 75%**: Nguy cơ lưu ban trực tiếp do nghỉ học quá 45 buổi học quy định bởi *Điều 12 Thông tư 22*.
                2.  **Có điểm liệt môn học dưới 3.5**: Bắt buộc xếp loại *Chưa đạt* học kỳ/cả năm.
                3.  **Có từ 2 môn nhận xét "Chưa đạt" trở lên**: Bắt buộc xếp loại *Chưa đạt*.
                4.  **GPA yếu kém dưới 5.0 hoặc số môn học đạt từ 5.0 trở lên ít hơn 6 môn**: Không đủ tiêu chuẩn xếp loại đạt.

                #### 📊 Phân cấp mức độ rủi ro hệ thống
                *   🔴 **Nguy cơ Cao (Critical)**: Điểm rủi ro $\ge 75$ hoặc dính bất kỳ vi phạm tự động nào ở trên. *Hành động: Kích hoạt kế hoạch can thiệp học đường khẩn cấp.*
                *   🟠 **Cần hỗ trợ (Warning)**: Điểm rủi ro từ $45$ đến $74$. *Hành động: Giáo viên chủ nhiệm gặp gỡ, trao đổi sát sao và lên phương án kèm cặp.*
                *   🟡 **Cần theo dõi (Monitor)**: Điểm rủi ro từ $25$ đến $44$. *Hành động: Theo dõi tiến trình làm bài tập và nhắc nhở trên lớp.*
                *   🟢 **An toàn (Safe)**: Điểm rủi ro dưới $25$. *Hành động: Khuyến khích tiếp tục phát huy và hỗ trợ các bạn cùng lớp.*
                """)


            # --- KPI Cards ---
            total_students = len(result_df)
            num_critical = len(result_df[result_df["risk_level"] == "🔴 Cao"])
            num_warning = len(result_df[result_df["risk_level"] == "🟠 Cảnh báo"])
            num_monitor = len(result_df[result_df["risk_level"] == "🟡 Theo dõi"])
            num_safe = len(result_df[result_df["risk_level"] == "🟢 An toàn"])

            # Render KPI
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                st.markdown(f"""
                <div style="background-color: #FEE2E2; border-left: 5px solid #EF4444; padding: 15px; border-radius: 8px;">
                    <h5 style="color: #991B1B; margin: 0; font-weight: 700; font-size: 0.95rem;">🔴 Nguy cơ Cao</h5>
                    <h2 style="color: #EF4444; margin: 5px 0 0 0; font-weight: 800; font-size: 1.8rem;">{num_critical} <span style="font-size: 0.85rem; font-weight: 400; color: #7F1D1D;">({num_critical/total_students*100:.1f}%)</span></h2>
                    <p style="color: #7F1D1D; margin: 5px 0 0 0; font-size: 0.75rem; font-weight: 500;">Cần can thiệp ngay lập tức</p>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[1]:
                st.markdown(f"""
                <div style="background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 8px;">
                    <h5 style="color: #92400E; margin: 0; font-weight: 700; font-size: 0.95rem;">🟠 Cần hỗ trợ</h5>
                    <h2 style="color: #F59E0B; margin: 5px 0 0 0; font-weight: 800; font-size: 1.8rem;">{num_warning} <span style="font-size: 0.85rem; font-weight: 400; color: #78350F;">({num_warning/total_students*100:.1f}%)</span></h2>
                    <p style="color: #78350F; margin: 5px 0 0 0; font-size: 0.75rem; font-weight: 500;">Cần giáo viên kèm cặp</p>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[2]:
                st.markdown(f"""
                <div style="background-color: #DBEAFE; border-left: 5px solid #3B82F6; padding: 15px; border-radius: 8px;">
                    <h5 style="color: #1E40AF; margin: 0; font-weight: 700; font-size: 0.95rem;">🟡 Cần theo dõi</h5>
                    <h2 style="color: #3B82F6; margin: 5px 0 0 0; font-weight: 800; font-size: 1.8rem;">{num_monitor} <span style="font-size: 0.85rem; font-weight: 400; color: #1E3A8A;">({num_monitor/total_students*100:.1f}%)</span></h2>
                    <p style="color: #1E3A8A; margin: 5px 0 0 0; font-size: 0.75rem; font-weight: 500;">Có dấu hiệu sa sút nhẹ</p>
                </div>
                """, unsafe_allow_html=True)
            with kpi_cols[3]:
                st.markdown(f"""
                <div style="background-color: #D1FAE5; border-left: 5px solid #10B981; padding: 15px; border-radius: 8px;">
                    <h5 style="color: #065F46; margin: 0; font-weight: 700; font-size: 0.95rem;">🟢 An toàn</h5>
                    <h2 style="color: #10B981; margin: 5px 0 0 0; font-weight: 800; font-size: 1.8rem;">{num_safe} <span style="font-size: 0.85rem; font-weight: 400; color: #064E3B;">({num_safe/total_students*100:.1f}%)</span></h2>
                    <p style="color: #064E3B; margin: 5px 0 0 0; font-size: 0.75rem; font-weight: 500;">Học tập tích cực, ổn định</p>
                </div>
                """, unsafe_allow_html=True)

            # --- Visualizations ---
            st.write("")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(plot_risk_level_distribution(result_df), use_container_width=True)
            with chart_col2:
                st.plotly_chart(plot_risk_by_class(result_df), use_container_width=True)

            # --- Emergency Action Report Export ---
            st.subheader("📥 Xuất báo cáo danh sách can thiệp khẩn cấp")
            emergency_df = result_df[result_df["risk_level"].isin(["🔴 Cao", "🟠 Cảnh báo"])].copy()

            col_bdown1, col_bdown2 = st.columns(2)
            with col_bdown1:
                st.download_button(
                    "📥 Tải danh sách can thiệp (CSV)",
                    emergency_df.to_csv(index=False, encoding="utf-8-sig"),
                    "danh_sach_can_thiep_khan_cap.csv", "text/csv",
                    use_container_width=True,
                    help="Chỉ xuất danh sách những học sinh có rủi ro Cao hoặc Cảnh báo để can thiệp kịp thời."
                )
            with col_bdown2:
                try:
                    excel_emergency = to_excel(emergency_df)
                    st.download_button(
                        "📥 Tải danh sách can thiệp (Excel)",
                        excel_emergency, "danh_sach_can_thiep_khan_cap.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Tải báo cáo can thiệp khẩn cấp dưới dạng Excel (.xlsx)."
                    )
                except Exception as e:
                    st.error(f"Lỗi xuất Excel can thiệp: {e}")

            # --- Priority Alerts List ---
            st.subheader("🔴 Danh sách cần can thiệp học thuật ưu tiên (Nguy cơ Cao & Cần hỗ trợ)")
            if not emergency_df.empty:
                disp_emergency = emergency_df.copy()
                cols_to_show = ["student_id", "class_name", "risk_level", "risk_score", "avg_score", "min_score", "attendance_rate", "assignment_completion_rate"]
                
                # Chỉ lấy các cột thực tế tồn tại trong dataframe để hiển thị tránh KeyError
                available_show_cols = [col for col in cols_to_show if col in disp_emergency.columns]
                disp_emergency = disp_emergency[available_show_cols]
                
                rename_map = {
                    "student_id": "Mã học sinh",
                    "class_name": "Lớp",
                    "risk_level": "Mức độ rủi ro",
                    "risk_score": "Điểm rủi ro",
                    "avg_score": "ĐTB chung",
                    "min_score": "Điểm liệt thấp nhất",
                    "attendance_rate": "Chuyên cần (%)",
                    "assignment_completion_rate": "Hoàn thành BT (%)"
                }
                # Lọc lại bản đồ rename chỉ với cột thực tế có mặt
                rename_map = {k: v for k, v in rename_map.items() if k in available_show_cols}
                disp_emergency = disp_emergency.rename(columns=rename_map)
                
                st.dataframe(disp_emergency, use_container_width=True, hide_index=True)

                # --- Tra cứu chi tiết ---
                st.write("")
                with st.expander("🔍 Tra cứu chi tiết cờ cảnh báo & Kế hoạch can thiệp từng học sinh", expanded=False):
                    if "student_id" in emergency_df.columns:
                        student_ids_list = emergency_df["student_id"].tolist()
                        selected_sid = st.selectbox("Chọn Mã học sinh để tra cứu chi tiết:", student_ids_list)
                        stud_data = emergency_df[emergency_df["student_id"] == selected_sid].iloc[0]
                    else:
                        st.info("ℹ️ Không tìm thấy cột 'student_id' trong file. Tra cứu theo dòng dữ liệu:")
                        student_ids_list = [f"Học sinh số {i+1} (Dòng {idx+1})" for idx, i in enumerate(emergency_df.index)]
                        selected_sid_label = st.selectbox("Chọn học sinh để tra cứu:", student_ids_list)
                        idx_sel = student_ids_list.index(selected_sid_label)
                        stud_data = emergency_df.iloc[idx_sel]
                        selected_sid = stud_data.get("student_id", f"Học sinh (Dòng {idx_sel+1})")

                    if selected_sid:
                        class_lbl = stud_data.get("class_name", "Chung (Không chia lớp)")
                        st.markdown(f"#### 👤 Học sinh: `{selected_sid}` | Lớp: `{class_lbl}`")


                        col_stud1, col_stud2 = st.columns(2)
                        col_stud1.metric("🔴 Điểm rủi ro học đường", f"{stud_data['risk_score']} / 100")
                        col_stud2.metric("🎯 AI Dự báo phân loại", stud_data["final_prediction"])

                        st.markdown("##### 🚨 Các cờ cảnh báo rủi ro cụ thể:")
                        flags = stud_data["warning_flags_str"].split("; ")
                        for f in flags:
                            if f:
                                st.write(f"- {f}")

                        st.markdown("##### 🤝 Khuyến nghị can thiệp cụ thể dành cho Giáo viên:")
                        recs = stud_data["recommendations_str"].split("; ")
                        for r in recs:
                            if r:
                                st.write(f"- {r}")
            else:
                st.success("🎉 Tuyệt vời! Không có học sinh nào nằm trong diện rủi ro Cao hay Cần hỗ trợ học tập.")

            # --- Full Dataset Predictions ---
            st.subheader("📋 Bảng kết quả dự đoán & Chỉ số rủi ro đầy đủ")
            full_display_cols = ["student_id", "class_name", "final_prediction", "risk_level", "risk_score", "attendance_rate", "avg_score"]
            available_full_cols = [col for col in full_display_cols if col in result_df.columns]
            st.dataframe(result_df[available_full_cols], use_container_width=True, hide_index=True)


            # Tải toàn bộ
            col_fdown1, col_fdown2 = st.columns(2)
            with col_fdown1:
                st.download_button(
                    "📥 Tải toàn bộ kết quả dự đoán (CSV)",
                    result_df.to_csv(index=False, encoding="utf-8-sig"),
                    "toan_bo_ket_qua_du_doan.csv", "text/csv",
                    use_container_width=True
                )
            with col_fdown2:
                try:
                    excel_full = to_excel(result_df)
                    st.download_button(
                        "📥 Tải toàn bộ kết quả dự đoán (Excel)",
                        excel_full, "toan_bo_ket_qua_du_doan.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Lỗi xuất Excel kết quả dự đoán: {e}")



# ============================================================
# TAB 7: GIẢI THÍCH MÔ HÌNH
# ============================================================

with tab7:
    st.header("🔍 Giải thích mô hình")
    metrics = load_metrics_if_exists()

    if metrics is None:
        st.warning("⚠️ Chưa có mô hình. Hãy huấn luyện trước.")
    else:
        # Feature Importance
        st.subheader("📊 Feature Importance — Random Forest")
        rf_m = metrics.get("random_forest", {})
        if "feature_importances" in rf_m:
            fi_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Tên tiếng Việt": [FEATURE_NAMES_VI.get(f, f) for f in FEATURE_COLUMNS],
                "Importance": rf_m["feature_importances"],
            }).sort_values("Importance", ascending=False)

            st.plotly_chart(
                plot_feature_importance(rf_m["feature_importances"], FEATURE_COLUMNS),
                use_container_width=True,
                key="fi_tab7",
            )
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

            st.write(explain_feature_importance(rf_m["feature_importances"], FEATURE_COLUMNS))

        # Decision Tree Summary
        st.markdown("---")
        st.subheader("🌳 Tóm tắt Decision Tree")
        try:
            dt_model, _, _ = load_models()
            if dt_model is not None:
                st.markdown(explain_decision_tree_summary(dt_model))
        except Exception:
            st.info("Không thể load Decision Tree model.")

        # Rule Engine Logic
        st.markdown("---")
        st.subheader("📐 Logic Rule Engine")
        st.markdown(explain_rule_engine_logic())

        # Số hóa chỉ số định tính trong Giáo dục
        st.markdown("---")
        st.subheader("📐 Số hóa & Quy đổi các chỉ số định tính")
        st.markdown(explain_qualitative_metrics())


        # Confusion Matrix
        st.markdown("---")
        st.subheader("🔲 Confusion Matrix")
        cm_cols = st.columns(2)
        dt_m = metrics.get("decision_tree", {})
        with cm_cols[0]:
            if "confusion_matrix" in dt_m:
                st.plotly_chart(
                    plot_confusion_matrix(dt_m["confusion_matrix"], dt_m.get("class_names", CLASS_LABELS), "Decision Tree"),
                    use_container_width=True,
                    key="cm_dt_tab7",
                )
                st.caption("Các ô trên đường chéo chính là dự đoán đúng; các ô ngoài đường chéo là trường hợp mô hình nhầm nhóm.")
        with cm_cols[1]:
            if "confusion_matrix" in rf_m:
                st.plotly_chart(
                    plot_confusion_matrix(rf_m["confusion_matrix"], rf_m.get("class_names", CLASS_LABELS), "Random Forest"),
                    use_container_width=True,
                    key="cm_rf_tab7",
                )
                st.caption("Các ô trên đường chéo chính là dự đoán đúng; các ô ngoài đường chéo là trường hợp mô hình nhầm nhóm.")

        # Tại sao không Deep Learning
        st.markdown("---")
        st.subheader("❓ Vì sao không dùng Deep Learning?")
        st.write("""
        - **Dữ liệu dạng bảng** (tabular data) — Decision Tree và Random Forest hoạt động tốt.
        - **Dataset nhỏ** (300-500 mẫu) — Deep Learning cần hàng nghìn đến hàng triệu mẫu.
        - **Tính giải thích** — Decision Tree dễ giải thích hơn Neural Network.
        - **Phù hợp bối cảnh demo** — không cần GPU hay thời gian huấn luyện lâu.
        - **Overfitting** — Deep Learning dễ overfit trên dữ liệu nhỏ.
        """)

# ============================================================
# TAB 8: PHƯƠNG PHÁP & GIỚI HẠN
# ============================================================

with tab8:
    st.header("📚 Phương pháp & Giới hạn")

    st.subheader("🎯 Mục tiêu dự án")
    st.write("""
    Hệ thống **Smart Student Classification System** là demo học thuật phục vụ báo cáo cuối kỳ.
    Mục tiêu là minh họa cách kết hợp **Rule Engine** (theo quy định Bộ GD&ĐT) và
    **Machine Learning có giám sát** (Decision Tree + Random Forest) để phân loại kết quả học tập.
    """)

    st.subheader("📊 Phương pháp")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### Rule Engine (BGDDT)")
        st.write("""
        - **Không phải Machine Learning** — đây là bộ luật cố định.
        - Tính DTB_mhk, DTB_mcn.
        - Phân loại: **Tốt / Khá / Đạt / Chưa đạt**.
        - Tạo nhãn `learning_result_label` cho ML.
        """)
    with col_m2:
        st.markdown("#### Supervised Machine Learning")
        st.write("""
        - **Học từ dữ liệu đã có nhãn** (do Rule Engine tạo).
        - Bài toán **phân loại nhiều lớp** (multi-class classification).
        - **Decision Tree**: dễ giải thích, phù hợp demo.
        - **Random Forest**: ổn định, cung cấp feature importance.
        """)

    st.subheader("📐 Kiến trúc hệ thống")
    st.code("""
    Raw Data → Validation → Calculate DTB_mhk/mcn
        → Feature Engineering → Rule Engine (labels)
        → Supervised ML Training → Evaluation
        → Prediction + Explanation → Streamlit UI
    """)

    st.subheader("⚠️ Giới hạn")
    st.warning("""
    **Giới hạn cần lưu ý:**
    - Dataset là **dữ liệu mô phỏng**, không phải dữ liệu thực.
    - Hệ thống **không thay thế** đánh giá chuyên môn của giáo viên.
    - ML **học từ nhãn do Rule Engine tạo** — không phải nhãn từ giáo viên thật.
    - Không sử dụng **Deep Learning** hay **Neural Network**.
    - Kết quả chỉ mang tính **tham khảo hỗ trợ**, không dùng làm quyết định chính thức.
    """)

    st.subheader("🔒 Đạo đức & bảo mật")
    st.write("""
    - Toàn bộ dữ liệu là **hoàn toàn giả lập**.
    - Nếu triển khai thực tế, cần tuân thủ quy định bảo vệ dữ liệu cá nhân.
    - AI chỉ là **công cụ hỗ trợ**, không thay thế quyết định giáo dục.
    - Cần có sự đồng ý của phụ huynh/giám hộ khi dùng dữ liệu thật.
    """)

    st.subheader("📖 Kiến thức cần nắm")
    with st.expander("Học máy có giám sát (Supervised ML) là gì?"):
        st.write("""
        Supervised ML là phương pháp học từ dữ liệu đã có nhãn (label).
        - **Input X**: các đặc trưng (features) của mỗi mẫu.
        - **Output y**: nhãn đúng đã biết trước.
        - Mô hình học mối quan hệ X → y từ training data.
        - Sau đó dự đoán nhãn cho dữ liệu mới chưa có nhãn.
        """)
    with st.expander("Decision Tree hoạt động như thế nào?"):
        st.write("""
        Decision Tree chia dữ liệu bằng chuỗi câu hỏi dạng "Feature X ≤ ngưỡng?".
        - Mỗi nút (node) là một câu hỏi.
        - Mỗi nhánh là câu trả lời (Có/Không).
        - Mỗi lá (leaf) là một nhãn phân loại.
        - Ưu điểm: dễ hiểu, dễ visualize.
        - Nhược điểm: dễ overfit nếu cây quá sâu.
        """)
    with st.expander("Random Forest hoạt động như thế nào?"):
        st.write("""
        Random Forest = tập hợp nhiều Decision Tree.
        - Mỗi cây được train trên một phần dữ liệu ngẫu nhiên (bootstrap).
        - Mỗi cây chỉ dùng một tập con features ngẫu nhiên.
        - Kết quả cuối cùng = bỏ phiếu đa số (majority voting).
        - Ưu điểm: ổn định, ít overfit, cung cấp feature importance.
        - Nhược điểm: khó giải thích hơn 1 cây đơn.
        """)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Thông tin hệ thống")
    st.markdown("---")
    st.markdown("**Phiên bản:** v2.0")
    st.markdown("**Năm học:** 2025-2026")
    st.markdown("**Nhãn phân loại:**")
    st.markdown("🟢 Tốt | 🔵 Khá | 🟡 Đạt | 🔴 Chưa đạt")
    st.markdown("---")
    st.markdown("**Mô hình ML:**")
    st.markdown("🌳 Decision Tree | 🌲 Random Forest")
    st.markdown("---")

    # Status indicators
    features_exists = STUDENT_FEATURES_FILE.exists()
    models_exist = MODELS_DIR.joinpath("decision_tree.pkl").exists()

    st.markdown(f"📁 Dataset: {'✅' if features_exists else '❌'}")
    st.markdown(f"🤖 Models: {'✅' if models_exist else '❌'}")

    if features_exists:
        df = pd.read_csv(STUDENT_FEATURES_FILE, encoding="utf-8-sig")
        st.markdown(f"👥 Học sinh: **{len(df)}**")

    st.markdown("---")
    st.caption("Demo học thuật — Báo cáo cuối kỳ")
