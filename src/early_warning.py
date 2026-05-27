"""
early_warning.py — Hệ thống Cảnh báo sớm (Early Warning System - EWS).

Cung cấp các hàm để tính toán điểm rủi ro (risk_score), mức độ rủi ro (risk_level),
các cờ cảnh báo rủi ro (warning_flags) và khuyến nghị can thiệp học đường.
Công thức bám sát quy chế loại trừ của Thông tư 22/2021/TT-BGDĐT.
"""

import math
from typing import Union, Dict, List
import pandas as pd
import numpy as np



def calculate_student_risk(row: Union[dict, pd.Series]) -> dict:
    """
    Tính toán chi tiết các chỉ số rủi ro cho một học sinh.

    Args:
        row: Dictionary hoặc Series chứa các đặc trưng học sinh.

    Returns:
        Dict gồm:
        - risk_score: Điểm rủi ro (0-100)
        - risk_level: Mức độ rủi ro ("🔴 Cao", "🟠 Cảnh báo", "🟡 Theo dõi", "🟢 An toàn")
        - warning_flags: Danh sách chuỗi cảnh báo cụ thể
        - recommendations: Danh sách khuyến nghị hành động của GV
    """
    # Lấy các tham số hành vi và điểm số
    attendance = float(row.get("attendance_rate", 100.0))
    min_score = float(row.get("min_score", 10.0))
    comment_fail = int(row.get("comment_not_pass_count", 0))
    avg_score = float(row.get("avg_score", 10.0))
    count_ge_5 = int(row.get("count_score_ge_5", 8))
    assignment = float(row.get("assignment_completion_rate", 100.0))
    progress = float(row.get("progress_delta", 0.0))
    behavior = float(row.get("behavior_score", 10.0))

    # Xử lý các giá trị NaN tiềm ẩn
    if math.isnan(attendance): attendance = 100.0
    if math.isnan(min_score): min_score = 10.0
    if math.isnan(avg_score): avg_score = 10.0
    if math.isnan(assignment): assignment = 100.0
    if math.isnan(progress): progress = 0.0
    if math.isnan(behavior): behavior = 10.0

    warning_flags = []
    recommendations = []

    # ==========================================
    # 1. TÍNH TOÁN CÁC CẤU PHẦN RỦI RO (0-100)
    # ==========================================

    # a. Chuyên cần (30% weight) - Điều 12 TT22: vắng quá 45 buổi (attendance < 75%)
    is_critical_attendance = attendance < 75.0
    if is_critical_attendance:
        r_attendance = 100.0
        warning_flags.append(f"🚨 CHUYÊN CẦN NGUY CẤP ({attendance}%): Vắng quá 45 buổi học (Vi phạm Điều 12 Thông tư 22, nguy cơ lưu ban thẳng).")
        recommendations.append("🤝 Tổ chức gặp trực tiếp phụ huynh, yêu cầu cam kết chuyên cần và phối hợp với gia đình kèm cặp đi học.")
    elif attendance < 85.0:
        r_attendance = 60.0 + (85.0 - attendance) * 4.0
        warning_flags.append(f"⚠️ Chuyên cần thấp ({attendance}%): Nguy cơ mất gốc kiến thức và vi phạm quy chế nghỉ học.")
        recommendations.append("📞 Gọi điện trao đổi ngay với phụ huynh tìm hiểu nguyên nhân vắng học thường xuyên.")
    else:
        r_attendance = max(0.0, (100.0 - attendance) * 4.0)

    # b. Điểm liệt (25% weight) - Điểm thành phần môn dưới 3.5
    is_critical_score = min_score < 3.5
    if is_critical_score:
        r_min_score = 100.0
        warning_flags.append(f"🚨 ĐIỂM LIỆT MÔN HỌC ({min_score}): Có môn học điểm trung bình dưới 3.5 (bắt buộc xếp loại Chưa đạt).")
        recommendations.append("📘 Lên danh sách phụ đạo học tập buổi chiều ngay lập tức đối với môn bị điểm liệt.")
    elif min_score < 5.0:
        r_min_score = 50.0 + (5.0 - min_score) * 33.3
        warning_flags.append(f"⚠️ Có môn dưới trung bình ({min_score}): Cần cải thiện nhanh để tránh bị kéo xuống điểm liệt.")
        recommendations.append("👨‍🏫 Yêu cầu giáo viên bộ môn có bài kiểm tra phụ hoặc hướng dẫn thêm các dạng bài cơ bản.")
    else:
        r_min_score = 0.0

    # c. Nhận xét (15% weight) - Môn nhận xét Chưa đạt >= 2
    is_critical_comments = comment_fail >= 2
    if is_critical_comments:
        r_comments = 100.0
        warning_flags.append(f"🚨 MÔN NHẬN XÉT YẾU ({comment_fail} môn): Có từ 2 môn nhận xét Chưa đạt trở lên (bắt buộc xếp loại Chưa đạt).")
        recommendations.append("🎨 Động viên học sinh hoàn thành các sản phẩm thực hành hoặc kiểm tra bổ sung môn nhận xét.")
    elif comment_fail == 1:
        r_comments = 50.0
        warning_flags.append("⚠️ Có 1 môn nhận xét Chưa đạt: Chạm ngưỡng giới hạn xếp loại đạt.")
        recommendations.append("💬 Gặp giáo viên bộ môn nhận xét (Thể dục/Nghệ thuật...) để xin kiểm tra bù phục hồi điểm.")
    else:
        r_comments = 0.0

    # d. ĐTB & Số môn đạt (15% weight) - GPA < 5.0 hoặc số môn học ĐTB >= 5.0 dưới 6 môn
    is_critical_gpa = avg_score < 5.0 or count_ge_5 < 6
    if is_critical_gpa:
        r_gpa = 100.0
        if avg_score < 5.0:
            warning_flags.append(f"🚨 GPA YẾU KÉM ({avg_score}): Điểm trung bình chung học kỳ dưới 5.0.")
        if count_ge_5 < 6:
            warning_flags.append(f"🚨 SỐ MÔN TRÊN TRUNG BÌNH THẤP ({count_ge_5}/8 môn): Có ít hơn 6 môn đạt điểm số từ 5.0 trở lên.")
        recommendations.append("📅 Lập thời gian biểu kèm cặp học tập 1-1 (bạn học tốt hỗ trợ bạn yếu hoặc giáo viên hỗ trợ).")
    elif avg_score < 6.5:
        r_gpa = 50.0 + (6.5 - avg_score) * 33.3
        warning_flags.append(f"⚠️ Học lực trung bình yếu (GPA {avg_score}): Chưa đạt tiêu chuẩn xếp loại khá trở lên.")
        recommendations.append("📖 Khuyến khích học sinh tập trung ôn thi học kỳ củng cố GPA.")
    else:
        r_gpa = 0.0

    # e. Bài tập (10% weight) - Hoàn thành bài tập về nhà thấp
    if assignment < 60.0:
        r_assignment = 100.0
        warning_flags.append(f"⚠️ LƯỜI HỌC/THIẾU BÀI TẬP ({assignment}%): Tỷ lệ nộp bài tập cực kỳ thấp, học sinh bỏ bê học tập.")
        recommendations.append("✏️ Yêu cầu chép phạt bài tập đầy đủ và giám sát nộp bài tập hàng ngày tại lớp.")
    elif assignment < 80.0:
        r_assignment = 50.0 + (80.0 - assignment) * 2.5
        warning_flags.append(f"⚠️ Bài tập thiếu hụt ({assignment}%): Chưa tự giác nộp bài đúng hạn.")
        recommendations.append("📝 Cử lớp phó học tập theo dõi sát tiến độ nộp bài của học sinh này.")
    else:
        r_assignment = max(0.0, (100.0 - assignment) * 2.5)

    # f. Sa sút tiến độ (5% weight)
    if progress <= -2.0:
        r_progress = 100.0
        warning_flags.append(f"📉 SA SÚT HỌC LỰC MẠNH ({progress:+.1f}): Kết quả học tập giảm nghiêm trọng so với kỳ trước.")
        recommendations.append("🧠 Tìm hiểu nguyên nhân tâm lý, áp lực gia đình hoặc vấn đề cá nhân đang tác động tiêu cực đến học sinh.")
    elif progress < 0.0:
        r_progress = -progress * 50.0
        warning_flags.append(f"📉 Kết quả giảm sút nhẹ ({progress:+.1f}) so với kỳ trước.")
    else:
        r_progress = 0.0

    # ==========================================
    # 2. TÍNH ĐIỂM RỦI RO TỔNG HỢP (WEIGHTED SCORE)
    # ==========================================
    risk_score = (
        0.30 * r_attendance +
        0.25 * r_min_score +
        0.15 * r_comments +
        0.15 * r_gpa +
        0.10 * r_assignment +
        0.05 * r_progress
    )
    risk_score = round(max(0.0, min(100.0, risk_score)), 1)

    # ==========================================
    # 3. PHÂN CẤP MỨC ĐỘ RỦI RO (RULE AUTOMATION)
    # ==========================================
    # Áp dụng quy chế cứng: Nếu vi phạm thẳng điều kiện Chưa đạt bắt buộc của BGDĐT -> Cao
    is_legal_failure = is_critical_attendance or is_critical_score or is_critical_comments or is_critical_gpa
    
    if is_legal_failure or risk_score >= 75.0:
        risk_level = "🔴 Cao"
        if not warning_flags:
            warning_flags.append("🚨 Học sinh có điểm rủi ro hành vi tổng hợp ở mức cực kỳ cao.")
        if not recommendations:
            recommendations.append("🚨 Cần kích hoạt ngay lập tức kế hoạch hỗ trợ can thiệp giáo dục cá nhân hóa.")
    elif risk_score >= 45.0:
        risk_level = "🟠 Cảnh báo"
        recommendations.append("📋 Theo dõi và nhắc nhở hàng tuần, lập mục tiêu tiến bộ ngắn hạn.")
    elif risk_score >= 25.0:
        risk_level = "🟡 Theo dõi"
        recommendations.append("💬 Thường xuyên kiểm tra vở ghi bài và động viên khích lệ học sinh phát biểu.")
    else:
        risk_level = "🟢 An toàn"
        warning_flags.append("🟢 Học tập ổn định và tích cực.")
        recommendations.append("🌟 Biểu dương trước lớp, khuyến khích giữ vững phong độ và hỗ trợ các bạn học yếu trong lớp.")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "warning_flags": warning_flags,
        "recommendations": recommendations
    }


def analyze_batch_ews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng EWS cảnh báo sớm hàng loạt trên tập dữ liệu học sinh.

    Args:
        df: DataFrame học sinh.

    Returns:
        DataFrame gốc kèm các cột rủi ro mới: risk_score, risk_level, warning_flags_str,
        sắp xếp theo thứ tự ưu tiên can thiệp (Cao -> Cảnh báo -> Theo dõi -> An toàn) 
        và điểm rủi ro giảm dần.
    """
    result_df = df.copy()

    risk_scores = []
    risk_levels = []
    warning_flags_list = []
    recs_list = []

    for _, row in result_df.iterrows():
        risk_analysis = calculate_student_risk(row)
        risk_scores.append(risk_analysis["risk_score"])
        risk_levels.append(risk_analysis["risk_level"])
        warning_flags_list.append("; ".join(risk_analysis["warning_flags"]))
        recs_list.append("; ".join(risk_analysis["recommendations"]))

    result_df["risk_score"] = risk_scores
    result_df["risk_level"] = risk_levels
    result_df["warning_flags_str"] = warning_flags_list
    result_df["recommendations_str"] = recs_list

    # Ánh xạ giá trị sắp xếp mức rủi ro
    level_order = {"🔴 Cao": 0, "🟠 Cảnh báo": 1, "🟡 Theo dõi": 2, "🟢 An toàn": 3}
    result_df["_level_priority"] = result_df["risk_level"].map(level_order)

    # Sắp xếp theo mức rủi ro ưu tiên và sau đó là điểm rủi ro
    result_df = result_df.sort_values(
        by=["_level_priority", "risk_score"],
        ascending=[True, False]
    ).drop(columns=["_level_priority"])

    return result_df
