"""
recommendations.py — Khuyến nghị hỗ trợ học sinh theo nhãn phân loại.

Module này sinh khuyến nghị tiếng Việt cụ thể cho từng mức kết quả học tập.
"""

from typing import Optional


def generate_recommendations(
    label: str,
    features: Optional[dict] = None,
) -> list[str]:
    """
    Sinh danh sách khuyến nghị hỗ trợ theo nhãn phân loại.

    Args:
        label: Nhãn phân loại (Tốt/Khá/Đạt/Chưa đạt).
        features: Dictionary feature values (để cá nhân hóa khuyến nghị).

    Returns:
        Danh sách các khuyến nghị tiếng Việt.
    """
    recommendations = []

    if label == "Tốt":
        recommendations = [
            "🌟 Giao nhiệm vụ nâng cao và thử thách để phát huy tiềm năng.",
            "📚 Khuyến khích tham gia các cuộc thi học thuật và dự án nghiên cứu.",
            "🤝 Khuyến khích hỗ trợ bạn học cùng lớp (peer tutoring).",
            "🎯 Đặt mục tiêu cải thiện các môn chưa đạt điểm 9-10.",
            "💡 Phát triển kỹ năng tự học và tư duy phản biện.",
        ]

    elif label == "Khá":
        recommendations = [
            "📈 Duy trì thói quen học tập đều đặn hiện tại.",
            "🎯 Tập trung cải thiện các môn còn dưới điểm 8.0.",
            "📖 Tăng thời gian ôn bài và làm bài tập bổ sung.",
            "👥 Tham gia nhóm học tập để nâng cao hiệu quả.",
            "📝 Rèn luyện kỹ năng giải đề thi để chuẩn bị cho kỳ thi.",
        ]

    elif label == "Đạt":
        recommendations = [
            "📋 Theo dõi sát các môn yếu và lập kế hoạch học bù.",
            "✏️ Bổ sung bài tập củng cố kiến thức nền tảng.",
            "⏰ Cải thiện tỷ lệ chuyên cần và hoàn thành bài tập.",
            "👨‍🏫 Trao đổi với giáo viên bộ môn để được hỗ trợ thêm.",
            "📅 Xây dựng thời gian biểu học tập cụ thể hàng ngày.",
        ]

    elif label == "Chưa đạt":
        recommendations = [
            "🚨 Cần lập kế hoạch hỗ trợ cá nhân ngay lập tức.",
            "📝 Phụ đạo các môn yếu, đặc biệt môn có ĐTB dưới 5.0.",
            "👪 Trao đổi với phụ huynh/giám hộ về tình hình học tập.",
            "👨‍🏫 Gặp giáo viên chủ nhiệm để bàn giải pháp hỗ trợ.",
            "📊 Theo dõi chuyên cần hàng tuần — cần cải thiện tỷ lệ đi học.",
            "💪 Khuyến khích và động viên, tránh tạo áp lực tiêu cực.",
        ]

    # Thêm khuyến nghị cá nhân hóa nếu có features
    if features:
        personal = _generate_personal_recommendations(label, features)
        recommendations.extend(personal)

    return recommendations


def _generate_personal_recommendations(
    label: str,
    features: dict,
) -> list[str]:
    """Sinh khuyến nghị cá nhân hóa dựa trên feature values."""
    personal = []

    attendance = features.get("attendance_rate", 100)
    assignment = features.get("assignment_completion_rate", 100)
    min_score = features.get("min_score", 10)
    behavior = features.get("behavior_score", 10)
    comment_fail = features.get("comment_not_pass_count", 0)
    progress = features.get("progress_delta", 0)

    if attendance < 75:
        personal.append(f"🚨 Tỷ lệ chuyên cần thấp ({attendance}%) — học sinh nguy cơ bị giữ lại lớp do vắng quá 45 buổi học quy định bởi Điều 12 Thông tư 22/2021/TT-BGDĐT.")

    if assignment < 60:
        personal.append(f"⚠️ Tỷ lệ hoàn thành bài tập chỉ {assignment}% — cần cải thiện đáng kể.")

    if min_score < 3.5:
        personal.append(f"❌ Có môn điểm rất thấp ({min_score}) — cần phụ đạo ngay môn này.")

    if behavior < 5.0:
        personal.append(f"⚠️ Điểm hành vi thấp ({behavior}/10) — cần trao đổi với phụ huynh.")

    if comment_fail >= 2:
        personal.append(f"⚠️ Có {comment_fail} môn nhận xét Chưa đạt — cần cải thiện.")

    if progress < -2.0:
        personal.append(f"📉 Điểm giảm so với kỳ trước ({progress:+.1f}) — cần tìm hiểu nguyên nhân.")
    elif progress > 1.5:
        personal.append(f"📈 Có tiến bộ tốt so với kỳ trước ({progress:+.1f}) — tiếp tục phát huy!")

    return personal


def get_risk_level(label: str) -> tuple[str, str]:
    """
    Xác định mức độ rủi ro theo nhãn.

    Returns:
        (risk_level, risk_color)
    """
    risk_map = {
        "Tốt": ("Rất thấp", "#10B981"),
        "Khá": ("Thấp", "#3B82F6"),
        "Đạt": ("Trung bình", "#F59E0B"),
        "Chưa đạt": ("Cao", "#EF4444"),
    }
    return risk_map.get(label, ("Không xác định", "#6B7280"))
