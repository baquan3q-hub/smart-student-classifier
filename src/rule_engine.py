"""
rule_engine.py — Rule Engine phân loại kết quả học tập theo Bộ GD&ĐT.

Module này implement:
- Tính điểm trung bình môn học kỳ (DTB_mhk)
- Tính điểm trung bình môn cả năm (DTB_mcn)
- Phân loại kết quả học tập: Tốt / Khá / Đạt / Chưa đạt
- Giải thích lý do phân loại

Công thức theo quy định Bộ GD&ĐT Việt Nam.
"""

import math
from typing import Optional


def calculate_dtb_mhk(
    regular_scores: list[float],
    midterm_score: float,
    final_score: Optional[float] = None,
) -> float:
    """
    Tính Điểm Trung Bình Môn Học Kỳ (DTB_mhk).

    Công thức đầy đủ:
        DTB_mhk = (sum(regular_scores) + 2 × midterm_score + 3 × final_score)
                  / (số_điểm_thường_xuyên + 5)

    Công thức giữa kỳ (nếu chưa có final_score):
        DTB_mhk_gk = (sum(regular_scores) + 2 × midterm_score)
                     / (số_điểm_thường_xuyên + 2)

    Args:
        regular_scores: Danh sách điểm đánh giá thường xuyên.
        midterm_score: Điểm giữa kỳ.
        final_score: Điểm cuối kỳ (tùy chọn).

    Returns:
        Điểm trung bình môn học kỳ, làm tròn 1 chữ số thập phân.

    Raises:
        ValueError: Nếu regular_scores rỗng hoặc điểm ngoài phạm vi [0, 10].
    """
    if not regular_scores:
        raise ValueError("regular_scores phải có ít nhất 1 điểm.")

    # Validate ranges for available scores
    all_scores = list(regular_scores) + [midterm_score]
    
    # Check if final_score is provided and is not NaN
    has_final = final_score is not None and not (isinstance(final_score, float) and math.isnan(final_score))
    if has_final:
        all_scores.append(final_score)

    for score in all_scores:
        if not (0.0 <= score <= 10.0):
            raise ValueError(f"Điểm phải trong khoảng [0, 10]. Nhận được: {score}")

    n = len(regular_scores)
    if has_final:
        numerator = sum(regular_scores) + 2 * midterm_score + 3 * final_score
        denominator = n + 5
    else:
        # Công thức tính điểm giữa kỳ: regular hệ số 1, midterm hệ số 2
        numerator = sum(regular_scores) + 2 * midterm_score
        denominator = n + 2
        
    return round(numerator / denominator, 1)


def calculate_dtb_mcn(
    dtb_hk1: float,
    dtb_hk2: float,
) -> float:
    """
    Tính Điểm Trung Bình Môn Cả Năm (DTB_mcn).

    Công thức:
        DTB_mcn = (DTB_mhk_HK1 + 2 × DTB_mhk_HK2) / 3

    Args:
        dtb_hk1: Điểm trung bình môn học kỳ 1.
        dtb_hk2: Điểm trung bình môn học kỳ 2.

    Returns:
        Điểm trung bình môn cả năm, làm tròn 1 chữ số thập phân.

    Raises:
        ValueError: Nếu điểm ngoài phạm vi [0, 10].
    """
    for score in [dtb_hk1, dtb_hk2]:
        if not (0.0 <= score <= 10.0):
            raise ValueError(f"DTB phải trong khoảng [0, 10]. Nhận được: {score}")

    return round((dtb_hk1 + 2 * dtb_hk2) / 3, 1)


def _evaluate_standard(
    score_averages: list[float],
    comment_statuses: list[str],
) -> str:
    """Đánh giá học lực tiêu chuẩn chưa áp dụng luật điều chỉnh."""
    comment_not_pass_count = sum(
        1 for status in comment_statuses if status == "Chưa đạt"
    )
    all_comments_pass = comment_not_pass_count == 0

    count_ge_8 = sum(1 for score in score_averages if score >= 8.0)
    count_ge_6_5 = sum(1 for score in score_averages if score >= 6.5)
    count_ge_5 = sum(1 for score in score_averages if score >= 5.0)
    count_lt_3_5 = sum(1 for score in score_averages if score < 3.5)

    all_ge_6_5 = all(score >= 6.5 for score in score_averages) if score_averages else False
    all_ge_5 = all(score >= 5.0 for score in score_averages) if score_averages else False

    # Mức Tốt
    if all_comments_pass and all_ge_6_5 and count_ge_8 >= 6:
        return "Tốt"

    # Mức Khá
    if all_comments_pass and all_ge_5 and count_ge_6_5 >= 6:
        return "Khá"

    # Mức Đạt
    if comment_not_pass_count <= 1 and count_ge_5 >= 6 and count_lt_3_5 == 0:
        return "Đạt"

    # Mức Chưa đạt
    return "Chưa đạt"


def classify_learning_result(
    score_averages: list[float],
    comment_statuses: list[str],
    attendance_rate: float = 100.0,
) -> str:
    """
    Phân loại kết quả học tập theo quy định Bộ GD&ĐT.

    Luật phân loại (theo thứ tự ưu tiên):

    0. Chuyên cần (Điều 12 Thông tư 22):
       - Nghỉ học quá 45 buổi (chuyên cần < 75%) -> Bắt buộc xếp loại Chưa đạt (lưu ban).

    1. Tốt:
       - Tất cả môn nhận xét đều Đạt.
       - Tất cả môn điểm số có ĐTB >= 6.5.
       - Có ít nhất 6 môn điểm số có ĐTB >= 8.0.

    2. Khá:
       - Tất cả môn nhận xét đều Đạt.
       - Tất cả môn điểm số có ĐTB >= 5.0.
       - Có ít nhất 6 môn điểm số có ĐTB >= 6.5.

    3. Đạt:
       - Có nhiều nhất 1 môn nhận xét Chưa đạt.
       - Có ít nhất 6 môn điểm số có ĐTB >= 5.0.
       - Không có môn điểm số nào có ĐTB < 3.5.

    4. Chưa đạt:
       - Các trường hợp còn lại.

    Args:
        score_averages: Danh sách ĐTB các môn đánh giá bằng điểm số.
        comment_statuses: Danh sách trạng thái các môn đánh giá bằng nhận xét
                         ("Đạt" hoặc "Chưa đạt").
        attendance_rate: Tỷ lệ chuyên cần (%)

    Returns:
        Nhãn phân loại: "Tốt", "Khá", "Đạt", hoặc "Chưa đạt".
    """
    # --- Áp dụng luật chuyên cần (Thông tư 22/2021/TT-BGDĐT) ---
    if attendance_rate < 75.0:
        return "Chưa đạt"

    base_label = _evaluate_standard(score_averages, comment_statuses)

    # --- Áp dụng quy tắc điều chỉnh lên mức liền kề (Điều 9 Khoản 2 Điểm d Thông tư 22) ---
    # Nếu bị hạ kết quả học tập từ 2 mức xếp loại trở lên so với Tốt hoặc Khá chỉ do 1 môn học duy nhất
    levels = ["Chưa đạt", "Đạt", "Khá", "Tốt"]
    base_idx = levels.index(base_label)
    
    adjusted = False
    adjusted_label = base_label
    
    # 1. Thử loại bỏ từng môn điểm số để kiểm tra
    for idx in range(len(score_averages)):
        sub_scores = score_averages[:idx] + score_averages[idx+1:]
        sub_label = _evaluate_standard(sub_scores, comment_statuses)
        sub_idx = levels.index(sub_label)
        
        # Nếu việc loại bỏ môn này giúp kết quả tăng lên từ 2 bậc trở lên (ví dụ: từ Đạt lên Tốt, từ Chưa đạt lên Khá/Tốt)
        if sub_idx >= base_idx + 2:
            adjusted = True
            adjusted_label = levels[base_idx + 1]  # Điều chỉnh lên mức liền kề của xếp loại thực tế
            break

    # GHI CHÚ: KHÔNG áp dụng quy tắc điều chỉnh nâng bậc cho các môn đánh giá bằng nhận xét.
    # Theo Thông tư 22/2021/TT-BGDĐT: Để được xếp loại Khá trở lên, BẮT BUỘC tất cả
    # các môn nhận xét phải xếp loại "Đạt". Quy tắc điều chỉnh (Điều 9 Khoản 2 Điểm d)
    # chỉ áp dụng cho các môn đánh giá bằng điểm số.

    return adjusted_label


def get_classification_details(
    score_averages: list[float],
    comment_statuses: list[str],
    attendance_rate: float = 100.0,
) -> dict:
    """
    Trả về chi tiết phân tích cho Rule Engine.

    Args:
        score_averages: Danh sách ĐTB các môn điểm số.
        comment_statuses: Danh sách trạng thái các môn nhận xét.
        attendance_rate: Tỷ lệ chuyên cần (%)

    Returns:
        Dictionary chứa các chỉ số và kết quả phân loại.
    """
    comment_not_pass_count = sum(
        1 for s in comment_statuses if s == "Chưa đạt"
    )
    comment_pass_count = sum(
        1 for s in comment_statuses if s == "Đạt"
    )

    count_ge_8 = sum(1 for s in score_averages if s >= 8.0)
    count_ge_6_5 = sum(1 for s in score_averages if s >= 6.5)
    count_ge_5 = sum(1 for s in score_averages if s >= 5.0)
    count_lt_3_5 = sum(1 for s in score_averages if s < 3.5)

    label = classify_learning_result(score_averages, comment_statuses, attendance_rate)

    return {
        "label": label,
        "num_score_subjects": len(score_averages),
        "num_comment_subjects": len(comment_statuses),
        "avg_score": round(sum(score_averages) / len(score_averages), 1) if score_averages else 0.0,
        "min_score": min(score_averages) if score_averages else 0.0,
        "max_score": max(score_averages) if score_averages else 0.0,
        "count_score_ge_8": count_ge_8,
        "count_score_ge_6_5": count_ge_6_5,
        "count_score_ge_5": count_ge_5,
        "count_score_lt_3_5": count_lt_3_5,
        "comment_pass_count": comment_pass_count,
        "comment_not_pass_count": comment_not_pass_count,
        "attendance_rate": attendance_rate,
    }


def get_classification_reason(
    score_averages: list[float],
    comment_statuses: list[str],
    label: Optional[str] = None,
    attendance_rate: float = 100.0,
) -> str:
    """
    Sinh lý do phân loại bằng tiếng Việt dễ hiểu.

    Args:
        score_averages: Danh sách ĐTB các môn điểm số.
        comment_statuses: Danh sách trạng thái các môn nhận xét.
        label: Nhãn phân loại (nếu None sẽ tự tính).
        attendance_rate: Tỷ lệ chuyên cần (%)

    Returns:
        Chuỗi giải thích lý do phân loại bằng tiếng Việt.
    """
    if label is None:
        label = classify_learning_result(score_averages, comment_statuses, attendance_rate)

    details = get_classification_details(score_averages, comment_statuses, attendance_rate)
    reasons = []

    # Kiểm tra xem có điều chỉnh nâng bậc lên mức liền kề theo Điều 9 Thông tư 22 không
    base_label = _evaluate_standard(score_averages, comment_statuses)
    if label != base_label and attendance_rate >= 75.0:
        reasons.append(f"✨ **Điều chỉnh theo luật Bộ GD&ĐT (Điều 9 Khoản 2 Điểm d TT22):** Học sinh ban đầu bị hạ xuống mức `{base_label}` nhưng được điều chỉnh lên mức liền kề là **{label}** do chỉ có duy nhất 01 môn học kéo kết quả xuống.")

    if label == "Tốt":
        reasons.append("✅ Tất cả môn nhận xét đều Đạt.")
        reasons.append(f"✅ Tất cả {details['num_score_subjects']} môn điểm số có ĐTB ≥ 6.5.")
        reasons.append(f"✅ Có {details['count_score_ge_8']} môn đạt từ 8.0 trở lên (yêu cầu ≥ 6).")
        reasons.append(f"📊 Điểm trung bình chung: {details['avg_score']}")

    elif label == "Khá":
        reasons.append("✅ Tất cả môn nhận xét đều Đạt.")
        reasons.append(f"✅ Tất cả {details['num_score_subjects']} môn điểm số có ĐTB ≥ 5.0.")
        reasons.append(f"✅ Có {details['count_score_ge_6_5']} môn đạt từ 6.5 trở lên (yêu cầu ≥ 6).")
        if details['count_score_ge_8'] < 6:
            reasons.append(f"ℹ️ Chỉ có {details['count_score_ge_8']} môn đạt từ 8.0 (chưa đủ 6 để xếp Tốt).")
        reasons.append(f"📊 Điểm trung bình chung: {details['avg_score']}")

    elif label == "Đạt":
        if details['comment_not_pass_count'] == 0:
            reasons.append("✅ Tất cả môn nhận xét đều Đạt.")
        else:
            reasons.append(f"⚠️ Có {details['comment_not_pass_count']} môn nhận xét Chưa đạt (cho phép tối đa 1).")
        reasons.append(f"✅ Có {details['count_score_ge_5']} môn điểm số có ĐTB ≥ 5.0 (yêu cầu ≥ 6).")
        reasons.append("✅ Không có môn nào có ĐTB dưới 3.5.")
        if details['count_score_ge_6_5'] < 6:
            reasons.append(f"ℹ️ Chỉ có {details['count_score_ge_6_5']} môn đạt từ 6.5 (chưa đủ 6 để xếp Khá).")
        reasons.append(f"📊 Điểm trung bình chung: {details['avg_score']}")

    else:  # Chưa đạt
        if attendance_rate < 75.0:
            reasons.append(f"❌ Nghỉ học quá nhiều: Chuyên cần đạt {attendance_rate}% (vi phạm Điều 12 Thông tư 22, nghỉ quá 45 buổi học buộc lưu ban).")
        if details['comment_not_pass_count'] > 1:
            reasons.append(f"❌ Có {details['comment_not_pass_count']} môn nhận xét Chưa đạt (cho phép tối đa 1).")
        if details['count_score_ge_5'] < 6:
            reasons.append(f"❌ Chỉ có {details['count_score_ge_5']} môn điểm số có ĐTB ≥ 5.0 (yêu cầu ≥ 6).")
        if details['count_score_lt_3_5'] > 0:
            reasons.append(f"❌ Có {details['count_score_lt_3_5']} môn có ĐTB dưới 3.5.")
        if details['min_score'] < 5.0 and details['count_score_lt_3_5'] == 0:
            reasons.append(f"⚠️ Điểm thấp nhất: {details['min_score']}")
        reasons.append(f"📊 Điểm trung bình chung: {details['avg_score']}")

    return "\n".join(reasons)
