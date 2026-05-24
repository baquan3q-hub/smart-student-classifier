"""
rule_engine.py — Rule Engine phân loại kết quả học tập theo Bộ GD&ĐT.

Module này implement:
- Tính điểm trung bình môn học kỳ (DTB_mhk)
- Tính điểm trung bình môn cả năm (DTB_mcn)
- Phân loại kết quả học tập: Tốt / Khá / Đạt / Chưa đạt
- Giải thích lý do phân loại

Công thức theo quy định Bộ GD&ĐT Việt Nam.
"""

from typing import Optional


def calculate_dtb_mhk(
    regular_scores: list[float],
    midterm_score: float,
    final_score: float,
) -> float:
    """
    Tính Điểm Trung Bình Môn Học Kỳ (DTB_mhk).

    Công thức:
        DTB_mhk = (sum(regular_scores) + 2 × midterm_score + 3 × final_score)
                  / (số_điểm_thường_xuyên + 5)

    Args:
        regular_scores: Danh sách điểm đánh giá thường xuyên.
        midterm_score: Điểm giữa kỳ.
        final_score: Điểm cuối kỳ.

    Returns:
        Điểm trung bình môn học kỳ, làm tròn 1 chữ số thập phân.

    Raises:
        ValueError: Nếu regular_scores rỗng hoặc điểm ngoài phạm vi [0, 10].
    """
    if not regular_scores:
        raise ValueError("regular_scores phải có ít nhất 1 điểm.")

    # Validate ranges
    all_scores = regular_scores + [midterm_score, final_score]
    for score in all_scores:
        if not (0.0 <= score <= 10.0):
            raise ValueError(f"Điểm phải trong khoảng [0, 10]. Nhận được: {score}")

    n = len(regular_scores)
    numerator = sum(regular_scores) + 2 * midterm_score + 3 * final_score
    denominator = n + 5
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


def classify_learning_result(
    score_averages: list[float],
    comment_statuses: list[str],
) -> str:
    """
    Phân loại kết quả học tập theo quy định Bộ GD&ĐT.

    Luật phân loại (theo thứ tự ưu tiên):

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

    Returns:
        Nhãn phân loại: "Tốt", "Khá", "Đạt", hoặc "Chưa đạt".
    """
    # --- Tính toán các chỉ số ---
    comment_not_pass_count = sum(
        1 for status in comment_statuses if status == "Chưa đạt"
    )
    all_comments_pass = comment_not_pass_count == 0

    count_ge_8 = sum(1 for score in score_averages if score >= 8.0)
    count_ge_6_5 = sum(1 for score in score_averages if score >= 6.5)
    count_ge_5 = sum(1 for score in score_averages if score >= 5.0)
    count_lt_3_5 = sum(1 for score in score_averages if score < 3.5)

    all_ge_6_5 = all(score >= 6.5 for score in score_averages)
    all_ge_5 = all(score >= 5.0 for score in score_averages)

    # --- Áp dụng luật phân loại ---

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


def get_classification_details(
    score_averages: list[float],
    comment_statuses: list[str],
) -> dict:
    """
    Trả về chi tiết phân tích cho Rule Engine.

    Args:
        score_averages: Danh sách ĐTB các môn điểm số.
        comment_statuses: Danh sách trạng thái các môn nhận xét.

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

    label = classify_learning_result(score_averages, comment_statuses)

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
    }


def get_classification_reason(
    score_averages: list[float],
    comment_statuses: list[str],
    label: Optional[str] = None,
) -> str:
    """
    Sinh lý do phân loại bằng tiếng Việt dễ hiểu.

    Args:
        score_averages: Danh sách ĐTB các môn điểm số.
        comment_statuses: Danh sách trạng thái các môn nhận xét.
        label: Nhãn phân loại (nếu None sẽ tự tính).

    Returns:
        Chuỗi giải thích lý do phân loại bằng tiếng Việt.
    """
    if label is None:
        label = classify_learning_result(score_averages, comment_statuses)

    details = get_classification_details(score_averages, comment_statuses)
    reasons = []

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
        if details['comment_not_pass_count'] > 1:
            reasons.append(f"❌ Có {details['comment_not_pass_count']} môn nhận xét Chưa đạt (cho phép tối đa 1).")
        if details['count_score_ge_5'] < 6:
            reasons.append(f"❌ Chỉ có {details['count_score_ge_5']} môn điểm số có ĐTB ≥ 5.0 (yêu cầu ≥ 6).")
        if details['count_score_lt_3_5'] > 0:
            reasons.append(f"❌ Có {details['count_score_lt_3_5']} môn có ĐTB dưới 3.5.")
        if details['min_score'] < 5.0:
            reasons.append(f"⚠️ Điểm thấp nhất: {details['min_score']}")
        reasons.append(f"📊 Điểm trung bình chung: {details['avg_score']}")

    return "\n".join(reasons)
