"""
feature_engineering.py — Kỹ thuật đặc trưng: aggregate dữ liệu thô thành features cho ML.

Module này xử lý:
- Tổng hợp điểm theo môn → chỉ số thống kê (avg, min, max, std, count thresholds)
- Tổng hợp nhận xét → comment_pass_count, comment_not_pass_count
- Merge với dữ liệu hành vi → student_features.csv hoàn chỉnh
- Gán nhãn learning_result_label bằng Rule Engine
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.rule_engine import classify_learning_result


def _aggregate_scores(
    scores_df: pd.DataFrame,
    student_id: str,
    semester: Optional[str] = None,
) -> dict:
    """
    Tổng hợp điểm DTB_mhk của một học sinh.

    Args:
        scores_df: DataFrame chứa điểm theo môn.
        student_id: Mã học sinh.
        semester: Lọc theo kỳ (None = tất cả).

    Returns:
        Dictionary chứa các chỉ số thống kê điểm.
    """
    mask = scores_df["student_id"] == student_id
    if semester:
        mask = mask & (scores_df["semester"] == semester)

    student_scores = scores_df[mask]

    if student_scores.empty:
        return {
            "num_score_subjects": 0,
            "avg_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "std_score": 0.0,
            "count_score_ge_8": 0,
            "count_score_ge_6_5": 0,
            "count_score_ge_5": 0,
            "count_score_lt_3_5": 0,
            "score_averages": [],
        }

    dtb_values = student_scores["dtb_mhk"].astype(float).tolist()

    return {
        "num_score_subjects": len(dtb_values),
        "avg_score": round(np.mean(dtb_values), 1),
        "min_score": round(min(dtb_values), 1),
        "max_score": round(max(dtb_values), 1),
        "std_score": round(float(np.std(dtb_values)), 2) if len(dtb_values) > 1 else 0.0,
        "count_score_ge_8": sum(1 for s in dtb_values if s >= 8.0),
        "count_score_ge_6_5": sum(1 for s in dtb_values if s >= 6.5),
        "count_score_ge_5": sum(1 for s in dtb_values if s >= 5.0),
        "count_score_lt_3_5": sum(1 for s in dtb_values if s < 3.5),
        "score_averages": dtb_values,
    }


def _aggregate_comments(
    comments_df: pd.DataFrame,
    student_id: str,
    semester: Optional[str] = None,
) -> dict:
    """
    Tổng hợp nhận xét của một học sinh.

    Args:
        comments_df: DataFrame chứa nhận xét theo môn.
        student_id: Mã học sinh.
        semester: Lọc theo kỳ (None = tất cả).

    Returns:
        Dictionary chứa số lượng Đạt/Chưa đạt.
    """
    mask = comments_df["student_id"] == student_id
    if semester:
        mask = mask & (comments_df["semester"] == semester)

    student_comments = comments_df[mask]

    if student_comments.empty:
        return {
            "num_comment_subjects": 0,
            "comment_pass_count": 0,
            "comment_not_pass_count": 0,
            "comment_statuses": [],
        }

    statuses = student_comments["comment_status"].tolist()

    return {
        "num_comment_subjects": len(statuses),
        "comment_pass_count": sum(1 for s in statuses if s == "Đạt"),
        "comment_not_pass_count": sum(1 for s in statuses if s == "Chưa đạt"),
        "comment_statuses": statuses,
    }


def build_student_features(
    profiles_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    period: str = "YEAR",
) -> pd.DataFrame:
    """
    Xây dựng bảng student_features.csv từ 3 bảng raw data.

    Pipeline:
    1. Lấy danh sách unique student_ids.
    2. Với mỗi student: aggregate scores + comments theo cả năm.
    3. Merge với behavior data từ profiles.
    4. Gán nhãn learning_result_label bằng Rule Engine.

    Args:
        profiles_df: DataFrame hồ sơ học sinh.
        scores_df: DataFrame điểm theo môn.
        comments_df: DataFrame nhận xét theo môn.
        period: Kỳ đánh giá ("HK1", "HK2", "YEAR").

    Returns:
        DataFrame chứa features cho ML (22 cột + target).
    """
    student_ids = profiles_df["student_id"].unique()
    features_rows = []

    for sid in student_ids:
        # Lấy thông tin profile (dùng dòng cuối cùng = HK2 nếu có)
        student_profiles = profiles_df[profiles_df["student_id"] == sid]
        if student_profiles.empty:
            continue

        # Lấy dòng profile cuối cùng (HK2 nếu có, hoặc HK1)
        profile = student_profiles.iloc[-1]

        # Aggregate scores (tất cả kỳ nếu YEAR)
        semester_filter = None if period == "YEAR" else period
        score_agg = _aggregate_scores(scores_df, sid, semester_filter)
        comment_agg = _aggregate_comments(comments_df, sid, semester_filter)

        # Classify bằng Rule Engine
        label = classify_learning_result(
            score_agg["score_averages"],
            comment_agg["comment_statuses"],
        )

        # Tính progress_delta
        progress_delta = float(profile.get("progress_delta", 0.0))

        features_rows.append({
            "student_id": sid,
            "class_name": profile["class_name"],
            "school_year": profile["school_year"],
            "period": period,
            "num_score_subjects": score_agg["num_score_subjects"],
            "num_comment_subjects": comment_agg["num_comment_subjects"],
            "avg_score": score_agg["avg_score"],
            "min_score": score_agg["min_score"],
            "max_score": score_agg["max_score"],
            "std_score": score_agg["std_score"],
            "count_score_ge_8": score_agg["count_score_ge_8"],
            "count_score_ge_6_5": score_agg["count_score_ge_6_5"],
            "count_score_ge_5": score_agg["count_score_ge_5"],
            "count_score_lt_3_5": score_agg["count_score_lt_3_5"],
            "comment_pass_count": comment_agg["comment_pass_count"],
            "comment_not_pass_count": comment_agg["comment_not_pass_count"],
            "attendance_rate": float(profile.get("attendance_rate", 0.0)),
            "assignment_completion_rate": float(profile.get("assignment_completion_rate", 0.0)),
            "participation_score": float(profile.get("participation_score", 0.0)),
            "behavior_score": float(profile.get("behavior_score", 0.0)),
            "teacher_evaluation_score": float(profile.get("teacher_evaluation_score", 0.0)),
            "progress_delta": progress_delta,
            "learning_result_label": label,
        })

    features_df = pd.DataFrame(features_rows)
    return features_df
