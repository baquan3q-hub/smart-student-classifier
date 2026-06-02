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

from src.config import SCORE_SUBJECTS, COMMENT_SUBJECTS
from src.rule_engine import classify_learning_result, calculate_dtb_mcn


def _aggregate_scores(
    scores_df: pd.DataFrame,
    student_id: str,
    semester: Optional[str] = None,
) -> dict:
    """
    Tổng hợp điểm DTB của một học sinh.
    - Nếu semester được truyền: Tổng hợp điểm DTB_mhk của học kỳ đó.
    - Nếu semester là None: Tính điểm trung bình cả năm (DTB_mcn) cho từng môn.

    Args:
        scores_df: DataFrame chứa điểm theo môn.
        student_id: Mã học sinh.
        semester: Lọc theo kỳ (None = cả năm).

    Returns:
        Dictionary chứa các chỉ số thống kê điểm.
    """
    student_scores = scores_df[scores_df["student_id"] == student_id]

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

    if semester:
        # Tổng hợp điểm cho 1 học kỳ
        term_scores = student_scores[student_scores["semester"] == semester]
        dtb_values = term_scores["dtb_mhk"].dropna().astype(float).tolist()
    else:
        # Tổng hợp điểm cả năm (YEAR) - Tính DTB_mcn cho mỗi môn
        dtb_values = []
        for subject in SCORE_SUBJECTS:
            hk1_row = student_scores[(student_scores["semester"] == "HK1") & (student_scores["subject_name"] == subject)]
            hk2_row = student_scores[(student_scores["semester"] == "HK2") & (student_scores["subject_name"] == subject)]
            
            if not hk1_row.empty and not hk2_row.empty:
                dtb_hk1 = float(hk1_row.iloc[0]["dtb_mhk"])
                dtb_hk2 = float(hk2_row.iloc[0]["dtb_mhk"])
                dtb_mcn = calculate_dtb_mcn(dtb_hk1, dtb_hk2)
                dtb_values.append(dtb_mcn)
            elif not hk2_row.empty:
                dtb_values.append(float(hk2_row.iloc[0]["dtb_mhk"]))
            elif not hk1_row.empty:
                dtb_values.append(float(hk1_row.iloc[0]["dtb_mhk"]))

    if not dtb_values:
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
    - Nếu semester được truyền: Tổng hợp nhận xét học kỳ đó.
    - Nếu semester là None: Tính kết quả nhận xét cả năm theo từng môn (dựa trên HK2).

    Args:
        comments_df: DataFrame chứa nhận xét theo môn.
        student_id: Mã học sinh.
        semester: Lọc theo kỳ (None = cả năm).

    Returns:
        Dictionary chứa số lượng Đạt/Chưa đạt.
    """
    student_comments = comments_df[comments_df["student_id"] == student_id]

    if student_comments.empty:
        return {
            "num_comment_subjects": 0,
            "comment_pass_count": 0,
            "comment_not_pass_count": 0,
            "comment_statuses": [],
        }

    if semester:
        # Tổng hợp cho 1 học kỳ
        term_comments = student_comments[student_comments["semester"] == semester]
        statuses = term_comments["comment_status"].dropna().tolist()
    else:
        # Tổng hợp nhận xét cả năm (YEAR)
        # Theo Điều 9 Thông tư 22: Kết quả môn nhận xét cả năm lấy theo kết quả học kỳ II.
        statuses = []
        for subject in COMMENT_SUBJECTS:
            hk1_row = student_comments[(student_comments["semester"] == "HK1") & (student_comments["subject_name"] == subject)]
            hk2_row = student_comments[(student_comments["semester"] == "HK2") & (student_comments["subject_name"] == subject)]
            
            if not hk2_row.empty:
                statuses.append(hk2_row.iloc[0]["comment_status"])
            elif not hk1_row.empty:
                statuses.append(hk1_row.iloc[0]["comment_status"])

    if not statuses:
        return {
            "num_comment_subjects": 0,
            "comment_pass_count": 0,
            "comment_not_pass_count": 0,
            "comment_statuses": [],
        }

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
        active_semester = profile["semester"]

        # Nếu là giữa kỳ (MID_SEMESTER), chỉ tổng hợp học kỳ hiện tại
        # để đảm bảo đúng số môn của 1 học kỳ (8 môn điểm số, 3 môn nhận xét)
        if period == "MID_SEMESTER":
            semester_filter = active_semester
        elif period == "YEAR":
            semester_filter = None
        else:
            semester_filter = period
            
        score_agg = _aggregate_scores(scores_df, sid, semester_filter)


        comment_agg = _aggregate_comments(comments_df, sid, semester_filter)

        # Classify bằng Rule Engine
        label = classify_learning_result(
            score_agg["score_averages"],
            comment_agg["comment_statuses"],
            attendance_rate=float(profile.get("attendance_rate", 100.0)),
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
            "is_mid_semester": 1 if period == "MID_SEMESTER" else 0,
            "learning_result_label": label,
        })

    features_df = pd.DataFrame(features_rows)
    return features_df
