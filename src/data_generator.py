"""
data_generator.py — Tạo dữ liệu mô phỏng cho hệ thống.

Module này sinh ra 3 file raw data và 1 file processed data:
- student_profiles_sample.csv (hồ sơ + hành vi)
- student_scores_sample.csv (điểm theo môn)
- student_comments_sample.csv (nhận xét theo môn)
- student_features.csv (features tổng hợp cho ML)

Dữ liệu hoàn toàn giả lập, KHÔNG sử dụng thông tin học sinh thật.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import (
    SCORE_SUBJECTS, COMMENT_SUBJECTS, SCHOOL_YEAR, SEMESTERS, CLASSES,
    NUM_STUDENTS, NUM_REGULAR_SCORES, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    ensure_directories,
)
from src.rule_engine import calculate_dtb_mhk, classify_learning_result
from src.utils import generate_vietnamese_name


def _generate_student_ids(num_students: int, start_idx: int = 1) -> list[str]:
    """Sinh danh sách mã học sinh."""
    return [f"HS{i:03d}" for i in range(start_idx, start_idx + num_students)]


def _assign_classes(student_ids: list[str], classes: list[str]) -> dict[str, str]:
    """Phân bổ học sinh vào các lớp đều nhau."""
    assignments = {}
    for i, sid in enumerate(student_ids):
        assignments[sid] = classes[i % len(classes)]
    return assignments


def _generate_student_tier(rng: np.random.Generator, data_profile: str = "balanced") -> str:
    """
    Sinh tier học lực cho học sinh dựa trên cấu hình phân phối dữ liệu.
    
    Các cấu hình:
    - balanced: tỷ lệ thực tế cân bằng.
    - high_performing: đa số học sinh học tốt (chọn lọc).
    - at_risk: đa số học sinh yếu kém, cần hỗ trợ.
    - mid_semester: tỷ lệ giống balanced nhưng sẽ ẩn điểm cuối kỳ ở bước sau.
    """
    if data_profile == "high_performing":
        tiers = ["tot", "kha", "dat", "chua_dat", "adversarial_tot", "adversarial_kha"]
        p = [0.50, 0.33, 0.11, 0.02, 0.03, 0.01]
    elif data_profile == "at_risk":
        tiers = ["tot", "kha", "dat", "chua_dat", "adversarial_tot", "adversarial_kha"]
        p = [0.03, 0.10, 0.30, 0.45, 0.06, 0.06]
    else:  # balanced hoặc mid_semester
        # Đã điều chỉnh tỷ lệ Chưa đạt thực tế về khoảng 10% tổng số lớp học
        # (chua_dat (6%) + adversarial_tot (2%) + adversarial_kha (2%) = 10%)
        tiers = ["tot", "kha", "dat", "chua_dat", "adversarial_tot", "adversarial_kha"]
        p = [0.25, 0.40, 0.25, 0.06, 0.02, 0.02]

    return rng.choice(tiers, p=p)


def _generate_scores_for_tier(
    tier: str,
    rng: np.random.Generator,
    num_regular: int = 3,
) -> tuple[list[float], float, float]:
    """
    Sinh điểm regular, midterm, final theo tier học lực.

    Returns:
        (regular_scores, midterm_score, final_score)
    """
    # Các học sinh nghịch cảnh sẽ có điểm học thuật giống hệt học lực thật (Tốt/Khá)
    effective_tier = "tot" if tier == "adversarial_tot" else ("kha" if tier == "adversarial_kha" else tier)

    if effective_tier == "tot":
        # Điểm cao: 7.0-10.0, xu hướng >= 8.0
        base = rng.uniform(7.5, 9.5)
        regular = [round(max(0, min(10, base + rng.normal(0, 0.7))), 1) for _ in range(num_regular)]
        midterm = round(max(0, min(10, base + rng.normal(0, 0.5))), 1)
        final = round(max(0, min(10, base + rng.normal(0.2, 0.5))), 1)

    elif effective_tier == "kha":
        # Điểm khá: 5.5-8.5, xu hướng 6.5-7.5
        base = rng.uniform(6.0, 8.0)
        regular = [round(max(0, min(10, base + rng.normal(0, 1.0))), 1) for _ in range(num_regular)]
        midterm = round(max(0, min(10, base + rng.normal(0, 0.8))), 1)
        final = round(max(0, min(10, base + rng.normal(0, 0.8))), 1)

    elif effective_tier == "dat":
        # Điểm trung bình: 4.0-7.0, xu hướng 5.0-6.0
        base = rng.uniform(4.5, 6.5)
        # Giảm độ lệch chuẩn từ 1.2/1.0 xuống 0.8/0.7 để tránh điểm liệt dưới 3.5 cho tier Đạt
        regular = [round(max(0, min(10, base + rng.normal(0, 0.8))), 1) for _ in range(num_regular)]
        midterm = round(max(0, min(10, base + rng.normal(0, 0.7))), 1)
        final = round(max(0, min(10, base + rng.normal(0, 0.7))), 1)

    else:  # chua_dat
        # Điểm thấp: 2.0-5.5, có thể có môn < 3.5
        base = rng.uniform(2.5, 5.0)
        regular = [round(max(0, min(10, base + rng.normal(0, 1.5))), 1) for _ in range(num_regular)]
        midterm = round(max(0, min(10, base + rng.normal(0, 1.2))), 1)
        final = round(max(0, min(10, base + rng.normal(-0.3, 1.2))), 1)

    # Clip tất cả điểm về [0, 10]
    regular = [round(max(0.0, min(10.0, s)), 1) for s in regular]
    midterm = round(max(0.0, min(10.0, midterm)), 1)
    final = round(max(0.0, min(10.0, final)), 1)

    return regular, midterm, final


def _generate_comment_for_tier(tier: str, rng: np.random.Generator) -> str:
    """Sinh trạng thái nhận xét theo tier."""
    if tier in ["tot", "adversarial_tot"]:
        return "Đạt"
    elif tier in ["kha", "adversarial_kha"]:
        return "Đạt"
    elif tier == "dat":
        # 85% Đạt, 15% Chưa đạt (cho phép tối đa 1 Chưa đạt)
        return rng.choice(["Đạt", "Chưa đạt"], p=[0.85, 0.15])
    else:  # chua_dat
        # 50% Đạt, 50% Chưa đạt
        return rng.choice(["Đạt", "Chưa đạt"], p=[0.50, 0.50])


def _generate_behavior_for_tier(
    tier: str,
    rng: np.random.Generator,
) -> dict:
    """Sinh dữ liệu hành vi/chuyên cần theo tier."""
    effective_tier = "tot" if tier == "adversarial_tot" else ("kha" if tier == "adversarial_kha" else tier)

    if effective_tier == "tot":
        total_sessions = rng.integers(80, 100)
        attend_rate = rng.uniform(90, 100)
        total_assignments = rng.integers(30, 50)
        assign_rate = rng.uniform(90, 100)
        participation = round(rng.uniform(7.5, 10.0), 1)
        behavior = round(rng.uniform(7.5, 10.0), 1)
        teacher_eval = round(rng.uniform(8.0, 10.0), 1)

    elif effective_tier == "kha":
        total_sessions = rng.integers(80, 100)
        # Nâng nhẹ attendance lên để tránh sát nút 75%
        attend_rate = rng.uniform(82, 97)
        total_assignments = rng.integers(30, 50)
        assign_rate = rng.uniform(78, 96)
        participation = round(rng.uniform(6.0, 8.5), 1)
        behavior = round(rng.uniform(6.0, 8.5), 1)
        teacher_eval = round(rng.uniform(6.0, 8.5), 1)

    elif effective_tier == "dat":
        total_sessions = rng.integers(80, 100)
        # Nâng lên tối thiểu 76% để tránh bị Chưa đạt do chuyên cần < 75% cho nhóm học sinh Đạt
        attend_rate = rng.uniform(76, 90)
        total_assignments = rng.integers(30, 50)
        assign_rate = rng.uniform(65, 85)
        participation = round(rng.uniform(4.5, 7.0), 1)
        behavior = round(rng.uniform(4.5, 7.0), 1)
        teacher_eval = round(rng.uniform(4.5, 7.0), 1)

    else:  # chua_dat
        total_sessions = rng.integers(80, 100)
        attend_rate = rng.uniform(40, 72)
        total_assignments = rng.integers(30, 50)
        assign_rate = rng.uniform(30, 65)
        participation = round(rng.uniform(2.0, 5.5), 1)
        behavior = round(rng.uniform(2.0, 5.5), 1)
        teacher_eval = round(rng.uniform(2.0, 5.5), 1)

    total_sessions = int(total_sessions)
    attended = int(round(total_sessions * attend_rate / 100))
    attended = max(0, min(total_sessions, attended))
    attendance_rate = round(attended / total_sessions * 100, 1)

    total_assignments = int(total_assignments)
    submitted = int(round(total_assignments * assign_rate / 100))
    submitted = max(0, min(total_assignments, submitted))
    assignment_completion_rate = round(submitted / total_assignments * 100, 1)

    # Clip rubric scores
    participation = round(max(1.0, min(10.0, participation)), 1)
    behavior = round(max(1.0, min(10.0, behavior)), 1)
    teacher_eval = round(max(1.0, min(10.0, teacher_eval)), 1)

    return {
        "total_sessions": total_sessions,
        "attended_sessions": attended,
        "attendance_rate": attendance_rate,
        "total_assignments": total_assignments,
        "submitted_assignments": submitted,
        "assignment_completion_rate": assignment_completion_rate,
        "participation_score": participation,
        "behavior_score": behavior,
        "teacher_evaluation_score": teacher_eval,
    }


def generate_raw_data(
    num_students: int = NUM_STUDENTS,
    random_state: int = 42,
    data_profile: str = "balanced",
    start_idx: int = 1,
    is_mid_sem: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sinh toàn bộ dữ liệu thô cho hệ thống.

    Args:
        num_students: Số lượng học sinh cần sinh.
        random_state: Seed cho reproducibility.
        data_profile: Cấu hình phân phối dữ liệu (balanced, high_performing, at_risk, mid_semester).
        start_idx: Chỉ số bắt đầu sinh mã học sinh (default: 1).

    Returns:
        (profiles_df, scores_df, comments_df)
    """
    rng = np.random.default_rng(random_state)

    student_ids = _generate_student_ids(num_students, start_idx)
    class_assignments = _assign_classes(student_ids, CLASSES)

    # Gán tier cho mỗi học sinh (cố định qua cả 2 kỳ)
    student_tiers = {sid: _generate_student_tier(rng, data_profile) for sid in student_ids}

    # Gán loại nghịch cảnh ngẫu nhiên (adversarial cases) cho một số học sinh tốt/khá
    # để mô hình AI học cách xử lý các điều kiện loại trừ (loại chuyên cần và nhận xét)
    student_adversarials = {}
    for sid in student_ids:
        if student_tiers[sid] in ["tot", "kha"]:
            r = rng.random()
            if r < 0.05:
                student_adversarials[sid] = "low_attendance"
            elif r < 0.10:
                student_adversarials[sid] = "single_comment"
            else:
                student_adversarials[sid] = "none"
        else:
            student_adversarials[sid] = "none"

    profiles_rows = []
    scores_rows = []
    comments_rows = []

    # is_mid_sem được lấy trực tiếp từ tham số truyền vào chứ không phụ thuộc vào data_profile nữa

    for sid in student_ids:
        tier = student_tiers[sid]
        class_name = class_assignments[sid]
        student_name = generate_vietnamese_name(int(sid[2:]))
        adv_type = student_adversarials.get(sid, "none")

        # Lưu DTB_mhk theo kỳ để tính previous_avg
        semester_avgs = {}

        for semester in SEMESTERS:
            # --- Sinh điểm cho từng môn điểm số ---
            dtb_list = []
            for subject in SCORE_SUBJECTS:
                regular, midterm, final = _generate_scores_for_tier(
                    tier, rng, NUM_REGULAR_SCORES
                )
                
                # Nếu là dữ liệu giữa kỳ, không có điểm thi cuối kỳ (gán NaN)
                if is_mid_sem:
                    final = np.nan
                    
                dtb_mhk = calculate_dtb_mhk(regular, midterm, final)
                dtb_list.append(dtb_mhk)

                scores_rows.append({
                    "student_id": sid,
                    "class_name": class_name,
                    "school_year": SCHOOL_YEAR,
                    "semester": semester,
                    "subject_name": subject,
                    "assessment_type": "score",
                    "regular_scores": ";".join(str(s) for s in regular),
                    "midterm_score": midterm,
                    "final_score": final,
                    "dtb_mhk": dtb_mhk,
                })

            # --- Sinh nhận xét cho từng môn nhận xét ---
            if adv_type == "single_comment":
                # Nghịch cảnh: Đúng 1 môn nhận xét Chưa đạt, 2 môn còn lại Đạt
                fail_subject = rng.choice(COMMENT_SUBJECTS)
                for subject in COMMENT_SUBJECTS:
                    status = "Chưa đạt" if subject == fail_subject else "Đạt"
                    comments_rows.append({
                        "student_id": sid,
                        "class_name": class_name,
                        "school_year": SCHOOL_YEAR,
                        "semester": semester,
                        "subject_name": subject,
                        "assessment_type": "comment",
                        "comment_status": status,
                    })
            elif tier in ["adversarial_tot", "adversarial_kha"]:
                # Chọn ngẫu nhiên 2 môn nhận xét làm "Chưa đạt", môn còn lại làm "Đạt"
                fail_subjects = rng.choice(COMMENT_SUBJECTS, size=2, replace=False)
                for subject in COMMENT_SUBJECTS:
                    status = "Chưa đạt" if subject in fail_subjects else "Đạt"
                    comments_rows.append({
                        "student_id": sid,
                        "class_name": class_name,
                        "school_year": SCHOOL_YEAR,
                        "semester": semester,
                        "subject_name": subject,
                        "assessment_type": "comment",
                        "comment_status": status,
                    })
            else:
                for subject in COMMENT_SUBJECTS:
                    status = _generate_comment_for_tier(tier, rng)
                    comments_rows.append({
                        "student_id": sid,
                        "class_name": class_name,
                        "school_year": SCHOOL_YEAR,
                        "semester": semester,
                        "subject_name": subject,
                        "assessment_type": "comment",
                        "comment_status": status,
                    })

            # Tính avg cho semester
            semester_avgs[semester] = round(np.mean(dtb_list), 1) if dtb_list else 0.0

            # --- Sinh profile/hành vi ---
            behavior = _generate_behavior_for_tier(tier, rng)

            if adv_type == "low_attendance":
                # Nghịch cảnh: Chuyên cần thấp dưới 75%
                low_att = round(rng.uniform(65.0, 74.0), 1)
                behavior["attendance_rate"] = low_att
                behavior["attended_sessions"] = int(round(behavior["total_sessions"] * low_att / 100))

            # previous_average_score: kỳ trước hoặc random
            if semester == "HK1":
                prev_avg = round(rng.uniform(
                    max(0, semester_avgs[semester] - 1.5),
                    min(10, semester_avgs[semester] + 1.5)
                ), 1)
            else:
                prev_avg = semester_avgs.get("HK1", 5.0)

            current_avg = semester_avgs[semester]
            progress_delta = round(current_avg - prev_avg, 1)
            progress_delta = round(max(-10.0, min(10.0, progress_delta)), 1)

            profiles_rows.append({
                "student_id": sid,
                "student_name": student_name,
                "class_name": class_name,
                "school_year": SCHOOL_YEAR,
                "semester": semester,
                "total_sessions": behavior["total_sessions"],
                "attended_sessions": behavior["attended_sessions"],
                "attendance_rate": behavior["attendance_rate"],
                "total_assignments": behavior["total_assignments"],
                "submitted_assignments": behavior["submitted_assignments"],
                "assignment_completion_rate": behavior["assignment_completion_rate"],
                "participation_score": behavior["participation_score"],
                "behavior_score": behavior["behavior_score"],
                "teacher_evaluation_score": behavior["teacher_evaluation_score"],
                "previous_average_score": prev_avg,
                "current_average_score": current_avg,
                "progress_delta": progress_delta,
            })

    profiles_df = pd.DataFrame(profiles_rows)
    scores_df = pd.DataFrame(scores_rows)
    comments_df = pd.DataFrame(comments_rows)

    return profiles_df, scores_df, comments_df


def generate_all_data(
    num_students: int = NUM_STUDENTS,
    random_state: int = 42,
    data_profile: str = "balanced",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sinh toàn bộ dữ liệu: raw + processed features.

    Args:
        num_students: Số lượng học sinh.
        random_state: Seed cho reproducibility.
        data_profile: Cấu hình phân phối dữ liệu (balanced, high_performing, at_risk, mid_semester).
        save: Có lưu file CSV hay không.

    Returns:
        (profiles_df, scores_df, comments_df, features_df)
    """
    ensure_directories()

    # Sinh raw data và processed features dựa trên data_profile
    from src.feature_engineering import build_student_features

    if data_profile == "mid_semester":
        # Sinh 100% dữ liệu giữa kỳ với profile balanced (phục vụ kịch bản đặc biệt)
        profiles_df, scores_df, comments_df = generate_raw_data(num_students, random_state, "balanced", 1, is_mid_sem=True)
        features_df = build_student_features(profiles_df, scores_df, comments_df, period="MID_SEMESTER")
    else:
        # Sinh 100% dữ liệu Cả năm (YEAR) để dữ liệu huấn luyện sạch và chính xác nhất
        profiles_df, scores_df, comments_df = generate_raw_data(num_students, random_state, data_profile, 1, is_mid_sem=False)
        features_df = build_student_features(profiles_df, scores_df, comments_df, period="YEAR")

    if save:
        # Lưu raw data
        profiles_df.to_csv(RAW_DATA_DIR / "student_profiles_sample.csv", index=False, encoding="utf-8-sig")
        scores_df.to_csv(RAW_DATA_DIR / "student_scores_sample.csv", index=False, encoding="utf-8-sig")
        comments_df.to_csv(RAW_DATA_DIR / "student_comments_sample.csv", index=False, encoding="utf-8-sig")

        # Lưu processed data
        features_df.to_csv(PROCESSED_DATA_DIR / "student_features.csv", index=False, encoding="utf-8-sig")

    return profiles_df, scores_df, comments_df, features_df

