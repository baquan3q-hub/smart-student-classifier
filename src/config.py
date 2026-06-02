"""
config.py — Cấu hình trung tâm cho Smart Student Classification System.

Chứa các hằng số, đường dẫn, danh sách feature columns, class labels,
và cấu hình môn học. Tất cả module khác import từ file này.
"""

from pathlib import Path
import json

# ============================================================
# ĐƯỜNG DẪN DỰ ÁN
# ============================================================

# Thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Thư mục models
MODELS_DIR = PROJECT_ROOT / "models"

# Thư mục reports
REPORTS_DIR = PROJECT_ROOT / "reports"

# Thư mục assets
ASSETS_DIR = PROJECT_ROOT / "assets"

# ============================================================
# FILE PATHS
# ============================================================

# Raw data files
STUDENT_PROFILES_FILE = RAW_DATA_DIR / "student_profiles_sample.csv"
STUDENT_SCORES_FILE = RAW_DATA_DIR / "student_scores_sample.csv"
STUDENT_COMMENTS_FILE = RAW_DATA_DIR / "student_comments_sample.csv"

# Processed data
STUDENT_FEATURES_FILE = PROCESSED_DATA_DIR / "student_features.csv"

# Subject config
SUBJECT_CONFIG_FILE = DATA_DIR / "subject_config.json"

# Model files
DECISION_TREE_MODEL_FILE = MODELS_DIR / "decision_tree.pkl"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"

# Report files
METRICS_FILE = REPORTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_FILE = REPORTS_DIR / "classification_report.txt"
BATCH_PREDICTIONS_FILE = REPORTS_DIR / "batch_predictions.csv"

# ============================================================
# NHÃN PHÂN LOẠI
# ============================================================

CLASS_LABELS = ["Tốt", "Khá", "Đạt", "Chưa đạt"]
TARGET_COLUMN = "learning_result_label"

# ============================================================
# FEATURE COLUMNS CHO MACHINE LEARNING
# ============================================================

FEATURE_COLUMNS = [
    "avg_score",
    "min_score",
    "max_score",
    "std_score",
    "count_score_ge_8",
    "count_score_ge_6_5",
    "count_score_ge_5",
    "count_score_lt_3_5",
    "comment_not_pass_count",
    "attendance_rate",
    "assignment_completion_rate",
    "participation_score",
    "behavior_score",
    "teacher_evaluation_score",
    "progress_delta",
    "is_mid_semester",
]

# ============================================================
# CẤU HÌNH MÔN HỌC
# ============================================================

# Danh sách 8 môn đánh giá bằng điểm số
SCORE_SUBJECTS = [
    "Toán",
    "Ngữ văn",
    "Ngoại ngữ",
    "Khoa học tự nhiên",
    "Lịch sử và Địa lí",
    "Tin học",
    "Công nghệ",
    "Giáo dục công dân",
]

# Danh sách 3 môn đánh giá bằng nhận xét
COMMENT_SUBJECTS = [
    "Giáo dục thể chất",
    "Nghệ thuật",
    "Hoạt động trải nghiệm hướng nghiệp",
]

# ============================================================
# CẤU HÌNH MACHINE LEARNING
# ============================================================

# Decision Tree hyperparameters
DT_PARAMS = {
    "criterion": "gini",
    "max_depth": 4,
    "min_samples_split": 10,
    "class_weight": "balanced",
    "random_state": 42,
}

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "class_weight": "balanced",
    "random_state": 42,
}

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================
# CẤU HÌNH DỮ LIỆU MÔ PHỎNG
# ============================================================

SCHOOL_YEAR = "2025-2026"
SEMESTERS = ["HK1", "HK2"]
CLASSES = [
    "6A1", "6A2", "6A3",
    "7A1", "7A2", "7A3",
    "8A1", "8A2", "8A3",
    "9A1", "9A2", "9A3"
]
NUM_STUDENTS = 1000
NUM_REGULAR_SCORES = 3  # Số điểm thường xuyên mỗi môn

# ============================================================
# VALIDATION RANGES
# ============================================================

SCORE_RANGE = (0.0, 10.0)
PERCENTAGE_RANGE = (0.0, 100.0)
RUBRIC_RANGE = (1.0, 10.0)
PROGRESS_DELTA_RANGE = (-10.0, 10.0)

# ============================================================
# SCHEMA DEFINITIONS
# ============================================================

STUDENT_PROFILES_COLUMNS = [
    "student_id", "student_name", "class_name", "school_year", "semester",
    "total_sessions", "attended_sessions", "attendance_rate",
    "total_assignments", "submitted_assignments", "assignment_completion_rate",
    "participation_score", "behavior_score", "teacher_evaluation_score",
    "previous_average_score", "current_average_score", "progress_delta",
]

STUDENT_SCORES_COLUMNS = [
    "student_id", "class_name", "school_year", "semester",
    "subject_name", "assessment_type", "regular_scores",
    "midterm_score", "final_score", "dtb_mhk",
]

STUDENT_COMMENTS_COLUMNS = [
    "student_id", "class_name", "school_year", "semester",
    "subject_name", "assessment_type", "comment_status",
]

STUDENT_FEATURES_COLUMNS = [
    "student_id", "class_name", "school_year", "period",
    "num_score_subjects", "num_comment_subjects",
    "avg_score", "min_score", "max_score", "std_score",
    "count_score_ge_8", "count_score_ge_6_5", "count_score_ge_5", "count_score_lt_3_5",
    "comment_pass_count", "comment_not_pass_count",
    "attendance_rate", "assignment_completion_rate",
    "participation_score", "behavior_score", "teacher_evaluation_score",
    "progress_delta", "learning_result_label",
]


def load_subject_config() -> dict:
    """Load cấu hình môn học từ file JSON."""
    if SUBJECT_CONFIG_FILE.exists():
        with open(SUBJECT_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback nếu file chưa tồn tại
    return {
        "score_subjects": SCORE_SUBJECTS,
        "comment_subjects": COMMENT_SUBJECTS,
        "school_year": SCHOOL_YEAR,
        "semesters": SEMESTERS,
        "classes": CLASSES,
        "num_students": NUM_STUDENTS,
        "num_regular_scores": NUM_REGULAR_SCORES,
    }


def ensure_directories() -> None:
    """Tạo các thư mục cần thiết nếu chưa tồn tại."""
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, ASSETS_DIR / "screenshots"]:
        dir_path.mkdir(parents=True, exist_ok=True)
