"""
preprocessing.py — Tiền xử lý dữ liệu cho Machine Learning.

Module này xử lý:
- Validate schema (kiểm tra đủ cột)
- Validate ranges (kiểm tra phạm vi giá trị)
- Handle missing values
- Prepare features (X) và target (y) cho training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import (
    FEATURE_COLUMNS, TARGET_COLUMN, STUDENT_FEATURES_FILE,
    SCORE_RANGE, PERCENTAGE_RANGE, RUBRIC_RANGE, PROGRESS_DELTA_RANGE,
    STUDENT_FEATURES_COLUMNS,
)


def load_features_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load dữ liệu features từ CSV.

    Args:
        filepath: Đường dẫn file (default: student_features.csv).

    Returns:
        DataFrame chứa features.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
    """
    if filepath is None:
        filepath = STUDENT_FEATURES_FILE

    if not filepath.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {filepath}\n"
            "Hãy tạo dataset mẫu trước bằng cách chuyển sang tab 'Dữ liệu'."
        )

    df = pd.read_csv(filepath, encoding="utf-8-sig")
    return df


def validate_schema(
    df: pd.DataFrame,
    expected_columns: Optional[list[str]] = None,
) -> tuple[bool, list[str]]:
    """
    Kiểm tra DataFrame có đủ cột cần thiết.

    Args:
        df: DataFrame cần validate.
        expected_columns: Danh sách cột yêu cầu.

    Returns:
        (is_valid, missing_columns)
    """
    if expected_columns is None:
        expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [col for col in expected_columns if col not in df.columns]
    return len(missing) == 0, missing


def validate_ranges(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Kiểm tra giá trị trong phạm vi cho phép.

    Args:
        df: DataFrame cần validate.

    Returns:
        (is_valid, list_of_warnings)
    """
    warnings = []

    range_checks = {
        "avg_score": SCORE_RANGE,
        "min_score": SCORE_RANGE,
        "max_score": SCORE_RANGE,
        "participation_score": RUBRIC_RANGE,
        "behavior_score": RUBRIC_RANGE,
        "teacher_evaluation_score": RUBRIC_RANGE,
        "attendance_rate": PERCENTAGE_RANGE,
        "assignment_completion_rate": PERCENTAGE_RANGE,
        "progress_delta": PROGRESS_DELTA_RANGE,
    }

    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns:
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                warnings.append(
                    f"Cột '{col}': {len(out_of_range)} giá trị ngoài phạm vi [{min_val}, {max_val}]"
                )

    return len(warnings) == 0, warnings


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý missing values trong DataFrame.

    Chiến lược:
    - Numeric: điền bằng median
    - Categorical: điền bằng mode

    Args:
        df: DataFrame cần xử lý.

    Returns:
        DataFrame đã xử lý missing values.
    """
    df = df.copy()

    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64, float, int]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)

    # Target: drop rows nếu missing
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])

    return df


def prepare_features_target(
    df: Optional[pd.DataFrame] = None,
    filepath: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Chuẩn bị X (features) và y (target) cho training.

    Args:
        df: DataFrame đầu vào (nếu None sẽ load từ file).
        filepath: Đường dẫn file CSV.

    Returns:
        (X, y) — DataFrame features và Series target.

    Raises:
        ValueError: Nếu thiếu cột hoặc dữ liệu rỗng.
    """
    if df is None:
        df = load_features_data(filepath)

    # Validate schema
    is_valid, missing = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Thiếu các cột: {missing}")

    # Handle missing values
    df = handle_missing_values(df)

    if df.empty:
        raise ValueError("DataFrame rỗng sau khi xử lý missing values.")

    # Extract X and y
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Validate ranges (chỉ cảnh báo, không block)
    is_valid_range, range_warnings = validate_ranges(X)
    if not is_valid_range:
        for w in range_warnings:
            print(f"⚠️ Warning: {w}")

    return X, y
