"""
train_models.py — Huấn luyện mô hình Machine Learning.

Module này implement pipeline huấn luyện:
1. Load processed dataset
2. Validate schema
3. Handle missing values
4. Split X/y
5. train_test_split (test_size=0.2, stratify=y)
6. Train Decision Tree
7. Train Random Forest
8. Evaluate
9. Save models bằng joblib
10. Save metrics
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Optional

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DT_PARAMS, RF_PARAMS, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, REPORTS_DIR,
    DECISION_TREE_MODEL_FILE, RANDOM_FOREST_MODEL_FILE, LABEL_ENCODER_FILE,
    METRICS_FILE, CLASSIFICATION_REPORT_FILE,
    ensure_directories,
)
from src.preprocessing import prepare_features_target
from src.evaluation import evaluate_model, generate_classification_report


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DecisionTreeClassifier:
    """
    Huấn luyện Decision Tree Classifier.

    Args:
        X_train: Features huấn luyện.
        y_train: Labels huấn luyện.

    Returns:
        Mô hình Decision Tree đã train.
    """
    model = DecisionTreeClassifier(**DT_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Huấn luyện Random Forest Classifier.

    Args:
        X_train: Features huấn luyện.
        y_train: Labels huấn luyện.

    Returns:
        Mô hình Random Forest đã train.
    """
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def save_models(
    dt_model: DecisionTreeClassifier,
    rf_model: RandomForestClassifier,
    label_encoder: LabelEncoder,
) -> None:
    """Lưu models và label encoder bằng joblib."""
    ensure_directories()
    joblib.dump(dt_model, DECISION_TREE_MODEL_FILE)
    joblib.dump(rf_model, RANDOM_FOREST_MODEL_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)


def load_models() -> tuple[Optional[DecisionTreeClassifier], Optional[RandomForestClassifier], Optional[LabelEncoder]]:
    """
    Load models đã lưu.

    Returns:
        (dt_model, rf_model, label_encoder) hoặc (None, None, None) nếu chưa có.
    """
    dt_model = None
    rf_model = None
    le = None

    if DECISION_TREE_MODEL_FILE.exists():
        dt_model = joblib.load(DECISION_TREE_MODEL_FILE)
    if RANDOM_FOREST_MODEL_FILE.exists():
        rf_model = joblib.load(RANDOM_FOREST_MODEL_FILE)
    if LABEL_ENCODER_FILE.exists():
        le = joblib.load(LABEL_ENCODER_FILE)

    return dt_model, rf_model, le


def train_all(
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Pipeline huấn luyện đầy đủ: load → split → train → evaluate → save.

    Args:
        df: DataFrame features (nếu None sẽ load từ file).

    Returns:
        Dictionary chứa metrics, models, và thông tin training.
    """
    ensure_directories()

    # 1. Prepare data
    X, y = prepare_features_target(df)

    # 2. Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    # 4. Train models
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 5. Evaluate
    dt_metrics = evaluate_model(dt_model, X_test, y_test, le.classes_, "Decision Tree")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, le.classes_, "Random Forest")

    # 6. Save models
    save_models(dt_model, rf_model, le)

    # 7. Save metrics
    all_metrics = {
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": len(X),
        "num_features": len(X.columns),
        "class_labels": le.classes_.tolist(),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # 8. Save classification report
    report_text = generate_classification_report(
        dt_model, rf_model, X_test, y_test, le.classes_
    )
    with open(CLASSIFICATION_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {
        "dt_model": dt_model,
        "rf_model": rf_model,
        "label_encoder": le,
        "dt_metrics": dt_metrics,
        "rf_metrics": rf_metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "all_metrics": all_metrics,
    }
