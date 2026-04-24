from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .experiments import _positive_probabilities
from .features import FEATURE_NAMES
from .modeling import create_model


ABLATION_SETS = {
    "full_features": FEATURE_NAMES,
    "without_base_similarity": [name for name in FEATURE_NAMES if name != "base_similarity"],
    "without_token_jaccard": [name for name in FEATURE_NAMES if name != "token_jaccard"],
    "without_sequence_similarity": [name for name in FEATURE_NAMES if name != "token_sequence_similarity"],
    "without_structure_features": [
        name
        for name in FEATURE_NAMES
        if name not in {"structure_cosine", "method_count_similarity", "loop_count_similarity", "branch_count_similarity"}
    ],
    "base_similarity_only": ["base_similarity"],
}


def _metrics(y_true, y_pred, y_score) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_ablation_experiment(
    pair_frame: pd.DataFrame,
    output_dir: str | Path,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> pd.DataFrame:
    """运行特征消融实验。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labeled = pair_frame.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)
    if labeled.empty or labeled["label"].nunique() < 2:
        raise ValueError("消融实验至少需要同时包含正负两类样本。")

    activities = sorted(labeled["activity"].unique())
    split_index = max(1, int(len(activities) * 0.7))
    train_activities = set(activities[:split_index])
    test_activities = set(activities[split_index:])
    if not test_activities:
        raise ValueError("题目数量不足，无法按题目划分训练/测试集合。")

    train_frame = labeled[labeled["activity"].isin(train_activities)].copy()
    test_frame = labeled[labeled["activity"].isin(test_activities)].copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("按题目划分后训练集或测试集为空。")

    rows: list[dict[str, object]] = []
    for ablation_name, feature_names in ABLATION_SETS.items():
        model = create_model(model_type=model_type, random_state=random_state)
        model.fit(train_frame[feature_names], train_frame["label"])
        predictions = model.predict(test_frame[feature_names])
        probabilities = _positive_probabilities(model, test_frame[feature_names])
        metrics = _metrics(test_frame["label"], predictions, probabilities)
        rows.append(
            {
                "ablation": ablation_name,
                "model_type": model_type,
                "feature_count": len(feature_names),
                "train_activities": len(train_activities),
                "test_activities": len(test_activities),
                "train_samples": int(len(train_frame)),
                "test_samples": int(len(test_frame)),
                **metrics,
            }
        )
        joblib.dump(
            {"model": model, "feature_names": feature_names},
            output / f"{ablation_name}_{model_type}.joblib",
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "ablation_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_question_split_experiment(
    pair_frame: pd.DataFrame,
    output_dir: str | Path,
    model_type: str = "random_forest",
    random_state: int = 42,
    include_base_similarity: bool = False,
) -> dict[str, object]:
    """运行更严格的按题目划分泛化实验。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labeled = pair_frame.dropna(subset=["label"]).copy()
    labeled["label"] = labeled["label"].astype(int)
    if labeled.empty or labeled["label"].nunique() < 2:
        raise ValueError("严格泛化实验至少需要同时包含正负两类样本。")

    feature_names = FEATURE_NAMES if include_base_similarity else [name for name in FEATURE_NAMES if name != "base_similarity"]
    activities = sorted(labeled["activity"].unique())
    split_index = max(1, int(len(activities) * 0.7))
    train_activities = set(activities[:split_index])
    test_activities = set(activities[split_index:])
    if not test_activities:
        raise ValueError("题目数量不足，无法按题目划分训练/测试集合。")

    train_frame = labeled[labeled["activity"].isin(train_activities)].copy()
    test_frame = labeled[labeled["activity"].isin(test_activities)].copy()
    model = create_model(model_type=model_type, random_state=random_state)
    model.fit(train_frame[feature_names], train_frame["label"])
    predictions = model.predict(test_frame[feature_names])
    probabilities = _positive_probabilities(model, test_frame[feature_names])
    result = {
        "model_type": model_type,
        "include_base_similarity": include_base_similarity,
        "feature_count": len(feature_names),
        "train_activities": len(train_activities),
        "test_activities": len(test_activities),
        "train_samples": int(len(train_frame)),
        "test_samples": int(len(test_frame)),
        **_metrics(test_frame["label"], predictions, probabilities),
    }
    joblib.dump({"model": model, "feature_names": feature_names}, output / "question_split_model.joblib")
    (output / "question_split_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
