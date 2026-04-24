from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FEATURE_NAMES
from .modeling import create_model


def _classification_metrics(y_true, y_pred, y_score) -> dict[str, object]:
    """统一计算分类指标。

    对比实验需要让规则基线和机器学习模型使用完全一致的指标口径，
    否则论文表格中的结果不可直接比较。
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _positive_probabilities(model, features: pd.DataFrame) -> list[float]:
    """读取二分类模型中“相似类”的预测概率。"""

    probabilities = model.predict_proba(features)
    classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["classifier"].classes_)
    if 1 not in classes:
        return [0.0 for _ in range(len(features))]
    positive_index = classes.index(1)
    return [float(row[positive_index]) for row in probabilities]


def run_comparison_experiment(
    pair_frame: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
    baseline_threshold: float = 0.85,
) -> pd.DataFrame:
    """运行规则基线、逻辑回归和随机森林的对比实验。

    所有方法复用同一份弱监督样本，并共享同一次训练/测试划分。
    规则基线不训练模型，直接用 `base_similarity` 与阈值判断是否相似；
    两个机器学习模型则使用相同特征训练，方便论文中说明建模收益。
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    labeled = pair_frame.dropna(subset=["label"]).copy()
    if labeled.empty or labeled["label"].nunique() < 2:
        raise ValueError("对比实验至少需要同时包含正负两类弱监督样本。")
    labeled["label"] = labeled["label"].astype(int)

    x = labeled[FEATURE_NAMES]
    y = labeled["label"]
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(
        x,
        y,
        labeled.index,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )

    rows: list[dict[str, object]] = []
    test_base_scores = labeled.loc[test_index, "base_similarity"].astype(float)
    baseline_predictions = (test_base_scores >= baseline_threshold).astype(int)
    baseline_metrics = _classification_metrics(y_test, baseline_predictions, test_base_scores)
    rows.append(
        {
            "method": "rule_baseline",
            "train_samples": 0,
            "test_samples": int(len(y_test)),
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "baseline_threshold": baseline_threshold,
            **baseline_metrics,
        }
    )

    for model_type in ["logistic_regression", "random_forest"]:
        model = create_model(model_type=model_type, random_state=random_state)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = _positive_probabilities(model, x_test)
        metrics = _classification_metrics(y_test, predictions, probabilities)
        rows.append(
            {
                "method": model_type,
                "train_samples": int(len(x_train)),
                "test_samples": int(len(x_test)),
                "positive_samples": int((y == 1).sum()),
                "negative_samples": int((y == 0).sum()),
                "baseline_threshold": "",
                **metrics,
            }
        )
        joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, output_path / f"{model_type}_model.joblib")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    (output_path / "comparison_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
