from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_pair_features


@dataclass(frozen=True)
class TrainingResult:
    """训练阶段的主要产物路径和指标。"""

    model_path: Path
    metrics_path: Path
    metrics: dict[str, object]


def create_model(model_type: str = "random_forest", random_state: int = 42):
    """创建传统机器学习二分类模型。

    RandomForest 对特征尺度不敏感，作为默认模型更稳健；LogisticRegression
    使用标准化流水线，便于论文中解释线性基线效果。
    """

    if model_type == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"不支持的模型类型: {model_type}")


def _positive_probabilities(model, features: pd.DataFrame) -> list[float]:
    """获取相似类别的预测概率。

    少数极端训练集可能只包含单一类别，此时模型没有完整两列概率，
    这里做兼容处理，让调用方得到稳定的 0/1 概率列表。
    """

    probabilities = model.predict_proba(features)
    classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["classifier"].classes_)
    if 1 not in classes:
        return [0.0 for _ in range(len(features))]
    positive_index = classes.index(1)
    return [float(row[positive_index]) for row in probabilities]


def train_model(
    pair_frame: pd.DataFrame,
    output_dir: str | Path,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> TrainingResult:
    """基于弱监督样本训练二分类模型，并保存模型和评估指标。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    labeled = pair_frame.dropna(subset=["label"]).copy()
    if labeled.empty:
        raise ValueError("没有可训练的弱监督样本，请降低阈值或扩大样本量。")
    labeled["label"] = labeled["label"].astype(int)
    if labeled["label"].nunique() < 2:
        raise ValueError("训练样本只有单一类别，请调整 high/low 阈值或增加数据。")

    x = labeled[FEATURE_NAMES]
    y = labeled["label"]
    # 小样本场景下如果直接按 25% 切分，训练集可能只剩单一类别，
    # 因此测试集大小会受类别数量保护，保证训练集至少包含两类样本。
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    if len(labeled) < 8 or class_counts.min() < 2:
        test_size = 0.5 if len(labeled) >= 4 else 1
    else:
        test_size = 0.25
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model = create_model(model_type=model_type, random_state=random_state)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = _positive_probabilities(model, x_test)
    metrics: dict[str, object] = {
        "model_type": model_type,
        "samples": int(len(labeled)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "positive_samples": int((y == 1).sum()),
        "negative_samples": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities)) if len(set(y_test)) == 2 else None

    model_path = output_path / "model.joblib"
    metrics_path = output_path / "metrics.json"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainingResult(model_path=model_path, metrics_path=metrics_path, metrics=metrics)


def load_model(model_path: str | Path):
    """加载训练后的模型包，包含模型对象和特征名。"""

    package = joblib.load(model_path)
    return package["model"], package.get("feature_names", FEATURE_NAMES)


def predict_pair_similarity(code_a: str, code_b: str, model_path: str | Path) -> dict[str, float]:
    """使用训练模型预测两段代码的相似概率。"""

    model, feature_names = load_model(model_path)
    features = extract_pair_features(code_a, code_b)
    frame = pd.DataFrame([features.as_dict()])[feature_names]
    probability = _positive_probabilities(model, frame)[0]
    result = features.as_dict()
    result["model_similarity"] = probability
    return result


def score_pair_frame(pair_frame: pd.DataFrame, model_path: str | Path) -> pd.DataFrame:
    """给特征表批量追加模型相似度分数。"""

    model, feature_names = load_model(model_path)
    scored = pair_frame.copy()
    scored["model_similarity"] = _positive_probabilities(model, scored[feature_names])
    return scored
