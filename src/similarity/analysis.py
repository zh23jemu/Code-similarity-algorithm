from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_paper_tables(
    data_dir: str | Path,
    training_pairs_path: str | Path,
    comparison_metrics_path: str | Path,
    similar_pairs_path: str | Path,
    review_summary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    """生成论文可直接使用的统计表。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    distribution = pd.read_csv(Path(data_dir) / "distribution_per_question.csv")
    distribution = distribution.sort_values("Number of submissions", ascending=False)
    distribution.to_csv(output / "table_question_distribution.csv", index=False, encoding="utf-8-sig")

    training_pairs = pd.read_csv(training_pairs_path)
    label_counts = training_pairs["label"].value_counts().rename_axis("label").reset_index(name="count")
    label_counts.to_csv(output / "table_label_distribution.csv", index=False, encoding="utf-8-sig")

    comparison_metrics = pd.read_csv(comparison_metrics_path)
    comparison_metrics.to_csv(output / "table_model_comparison.csv", index=False, encoding="utf-8-sig")

    similar_pairs = pd.read_csv(similar_pairs_path)
    similar_activity_distribution = (
        similar_pairs["activity"].value_counts().rename_axis("activity").reset_index(name="high_similarity_pairs")
    )
    similar_activity_distribution.to_csv(
        output / "table_high_similarity_activity_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    review_summary = pd.read_csv(review_summary_path)
    review_summary.to_csv(output / "table_manual_review_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "question_distribution": str(output / "table_question_distribution.csv"),
        "label_distribution": str(output / "table_label_distribution.csv"),
        "model_comparison": str(output / "table_model_comparison.csv"),
        "high_similarity_activity_distribution": str(output / "table_high_similarity_activity_distribution.csv"),
        "manual_review_summary": str(output / "table_manual_review_summary.csv"),
    }
    (output / "tables_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
