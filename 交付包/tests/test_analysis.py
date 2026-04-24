from pathlib import Path

import pandas as pd

from similarity.analysis import build_paper_tables


def test_build_paper_tables_writes_expected_files(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    pd.DataFrame(
        [{"Question (activity)": "q1", "Number of submissions": 10}]
    ).to_csv(data_dir / "distribution_per_question.csv", index=False)

    pd.DataFrame([{"label": 1}, {"label": 0}, {"label": 1}]).to_csv(tmp_path / "training.csv", index=False)
    pd.DataFrame([{"method": "rf", "accuracy": 1.0}]).to_csv(tmp_path / "compare.csv", index=False)
    pd.DataFrame([{"activity": "q1"}, {"activity": "q1"}, {"activity": "q2"}]).to_csv(tmp_path / "similar.csv", index=False)
    pd.DataFrame([{"指标": "抽样总数", "数值": 100}]).to_csv(tmp_path / "review.csv", index=False)

    manifest = build_paper_tables(
        data_dir=data_dir,
        training_pairs_path=tmp_path / "training.csv",
        comparison_metrics_path=tmp_path / "compare.csv",
        similar_pairs_path=tmp_path / "similar.csv",
        review_summary_path=tmp_path / "review.csv",
        output_dir=tmp_path / "tables",
    )

    assert "question_distribution" in manifest
    assert (tmp_path / "tables" / "table_model_comparison.csv").exists()
