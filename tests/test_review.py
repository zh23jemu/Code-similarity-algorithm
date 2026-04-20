from pathlib import Path

import pandas as pd

from similarity.review import create_review_sample, summarize_manual_review


def test_create_review_sample_adds_review_columns(tmp_path: Path):
    source = tmp_path / "similar_pairs.csv"
    output = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {"activity": "q1", "left_submission_id": "1", "right_submission_id": "2", "left_user_id": "u1", "right_user_id": "u2", "model_similarity": 0.99, "base_similarity": 0.95},
            {"activity": "q1", "left_submission_id": "3", "right_submission_id": "4", "left_user_id": "u3", "right_user_id": "u4", "model_similarity": 0.90, "base_similarity": 0.88},
        ]
    ).to_csv(source, index=False)

    sample = create_review_sample(source, output, top_k=1, random_k=1)

    assert output.exists()
    assert len(sample) == 2
    assert {"review_decision", "reviewer", "review_note"}.issubset(sample.columns)


def test_summarize_manual_review_writes_summary_and_analysis(tmp_path: Path):
    source = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {"model_similarity": 0.99, "base_similarity": 0.95, "review_decision": "similar"},
            {"model_similarity": 0.91, "base_similarity": 0.88, "review_decision": "not_similar"},
            {"model_similarity": 0.90, "base_similarity": 0.86, "review_decision": "uncertain"},
            {"model_similarity": 0.89, "base_similarity": 0.85, "review_decision": ""},
        ]
    ).to_csv(source, index=False)

    summary = summarize_manual_review(source, tmp_path)

    assert summary["reviewed_rows"] == 3
    assert summary["pending_rows"] == 1
    assert (tmp_path / "manual_review_summary.csv").exists()
    assert (tmp_path / "manual_review_analysis.md").exists()
