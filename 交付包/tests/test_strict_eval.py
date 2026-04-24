from pathlib import Path

import pandas as pd

from similarity.strict_eval import run_ablation_experiment, run_question_split_experiment


def _sample_frame() -> pd.DataFrame:
    rows = []
    activities = ["a1", "a2", "a3", "a4", "a5", "a6"]
    for idx, activity in enumerate(activities):
        rows.append(
            {
                "activity": activity,
                "label": 1 if idx % 2 == 0 else 0,
                "token_jaccard": 0.95 if idx % 2 == 0 else 0.30,
                "token_sequence_similarity": 0.94 if idx % 2 == 0 else 0.25,
                "keyword_cosine": 0.92 if idx % 2 == 0 else 0.35,
                "operator_cosine": 0.91 if idx % 2 == 0 else 0.40,
                "structure_cosine": 0.90 if idx % 2 == 0 else 0.20,
                "length_similarity": 0.88 if idx % 2 == 0 else 0.50,
                "line_count_similarity": 0.87 if idx % 2 == 0 else 0.45,
                "method_count_similarity": 0.86 if idx % 2 == 0 else 0.40,
                "loop_count_similarity": 0.85 if idx % 2 == 0 else 0.35,
                "branch_count_similarity": 0.84 if idx % 2 == 0 else 0.30,
                "base_similarity": 0.93 if idx % 2 == 0 else 0.28,
            }
        )
    return pd.DataFrame(rows)


def test_run_ablation_experiment_outputs_summary(tmp_path: Path):
    summary = run_ablation_experiment(_sample_frame(), tmp_path)

    assert "full_features" in set(summary["ablation"])
    assert (tmp_path / "ablation_metrics.csv").exists()


def test_run_question_split_experiment_outputs_metrics(tmp_path: Path):
    result = run_question_split_experiment(_sample_frame(), tmp_path, include_base_similarity=False)

    assert "accuracy" in result
    assert (tmp_path / "question_split_metrics.json").exists()
