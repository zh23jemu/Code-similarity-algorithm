from pathlib import Path

import pandas as pd

from similarity.cli import build_parser
from similarity.data import Submission
from similarity.modeling import predict_pair_similarity, train_model
from similarity.pairs import build_pair_frame, iter_activity_pairs


def test_training_pipeline_on_small_samples(tmp_path: Path):
    submissions = [
        Submission("1", "u1", "q1", "class A { int f(){ return 1; } }"),
        Submission("2", "u2", "q1", "class B { int f(){ return 2; } }"),
        Submission("3", "u3", "q1", "class C { int f(){ return 3; } }"),
        Submission("4", "u4", "q1", "class D { void g(){ for(int i=0;i<5;i++){ System.out.println(i); } } }"),
        Submission("5", "u5", "q1", "class E { void h(){ while(true){ break; } } }"),
        Submission("6", "u6", "q1", "class F { String s(){ return \"abc\"; } }"),
    ]
    pairs = iter_activity_pairs(submissions, max_pairs_per_question=10)
    frame = build_pair_frame(pairs, high_threshold=0.75, low_threshold=0.65)

    result = train_model(frame, tmp_path, model_type="logistic_regression")

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    prediction = predict_pair_similarity(submissions[0].code, submissions[1].code, result.model_path)
    assert 0.0 <= prediction["model_similarity"] <= 1.0


def test_cross_user_pair_filter_excludes_same_user():
    submissions = [
        Submission("1", "u1", "q1", "class A {}"),
        Submission("2", "u1", "q1", "class B {}"),
        Submission("3", "u2", "q1", "class C {}"),
    ]

    pairs = iter_activity_pairs(submissions, cross_user_only=True)

    assert pairs
    assert all(pair.left.user_id != pair.right.user_id for pair in pairs)


def test_review_cli_accepts_stratified_arguments(tmp_path: Path):
    source = tmp_path / "scored_pairs.csv"
    pd.DataFrame(
        [
            {"activity": "q1", "left_submission_id": "1", "right_submission_id": "2", "left_user_id": "u1", "right_user_id": "u2", "model_similarity": 0.95, "base_similarity": 0.90},
            {"activity": "q1", "left_submission_id": "3", "right_submission_id": "4", "left_user_id": "u3", "right_user_id": "u4", "model_similarity": 0.70, "base_similarity": 0.66},
            {"activity": "q1", "left_submission_id": "5", "right_submission_id": "6", "left_user_id": "u5", "right_user_id": "u6", "model_similarity": 0.30, "base_similarity": 0.28},
        ]
    ).to_csv(source, index=False)
    output = tmp_path / "manual_review_sample.csv"

    parser = build_parser()
    args = parser.parse_args(
        [
            "create-review-sample",
            "--similar-pairs",
            str(source),
            "--output",
            str(output),
            "--high-k",
            "1",
            "--medium-k",
            "1",
            "--low-k",
            "1",
        ]
    )
    args.func(args)

    sample = pd.read_csv(output)
    assert len(sample) == 3
    assert set(sample["review_stratum"]) == {"high", "medium", "low"}
