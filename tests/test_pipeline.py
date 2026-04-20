from pathlib import Path

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
