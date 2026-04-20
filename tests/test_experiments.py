from pathlib import Path

from similarity.data import Submission
from similarity.experiments import run_comparison_experiment
from similarity.pairs import build_pair_frame, iter_activity_pairs


def test_comparison_experiment_writes_metrics(tmp_path: Path):
    submissions = [
        Submission("1", "u1", "q1", "class A { int f(){ return 1; } }"),
        Submission("2", "u2", "q1", "class B { int f(){ return 2; } }"),
        Submission("3", "u3", "q1", "class C { int f(){ return 3; } }"),
        Submission("4", "u4", "q1", "class D { void g(){ for(int i=0;i<5;i++){ System.out.println(i); } } }"),
        Submission("5", "u5", "q1", "class E { void h(){ while(true){ break; } } }"),
        Submission("6", "u6", "q1", "class F { String s(){ return \"abc\"; } }"),
    ]
    pairs = iter_activity_pairs(submissions, max_pairs_per_question=15)
    frame = build_pair_frame(pairs, high_threshold=0.75, low_threshold=0.65)

    summary = run_comparison_experiment(frame, tmp_path, baseline_threshold=0.75)

    assert set(summary["method"]) == {"rule_baseline", "logistic_regression", "random_forest"}
    assert (tmp_path / "comparison_metrics.csv").exists()
    assert (tmp_path / "comparison_metrics.json").exists()
