from pathlib import Path

from similarity.cli import build_parser
from similarity.data import Submission
from similarity.modeling import train_model
from similarity.pairs import build_pair_frame, iter_activity_pairs
from similarity.webapp import build_compare_payload, build_status_payload


def test_build_status_payload_reads_output_dir(tmp_path: Path):
    """验证前端页面状态接口会从指定输出目录读取模型和参数。"""

    output_dir = tmp_path / "outputs_demo"
    output_dir.mkdir()

    (output_dir / "metrics.json").write_text(
        '{"model_type":"random_forest","samples":5000,"positive_samples":1250,"negative_samples":3780,"accuracy":0.98,"f1":0.97}',
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        '{"data_dir":"dataset","model_type":"random_forest","high_threshold":0.90,"low_threshold":0.75,"similarity_threshold":0.90,"max_pairs_per_question":2000}',
        encoding="utf-8",
    )
    (output_dir / "model.joblib").write_bytes(b"demo")

    payload = build_status_payload(tmp_path, output_dir="outputs_demo")

    assert payload["outputDir"] == "outputs_demo"
    assert payload["modelLoaded"] is True
    assert payload["summaryItems"][0]["value"] == 1250
    assert payload["configItems"][0]["value"] == "dataset"


def test_build_compare_payload_returns_similarity_scores(tmp_path: Path):
    """验证前端检测接口会返回模型相似度与各项特征分数。"""

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

    payload = build_compare_payload(
        "class SumA { int sum(int n){ int s=0; for(int i=1;i<=n;i++){ s+=i; } return s; } }",
        "class SumB { int sum(int n){ int result=0; for(int i=1;i<=n;i++){ result=result+i; } return result; } }",
        result.model_path,
    )

    assert 0.0 <= payload["summary"]["model_similarity"] <= 1.0
    assert payload["features"]
    assert any(item["label"] == "基础规则相似度" for item in payload["features"])


def test_cli_registers_serve_webui_command():
    """验证命令行已注册新的前端页面启动命令。"""

    parser = build_parser()
    args = parser.parse_args(["serve-webui", "--output-dir", "outputs_cross_user", "--port", "8123"])

    assert args.command == "serve-webui"
    assert args.output_dir == "outputs_cross_user"
    assert args.port == 8123
    assert callable(args.func)
