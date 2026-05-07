from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import build_paper_tables
from .data import inspect_dataset, load_submissions
from .experiments import run_comparison_experiment
from .modeling import predict_pair_similarity, score_pair_frame, train_model
from .pairs import build_pair_frame, iter_activity_pairs
from .report import write_html_report, write_json
from .review import create_review_sample, summarize_manual_review
from .strict_eval import run_ablation_experiment, run_question_split_experiment
from .webapp import run_webui_server


def _print_json(data: dict[str, object]) -> None:
    """命令行统一使用中文友好的 JSON 输出。"""

    print(json.dumps(data, ensure_ascii=False, indent=2))


def command_inspect_data(args: argparse.Namespace) -> None:
    """查看数据集规模和题目分布。"""

    _print_json(inspect_dataset(args.data_dir))


def _build_training_frame(args: argparse.Namespace):
    """从原始提交构造弱监督训练样本。"""

    submissions = load_submissions(args.data_dir)
    pairs = iter_activity_pairs(
        submissions,
        question_id=args.question_id,
        max_pairs_per_question=args.max_pairs_per_question,
        random_state=args.random_state,
        cross_user_only=args.cross_user_only,
    )
    return build_pair_frame(
        pairs,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        include_unlabeled=False,
    )


def command_train(args: argparse.Namespace) -> None:
    """训练模型并保存训练样本、模型和评估指标。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = _build_training_frame(args)
    training_pairs_path = output_dir / "training_pairs.csv"
    pair_frame.to_csv(training_pairs_path, index=False, encoding="utf-8-sig")
    result = train_model(pair_frame, output_dir=output_dir, model_type=args.model_type, random_state=args.random_state)
    write_json(
        output_dir / "run_config.json",
        {
            "command": "train",
            "data_dir": str(args.data_dir),
            "high_threshold": args.high_threshold,
            "low_threshold": args.low_threshold,
            "max_pairs_per_question": args.max_pairs_per_question,
            "model_type": args.model_type,
        },
    )
    _print_json(
        {
            "training_pairs": str(training_pairs_path),
            "model": str(result.model_path),
            "metrics": str(result.metrics_path),
            "metrics_detail": result.metrics,
        }
    )


def command_predict(args: argparse.Namespace) -> None:
    """使用已有模型批量预测同题代码对，并输出高相似结果。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    submissions = load_submissions(args.data_dir)
    pairs = iter_activity_pairs(
        submissions,
        question_id=args.question_id,
        max_pairs_per_question=args.max_pairs_per_question,
        random_state=args.random_state,
        cross_user_only=args.cross_user_only,
    )
    pair_frame = build_pair_frame(pairs, include_unlabeled=True)
    scored = score_pair_frame(pair_frame, args.model)
    scored_path = output_dir / "scored_pairs.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    similar = scored[scored["model_similarity"] >= args.threshold].sort_values("model_similarity", ascending=False)
    similar_path = output_dir / "similar_pairs.csv"
    similar.to_csv(similar_path, index=False, encoding="utf-8-sig")
    write_html_report(
        output_dir / "report.html",
        similar,
        metrics=None,
        top_k=args.top_k,
        run_config={
            "command": "predict",
            "data_dir": str(args.data_dir),
            "model": str(args.model),
            "threshold": args.threshold,
            "max_pairs_per_question": args.max_pairs_per_question,
            "question_id": args.question_id,
            "cross_user_only": args.cross_user_only,
        },
    )
    _print_json({"scored_pairs": str(scored_path), "similar_pairs": str(similar_path), "count": int(len(similar))})


def command_run_pipeline(args: argparse.Namespace) -> None:
    """一键完成样本构造、模型训练、预测和报告生成。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = _build_training_frame(args)
    training_pairs_path = output_dir / "training_pairs.csv"
    pair_frame.to_csv(training_pairs_path, index=False, encoding="utf-8-sig")

    result = train_model(pair_frame, output_dir=output_dir, model_type=args.model_type, random_state=args.random_state)
    scored = score_pair_frame(pair_frame, result.model_path)
    scored_path = output_dir / "scored_pairs.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    similar = scored[scored["model_similarity"] >= args.threshold].sort_values("model_similarity", ascending=False)
    similar_path = output_dir / "similar_pairs.csv"
    similar.to_csv(similar_path, index=False, encoding="utf-8-sig")

    run_config = {
        "command": "run-pipeline",
        "data_dir": str(args.data_dir),
        "high_threshold": args.high_threshold,
        "low_threshold": args.low_threshold,
        "similarity_threshold": args.threshold,
        "max_pairs_per_question": args.max_pairs_per_question,
        "model_type": args.model_type,
        "question_id": args.question_id,
        "cross_user_only": args.cross_user_only,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_html_report(output_dir / "report.html", similar, metrics=result.metrics, top_k=args.top_k, run_config=run_config)
    _print_json(
        {
            "training_pairs": str(training_pairs_path),
            "model": str(result.model_path),
            "metrics": str(result.metrics_path),
            "scored_pairs": str(scored_path),
            "similar_pairs": str(similar_path),
            "report": str(output_dir / "report.html"),
        }
    )


def command_compare_files(args: argparse.Namespace) -> None:
    """比较两个本地 Java 文件，输出模型相似度和分项特征。"""

    code_a = Path(args.left).read_text(encoding=args.encoding)
    code_b = Path(args.right).read_text(encoding=args.encoding)
    _print_json(predict_pair_similarity(code_a, code_b, args.model))


def command_compare_models(args: argparse.Namespace) -> None:
    """运行规则基线、逻辑回归和随机森林的对比实验。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = _build_training_frame(args)
    training_pairs_path = output_dir / "comparison_training_pairs.csv"
    pair_frame.to_csv(training_pairs_path, index=False, encoding="utf-8-sig")
    summary = run_comparison_experiment(
        pair_frame,
        output_dir=output_dir,
        random_state=args.random_state,
        baseline_threshold=args.baseline_threshold,
    )
    write_json(
        output_dir / "comparison_config.json",
        {
            "command": "compare-models",
            "data_dir": str(args.data_dir),
            "high_threshold": args.high_threshold,
            "low_threshold": args.low_threshold,
            "baseline_threshold": args.baseline_threshold,
            "max_pairs_per_question": args.max_pairs_per_question,
            "question_id": args.question_id,
        },
    )
    _print_json(
        {
            "training_pairs": str(training_pairs_path),
            "comparison_csv": str(output_dir / "comparison_metrics.csv"),
            "comparison_json": str(output_dir / "comparison_metrics.json"),
            "methods": summary["method"].tolist(),
        }
    )


def command_create_review_sample(args: argparse.Namespace) -> None:
    """从相似代码对结果中导出人工复核抽样表。"""

    stratified_counts = None
    if args.high_k or args.medium_k or args.low_k:
        stratified_counts = {
            "high": args.high_k,
            "medium": args.medium_k,
            "low": args.low_k,
        }
    sample = create_review_sample(
        similar_pairs_path=args.similar_pairs,
        output_path=args.output,
        top_k=args.top_k,
        random_k=args.random_k,
        random_state=args.random_state,
        score_column=args.score_column,
        stratified_counts=stratified_counts,
    )
    _print_json(
        {
            "review_sample": str(args.output),
            "rows": int(len(sample)),
            "sampling_mode": "stratified" if stratified_counts else "top_and_random",
        }
    )


def command_summarize_review(args: argparse.Namespace) -> None:
    """统计人工复核表，并生成论文分析段落。"""

    summary = summarize_manual_review(args.review_sample, args.output_dir)
    _print_json(
        {
            "summary_csv": str(Path(args.output_dir) / "manual_review_summary.csv"),
            "analysis_md": str(Path(args.output_dir) / "manual_review_analysis.md"),
            "summary": summary,
        }
    )


def command_ablation_experiment(args: argparse.Namespace) -> None:
    """运行特征消融实验。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = _build_training_frame(args)
    pair_frame.to_csv(output_dir / "ablation_training_pairs.csv", index=False, encoding="utf-8-sig")
    summary = run_ablation_experiment(pair_frame, output_dir=output_dir, model_type=args.model_type, random_state=args.random_state)
    _print_json(
        {
            "ablation_csv": str(output_dir / "ablation_metrics.csv"),
            "ablation_json": str(output_dir / "ablation_metrics.json"),
            "ablations": summary["ablation"].tolist(),
        }
    )


def command_question_split_experiment(args: argparse.Namespace) -> None:
    """运行按题目划分的严格泛化实验。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = _build_training_frame(args)
    pair_frame.to_csv(output_dir / "question_split_training_pairs.csv", index=False, encoding="utf-8-sig")
    result = run_question_split_experiment(
        pair_frame,
        output_dir=output_dir,
        model_type=args.model_type,
        random_state=args.random_state,
        include_base_similarity=args.include_base_similarity,
    )
    _print_json({"metrics_json": str(output_dir / "question_split_metrics.json"), "result": result})


def command_build_paper_tables(args: argparse.Namespace) -> None:
    """生成论文用统计表。"""

    manifest = build_paper_tables(
        data_dir=args.data_dir,
        training_pairs_path=args.training_pairs,
        comparison_metrics_path=args.comparison_metrics,
        similar_pairs_path=args.similar_pairs,
        review_summary_path=args.review_summary,
        output_dir=args.output_dir,
    )
    _print_json(manifest)


def command_serve_webui(args: argparse.Namespace) -> None:
    """启动本地前端检测页面，支持输入两段代码后一键检测相似度。"""

    run_webui_server(
        root_dir=Path.cwd(),
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
    )


def command_reproduce_paper(args: argparse.Namespace) -> None:
    """一键复现论文主要实验与统计表。"""

    base_output = Path(args.output_dir)
    cross_user_output = base_output / "cross_user_pipeline"
    compare_output = base_output / "cross_user_compare"
    ablation_output = base_output / "ablation"
    question_split_output = base_output / "question_split"
    tables_output = base_output / "paper_tables"

    pipeline_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=str(cross_user_output),
        question_id=args.question_id,
        max_pairs_per_question=args.max_pairs_per_question,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        model_type=args.model_type,
        random_state=args.random_state,
        threshold=args.threshold,
        top_k=args.top_k,
        cross_user_only=True,
    )
    command_run_pipeline(pipeline_args)

    compare_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=str(compare_output),
        question_id=args.question_id,
        max_pairs_per_question=args.max_pairs_per_question,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        model_type=args.model_type,
        random_state=args.random_state,
        baseline_threshold=args.high_threshold,
        cross_user_only=True,
    )
    command_compare_models(compare_args)

    review_sample_args = argparse.Namespace(
        similar_pairs=str(cross_user_output / "similar_pairs.csv"),
        output=str(cross_user_output / "manual_review_sample.csv"),
        top_k=50,
        random_k=50,
        random_state=args.random_state,
    )
    command_create_review_sample(review_sample_args)

    summarize_review_args = argparse.Namespace(
        review_sample=str(cross_user_output / "manual_review_sample.csv"),
        output_dir=str(cross_user_output),
    )
    command_summarize_review(summarize_review_args)

    ablation_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=str(ablation_output),
        question_id=args.question_id,
        max_pairs_per_question=args.max_pairs_per_question,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        model_type=args.model_type,
        random_state=args.random_state,
        cross_user_only=True,
    )
    command_ablation_experiment(ablation_args)

    question_split_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=str(question_split_output),
        question_id=args.question_id,
        max_pairs_per_question=args.max_pairs_per_question,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        model_type=args.model_type,
        random_state=args.random_state,
        cross_user_only=True,
        include_base_similarity=False,
    )
    command_question_split_experiment(question_split_args)

    table_args = argparse.Namespace(
        data_dir=args.data_dir,
        training_pairs=str(cross_user_output / "training_pairs.csv"),
        comparison_metrics=str(compare_output / "comparison_metrics.csv"),
        similar_pairs=str(cross_user_output / "similar_pairs.csv"),
        review_summary=str(cross_user_output / "manual_review_summary.csv"),
        output_dir=str(tables_output),
    )
    command_build_paper_tables(table_args)

    # 在人工复核摘要和论文统计表准备完成后，重新生成一次答辩看板，
    # 让最终 report.html 能展示更完整的复核与分布信息。
    similar_frame = _read_csv(cross_user_output / "similar_pairs.csv")
    metrics_data = json.loads((cross_user_output / "metrics.json").read_text(encoding="utf-8"))
    run_config = json.loads((cross_user_output / "run_config.json").read_text(encoding="utf-8"))
    write_html_report(
        cross_user_output / "report.html",
        similar_frame,
        metrics=metrics_data,
        top_k=args.top_k,
        run_config=run_config,
    )

    _print_json(
        {
            "pipeline_dir": str(cross_user_output),
            "compare_dir": str(compare_output),
            "ablation_dir": str(ablation_output),
            "question_split_dir": str(question_split_output),
            "paper_tables_dir": str(tables_output),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(prog="similarity", description="基于弱监督训练建模的 Java 代码相似度检测工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="查看数据集概况")
    inspect_parser.add_argument("--data-dir", default="dataset")
    inspect_parser.set_defaults(func=command_inspect_data)

    train_parser = subparsers.add_parser("train", help="构造弱监督样本并训练模型")
    _add_common_training_args(train_parser)
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="使用已有模型批量预测相似代码对")
    _add_common_prediction_args(predict_parser)
    predict_parser.set_defaults(func=command_predict)

    pipeline_parser = subparsers.add_parser("run-pipeline", help="一键完成训练、预测和报告生成")
    _add_common_training_args(pipeline_parser)
    pipeline_parser.add_argument("--threshold", type=float, default=0.85, help="输出高相似代码对的模型概率阈值")
    pipeline_parser.add_argument("--top-k", type=int, default=100, help="HTML 报告展示的最高相似代码对数量")
    pipeline_parser.set_defaults(func=command_run_pipeline)

    compare_models_parser = subparsers.add_parser("compare-models", help="对比规则基线、逻辑回归和随机森林")
    _add_common_training_args(compare_models_parser)
    compare_models_parser.add_argument("--baseline-threshold", type=float, default=0.85, help="规则基线判定为相似的阈值")
    compare_models_parser.set_defaults(func=command_compare_models)

    compare_parser = subparsers.add_parser("compare-files", help="比较两个 Java 文件")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    compare_parser.add_argument("--model", default="outputs/model.joblib")
    compare_parser.add_argument("--encoding", default="utf-8")
    compare_parser.set_defaults(func=command_compare_files)

    review_parser = subparsers.add_parser("create-review-sample", help="从高相似结果中生成可人工复核的抽样表")
    review_parser.add_argument("--similar-pairs", default="outputs/similar_pairs.csv")
    review_parser.add_argument("--output", default="outputs/manual_review_sample.csv")
    review_parser.add_argument("--top-k", type=int, default=50)
    review_parser.add_argument("--random-k", type=int, default=50)
    review_parser.add_argument("--high-k", type=int, default=0, help="分层抽样时高分层样本数，通常建议输入 scored_pairs.csv")
    review_parser.add_argument("--medium-k", type=int, default=0, help="分层抽样时中分层样本数")
    review_parser.add_argument("--low-k", type=int, default=0, help="分层抽样时低分层样本数")
    review_parser.add_argument("--score-column", default="model_similarity", help="分层抽样使用的分数字段")
    review_parser.add_argument("--random-state", type=int, default=42)
    review_parser.set_defaults(func=command_create_review_sample)

    summarize_review_parser = subparsers.add_parser("summarize-review", help="统计人工复核结果并生成论文分析段落")
    summarize_review_parser.add_argument("--review-sample", default="outputs/manual_review_sample.csv")
    summarize_review_parser.add_argument("--output-dir", default="outputs")
    summarize_review_parser.set_defaults(func=command_summarize_review)

    ablation_parser = subparsers.add_parser("ablation-experiment", help="运行特征消融实验")
    _add_common_training_args(ablation_parser)
    ablation_parser.set_defaults(func=command_ablation_experiment)

    question_split_parser = subparsers.add_parser("question-split-experiment", help="运行按题目划分的严格泛化实验")
    _add_common_training_args(question_split_parser)
    question_split_parser.add_argument("--include-base-similarity", action="store_true", help="严格实验中保留 base_similarity 特征")
    question_split_parser.set_defaults(func=command_question_split_experiment)

    table_parser = subparsers.add_parser("build-paper-tables", help="生成论文用统计表")
    table_parser.add_argument("--data-dir", default="dataset")
    table_parser.add_argument("--training-pairs", default="outputs_cross_user/training_pairs.csv")
    table_parser.add_argument("--comparison-metrics", default="outputs_cross_user_compare/comparison_metrics.csv")
    table_parser.add_argument("--similar-pairs", default="outputs_cross_user/similar_pairs.csv")
    table_parser.add_argument("--review-summary", default="outputs_cross_user/manual_review_summary.csv")
    table_parser.add_argument("--output-dir", default="paper_tables")
    table_parser.set_defaults(func=command_build_paper_tables)

    serve_webui_parser = subparsers.add_parser("serve-webui", help="启动本地代码相似度检测前端页面")
    serve_webui_parser.add_argument("--output-dir", default="outputs", help="默认读取的输出目录")
    serve_webui_parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    serve_webui_parser.add_argument("--port", type=int, default=8000, help="服务端口")
    serve_webui_parser.set_defaults(func=command_serve_webui)

    # 兼容之前的命令名，避免已有使用方式失效。
    serve_dashboard_parser = subparsers.add_parser("serve-dashboard", help="兼容旧命令名，启动本地代码相似度检测前端页面")
    serve_dashboard_parser.add_argument("--output-dir", default="outputs", help="默认读取的输出目录")
    serve_dashboard_parser.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    serve_dashboard_parser.add_argument("--port", type=int, default=8000, help="服务端口")
    serve_dashboard_parser.set_defaults(func=command_serve_webui)

    reproduce_parser = subparsers.add_parser("reproduce-paper", help="一键复现论文主要实验与统计表")
    _add_common_training_args(reproduce_parser)
    reproduce_parser.add_argument("--threshold", type=float, default=0.90)
    reproduce_parser.add_argument("--top-k", type=int, default=100)
    reproduce_parser.set_defaults(func=command_reproduce_paper)
    return parser


def _read_csv(path: str | Path):
    """读取命令行阶段生成的 CSV，用于组合后续报告流程。"""

    return pd.read_csv(path, encoding="utf-8-sig")


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """训练和流水线共用参数。"""

    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--question-id", default=None, help="只处理指定题目 activity")
    parser.add_argument("--max-pairs-per-question", type=int, default=2000)
    parser.add_argument("--high-threshold", type=float, default=0.85)
    parser.add_argument("--low-threshold", type=float, default=0.35)
    parser.add_argument("--model-type", choices=["random_forest", "logistic_regression"], default="random_forest")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cross-user-only", action="store_true", help="只构造不同用户之间的代码对")


def _add_common_prediction_args(parser: argparse.ArgumentParser) -> None:
    """预测命令共用参数。"""

    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", default="outputs/model.joblib")
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--max-pairs-per-question", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cross-user-only", action="store_true", help="只预测不同用户之间的代码对")


def main(argv: list[str] | None = None) -> None:
    """命令行主入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
