from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import inspect_dataset, load_submissions
from .modeling import predict_pair_similarity, score_pair_frame, train_model
from .pairs import build_pair_frame, iter_activity_pairs
from .report import write_html_report, write_json


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
    )
    pair_frame = build_pair_frame(pairs, include_unlabeled=True)
    scored = score_pair_frame(pair_frame, args.model)
    similar = scored[scored["model_similarity"] >= args.threshold].sort_values("model_similarity", ascending=False)
    similar_path = output_dir / "similar_pairs.csv"
    similar.to_csv(similar_path, index=False, encoding="utf-8-sig")
    write_html_report(output_dir / "report.html", similar, metrics=None, top_k=args.top_k)
    _print_json({"similar_pairs": str(similar_path), "count": int(len(similar))})


def command_run_pipeline(args: argparse.Namespace) -> None:
    """一键完成样本构造、模型训练、预测和报告生成。"""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = _build_training_frame(args)
    training_pairs_path = output_dir / "training_pairs.csv"
    pair_frame.to_csv(training_pairs_path, index=False, encoding="utf-8-sig")

    result = train_model(pair_frame, output_dir=output_dir, model_type=args.model_type, random_state=args.random_state)
    scored = score_pair_frame(pair_frame, result.model_path)
    similar = scored[scored["model_similarity"] >= args.threshold].sort_values("model_similarity", ascending=False)
    similar_path = output_dir / "similar_pairs.csv"
    similar.to_csv(similar_path, index=False, encoding="utf-8-sig")

    write_html_report(output_dir / "report.html", similar, metrics=result.metrics, top_k=args.top_k)
    write_json(
        output_dir / "run_config.json",
        {
            "command": "run-pipeline",
            "data_dir": str(args.data_dir),
            "high_threshold": args.high_threshold,
            "low_threshold": args.low_threshold,
            "similarity_threshold": args.threshold,
            "max_pairs_per_question": args.max_pairs_per_question,
            "model_type": args.model_type,
            "question_id": args.question_id,
        },
    )
    _print_json(
        {
            "training_pairs": str(training_pairs_path),
            "model": str(result.model_path),
            "metrics": str(result.metrics_path),
            "similar_pairs": str(similar_path),
            "report": str(output_dir / "report.html"),
        }
    )


def command_compare_files(args: argparse.Namespace) -> None:
    """比较两个本地 Java 文件，输出模型相似度和分项特征。"""

    code_a = Path(args.left).read_text(encoding=args.encoding)
    code_b = Path(args.right).read_text(encoding=args.encoding)
    _print_json(predict_pair_similarity(code_a, code_b, args.model))


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

    compare_parser = subparsers.add_parser("compare-files", help="比较两个 Java 文件")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    compare_parser.add_argument("--model", default="outputs/model.joblib")
    compare_parser.add_argument("--encoding", default="utf-8")
    compare_parser.set_defaults(func=command_compare_files)
    return parser


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


def main(argv: list[str] | None = None) -> None:
    """命令行主入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
