from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper_figures"


def configure() -> None:
    plt.rcParams["font.sans-serif"] = ["Consolas", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220


def terminal_figure(title: str, lines: list[str], output_path: Path) -> None:
    """生成模拟命令行截图风格图片。"""

    height = max(4.0, 0.42 * len(lines) + 1.2)
    fig, ax = plt.subplots(figsize=(11.5, height))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        title,
        color="#93c5fd",
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        transform=ax.transAxes,
    )

    y = 0.88
    step = 0.82 / max(len(lines), 1)
    for line in lines:
        ax.text(
            0.02,
            y,
            line,
            color="#e5e7eb",
            fontsize=10.5,
            va="top",
            ha="left",
            family="Consolas",
            transform=ax.transAxes,
        )
        y -= step

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def main() -> None:
    configure()

    inspect_lines = [
        r"PS C:\Coding\Code-similarity-algorithm> .venv\Scripts\python.exe -m similarity inspect-data --data-dir dataset",
        "{",
        '  "submissions": 16074,',
        '  "activities": 99,',
        '  "users": 270,',
        '  "distribution_rows": 99,',
        '  "question_rows": 2635',
        "}",
    ]
    terminal_figure("命令行运行示意：数据集概况统计", inspect_lines, OUTPUT_DIR / "fig_cli_inspect_data.png")

    pipeline_lines = [
        r"PS C:\Coding\Code-similarity-algorithm> .venv\Scripts\python.exe -m similarity run-pipeline --data-dir dataset --output-dir outputs_cross_user --max-pairs-per-question 1000 --high-threshold 0.90 --low-threshold 0.75 --threshold 0.90 --model-type random_forest --cross-user-only",
        "{",
        r'  "training_pairs": "outputs_cross_user\\training_pairs.csv",',
        r'  "model": "outputs_cross_user\\model.joblib",',
        r'  "metrics": "outputs_cross_user\\metrics.json",',
        r'  "similar_pairs": "outputs_cross_user\\similar_pairs.csv",',
        r'  "report": "outputs_cross_user\\report.html"',
        "}",
    ]
    wrapped_pipeline = []
    for line in pipeline_lines:
        wrapped_pipeline.extend(textwrap.wrap(line, width=110) or [""])
    terminal_figure("命令行运行示意：模型训练与流水线输出", wrapped_pipeline, OUTPUT_DIR / "fig_cli_pipeline_run.png")

    compare_lines = [
        r"PS C:\Coding\Code-similarity-algorithm> .venv\Scripts\python.exe -m similarity compare-models --data-dir dataset --output-dir outputs_cross_user_compare --max-pairs-per-question 1000 --high-threshold 0.90 --low-threshold 0.75 --baseline-threshold 0.90 --cross-user-only",
        "{",
        r'  "training_pairs": "outputs_cross_user_compare\\comparison_training_pairs.csv",',
        r'  "comparison_csv": "outputs_cross_user_compare\\comparison_metrics.csv",',
        r'  "comparison_json": "outputs_cross_user_compare\\comparison_metrics.json",',
        r'  "methods": ["rule_baseline", "logistic_regression", "random_forest"]',
        "}",
    ]
    wrapped_compare = []
    for line in compare_lines:
        wrapped_compare.extend(textwrap.wrap(line, width=110) or [""])
    terminal_figure("命令行运行示意：模型对比实验", wrapped_compare, OUTPUT_DIR / "fig_cli_compare_models.png")

    question_split_lines = [
        r"PS C:\Coding\Code-similarity-algorithm> .venv\Scripts\python.exe -m similarity question-split-experiment --data-dir dataset --output-dir outputs_question_split --max-pairs-per-question 500 --high-threshold 0.90 --low-threshold 0.75 --model-type logistic_regression --cross-user-only",
        "{",
        r'  "metrics_json": "outputs_question_split\\question_split_metrics.json",',
        '  "result": {',
        '    "train_activities": 68,',
        '    "test_activities": 30,',
        '    "train_samples": 25086,',
        '    "test_samples": 10333,',
        '    "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0',
        "  }",
        "}",
    ]
    wrapped_question_split = []
    for line in question_split_lines:
        wrapped_question_split.extend(textwrap.wrap(line, width=110) or [""])
    terminal_figure("命令行运行示意：严格泛化实验", wrapped_question_split, OUTPUT_DIR / "fig_cli_question_split.png")

    pipeline_detail_lines = [
        r"PS C:\Coding\Code-similarity-algorithm> .venv\Scripts\python.exe -m similarity run-pipeline --data-dir dataset --output-dir outputs_cross_user --question-id warm_up_dsa --max-pairs-per-question 80 --high-threshold 0.80 --low-threshold 0.60 --threshold 0.80 --model-type logistic_regression --top-k 20",
        "{",
        r'  "training_pairs": "outputs_cross_user\\training_pairs.csv",',
        r'  "model": "outputs_cross_user\\model.joblib",',
        r'  "metrics": "outputs_cross_user\\metrics.json",',
        r'  "similar_pairs": "outputs_cross_user\\similar_pairs.csv",',
        r'  "report": "outputs_cross_user\\report.html"',
        "}",
    ]
    wrapped_pipeline_detail = []
    for line in pipeline_detail_lines:
        wrapped_pipeline_detail.extend(textwrap.wrap(line, width=110) or [""])
    terminal_figure("命令行运行示意：完整训练流水线", wrapped_pipeline_detail, OUTPUT_DIR / "fig_cli_full_pipeline.png")


if __name__ == "__main__":
    main()
