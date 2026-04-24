from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "paper_figures"


def configure_matplotlib() -> None:
    """设置中文字体和基础绘图风格。"""

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_label_distribution() -> None:
    df = pd.read_csv(PROJECT_ROOT / "paper_tables" / "table_label_distribution.csv")
    label_map = {"0": "负样本", "1": "正样本", 0: "负样本", 1: "正样本"}
    labels = [label_map.get(value, str(value)) for value in df["label"]]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, df["count"], color=["#d97757", "#2a9d8f"])
    plt.title("训练样本正负类别分布")
    plt.ylabel("样本数量")
    for bar, value in zip(bars, df["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    save_current(OUTPUT_DIR / "fig_label_distribution.png")


def plot_model_comparison() -> None:
    df = pd.read_csv(PROJECT_ROOT / "paper_tables" / "table_model_comparison.csv")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    methods = df["method"].tolist()
    x = range(len(methods))
    width = 0.15

    plt.figure(figsize=(10, 5))
    for idx, metric in enumerate(metrics):
        plt.bar([v + idx * width for v in x], df[metric], width=width, label=metric.upper())
    plt.xticks([v + width * 2 for v in x], methods, rotation=10)
    plt.ylim(0.0, 1.05)
    plt.title("模型对比实验结果")
    plt.ylabel("指标值")
    plt.legend()
    save_current(OUTPUT_DIR / "fig_model_comparison.png")


def plot_ablation() -> None:
    df = pd.read_csv(PROJECT_ROOT / "outputs_ablation" / "ablation_metrics.csv")
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df["ablation"], df["f1"], color="#457b9d")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.95, 1.01)
    plt.title("特征消融实验（F1）")
    plt.ylabel("F1 值")
    for bar, value in zip(bars, df["f1"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    save_current(OUTPUT_DIR / "fig_ablation_f1.png")


def plot_activity_distribution() -> None:
    df = pd.read_csv(PROJECT_ROOT / "paper_tables" / "table_high_similarity_activity_distribution.csv").head(10)
    plt.figure(figsize=(10, 5))
    plt.bar(df["activity"], df["high_similarity_pairs"], color="#8d99ae")
    plt.xticks(rotation=30, ha="right")
    plt.title("高相似代码对题目分布（Top 10）")
    plt.ylabel("高相似代码对数量")
    save_current(OUTPUT_DIR / "fig_high_similarity_activity_top10.png")


def plot_similarity_distribution() -> None:
    df = pd.read_csv(PROJECT_ROOT / "outputs_cross_user" / "similar_pairs.csv")
    plt.figure(figsize=(8, 4.5))
    plt.hist(df["model_similarity"], bins=20, color="#264653", edgecolor="white")
    plt.title("高相似代码对模型相似度分布")
    plt.xlabel("模型相似度")
    plt.ylabel("代码对数量")
    save_current(OUTPUT_DIR / "fig_similarity_distribution.png")


def plot_manual_review() -> None:
    df = pd.read_csv(PROJECT_ROOT / "paper_tables" / "table_manual_review_summary.csv")
    mapping = {
        "人工确认相似数量": "确认相似",
        "人工判断不相似数量": "判断不相似",
        "人工判断不确定数量": "不确定",
    }
    selected = df[df["指标"].isin(mapping.keys())].copy()
    selected["标签"] = selected["指标"].map(mapping)
    plt.figure(figsize=(6, 4))
    bars = plt.bar(selected["标签"], selected["数值"], color=["#2a9d8f", "#e76f51", "#e9c46a"])
    plt.title("人工辅助复核结果分布")
    plt.ylabel("数量")
    for bar, value in zip(bars, selected["数值"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(int(value)), ha="center", va="bottom")
    save_current(OUTPUT_DIR / "fig_manual_review_distribution.png")


def plot_question_split_metrics() -> None:
    data = json.loads((PROJECT_ROOT / "outputs_question_split" / "question_split_metrics.json").read_text(encoding="utf-8"))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [data[item] for item in metrics]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(metrics, values, color="#6d597a")
    plt.ylim(0.95, 1.01)
    plt.title("按题目划分严格泛化实验结果")
    plt.ylabel("指标值")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    save_current(OUTPUT_DIR / "fig_question_split_metrics.png")


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_label_distribution()
    plot_model_comparison()
    plot_ablation()
    plot_activity_distribution()
    plot_similarity_distribution()
    plot_manual_review()
    plot_question_split_metrics()
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
