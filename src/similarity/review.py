from __future__ import annotations

from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = [
    "review_decision",
    "reviewer",
    "review_note",
]

DEFAULT_STRATA_BINS = [
    ("high", 0.85, 1.0),
    ("medium", 0.60, 0.85),
    ("low", 0.0, 0.60),
]


def _normalize_strata_counts(
    stratified_counts: dict[str, int] | None,
    top_k: int,
    random_k: int,
) -> dict[str, int] | None:
    """统一整理分层抽样参数。

    兼容旧版 `top_k + random_k` 调用方式，同时支持显式传入
    `high / medium / low` 三层抽样数量。
    """

    if stratified_counts is None:
        return None
    normalized = {
        "high": max(0, int(stratified_counts.get("high", 0))),
        "medium": max(0, int(stratified_counts.get("medium", 0))),
        "low": max(0, int(stratified_counts.get("low", 0))),
    }
    if sum(normalized.values()) == 0:
        return None
    return normalized


def _sample_stratum(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """对单个分层区间执行稳定抽样。

    当某一层数据量不足时，直接返回该层全部样本，避免人工验证因为
    抽样数量超限而失败。
    """

    if sample_size <= 0 or frame.empty:
        return frame.head(0).copy()
    if len(frame) <= sample_size:
        return frame.copy()
    return frame.sample(n=sample_size, random_state=random_state)


def _build_stratified_sample(
    frame: pd.DataFrame,
    score_column: str,
    stratified_counts: dict[str, int],
    random_state: int,
) -> pd.DataFrame:
    """基于分数区间生成高/中/低分层人工复核样本。

    分层抽样的目标不是继续放大高分样本，而是让人工复核覆盖高分、
    中间模糊区间与低分区间，从而帮助论文更客观地讨论误报与漏报风险。
    """

    sampled_frames: list[pd.DataFrame] = []
    used_indices: set[int] = set()
    for offset, (stratum_name, lower_bound, upper_bound) in enumerate(DEFAULT_STRATA_BINS):
        sample_size = stratified_counts.get(stratum_name, 0)
        if sample_size <= 0:
            continue
        if stratum_name == "high":
            mask = frame[score_column].between(lower_bound, upper_bound, inclusive="both")
        else:
            mask = frame[score_column].between(lower_bound, upper_bound, inclusive="left")
        candidates = frame[mask & ~frame.index.isin(used_indices)].copy()
        sampled = _sample_stratum(candidates, sample_size, random_state + offset)
        if sampled.empty:
            continue
        sampled["review_stratum"] = stratum_name
        sampled_frames.append(sampled)
        used_indices.update(sampled.index.tolist())

    if not sampled_frames:
        return frame.head(0).copy()
    return pd.concat(sampled_frames, ignore_index=True)


def create_review_sample(
    similar_pairs_path: str | Path,
    output_path: str | Path,
    top_k: int = 50,
    random_k: int = 50,
    random_state: int = 42,
    score_column: str = "model_similarity",
    stratified_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """从高相似结果中生成可人工复核的抽样表。

    表格会保留模型分数和提交元数据，并额外添加人工复核列：
    `review_decision` 可填写 `similar`、`not_similar` 或 `uncertain`；
    `reviewer` 填写复核人；`review_note` 记录原因。这样论文中可以
    补充“人工抽样检查 Top 结果”的可信度分析。

    若传入 `stratified_counts`，则改用高/中/低分层抽样；此时建议输入
    完整打分结果 `scored_pairs.csv`，便于人工样本覆盖不同风险区间。
    """

    frame = pd.read_csv(similar_pairs_path)
    if score_column not in frame.columns:
        raise ValueError(f"复核样本文件缺少分数字段: {score_column}")

    normalized_strata = _normalize_strata_counts(stratified_counts, top_k=top_k, random_k=random_k)
    if frame.empty:
        sample = frame.copy()
    elif normalized_strata is not None:
        sample = _build_stratified_sample(
            frame=frame,
            score_column=score_column,
            stratified_counts=normalized_strata,
            random_state=random_state,
        )
    else:
        sorted_frame = frame.sort_values("model_similarity", ascending=False)
        top_sample = sorted_frame.head(top_k)
        remaining = sorted_frame.iloc[top_k:]
        random_sample = (
            remaining.sample(n=min(random_k, len(remaining)), random_state=random_state)
            if random_k > 0 and not remaining.empty
            else remaining.head(0)
        )
        sample = pd.concat([top_sample, random_sample], ignore_index=True)

    for column in REVIEW_COLUMNS:
        if column not in sample.columns:
            sample[column] = ""

    preferred_columns = [
        "activity",
        "left_submission_id",
        "right_submission_id",
        "left_user_id",
        "right_user_id",
        "model_similarity",
        "base_similarity",
        "review_stratum",
        *REVIEW_COLUMNS,
    ]
    ordered_columns = [column for column in preferred_columns if column in sample.columns]
    remaining_columns = [column for column in sample.columns if column not in ordered_columns]
    sample = sample[ordered_columns + remaining_columns]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False, encoding="utf-8-sig")
    return sample


def summarize_manual_review(review_sample_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """统计人工复核结果，并生成论文实验分析段落。

    如果复核表尚未填写，本函数仍会生成“待复核”统计，帮助论文先占位；
    当 `review_decision` 填入 `similar`、`not_similar`、`uncertain` 后，
    同一命令即可得到人工确认率等真实统计结果。
    """

    frame = pd.read_csv(review_sample_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decisions = frame["review_decision"].fillna("").astype(str).str.strip().str.lower()
    valid_mask = decisions.isin({"similar", "not_similar", "uncertain"})
    reviewed = frame[valid_mask].copy()
    valid_decisions = decisions[valid_mask]

    total_rows = int(len(frame))
    reviewed_rows = int(len(reviewed))
    pending_rows = total_rows - reviewed_rows
    similar_count = int((valid_decisions == "similar").sum())
    not_similar_count = int((valid_decisions == "not_similar").sum())
    uncertain_count = int((valid_decisions == "uncertain").sum())
    confirmed_rate = similar_count / reviewed_rows if reviewed_rows else None
    rejected_rate = not_similar_count / reviewed_rows if reviewed_rows else None
    uncertain_rate = uncertain_count / reviewed_rows if reviewed_rows else None

    summary = {
        "total_rows": total_rows,
        "reviewed_rows": reviewed_rows,
        "pending_rows": pending_rows,
        "similar_count": similar_count,
        "not_similar_count": not_similar_count,
        "uncertain_count": uncertain_count,
        "confirmed_rate": confirmed_rate,
        "rejected_rate": rejected_rate,
        "uncertain_rate": uncertain_rate,
        "mean_model_similarity": float(frame["model_similarity"].mean()) if "model_similarity" in frame else None,
        "mean_base_similarity": float(frame["base_similarity"].mean()) if "base_similarity" in frame else None,
    }

    table = pd.DataFrame(
        [
            {"指标": "抽样总数", "数值": total_rows},
            {"指标": "已复核数量", "数值": reviewed_rows},
            {"指标": "待复核数量", "数值": pending_rows},
            {"指标": "人工确认相似数量", "数值": similar_count},
            {"指标": "人工判断不相似数量", "数值": not_similar_count},
            {"指标": "人工判断不确定数量", "数值": uncertain_count},
            {"指标": "人工确认率", "数值": "" if confirmed_rate is None else round(confirmed_rate, 4)},
            {"指标": "人工否定率", "数值": "" if rejected_rate is None else round(rejected_rate, 4)},
            {"指标": "人工不确定率", "数值": "" if uncertain_rate is None else round(uncertain_rate, 4)},
            {"指标": "平均模型相似度", "数值": "" if summary["mean_model_similarity"] is None else round(summary["mean_model_similarity"], 4)},
            {"指标": "平均基础相似度", "数值": "" if summary["mean_base_similarity"] is None else round(summary["mean_base_similarity"], 4)},
        ]
    )
    table.to_csv(output / "manual_review_summary.csv", index=False, encoding="utf-8-sig")

    paragraph = build_review_analysis_paragraph(summary)
    (output / "manual_review_analysis.md").write_text(paragraph, encoding="utf-8")
    return summary


def build_review_analysis_paragraph(summary: dict[str, object]) -> str:
    """根据人工复核统计生成可放入论文的实验分析段落。"""

    total_rows = int(summary["total_rows"])
    reviewed_rows = int(summary["reviewed_rows"])
    pending_rows = int(summary["pending_rows"])
    mean_model = summary["mean_model_similarity"]
    mean_base = summary["mean_base_similarity"]
    mean_model_text = "暂无" if mean_model is None else f"{mean_model:.4f}"
    mean_base_text = "暂无" if mean_base is None else f"{mean_base:.4f}"

    if reviewed_rows == 0:
        return (
            "### 人工复核结果分析\n\n"
            f"为提高实验结论的可信度，本文在跨用户过滤后的高相似代码对中抽取 {total_rows} 组样本，"
            "用于人工复核模型输出结果。抽样样本同时包含模型相似度最高的 Top 样本和随机抽取样本，"
            "可以兼顾高置信结果与一般高相似结果的检查。当前复核表尚未填写人工结论，"
            f"待复核样本数量为 {pending_rows}。从待复核样本的模型输出看，平均模型相似度为 {mean_model_text}，"
            f"平均基础相似度为 {mean_base_text}，说明这些样本在模型判断和规则特征层面均具有较高相似性。"
            "后续可在 `review_decision` 字段中填写 `similar`、`not_similar` 或 `uncertain`，"
            "再重新生成统计表，以获得人工确认率、人工否定率和不确定率。"
        )

    confirmed_rate = float(summary["confirmed_rate"])
    rejected_rate = float(summary["rejected_rate"])
    uncertain_rate = float(summary["uncertain_rate"])
    return (
        "### 人工复核结果分析\n\n"
        f"为提高实验结论的可信度，本文在跨用户过滤后的高相似代码对中抽取 {total_rows} 组样本进行人工复核，"
        f"其中已完成人工判断 {reviewed_rows} 组，待复核 {pending_rows} 组。复核结果显示，"
        f"人工确认相似比例为 {confirmed_rate:.2%}，人工判断不相似比例为 {rejected_rate:.2%}，"
        f"不确定比例为 {uncertain_rate:.2%}。抽样样本的平均模型相似度为 {mean_model_text}，"
        f"平均基础相似度为 {mean_base_text}。该结果表明，跨用户过滤后的高相似代码对不仅在模型概率上具有较高得分，"
        "也能够通过人工复核进一步验证其合理性。由于本文采用弱监督方式构造训练标签，人工复核结果可作为对伪标签实验的补充验证，"
        "有助于降低仅依赖自动指标造成的评价偏差。"
    )
