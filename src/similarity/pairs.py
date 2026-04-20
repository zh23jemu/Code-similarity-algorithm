from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Iterable

import pandas as pd

from .data import Submission, group_by_activity
from .features import FEATURE_NAMES, extract_pair_features


@dataclass(frozen=True)
class CodePair:
    """训练或预测阶段使用的代码对。"""

    activity: str
    left: Submission
    right: Submission


def iter_activity_pairs(
    submissions: Iterable[Submission],
    question_id: str | None = None,
    max_pairs_per_question: int = 2000,
    random_state: int = 42,
) -> list[CodePair]:
    """按题目生成代码对，并对大题目进行抽样以控制计算量。

    数据集中某些题目提交量很大，完全两两组合会产生数十万代码对。
    这里先枚举同题代码对，再用固定随机种子抽样，保证实验可复现。
    """

    random = Random(random_state)
    grouped = group_by_activity(submissions)
    pairs: list[CodePair] = []
    for activity, items in grouped.items():
        if question_id and activity != question_id:
            continue
        candidates = [CodePair(activity=activity, left=left, right=right) for left, right in combinations(items, 2)]
        if max_pairs_per_question > 0 and len(candidates) > max_pairs_per_question:
            candidates = random.sample(candidates, max_pairs_per_question)
        pairs.extend(candidates)
    return pairs


def build_pair_frame(
    pairs: Iterable[CodePair],
    high_threshold: float = 0.85,
    low_threshold: float = 0.35,
    include_unlabeled: bool = False,
    adaptive_labeling: bool = True,
) -> pd.DataFrame:
    """将代码对转换为特征表，并按弱监督规则生成标签。

    `base_similarity >= high_threshold` 标为相似，`<= low_threshold` 标为
    不相似，中间样本默认丢弃，减少噪声标签对模型训练的影响。
    """

    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    for pair in pairs:
        features = extract_pair_features(pair.left.code, pair.right.code)
        label: int | None
        if features.base_similarity >= high_threshold:
            label = 1
        elif features.base_similarity <= low_threshold:
            label = 0
        else:
            label = None

        row: dict[str, object] = {
            "activity": pair.activity,
            "left_submission_id": pair.left.submission_id,
            "right_submission_id": pair.right.submission_id,
            "left_user_id": pair.left.user_id,
            "right_user_id": pair.right.user_id,
            "label": label,
        }
        row.update(features.as_dict())
        raw_rows.append(row)

    # 某些题目或小样本抽样下，固定阈值可能只产生单一类别。
    # 为了让“弱监督训练”在本科实验中更稳定，这里用分位数做兜底：
    # 高分位样本标为相似、低分位样本标为不相似，中间仍保持未标注。
    labels = {row["label"] for row in raw_rows if row["label"] is not None}
    if adaptive_labeling and raw_rows and len(labels) < 2 and not include_unlabeled:
        scores = sorted(float(row["base_similarity"]) for row in raw_rows)
        low_index = max(0, int(len(scores) * 0.25) - 1)
        high_index = min(len(scores) - 1, int(len(scores) * 0.75))
        adaptive_low = scores[low_index]
        adaptive_high = scores[high_index]
        for row in raw_rows:
            score = float(row["base_similarity"])
            if score <= adaptive_low:
                row["label"] = 0
            elif score >= adaptive_high:
                row["label"] = 1
            else:
                row["label"] = None

    for row in raw_rows:
        if row["label"] is None and not include_unlabeled:
            continue
        rows.append(row)

    columns = [
        "activity",
        "left_submission_id",
        "right_submission_id",
        "left_user_id",
        "right_user_id",
        "label",
        *FEATURE_NAMES,
    ]
    return pd.DataFrame(rows, columns=columns)
