from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Submission:
    """单份 Java 提交记录。

    字段尽量贴近数据集原始含义，后续训练和报告都围绕该结构传递，
    可以避免不同模块反复依赖 CSV 的原始列名。
    """

    submission_id: str
    user_id: str
    activity: str
    code: str
    result: str = ""
    timestamp: float | None = None


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], file_name: str) -> None:
    """检查 CSV 是否包含必要列，提前给出清晰错误信息。"""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{file_name} 缺少必要字段: {', '.join(missing)}")


def load_submissions(data_dir: str | Path) -> list[Submission]:
    """读取 Java 提交 CSV，并转换为统一的提交对象列表。

    数据集中 `activity` 表示题目名称，`user` 表示学生匿名编号，`code`
    表示 Java 源码。这里会过滤空代码，避免训练阶段出现无意义样本。
    """

    path = Path(data_dir) / "java_submissions.csv"
    frame = pd.read_csv(path)
    _require_columns(frame, ["index", "user", "activity", "code"], path.name)

    submissions: list[Submission] = []
    for row in frame.itertuples(index=False):
        code = getattr(row, "code")
        if not isinstance(code, str) or not code.strip():
            continue
        timestamp = getattr(row, "timestamp", None)
        submissions.append(
            Submission(
                submission_id=str(getattr(row, "index")),
                user_id=str(getattr(row, "user")),
                activity=str(getattr(row, "activity")),
                code=code,
                result=str(getattr(row, "result", "")),
                timestamp=float(timestamp) if pd.notna(timestamp) else None,
            )
        )
    return submissions


def group_by_activity(submissions: Iterable[Submission]) -> dict[str, list[Submission]]:
    """按题目名称分组，默认相似度比较只在同一题目内部进行。"""

    grouped: dict[str, list[Submission]] = {}
    for submission in submissions:
        grouped.setdefault(submission.activity, []).append(submission)
    return grouped


def inspect_dataset(data_dir: str | Path) -> dict[str, object]:
    """汇总数据集概况，供命令行和论文实验描述使用。"""

    submissions = load_submissions(data_dir)
    grouped = group_by_activity(submissions)
    distribution_path = Path(data_dir) / "distribution_per_question.csv"
    questions_path = Path(data_dir) / "tst_questions.csv"

    return {
        "submissions": len(submissions),
        "activities": len(grouped),
        "users": len({item.user_id for item in submissions}),
        "distribution_rows": len(pd.read_csv(distribution_path)) if distribution_path.exists() else 0,
        # `tst_questions.csv` 在该数据集中使用 `~` 和多行文本混合保存题面，
        # 不是标准逗号 CSV；这里只统计非空文本行，避免数据概况命令因解析题面失败。
        "question_rows": (
            sum(1 for line in questions_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
            if questions_path.exists()
            else 0
        ),
        "largest_activities": [
            {"activity": activity, "submissions": len(items)}
            for activity, items in sorted(grouped.items(), key=lambda pair: len(pair[1]), reverse=True)[:10]
        ],
    }
