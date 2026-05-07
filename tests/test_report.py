from pathlib import Path

import pandas as pd

from similarity.report import write_html_report


def test_write_html_report_generates_dashboard_html(tmp_path: Path):
    """验证报告生成器会输出答辩看板需要的主要结构和转义内容。"""

    report_path = tmp_path / "report.html"
    scored_pairs = pd.DataFrame(
        [
            {
                "activity": "<demo_activity>",
                "left_submission_id": "1",
                "right_submission_id": "2",
                "left_user_id": "user_a",
                "right_user_id": "user_b",
                "label": 1,
                "token_jaccard": 0.91,
                "token_sequence_similarity": 0.88,
                "keyword_cosine": 0.93,
                "operator_cosine": 0.92,
                "structure_cosine": 0.94,
                "length_similarity": 0.82,
                "line_count_similarity": 0.80,
                "method_count_similarity": 1.0,
                "loop_count_similarity": 0.75,
                "branch_count_similarity": 0.67,
                "base_similarity": 0.89,
                "model_similarity": 0.97,
            }
        ]
    )
    metrics = {
        "model_type": "random_forest",
        "accuracy": 0.98,
        "precision": 0.97,
        "recall": 0.96,
        "f1": 0.965,
    }
    run_config = {
        "command": "run-pipeline",
        "data_dir": "dataset",
        "similarity_threshold": 0.85,
    }

    write_html_report(report_path, scored_pairs, metrics=metrics, top_k=50, run_config=run_config)

    html_text = report_path.read_text(encoding="utf-8")
    assert "代码相似度检测答辩前端看板" in html_text
    assert "高相似代码对看板" in html_text
    assert "renderPairs()" in html_text
    assert "<demo_activity>" in html_text
    assert '"similarity_threshold": 0.85' in html_text
    assert "模型评估与规则依据" in html_text


def test_write_html_report_handles_missing_optional_tables(tmp_path: Path):
    """验证缺少复核或论文表格时，页面仍能生成并给出空状态提示。"""

    report_path = tmp_path / "report.html"
    scored_pairs = pd.DataFrame(
        columns=[
            "activity",
            "left_submission_id",
            "right_submission_id",
            "left_user_id",
            "right_user_id",
            "label",
            "token_jaccard",
            "token_sequence_similarity",
            "keyword_cosine",
            "operator_cosine",
            "structure_cosine",
            "length_similarity",
            "line_count_similarity",
            "method_count_similarity",
            "loop_count_similarity",
            "branch_count_similarity",
            "base_similarity",
            "model_similarity",
        ]
    )

    write_html_report(report_path, scored_pairs, metrics=None, top_k=20, run_config=None)

    html_text = report_path.read_text(encoding="utf-8")
    assert "当前没有题目分布数据" in html_text
    assert "当前没有模型对比实验表" in html_text
