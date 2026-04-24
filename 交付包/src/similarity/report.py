from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def write_json(path: str | Path, data: dict[str, object]) -> None:
    """写入 UTF-8 JSON 文件，便于保存运行配置和数据概况。"""

    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html_report(
    path: str | Path,
    scored_pairs: pd.DataFrame,
    metrics: dict[str, object] | None = None,
    top_k: int = 100,
) -> None:
    """生成静态 HTML 报告。

    报告采用表格布局，不把文字叠放在图片或复杂背景上，保证后续写论文
    截图时重要文本和数值都清晰可读。
    """

    top_pairs = scored_pairs.sort_values("model_similarity", ascending=False).head(top_k)
    metrics_html = ""
    if metrics:
        metrics_rows = "".join(
            f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in metrics.items()
        )
        metrics_html = f"<section><h2>模型评估指标</h2><table>{metrics_rows}</table></section>"

    pair_rows = []
    for row in top_pairs.itertuples(index=False):
        pair_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.activity))}</td>"
            f"<td>{html.escape(str(row.left_submission_id))}</td>"
            f"<td>{html.escape(str(row.right_submission_id))}</td>"
            f"<td>{html.escape(str(row.left_user_id))}</td>"
            f"<td>{html.escape(str(row.right_user_id))}</td>"
            f"<td>{float(row.model_similarity):.4f}</td>"
            f"<td>{float(row.base_similarity):.4f}</td>"
            "</tr>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>代码相似度检测报告</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --card: #fffaf0;
      --ink: #1f2933;
      --line: #d6c7a8;
      --accent: #1f6f5b;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(135deg, var(--bg), #eaf3ef);
      color: var(--ink);
      font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      line-height: 1.6;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 18px 45px rgba(45, 37, 20, 0.12);
    }}
    h1, h2 {{ margin-top: 0; }}
    h1 {{ color: var(--accent); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 28px;
      background: white;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{ background: #edf7f2; }}
    @media (max-width: 760px) {{
      body {{ padding: 12px; }}
      main {{ padding: 16px; }}
      table {{ display: block; overflow-x: auto; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>代码相似度检测报告</h1>
  <p>本报告展示传统机器学习模型预测出的高相似 Java 代码提交对。模型相似度来自二分类模型的相似类概率，基础相似度来自弱监督规则特征。</p>
  {metrics_html}
  <section>
    <h2>高相似代码对 Top {top_k}</h2>
    <table>
      <thead>
        <tr>
          <th>题目</th><th>提交 A</th><th>提交 B</th><th>用户 A</th><th>用户 B</th><th>模型相似度</th><th>基础相似度</th>
        </tr>
      </thead>
      <tbody>
        {''.join(pair_rows)}
      </tbody>
    </table>
  </section>
</main>
</body>
</html>
"""
    Path(path).write_text(document, encoding="utf-8")
