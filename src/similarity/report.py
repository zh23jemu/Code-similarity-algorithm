from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_json(path: str | Path, data: dict[str, object]) -> None:
    """写入 UTF-8 JSON 文件，便于保存运行配置和数据概况。"""

    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_optional_json(path: Path) -> dict[str, object] | None:
    """读取可选 JSON 文件；文件不存在或内容异常时返回空值。"""

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    """读取可选 CSV 文件；文件不存在或解析失败时返回空值。"""

    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _sanitize_value(value: object) -> object:
    """将 DataFrame 中的值转换为可安全序列化、可稳定展示的基础类型。"""

    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _frame_to_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, object]]:
    """将表格转换为 JSON 兼容字典列表，供接口与静态页面复用。"""

    if frame.empty:
        return []
    view = frame.head(limit) if limit is not None else frame
    records: list[dict[str, object]] = []
    for row in view.to_dict(orient="records"):
        records.append({str(key): _sanitize_value(value) for key, value in row.items()})
    return records


def _metrics_cards(metrics: dict[str, object] | None, pair_count: int, activity_count: int, top_k: int) -> list[dict[str, object]]:
    """构建总览指标卡，优先突出答辩展示最关心的信息。"""

    cards = [
        {
            "label": "高相似代码对",
            "value": f"{pair_count}",
            "hint": f"当前报告纳入 {min(pair_count, top_k)} 条 Top 结果展示",
        },
        {
            "label": "涉及题目数",
            "value": f"{activity_count}",
            "hint": "用于展示高相似结果覆盖的题目范围",
        },
    ]
    if metrics:
        accuracy = metrics.get("accuracy")
        f1_score = metrics.get("f1")
        model_type = metrics.get("model_type", "未知模型")
        if isinstance(accuracy, (int, float)):
            cards.append(
                {
                    "label": "准确率 Accuracy",
                    "value": f"{float(accuracy):.4f}",
                    "hint": f"模型类型：{model_type}",
                }
            )
        if isinstance(f1_score, (int, float)):
            cards.append(
                {
                    "label": "F1 值",
                    "value": f"{float(f1_score):.4f}",
                    "hint": "综合衡量精确率与召回率",
                }
            )
    return cards


def _activity_distribution(frame: pd.DataFrame) -> list[dict[str, object]]:
    """统计高相似代码对在不同题目中的分布。"""

    if frame.empty or "activity" not in frame.columns:
        return []
    counts = (
        frame["activity"]
        .astype(str)
        .value_counts()
        .head(10)
        .reset_index()
    )
    counts.columns = ["activity", "high_similarity_pairs"]
    return _frame_to_records(counts)


def _feature_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    """汇总关键特征均值，帮助答辩时解释模型依据。"""

    feature_columns = [
        ("model_similarity", "模型相似度"),
        ("base_similarity", "基础规则相似度"),
        ("token_jaccard", "Token Jaccard"),
        ("token_sequence_similarity", "Token 序列相似度"),
        ("structure_cosine", "结构相似度"),
        ("keyword_cosine", "关键字余弦相似度"),
    ]
    rows: list[dict[str, object]] = []
    for column, label in feature_columns:
        if column not in frame.columns or frame.empty:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "feature": label,
                "column": column,
                "mean": round(float(series.mean()), 4),
                "max": round(float(series.max()), 4),
                "min": round(float(series.min()), 4),
            }
        )
    return rows


def _metrics_rows(metrics: dict[str, object] | None) -> list[dict[str, object]]:
    """将评估指标转为前端便于渲染的键值列表。"""

    if not metrics:
        return []
    rows: list[dict[str, object]] = []
    for key, value in metrics.items():
        rows.append({"key": str(key), "value": _sanitize_value(value)})
    return rows


def _read_related_tables(root_dir: Path, output_dir: Path) -> dict[str, list[dict[str, object]]]:
    """读取论文统计表和人工复核摘要。"""

    review_summary_frame = _read_optional_csv(output_dir / "manual_review_summary.csv")
    review_summary = _frame_to_records(review_summary_frame) if review_summary_frame is not None else []

    table_map: dict[str, list[dict[str, object]]] = {
        "reviewSummary": review_summary,
        "questionDistribution": [],
        "labelDistribution": [],
        "modelComparison": [],
        "paperActivityDistribution": [],
    }

    manifest_path = root_dir / "paper_tables" / "tables_manifest.json"
    manifest = _read_optional_json(manifest_path)
    if not manifest:
        return table_map

    mapping = {
        "question_distribution": "questionDistribution",
        "label_distribution": "labelDistribution",
        "model_comparison": "modelComparison",
        "high_similarity_activity_distribution": "paperActivityDistribution",
    }
    for manifest_key, payload_key in mapping.items():
        rel_path = manifest.get(manifest_key)
        if not isinstance(rel_path, str):
            continue
        normalized = rel_path.replace("\\", "/")
        csv_path = root_dir / normalized
        frame = _read_optional_csv(csv_path)
        if frame is not None:
            table_map[payload_key] = _frame_to_records(frame.head(10))
    return table_map


def build_dashboard_payload(
    root_dir: str | Path,
    output_dir: str | Path = "outputs",
    top_k: int = 100,
) -> dict[str, object]:
    """根据输出目录构建答辩看板的数据载荷。"""

    root_path = Path(root_dir)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root_path / output_path

    similar_frame = _read_optional_csv(output_path / "similar_pairs.csv")
    metrics = _read_optional_json(output_path / "metrics.json")
    run_config = _read_optional_json(output_path / "run_config.json") or {}

    if similar_frame is None:
        similar_frame = pd.DataFrame(
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

    if "model_similarity" in similar_frame.columns and not similar_frame.empty:
        top_pairs = similar_frame.sort_values("model_similarity", ascending=False).head(top_k).copy()
    else:
        top_pairs = similar_frame.head(top_k).copy()

    summary_cards = _metrics_cards(
        metrics=metrics,
        pair_count=int(len(similar_frame)),
        activity_count=int(top_pairs["activity"].nunique()) if not top_pairs.empty and "activity" in top_pairs.columns else 0,
        top_k=top_k,
    )
    payload: dict[str, object] = {
        "title": "代码相似度检测答辩看板",
        "summaryCards": summary_cards,
        "metrics": _metrics_rows(metrics),
        "runConfig": run_config,
        "pairs": _frame_to_records(top_pairs, limit=top_k),
        "activityDistribution": _activity_distribution(similar_frame),
        "featureSummary": _feature_summary(similar_frame),
        "meta": {
            "topK": top_k,
            "pairCount": int(len(similar_frame)),
            "generatedFrom": output_path.name,
            "outputDir": str(output_path.relative_to(root_path)) if output_path.is_relative_to(root_path) else str(output_path),
        },
    }
    payload.update(_read_related_tables(root_path, output_path))
    return payload


def _dashboard_html() -> str:
    """返回 Web 看板主页 HTML，前端通过接口实时拉取数据。"""

    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>代码相似度检测答辩看板</title>
  <style>
    :root {
      --bg: #efe6d1;
      --paper: rgba(255, 251, 244, 0.92);
      --panel: rgba(255, 255, 255, 0.82);
      --ink: #1f2430;
      --muted: #5f6573;
      --line: rgba(92, 74, 37, 0.16);
      --accent: #17594a;
      --accent-soft: #d6ebe4;
      --accent-strong: #0f3f35;
      --warm: #b46b2c;
      --danger: #b94141;
      --shadow: 0 18px 40px rgba(46, 37, 25, 0.12);
      --radius-xl: 24px;
      --radius-lg: 18px;
      --radius-md: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(255, 255, 255, 0.7), transparent 35%),
        linear-gradient(135deg, #efe5d0 0%, #f5f1e8 45%, #deebe5 100%);
      min-height: 100vh;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.14) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.14) 1px, transparent 1px);
      background-size: 26px 26px;
      mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.18), transparent 65%);
    }
    main {
      width: min(1280px, calc(100vw - 32px));
      margin: 24px auto 40px;
      padding: 28px;
      background: var(--paper);
      border: 1px solid rgba(255, 255, 255, 0.7);
      border-radius: 32px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
      gap: 20px;
      margin-bottom: 24px;
      align-items: start;
    }
    .hero-main {
      display: grid;
      gap: 16px;
      align-content: start;
      min-width: 0;
    }
    .hero-card, .hero-side, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-xl);
      box-shadow: 0 10px 24px rgba(61, 49, 29, 0.08);
    }
    .hero-card {
      padding: 28px;
      position: relative;
      overflow: hidden;
    }
    .hero-card::after {
      content: "";
      position: absolute;
      right: -60px;
      top: -40px;
      width: 220px;
      height: 220px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(23, 89, 74, 0.18), transparent 70%);
    }
    .hero-side {
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(23, 89, 74, 0.08);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    h1 {
      margin: 16px 0 12px;
      font-size: clamp(30px, 5vw, 46px);
      line-height: 1.12;
      letter-spacing: -0.03em;
      max-width: 92%;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.8;
      max-width: 92%;
    }
    .flow-list, .config-list { display: grid; gap: 10px; }
    .flow-item, .config-item {
      display: flex;
      gap: 12px;
      align-items: flex-start;
    }
    .flow-index {
      flex: none;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      font-weight: 700;
      background: var(--accent-soft);
      color: var(--accent-strong);
      margin-top: 2px;
    }
    .config-key {
      color: var(--muted);
      min-width: 108px;
      font-size: 14px;
    }
    .config-value {
      font-weight: 700;
      word-break: break-word;
      flex: 1;
    }
    .status-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
      align-items: center;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(23, 89, 74, 0.08);
      color: var(--accent-strong);
      font-size: 13px;
      font-weight: 700;
    }
    .status-pill button {
      border: none;
      background: var(--accent);
      color: white;
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
    }
    .metric-card {
      padding: 18px;
      border-radius: var(--radius-lg);
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(240,248,244,0.75));
      border: 1px solid rgba(23, 89, 74, 0.12);
      box-shadow: 0 10px 18px rgba(54, 41, 18, 0.06);
    }
    .metric-label { color: var(--muted); font-size: 14px; }
    .metric-value {
      margin: 8px 0 6px;
      font-size: clamp(26px, 3vw, 36px);
      font-weight: 800;
      color: var(--accent-strong);
    }
    .metric-hint { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .dashboard-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(300px, 0.9fr);
      gap: 20px;
      margin-bottom: 24px;
    }
    .panel { padding: 22px; }
    .panel h2 { margin: 0 0 6px; font-size: 22px; }
    .panel-intro {
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.7;
    }
    .table-shell {
      overflow-x: auto;
      border-radius: var(--radius-md);
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.85);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 560px;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid rgba(92, 74, 37, 0.08);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
      line-height: 1.6;
      word-break: break-word;
    }
    th {
      position: sticky;
      top: 0;
      background: #edf6f2;
      color: var(--accent-strong);
      font-size: 13px;
      letter-spacing: 0.03em;
      z-index: 1;
    }
    tbody tr:hover { background: rgba(214, 235, 228, 0.28); }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    .mini-card {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: rgba(255,255,255,0.78);
      padding: 16px;
    }
    .mini-card h3 { margin: 0 0 8px; font-size: 16px; }
    .bar-list { display: grid; gap: 12px; }
    .bar-item { display: grid; gap: 6px; }
    .bar-label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
    }
    .bar-track {
      height: 10px;
      border-radius: 999px;
      background: rgba(23, 89, 74, 0.10);
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #1a7c66, #d9a24d);
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 180px 180px 160px;
      gap: 14px;
      margin-bottom: 16px;
    }
    .field { display: grid; gap: 6px; }
    .field label {
      font-size: 13px;
      color: var(--muted);
      font-weight: 700;
    }
    .field input, .field select {
      width: 100%;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid rgba(92, 74, 37, 0.16);
      background: rgba(255,255,255,0.9);
      color: var(--ink);
      font: inherit;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(23, 89, 74, 0.1);
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
    }
    .score { font-variant-numeric: tabular-nums; font-weight: 700; }
    .score.high { color: var(--accent-strong); }
    .score.mid { color: var(--warm); }
    .score.low { color: var(--danger); }
    details {
      border: 1px solid rgba(92, 74, 37, 0.12);
      border-radius: 14px;
      background: rgba(255,255,255,0.82);
      padding: 12px 14px;
    }
    summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--accent-strong);
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .detail-item {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(214, 235, 228, 0.25);
      border: 1px solid rgba(23, 89, 74, 0.08);
    }
    .detail-name {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .detail-value {
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    .empty {
      padding: 18px;
      border-radius: 14px;
      border: 1px dashed rgba(92, 74, 37, 0.18);
      background: rgba(255,255,255,0.65);
      color: var(--muted);
      text-align: center;
    }
    .footnote {
      margin-top: 20px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.8;
    }
    @media (max-width: 1080px) {
      .hero, .dashboard-grid { grid-template-columns: 1fr; }
      .hero-main { gap: 14px; }
      .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .toolbar { grid-template-columns: 1fr; }
      .detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      main {
        width: min(100vw - 16px, 100%);
        margin: 8px auto 20px;
        padding: 14px;
        border-radius: 22px;
      }
      .hero-card, .hero-side, .panel { border-radius: 18px; }
      .cards, .mini-grid, .detail-grid { grid-template-columns: 1fr; }
      .hero-card, .hero-side, .panel, .metric-card { padding: 16px; }
      table { min-width: 480px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-main">
        <article class="hero-card">
          <div class="eyebrow">Defense Dashboard</div>
          <h1>代码相似度检测答辩前端看板</h1>
          <p>该页面改为前后端 Web 项目，页面每次刷新都会直接读取最新输出目录结果。你不需要重新生成 HTML，只要训练结果文件更新，前端就能拉到最新数据。</p>
          <div class="status-row">
            <span class="status-pill">当前输出目录：<strong id="current-output-dir">outputs</strong></span>
            <span class="status-pill">刷新时间：<strong id="refresh-time">--</strong></span>
            <span class="status-pill"><button id="refresh-button" type="button">立即刷新数据</button></span>
          </div>
        </article>

        <section class="cards" id="summary-cards"></section>
      </div>
      <aside class="hero-side">
        <div>
          <h2>系统流程</h2>
          <div class="flow-list">
            <div class="flow-item"><span class="flow-index">1</span><div><strong>读取数据集</strong><br>加载 Java 提交记录并按题目分组。</div></div>
            <div class="flow-item"><span class="flow-index">2</span><div><strong>提取相似特征</strong><br>计算 Token、结构、关键字与规则分数。</div></div>
            <div class="flow-item"><span class="flow-index">3</span><div><strong>弱监督训练建模</strong><br>使用传统机器学习模型学习相似概率。</div></div>
            <div class="flow-item"><span class="flow-index">4</span><div><strong>输出报告与复核样本</strong><br>前端实时读取结果文件并展示。</div></div>
          </div>
        </div>
        <div>
          <h2>运行配置</h2>
          <div class="config-list" id="run-config-list"></div>
        </div>
      </aside>
    </section>

    <section class="dashboard-grid">
      <article class="panel">
        <h2>模型评估与规则依据</h2>
        <p class="panel-intro">左侧表格展示本次输出目录中的模型指标，右侧摘要帮助答辩时说明模型为什么能区分相似代码。</p>
        <div class="mini-grid">
          <div class="mini-card">
            <h3>模型评估指标</h3>
            <div class="table-shell">
              <table>
                <thead><tr><th>指标</th><th>数值</th></tr></thead>
                <tbody id="metrics-body"></tbody>
              </table>
            </div>
          </div>
          <div class="mini-card">
            <h3>特征均值摘要</h3>
            <div class="table-shell">
              <table>
                <thead><tr><th>特征</th><th>均值</th><th>最高</th><th>最低</th></tr></thead>
                <tbody id="feature-body"></tbody>
              </table>
            </div>
          </div>
        </div>
      </article>

      <article class="panel">
        <h2>题目与人工复核概览</h2>
        <p class="panel-intro">这里优先展示答辩中容易讲清楚的总体分布与人工验证结果。</p>
        <div class="mini-grid">
          <div class="mini-card">
            <h3>高相似题目分布</h3>
            <div class="bar-list" id="activity-bars"></div>
          </div>
          <div class="mini-card">
            <h3>人工复核摘要</h3>
            <div class="table-shell">
              <table>
                <thead><tr><th>指标</th><th>数值</th></tr></thead>
                <tbody id="review-body"></tbody>
              </table>
            </div>
          </div>
        </div>
      </article>
    </section>

    <section class="dashboard-grid">
      <article class="panel">
        <h2>数据集与标签分布</h2>
        <p class="panel-intro">如果项目已经生成 `paper_tables`，这里会自动读取统计表；否则保留空状态提示，不影响页面使用。</p>
        <div class="mini-grid">
          <div class="mini-card">
            <h3>题目提交量 Top 10</h3>
            <div class="table-shell">
              <table>
                <thead><tr><th>题目</th><th>提交量</th></tr></thead>
                <tbody id="question-dist-body"></tbody>
              </table>
            </div>
          </div>
          <div class="mini-card">
            <h3>弱监督标签分布</h3>
            <div class="table-shell">
              <table>
                <thead><tr><th>标签</th><th>数量</th></tr></thead>
                <tbody id="label-dist-body"></tbody>
              </table>
            </div>
          </div>
        </div>
      </article>

      <article class="panel">
        <h2>模型对比实验</h2>
        <p class="panel-intro">用于展示规则基线、逻辑回归与随机森林在同一数据上的表现差异。</p>
        <div class="table-shell">
          <table>
            <thead><tr><th>方法</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1</th><th>ROC-AUC</th></tr></thead>
            <tbody id="comparison-body"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="panel">
      <h2>高相似代码对看板</h2>
      <p class="panel-intro">支持前端本地搜索、按题目筛选和排序。详情面板会展开单个代码对的关键特征，便于答辩现场解释模型判定依据。</p>
      <div class="toolbar">
        <div class="field">
          <label for="pair-search">搜索题目、提交编号或用户编号</label>
          <input id="pair-search" type="text" placeholder="例如 fibonacci / user_0210 / 12588">
        </div>
        <div class="field">
          <label for="activity-filter">按题目筛选</label>
          <select id="activity-filter"></select>
        </div>
        <div class="field">
          <label for="sort-filter">排序方式</label>
          <select id="sort-filter">
            <option value="model_desc">模型相似度从高到低</option>
            <option value="base_desc">基础相似度从高到低</option>
            <option value="activity_asc">题目名称升序</option>
          </select>
        </div>
        <div class="field">
          <label for="output-dir-input">输出目录</label>
          <select id="output-dir-input">
            <option value="outputs">outputs</option>
            <option value="outputs_cross_user">outputs_cross_user</option>
          </select>
        </div>
      </div>
      <div class="table-shell">
        <table>
          <thead>
            <tr>
              <th>题目</th>
              <th>提交对</th>
              <th>用户对</th>
              <th>模型相似度</th>
              <th>基础相似度</th>
              <th>标签</th>
              <th>详情</th>
            </tr>
          </thead>
          <tbody id="pairs-body"></tbody>
        </table>
      </div>
      <p class="footnote" id="pair-count-note"></p>
    </section>
  </main>

  <script>
    const state = {
      payload: null,
      allPairs: [],
      outputDir: "outputs",
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function scoreClass(score) {
      const numeric = Number(score);
      if (numeric >= 0.85) return "high";
      if (numeric >= 0.6) return "mid";
      return "low";
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") return "暂无";
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
      return String(value);
    }

    function renderEmpty(targetId, colspan, text) {
      document.getElementById(targetId).innerHTML = `<tr><td colspan="${colspan}"><div class="empty">${escapeHtml(text)}</div></td></tr>`;
    }

    function renderSummaryCards() {
      const cards = state.payload.summaryCards || [];
      document.getElementById("summary-cards").innerHTML = cards.map((card) => `
        <article class="metric-card">
          <div class="metric-label">${escapeHtml(card.label)}</div>
          <div class="metric-value">${escapeHtml(card.value)}</div>
          <div class="metric-hint">${escapeHtml(card.hint)}</div>
        </article>
      `).join("");
    }

    function renderRunConfig() {
      const target = document.getElementById("run-config-list");
      const config = state.payload.runConfig || {};
      const entries = Object.entries(config);
      if (!entries.length) {
        target.innerHTML = '<div class="empty">当前输出目录下没有可展示的运行配置。</div>';
        return;
      }
      target.innerHTML = entries.map(([key, value]) => `
        <div class="config-item">
          <div class="config-key">${escapeHtml(key)}</div>
          <div class="config-value">${escapeHtml(formatValue(value))}</div>
        </div>
      `).join("");
    }

    function renderMetrics() {
      const rows = state.payload.metrics || [];
      if (!rows.length) {
        renderEmpty("metrics-body", 2, "当前报告未携带 metrics.json 内容。");
        return;
      }
      document.getElementById("metrics-body").innerHTML = rows.map((row) => `
        <tr><td>${escapeHtml(row.key)}</td><td>${escapeHtml(formatValue(row.value))}</td></tr>
      `).join("");
    }

    function renderFeatureSummary() {
      const rows = state.payload.featureSummary || [];
      if (!rows.length) {
        renderEmpty("feature-body", 4, "没有可计算的特征摘要。");
        return;
      }
      document.getElementById("feature-body").innerHTML = rows.map((row) => `
        <tr>
          <td>${escapeHtml(row.feature)}</td>
          <td>${escapeHtml(formatValue(row.mean))}</td>
          <td>${escapeHtml(formatValue(row.max))}</td>
          <td>${escapeHtml(formatValue(row.min))}</td>
        </tr>
      `).join("");
    }

    function renderActivityBars() {
      const rows = (state.payload.paperActivityDistribution && state.payload.paperActivityDistribution.length)
        ? state.payload.paperActivityDistribution
        : state.payload.activityDistribution || [];
      const target = document.getElementById("activity-bars");
      if (!rows.length) {
        target.innerHTML = '<div class="empty">当前没有题目分布数据。</div>';
        return;
      }
      const maxValue = Math.max(...rows.map((row) => Number(row.high_similarity_pairs || row.count || 0)), 1);
      target.innerHTML = rows.slice(0, 8).map((row) => {
        const activity = row.activity || row.Question || row["Question (activity)"] || "未知题目";
        const rawValue = Number(row.high_similarity_pairs || row.count || row["Number of submissions"] || 0);
        const width = Math.max(6, Math.round((rawValue / maxValue) * 100));
        return `
          <div class="bar-item">
            <div class="bar-label"><span>${escapeHtml(activity)}</span><strong>${escapeHtml(formatValue(rawValue))}</strong></div>
            <div class="bar-track"><div class="bar-fill" style="width: ${width}%"></div></div>
          </div>
        `;
      }).join("");
    }

    function renderReviewSummary() {
      const rows = state.payload.reviewSummary || [];
      if (!rows.length) {
        renderEmpty("review-body", 2, "当前输出目录下没有人工复核汇总文件。");
        return;
      }
      document.getElementById("review-body").innerHTML = rows.map((row) => {
        const key = row["指标"] ?? row.metric ?? row.key ?? "指标";
        const value = row["数值"] ?? row.value ?? "";
        return `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(formatValue(value))}</td></tr>`;
      }).join("");
    }

    function renderSimpleTable(targetId, rows, columns, emptyText) {
      if (!rows.length) {
        renderEmpty(targetId, columns.length, emptyText);
        return;
      }
      document.getElementById(targetId).innerHTML = rows.map((row) => `
        <tr>${columns.map((column) => `<td>${escapeHtml(formatValue(row[column]))}</td>`).join("")}</tr>
      `).join("");
    }

    function renderModelComparison() {
      const rows = state.payload.modelComparison || [];
      if (!rows.length) {
        renderEmpty("comparison-body", 6, "当前没有模型对比实验表。");
        return;
      }
      document.getElementById("comparison-body").innerHTML = rows.map((row) => `
        <tr>
          <td>${escapeHtml(formatValue(row.method))}</td>
          <td>${escapeHtml(formatValue(row.accuracy))}</td>
          <td>${escapeHtml(formatValue(row.precision))}</td>
          <td>${escapeHtml(formatValue(row.recall))}</td>
          <td>${escapeHtml(formatValue(row.f1))}</td>
          <td>${escapeHtml(formatValue(row.roc_auc))}</td>
        </tr>
      `).join("");
    }

    function fillActivityFilter() {
      const select = document.getElementById("activity-filter");
      const activities = [...new Set(state.allPairs.map((item) => item.activity).filter(Boolean))].sort();
      select.innerHTML = ['<option value="">全部题目</option>', ...activities.map((activity) => `<option value="${escapeHtml(activity)}">${escapeHtml(activity)}</option>`)].join("");
    }

    function pairMatchesQuery(pair, query) {
      if (!query) return true;
      const text = [
        pair.activity,
        pair.left_submission_id,
        pair.right_submission_id,
        pair.left_user_id,
        pair.right_user_id,
      ].join(" ").toLowerCase();
      return text.includes(query);
    }

    function sortPairs(pairs, mode) {
      const copy = pairs.slice();
      if (mode === "base_desc") {
        return copy.sort((a, b) => Number(b.base_similarity || 0) - Number(a.base_similarity || 0));
      }
      if (mode === "activity_asc") {
        return copy.sort((a, b) => String(a.activity || "").localeCompare(String(b.activity || ""), "zh-CN"));
      }
      return copy.sort((a, b) => Number(b.model_similarity || 0) - Number(a.model_similarity || 0));
    }

    function pairDetailItems(pair) {
      return [
        ["Token Jaccard", pair.token_jaccard],
        ["Token 序列相似度", pair.token_sequence_similarity],
        ["结构相似度", pair.structure_cosine],
        ["关键字余弦相似度", pair.keyword_cosine],
        ["运算符余弦相似度", pair.operator_cosine],
        ["长度相似度", pair.length_similarity],
        ["行数相似度", pair.line_count_similarity],
        ["方法数量相似度", pair.method_count_similarity],
        ["循环数量相似度", pair.loop_count_similarity],
        ["分支数量相似度", pair.branch_count_similarity],
      ];
    }

    function renderPairs() {
      const query = document.getElementById("pair-search").value.trim().toLowerCase();
      const activity = document.getElementById("activity-filter").value;
      const sortMode = document.getElementById("sort-filter").value;

      let visiblePairs = state.allPairs.filter((pair) => pairMatchesQuery(pair, query));
      if (activity) visiblePairs = visiblePairs.filter((pair) => pair.activity === activity);
      visiblePairs = sortPairs(visiblePairs, sortMode);

      const target = document.getElementById("pairs-body");
      const note = document.getElementById("pair-count-note");
      if (!visiblePairs.length) {
        renderEmpty("pairs-body", 7, "当前筛选条件下没有可展示的高相似代码对。");
        note.textContent = "未命中任何代码对，请调整搜索词或筛选条件。";
        return;
      }

      target.innerHTML = visiblePairs.map((pair, index) => {
        const label = pair.label === null || pair.label === undefined || pair.label === "" ? "未标注" : String(pair.label);
        const details = pairDetailItems(pair).map(([name, value]) => `
          <div class="detail-item">
            <div class="detail-name">${escapeHtml(name)}</div>
            <div class="detail-value">${escapeHtml(formatValue(value))}</div>
          </div>
        `).join("");
        return `
          <tr>
            <td><span class="badge">${escapeHtml(pair.activity || "未知题目")}</span></td>
            <td>${escapeHtml(formatValue(pair.left_submission_id))} / ${escapeHtml(formatValue(pair.right_submission_id))}</td>
            <td>${escapeHtml(formatValue(pair.left_user_id))} / ${escapeHtml(formatValue(pair.right_user_id))}</td>
            <td><span class="score ${scoreClass(pair.model_similarity)}">${escapeHtml(formatValue(pair.model_similarity))}</span></td>
            <td><span class="score ${scoreClass(pair.base_similarity)}">${escapeHtml(formatValue(pair.base_similarity))}</span></td>
            <td>${escapeHtml(label)}</td>
            <td>
              <details${index < 2 ? " open" : ""}>
                <summary>展开特征详情</summary>
                <div class="detail-grid">${details}</div>
              </details>
            </td>
          </tr>
        `;
      }).join("");
      note.textContent = `当前展示 ${visiblePairs.length} 条代码对，共加载 ${state.allPairs.length} 条 Top 结果。`;
    }

    async function loadDashboardData(outputDir = state.outputDir) {
      const response = await fetch(`/api/dashboard?output_dir=${encodeURIComponent(outputDir)}`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`接口请求失败：${response.status}`);
      }
      state.payload = await response.json();
      state.allPairs = Array.isArray(state.payload.pairs) ? state.payload.pairs.slice() : [];
      state.outputDir = outputDir;
      document.getElementById("current-output-dir").textContent = state.payload.meta?.outputDir || outputDir;
      document.getElementById("refresh-time").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
      document.getElementById("output-dir-input").value = outputDir;
      renderSummaryCards();
      renderRunConfig();
      renderMetrics();
      renderFeatureSummary();
      renderActivityBars();
      renderReviewSummary();
      renderSimpleTable("question-dist-body", state.payload.questionDistribution || [], ["Question (activity)", "Number of submissions"], "当前没有题目提交量统计表。");
      renderSimpleTable("label-dist-body", state.payload.labelDistribution || [], ["label", "count"], "当前没有弱监督标签分布统计。");
      renderModelComparison();
      fillActivityFilter();
      renderPairs();
    }

    function bindEvents() {
      document.getElementById("pair-search").addEventListener("input", renderPairs);
      document.getElementById("activity-filter").addEventListener("change", renderPairs);
      document.getElementById("sort-filter").addEventListener("change", renderPairs);
      document.getElementById("output-dir-input").addEventListener("change", async (event) => {
        await loadDashboardData(event.target.value);
      });
      document.getElementById("refresh-button").addEventListener("click", async () => {
        await loadDashboardData(state.outputDir);
      });
    }

    bindEvents();
    loadDashboardData().catch((error) => {
      document.body.innerHTML = `<main><div class="empty">页面加载失败：${escapeHtml(error.message)}</div></main>`;
    });
  </script>
</body>
</html>
"""


def write_html_report(
    path: str | Path,
    scored_pairs: pd.DataFrame,
    metrics: dict[str, object] | None = None,
    top_k: int = 100,
    run_config: dict[str, object] | None = None,
) -> None:
    """生成静态报告。

    静态版本继续保留，方便现有 CLI 流程兼容；前端样式与 Web 项目保持一致，
    但数据仍在生成阶段直接写入 HTML。
    """

    report_path = Path(path)
    root_dir = report_path.parent.parent
    run_config = run_config or {}

    if "model_similarity" in scored_pairs.columns and not scored_pairs.empty:
        top_pairs = scored_pairs.sort_values("model_similarity", ascending=False).head(top_k).copy()
    else:
        top_pairs = scored_pairs.head(top_k).copy()

    summary_cards = _metrics_cards(
        metrics=metrics,
        pair_count=int(len(scored_pairs)),
        activity_count=int(top_pairs["activity"].nunique()) if not top_pairs.empty and "activity" in top_pairs.columns else 0,
        top_k=top_k,
    )
    payload = {
        "title": "代码相似度检测答辩看板",
        "summaryCards": summary_cards,
        "metrics": _metrics_rows(metrics),
        "runConfig": run_config,
        "pairs": _frame_to_records(top_pairs, limit=top_k),
        "activityDistribution": _activity_distribution(scored_pairs),
        "featureSummary": _feature_summary(scored_pairs),
        "meta": {
            "topK": top_k,
            "pairCount": int(len(scored_pairs)),
            "generatedFrom": report_path.parent.name,
            "outputDir": report_path.parent.name,
        },
    }
    payload.update(_read_related_tables(root_dir, report_path.parent))
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    document = _dashboard_html().replace(
        'loadDashboardData().catch((error) => {\n      document.body.innerHTML = `<main><div class="empty">页面加载失败：${escapeHtml(error.message)}</div></main>`;\n    });',
        f"""state.payload = {payload_json};
    state.allPairs = Array.isArray(state.payload.pairs) ? state.payload.pairs.slice() : [];
    document.getElementById("current-output-dir").textContent = state.payload.meta?.outputDir || "outputs";
    document.getElementById("refresh-time").textContent = "静态报告";
    renderSummaryCards();
    renderRunConfig();
    renderMetrics();
    renderFeatureSummary();
    renderActivityBars();
    renderReviewSummary();
    renderSimpleTable("question-dist-body", state.payload.questionDistribution || [], ["Question (activity)", "Number of submissions"], "当前没有题目提交量统计表。");
    renderSimpleTable("label-dist-body", state.payload.labelDistribution || [], ["label", "count"], "当前没有弱监督标签分布统计。");
    renderModelComparison();
    fillActivityFilter();
    renderPairs();""",
    )
    document = document.replace('id="refresh-button" type="button">立即刷新数据</button>', 'id="refresh-button" type="button" disabled>静态报告</button>')
    document = document.replace(
        '<select id="output-dir-input">\n            <option value="outputs">outputs</option>\n            <option value="outputs_cross_user">outputs_cross_user</option>\n          </select>',
        f'<select id="output-dir-input" disabled><option value="{report_path.parent.name}">{report_path.parent.name}</option></select>',
    )
    report_path.write_text(document, encoding="utf-8")


def get_dashboard_html() -> str:
    """返回 Web 服务主页 HTML。"""

    return _dashboard_html()
