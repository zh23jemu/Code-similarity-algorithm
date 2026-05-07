from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .modeling import predict_pair_similarity


def _json_bytes(data: dict[str, Any]) -> bytes:
    """将字典编码为 UTF-8 JSON 响应体。"""

    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def _read_optional_json(path: Path) -> dict[str, object] | None:
    """读取可选 JSON 文件；文件不存在或格式异常时返回空值。"""

    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_output_dir(root_dir: Path, output_dir: str) -> Path:
    """将前端传入的输出目录解析为项目内绝对路径。"""

    candidate = Path(output_dir)
    if not candidate.is_absolute():
        candidate = root_dir / candidate
    return candidate.resolve()


def _resolve_model_path(root_dir: Path, output_dir: str, model_path: str | None = None) -> Path:
    """解析模型文件路径，默认使用输出目录下的 `model.joblib`。"""

    if model_path:
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = root_dir / candidate
        return candidate.resolve()
    return (_resolve_output_dir(root_dir, output_dir) / "model.joblib").resolve()


def build_status_payload(root_dir: str | Path, output_dir: str = "outputs", model_path: str | None = None) -> dict[str, object]:
    """构建前端页面需要的模型状态、参数和样本摘要。"""

    root_path = Path(root_dir).resolve()
    output_path = _resolve_output_dir(root_path, output_dir)
    resolved_model_path = _resolve_model_path(root_path, output_dir, model_path)

    metrics = _read_optional_json(output_path / "metrics.json") or {}
    run_config = _read_optional_json(output_path / "run_config.json") or {}

    summary_items = [
        {
            "label": "正样本",
            "value": int(metrics.get("positive_samples", 0) or 0),
            "tone": "green",
        },
        {
            "label": "负样本",
            "value": int(metrics.get("negative_samples", 0) or 0),
            "tone": "blue",
        },
        {
            "label": "未标注",
            "value": max(int(metrics.get("samples", 0) or 0) - int(metrics.get("positive_samples", 0) or 0) - int(metrics.get("negative_samples", 0) or 0), 0),
            "tone": "orange",
        },
        {
            "label": "总样本",
            "value": int(metrics.get("samples", 0) or 0),
            "tone": "purple",
        },
    ]

    config_items = [
        {"key": "数据集目录", "value": str(run_config.get("data_dir", "dataset"))},
        {"key": "输出目录", "value": str(output_path.relative_to(root_path)) if output_path.is_relative_to(root_path) else str(output_path)},
        {"key": "模型类型", "value": str(run_config.get("model_type", metrics.get("model_type", "random_forest")))},
        {"key": "题目 ID / Activity", "value": str(run_config.get("question_id") or "留空表示全部题目")},
        {"key": "每题最大代码对数量", "value": str(run_config.get("max_pairs_per_question", 2000))},
        {"key": "高阈值 high_threshold", "value": str(run_config.get("high_threshold", 0.85))},
        {"key": "低阈值 low_threshold", "value": str(run_config.get("low_threshold", 0.35))},
        {"key": "输出相似度阈值 threshold", "value": str(run_config.get("similarity_threshold", 0.85))},
    ]

    return {
        "projectTitle": "Java 代码相似度检测",
        "modelLoaded": resolved_model_path.exists(),
        "modelPath": str(resolved_model_path.relative_to(root_path)) if resolved_model_path.is_relative_to(root_path) else str(resolved_model_path),
        "outputDir": str(output_path.relative_to(root_path)) if output_path.is_relative_to(root_path) else str(output_path),
        "metrics": metrics,
        "runConfig": run_config,
        "summaryItems": summary_items,
        "configItems": config_items,
    }


def build_compare_payload(code_a: str, code_b: str, model_path: str | Path) -> dict[str, object]:
    """对两段代码执行相似度预测，并整理为前端展示结果。"""

    result = predict_pair_similarity(code_a, code_b, model_path)
    feature_rows = [
        {"label": "模型相似度", "key": "model_similarity", "value": float(result.get("model_similarity", 0.0))},
        {"label": "基础规则相似度", "key": "base_similarity", "value": float(result.get("base_similarity", 0.0))},
        {"label": "Token Jaccard", "key": "token_jaccard", "value": float(result.get("token_jaccard", 0.0))},
        {"label": "Token 序列相似度", "key": "token_sequence_similarity", "value": float(result.get("token_sequence_similarity", 0.0))},
        {"label": "关键字余弦相似度", "key": "keyword_cosine", "value": float(result.get("keyword_cosine", 0.0))},
        {"label": "运算符余弦相似度", "key": "operator_cosine", "value": float(result.get("operator_cosine", 0.0))},
        {"label": "结构相似度", "key": "structure_cosine", "value": float(result.get("structure_cosine", 0.0))},
        {"label": "长度相似度", "key": "length_similarity", "value": float(result.get("length_similarity", 0.0))},
        {"label": "行数相似度", "key": "line_count_similarity", "value": float(result.get("line_count_similarity", 0.0))},
        {"label": "方法数量相似度", "key": "method_count_similarity", "value": float(result.get("method_count_similarity", 0.0))},
        {"label": "循环数量相似度", "key": "loop_count_similarity", "value": float(result.get("loop_count_similarity", 0.0))},
        {"label": "分支数量相似度", "key": "branch_count_similarity", "value": float(result.get("branch_count_similarity", 0.0))},
    ]
    return {
        "summary": {
            "model_similarity": float(result.get("model_similarity", 0.0)),
            "base_similarity": float(result.get("base_similarity", 0.0)),
        },
        "features": feature_rows,
        "raw": result,
    }


def get_compare_webui_html() -> str:
    """返回代码相似度检测前端页面 HTML。"""

    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Java 代码相似度检测</title>
  <style>
    :root {
      --bg: #f3f6ff;
      --paper: #ffffff;
      --panel: rgba(255, 255, 255, 0.92);
      --line: #dbe4ff;
      --line-strong: #bfd0ff;
      --ink: #1f2940;
      --muted: #69738b;
      --primary: #2f67f6;
      --primary-strong: #1e4fcb;
      --primary-soft: #edf2ff;
      --green: #27b36a;
      --green-soft: #ebfff3;
      --blue: #4f7ff7;
      --blue-soft: #eef4ff;
      --orange: #ff8e33;
      --orange-soft: #fff4ea;
      --purple: #8456f7;
      --purple-soft: #f4efff;
      --shadow: 0 18px 38px rgba(50, 79, 153, 0.12);
      --radius-xl: 26px;
      --radius-lg: 20px;
      --radius-md: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(114, 168, 255, 0.24), transparent 26%),
        linear-gradient(180deg, #f8faff 0%, #eef3ff 100%);
      min-height: 100vh;
    }
    .app {
      min-height: 100vh;
      padding: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(191, 208, 255, 0.8);
      border-radius: var(--radius-xl);
      box-shadow: var(--shadow);
    }
    .content {
      display: grid;
      gap: 18px;
      min-width: 0;
      width: min(1320px, 100%);
      margin: 0 auto;
    }
    .hero {
      padding: 24px 26px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .hero h2 {
      margin: 0;
      font-size: 34px;
      letter-spacing: -0.04em;
    }
    .hero p {
      margin: 10px 0 0;
      color: var(--muted);
      line-height: 1.8;
      max-width: 860px;
    }
    .hero-status {
      flex: none;
      border-radius: 999px;
      padding: 10px 16px;
      font-size: 14px;
      font-weight: 700;
      background: var(--green-soft);
      color: var(--green);
      white-space: nowrap;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(300px, 0.74fr) minmax(620px, 1.42fr) minmax(240px, 0.52fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      padding: 22px;
      min-width: 0;
    }
    .panel h3 {
      margin: 0 0 16px;
      font-size: 18px;
    }
    .config-list {
      display: grid;
      gap: 12px;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    .field label {
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
    }
    .field input,
    .field select,
    .field textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
      color: var(--ink);
      padding: 12px 14px;
      font: inherit;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .field input:focus,
    .field select:focus,
    .field textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(47, 103, 246, 0.12);
    }
    .field textarea {
      min-height: 320px;
      resize: vertical;
      font-family: "Cascadia Code", "Consolas", monospace;
      line-height: 1.65;
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 8px;
    }
    .btn {
      border: none;
      border-radius: 14px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn:disabled { cursor: not-allowed; opacity: 0.65; transform: none; }
    .btn.primary {
      background: linear-gradient(180deg, #2f67f6, #2458db);
      color: white;
      box-shadow: 0 12px 22px rgba(47, 103, 246, 0.26);
    }
    .btn.secondary {
      background: #edf2ff;
      color: #2f67f6;
    }
    .code-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 14px;
    }
    .helper {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.8;
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .stat-card {
      border-radius: 16px;
      padding: 16px;
      border: 1px solid rgba(191, 208, 255, 0.7);
      min-width: 0;
    }
    .stat-card .label {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
      font-weight: 700;
    }
    .stat-card .value {
      font-size: clamp(20px, 2vw, 26px);
      line-height: 1.05;
      font-weight: 800;
      color: var(--ink);
      letter-spacing: -0.03em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: clip;
    }
    .stat-green { background: var(--green-soft); }
    .stat-blue { background: var(--blue-soft); }
    .stat-orange { background: var(--orange-soft); }
    .stat-purple { background: var(--purple-soft); }
    .result-card {
      margin-top: 18px;
      border-radius: 18px;
      padding: 18px;
      background: linear-gradient(180deg, #f7f9ff, #eff4ff);
      border: 1px solid rgba(191, 208, 255, 0.8);
    }
    .result-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    .result-score {
      font-size: 42px;
      line-height: 1;
      font-weight: 900;
      color: #2d5be1;
      letter-spacing: -0.04em;
    }
    .result-sub {
      color: var(--muted);
      font-size: 13px;
    }
    .feature-table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 14px;
      overflow: hidden;
    }
    .feature-table th,
    .feature-table td {
      padding: 10px 12px;
      border-bottom: 1px solid #edf1ff;
      text-align: left;
      font-size: 14px;
    }
    .feature-table th {
      background: #f3f6ff;
      color: var(--muted);
      font-size: 13px;
    }
    .feature-table tr:last-child td { border-bottom: none; }
    .config-meta {
      margin-top: 14px;
      padding-top: 14px;
      border-top: 1px dashed var(--line-strong);
      display: grid;
      gap: 10px;
    }
    .meta-item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .meta-item strong {
      color: var(--ink);
      text-align: right;
    }
    .empty-box {
      border: 1px dashed var(--line-strong);
      border-radius: 16px;
      padding: 18px;
      color: var(--muted);
      background: #fafcff;
      line-height: 1.8;
    }
    .error-box {
      margin-top: 14px;
      border-radius: 14px;
      padding: 14px 16px;
      background: #fff2f2;
      color: #b94141;
      border: 1px solid #ffd2d2;
      line-height: 1.7;
    }
    @media (max-width: 1280px) {
      .layout { grid-template-columns: 1fr; }
      .code-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 900px) {
      .hero { flex-direction: column; align-items: flex-start; }
    }
    @media (max-width: 640px) {
      .app { padding: 10px; gap: 10px; }
      .panel, .hero { padding: 16px; }
      .stats-grid { grid-template-columns: 1fr 1fr; }
      .button-row { flex-direction: column; }
      .btn { width: 100%; }
      .stat-card { padding: 14px; }
      .stat-card .value { font-size: clamp(18px, 5vw, 22px); }
    }
  </style>
</head>
<body>
  <div class="app">
    <main class="content">
      <section class="panel hero">
        <div>
          <h2>Java 代码相似度检测</h2>
          <p>前端页面负责接收两段 Java 代码并发起检测请求；后端将复用现有 Python 检测能力，返回模型相似度和关键特征结果，实现“一键检测”演示。</p>
        </div>
        <div class="hero-status" id="hero-status">模型状态读取中...</div>
      </section>

      <section class="layout">
        <section class="panel">
          <h3>检测任务配置</h3>
          <div class="config-list">
            <div class="field">
              <label for="output-dir">输出目录</label>
              <select id="output-dir">
                <option value="outputs">outputs</option>
                <option value="outputs_cross_user">outputs_cross_user</option>
              </select>
            </div>
            <div class="field">
              <label for="model-path">模型文件路径</label>
              <input id="model-path" type="text" placeholder="默认自动使用输出目录下的 model.joblib">
            </div>
          </div>
          <div class="button-row">
            <button class="btn secondary" id="reload-status" type="button">重载模型状态</button>
            <button class="btn primary" id="run-compare" type="button">一键检测</button>
            <button class="btn secondary" id="fill-demo" type="button">填充示例</button>
          </div>
          <div class="config-meta" id="config-meta"></div>
        </section>

        <section class="panel">
          <h3>代码输入 / 文件检测</h3>
          <div class="code-grid">
            <div class="field">
              <label for="code-a">代码 A</label>
              <textarea id="code-a" placeholder="请粘贴第一段 Java 代码"></textarea>
            </div>
            <div class="field">
              <label for="code-b">代码 B</label>
              <textarea id="code-b" placeholder="请粘贴第二段 Java 代码"></textarea>
            </div>
          </div>
          <p class="helper">前端页面用于触发命令与展示交互；真实检测由后端调用项目现有的 Python 预测逻辑完成。建议先训练好模型，再在这里粘贴两段 Java 代码进行对比。</p>
          <div id="compare-error"></div>
          <div id="compare-result" class="empty-box">点击“一键检测”后，这里会展示模型相似度、基础规则相似度以及各项特征分数。</div>
        </section>

        <section class="panel">
          <h3>模型与样本状态</h3>
          <div class="stats-grid" id="summary-stats"></div>
        </section>
      </section>
    </main>
  </div>

  <script>
    const state = {
      status: null,
      compare: null,
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatMetric(value) {
      if (value === null || value === undefined || value === "") return "暂无";
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
      return String(value);
    }

    function scoreTone(score) {
      if (score >= 0.85) return "高度相似";
      if (score >= 0.6) return "中等相似";
      if (score >= 0.35) return "低度相似";
      return "差异较大";
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `请求失败：${response.status}`);
      }
      return payload;
    }

    function renderStatus() {
      const status = state.status;
      const heroStatus = document.getElementById("hero-status");
      const stats = document.getElementById("summary-stats");
      const meta = document.getElementById("config-meta");

      if (!status) {
        heroStatus.textContent = "模型状态读取失败";
        stats.innerHTML = '<div class="empty-box">当前无法读取模型状态。</div>';
        meta.innerHTML = "";
        return;
      }

      heroStatus.textContent = status.modelLoaded ? `模型已加载：${status.runConfig?.model_type || status.metrics?.model_type || "Random Forest"}` : "模型未加载";
      heroStatus.style.background = status.modelLoaded ? "var(--green-soft)" : "#fff2f2";
      heroStatus.style.color = status.modelLoaded ? "var(--green)" : "#b94141";

      const toneClass = {
        green: "stat-green",
        blue: "stat-blue",
        orange: "stat-orange",
        purple: "stat-purple",
      };
      stats.innerHTML = (status.summaryItems || []).map((item) => `
        <article class="stat-card ${toneClass[item.tone] || "stat-blue"}">
          <div class="label">${escapeHtml(item.label)}</div>
          <div class="value">${escapeHtml(formatMetric(item.value))}</div>
        </article>
      `).join("");

      meta.innerHTML = (status.configItems || []).map((item) => `
        <div class="meta-item">
          <span>${escapeHtml(item.key)}</span>
          <strong>${escapeHtml(item.value)}</strong>
        </div>
      `).join("") + `
        <div class="meta-item">
          <span>模型文件</span>
          <strong>${escapeHtml(status.modelPath || "暂无")}</strong>
        </div>
      `;
    }

    function renderCompareResult() {
      const container = document.getElementById("compare-result");
      const errorBox = document.getElementById("compare-error");
      errorBox.innerHTML = "";

      if (!state.compare) {
        container.className = "empty-box";
        container.innerHTML = "点击“一键检测”后，这里会展示模型相似度、基础规则相似度以及各项特征分数。";
        return;
      }

      const summary = state.compare.summary;
      const features = state.compare.features || [];
      container.className = "result-card";
      container.innerHTML = `
        <div class="result-head">
          <div>
            <div class="result-score">${escapeHtml(formatMetric(summary.model_similarity))}</div>
            <div class="result-sub">模型判断：${escapeHtml(scoreTone(summary.model_similarity))}</div>
          </div>
          <div class="result-sub">基础规则相似度：${escapeHtml(formatMetric(summary.base_similarity))}</div>
        </div>
        <table class="feature-table">
          <thead>
            <tr><th>特征项</th><th>数值</th></tr>
          </thead>
          <tbody>
            ${features.map((item) => `<tr><td>${escapeHtml(item.label)}</td><td>${escapeHtml(formatMetric(item.value))}</td></tr>`).join("")}
          </tbody>
        </table>
      `;
    }

    async function loadStatus() {
      const outputDir = document.getElementById("output-dir").value;
      const modelPath = document.getElementById("model-path").value.trim();
      const query = new URLSearchParams({ output_dir: outputDir });
      if (modelPath) query.set("model_path", modelPath);
      state.status = await requestJson(`/api/status?${query.toString()}`, { cache: "no-store" });
      renderStatus();
    }

    async function runCompare() {
      const codeA = document.getElementById("code-a").value.trim();
      const codeB = document.getElementById("code-b").value.trim();
      const outputDir = document.getElementById("output-dir").value;
      const modelPath = document.getElementById("model-path").value.trim();
      const errorBox = document.getElementById("compare-error");

      if (!codeA || !codeB) {
        errorBox.innerHTML = '<div class="error-box">请先输入两段 Java 代码，再执行一键检测。</div>';
        return;
      }

      const button = document.getElementById("run-compare");
      button.disabled = true;
      button.textContent = "检测中...";
      try {
        state.compare = await requestJson("/api/compare", {
          method: "POST",
          body: JSON.stringify({
            code_a: codeA,
            code_b: codeB,
            output_dir: outputDir,
            model_path: modelPath || null,
          }),
        });
        renderCompareResult();
      } catch (error) {
        errorBox.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
      } finally {
        button.disabled = false;
        button.textContent = "一键检测";
      }
    }

    function fillDemo() {
      document.getElementById("code-a").value = `public class SumA {\n    public static int sum(int n) {\n        int s = 0;\n        for (int i = 1; i <= n; i++) {\n            s += i;\n        }\n        return s;\n    }\n}`;
      document.getElementById("code-b").value = `public class SumB {\n    public static int sum(int n) {\n        int result = 0;\n        for (int i = 1; i <= n; i++) {\n            result = result + i;\n        }\n        return result;\n    }\n}`;
    }

    function bindEvents() {
      document.getElementById("reload-status").addEventListener("click", loadStatus);
      document.getElementById("run-compare").addEventListener("click", runCompare);
      document.getElementById("fill-demo").addEventListener("click", fillDemo);
      document.getElementById("output-dir").addEventListener("change", async () => {
        state.compare = null;
        renderCompareResult();
        await loadStatus();
      });
    }

    bindEvents();
    renderCompareResult();
    loadStatus().catch((error) => {
      document.getElementById("compare-error").innerHTML = `<div class="error-box">初始化状态失败：${escapeHtml(error.message)}</div>`;
    });
  </script>
</body>
</html>
"""


def run_webui_server(
    root_dir: str | Path,
    output_dir: str = "outputs",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """启动本地前端检测页面 Web 服务。"""

    root_path = Path(root_dir).resolve()
    page_html = get_compare_webui_html().encode("utf-8")

    class WebUiHandler(BaseHTTPRequestHandler):
        """提供前端页面、模型状态接口和代码对比接口。"""

        server_version = "SimilarityWebUI/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(HTTPStatus.OK, page_html, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                params = parse_qs(parsed.query)
                current_output = params.get("output_dir", [output_dir])[0]
                current_model = params.get("model_path", [None])[0]
                try:
                    payload = build_status_payload(root_path, current_output, current_model)
                except Exception as exc:  # pragma: no cover
                    self._send_bytes(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        _json_bytes({"error": f"读取模型状态失败: {exc}"}),
                        "application/json; charset=utf-8",
                    )
                    return
                self._send_bytes(HTTPStatus.OK, _json_bytes(payload), "application/json; charset=utf-8")
                return
            self._send_bytes(
                HTTPStatus.NOT_FOUND,
                _json_bytes({"error": "未找到对应资源"}),
                "application/json; charset=utf-8",
            )

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/compare":
                self._send_bytes(
                    HTTPStatus.NOT_FOUND,
                    _json_bytes({"error": "未找到对应资源"}),
                    "application/json; charset=utf-8",
                )
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(body.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                self._send_bytes(
                    HTTPStatus.BAD_REQUEST,
                    _json_bytes({"error": "请求体必须是合法的 JSON。"}),
                    "application/json; charset=utf-8",
                )
                return

            code_a = str(payload.get("code_a", "")).strip()
            code_b = str(payload.get("code_b", "")).strip()
            current_output = str(payload.get("output_dir", output_dir) or output_dir)
            current_model = payload.get("model_path")

            if not code_a or not code_b:
                self._send_bytes(
                    HTTPStatus.BAD_REQUEST,
                    _json_bytes({"error": "请同时提供代码 A 和代码 B。"}),
                    "application/json; charset=utf-8",
                )
                return

            resolved_model = _resolve_model_path(root_path, current_output, str(current_model).strip() if current_model else None)
            if not resolved_model.exists():
                self._send_bytes(
                    HTTPStatus.BAD_REQUEST,
                    _json_bytes({"error": f"模型文件不存在：{resolved_model}。请先训练模型或检查模型路径。"}),
                    "application/json; charset=utf-8",
                )
                return

            try:
                result = build_compare_payload(code_a, code_b, resolved_model)
            except Exception as exc:  # pragma: no cover
                self._send_bytes(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    _json_bytes({"error": f"执行代码相似度检测失败: {exc}"}),
                    "application/json; charset=utf-8",
                )
                return

            self._send_bytes(HTTPStatus.OK, _json_bytes(result), "application/json; charset=utf-8")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            """保持服务端安静，避免终端被每次刷新刷屏。"""

        def _send_bytes(self, status: HTTPStatus, content: bytes, content_type: str) -> None:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer((host, port), WebUiHandler)
    print(
        json.dumps(
            {
                "message": "代码相似度检测前端页面已启动",
                "url": f"http://{host}:{port}/",
                "root_dir": str(root_path),
                "default_output_dir": output_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止前端检测页面 Web 服务。")
    finally:
        server.server_close()
