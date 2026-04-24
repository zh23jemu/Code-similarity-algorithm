# 代码相似度检测训练建模项目

本项目用于本科毕业设计的代码部分，目标是基于 DSA Java 提交数据集实现“特征工程 + 弱监督标签构造 + 传统机器学习训练”的代码相似度检测流程。

## 环境准备

项目要求使用本地虚拟环境 `.venv`，不要直接使用系统 Python。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip.exe install -e . pytest
```

## 常用命令

查看数据集概况：

```powershell
.venv\Scripts\python.exe -m similarity inspect-data --data-dir dataset
```

训练模型：

```powershell
.venv\Scripts\python.exe -m similarity train --data-dir dataset --output-dir outputs --max-pairs-per-question 2000
```

一键运行完整流程：

```powershell
.venv\Scripts\python.exe -m similarity run-pipeline --data-dir dataset --output-dir outputs --high-threshold 0.85 --low-threshold 0.35
```

比较两个 Java 文件：

```powershell
.venv\Scripts\python.exe -m similarity compare-files a.java b.java --model outputs/model.joblib
```

生成更可信的分层人工复核样本：

```powershell
.venv\Scripts\python.exe -m similarity predict --data-dir dataset --output-dir outputs_cross_user --model outputs_cross_user/model.joblib --cross-user-only
.venv\Scripts\python.exe -m similarity create-review-sample --similar-pairs outputs_cross_user/scored_pairs.csv --output outputs_cross_user/manual_review_sample.csv --high-k 40 --medium-k 40 --low-k 40
```

其中 `scored_pairs.csv` 包含全部代码对分数，更适合按高分、中间分数、低分三层抽样；相比只从 `similar_pairs.csv` 抽样，更有利于人工复核评估误报风险。

## 输出文件

- `outputs/training_pairs.csv`：弱监督构造出的训练样本。
- `outputs/model.joblib`：训练后的传统机器学习模型。
- `outputs/metrics.json`：准确率、精确率、召回率、F1、ROC-AUC、混淆矩阵等指标。
- `outputs/scored_pairs.csv`：模型对全部候选代码对的打分结果，可用于分层人工复核抽样。
- `outputs/similar_pairs.csv`：模型预测出的高相似代码对。
- `outputs/report.html`：静态 HTML 可视化报告。
