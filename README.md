# 代码相似度检测训练建模项目

本项目用于本科毕业设计的代码部分，目标是基于 DSA Java 提交数据集实现“特征工程 + 弱监督标签构造 + 传统机器学习训练”的代码相似度检测流程。

项目主要流程包括：读取 Java 代码提交数据、构造候选代码对、执行代码预处理、提取相似度特征、通过弱监督规则生成训练标签、训练传统机器学习模型，并输出代码对相似概率、实验指标和复核样本。

## 一、环境准备

项目要求使用本地虚拟环境 `.venv`，不要直接使用系统 Python。

项目支持 Python 3.10 到 Python 3.13。Windows PowerShell 下推荐使用 Python 3.12 创建虚拟环境：

```powershell
py -3.12 -m venv .venv
```

如果电脑没有 Python 3.12，也可以改用已经安装且满足版本要求的 Python，例如：

```powershell
py -3.11 -m venv .venv
```

安装项目依赖：

```powershell
.venv\Scripts\pip.exe install -e . pytest
```

该命令会安装项目依赖，包括 `pandas`、`scikit-learn`、`joblib`，同时安装测试工具 `pytest`。

## 二、数据准备

项目默认从 `dataset` 目录读取代码提交数据。使用前需要将数据集放到项目根目录下的 `dataset` 文件夹中。

数据文件应包含代码提交相关字段，例如题目或活动编号、用户编号、提交编号和代码文本。具体字段会由 `src/similarity/data.py` 中的数据读取逻辑检查，如果缺少必要字段，程序会在运行时提示错误。

## 三、常用运行命令

后续命令都需要在项目根目录下执行，也就是包含 `pyproject.toml`、`README.md` 和 `src` 文件夹的目录。

### 1. 查看数据集概况

```powershell
.venv\Scripts\python.exe -m similarity inspect-data --data-dir dataset
```

该命令会输出数据集规模、题目数量、用户数量、提交数量等基础信息。

### 2. 训练相似度检测模型

```powershell
.venv\Scripts\python.exe -m similarity train --data-dir dataset --output-dir outputs --max-pairs-per-question 2000
```

训练过程会先构造代码对，再提取相似度特征，并根据弱监督规则生成训练标签，最后训练传统机器学习模型。

### 3. 一键运行完整流程

```powershell
.venv\Scripts\python.exe -m similarity run-pipeline --data-dir dataset --output-dir outputs --high-threshold 0.85 --low-threshold 0.35
```

其中，`--high-threshold` 是弱监督正样本阈值，基础相似度高于该值的代码对会被标为相似；`--low-threshold` 是弱监督负样本阈值，基础相似度低于该值的代码对会被标为不相似。

### 4. 比较两个 Java 文件

如果已经训练好模型，可以直接比较两个 Java 文件的相似度：

```powershell
.venv\Scripts\python.exe -m similarity compare-files a.java b.java --model outputs/model.joblib
```

其中，`a.java` 和 `b.java` 需要替换为实际 Java 文件路径。程序会输出规则特征分数和模型预测的相似概率。

### 5. 批量预测并生成复核样本

训练模型后，可以对数据集中的候选代码对进行批量预测：

```powershell
.venv\Scripts\python.exe -m similarity predict --data-dir dataset --output-dir outputs_cross_user --model outputs/model.joblib --cross-user-only
```

如果需要抽取人工复核样本，可以继续运行：

```powershell
.venv\Scripts\python.exe -m similarity create-review-sample --similar-pairs outputs_cross_user/scored_pairs.csv --output outputs_cross_user/manual_review_sample.csv --high-k 40 --medium-k 40 --low-k 40
```

其中，`scored_pairs.csv` 包含全部代码对分数，更适合按高分、中间分数、低分三层抽样；相比只从 `similar_pairs.csv` 抽样，更有利于人工复核评估误报风险。

### 6. 运行测试

如果需要确认代码环境和主要功能是否正常，可以运行测试：

```powershell
.venv\Scripts\python.exe -m pytest
```

测试通过说明预处理、特征提取、训练流程、实验分析和人工复核等核心功能可以正常工作。

## 四、主要输出文件

- `outputs/training_pairs.csv`：弱监督构造出的训练样本。
- `outputs/model.joblib`：训练后的传统机器学习模型。
- `outputs/metrics.json`：准确率、精确率、召回率、F1、ROC-AUC、混淆矩阵等指标。
- `outputs/scored_pairs.csv`：模型对全部候选代码对的打分结果，可用于分层人工复核抽样。
- `outputs/similar_pairs.csv`：模型预测出的高相似代码对。
- `outputs/report.html`：静态 HTML 可视化报告。

运行训练、预测和实验命令后，程序会生成输出目录，例如 `outputs`、`outputs_cross_user` 等。这些目录属于运行结果，不是项目源代码的一部分，可以根据需要保留、压缩或删除。

## 五、代码文件功能说明

### 1. 项目入口与命令行

#### `src/similarity/__init__.py`

项目包初始化文件，用于声明 `similarity` 是一个可导入的 Python 包。当前文件主要保存包级说明，不包含复杂业务逻辑。

#### `src/similarity/__main__.py`

模块方式运行项目时的入口文件。用户可以通过 `python -m similarity` 的方式进入命令行程序，实际命令解析和业务分发由 `cli.py` 完成。

#### `src/similarity/cli.py`

项目命令行主控文件，负责解析命令行参数并调用对应功能模块。该文件提供数据检查、模型训练、代码对预测、完整流水线运行、模型对比实验、人工复核样本生成、复核结果汇总、特征消融实验、按题目划分泛化实验以及论文表格生成等命令。

### 2. 数据读取与代码对构造

#### `src/similarity/data.py`

数据加载与数据概况分析模块。该文件定义提交记录的数据结构，负责从数据文件中读取学生代码提交，检查必要字段是否存在，并按照题目或活动编号对提交记录进行分组，同时提供数据集规模、题目数量、用户数量等统计信息。

#### `src/similarity/pairs.py`

代码对构造与弱监督标签生成模块。该文件负责在同一题目或活动下组合两两代码提交，提取代码对特征，并根据基础相似度分数生成弱监督标签：高于高阈值的样本标为相似，低于低阈值的样本标为不相似，中间不确定样本可选择丢弃或保留。

### 3. 预处理与特征工程

#### `src/similarity/preprocess.py`

Java 源代码预处理模块。主要功能包括移除注释、归一化字符串字面量、字符字面量和数字字面量、将 Java 代码切分为 Token 序列，并将普通标识符统一替换为 `IDENT`。这样可以降低变量名、常量值和注释文本对相似度判断的干扰。

#### `src/similarity/features.py`

代码相似度特征提取模块，是项目的核心算法文件之一。该文件针对一对代码提取 Token Jaccard 相似度、Token 序列相似度、关键字余弦相似度、运算符余弦相似度、结构相似度、长度相似度、行数相似度、方法数量相似度、循环数量相似度和分支数量相似度，并进一步计算用于弱监督标注的基础规则相似度。

### 4. 模型训练与预测

#### `src/similarity/modeling.py`

机器学习建模模块。该文件负责创建逻辑回归或随机森林模型，使用弱监督标签样本训练二分类模型，保存模型和评估指标，并在预测阶段输出两段代码的模型相似概率。项目最终可以将模型输出概率作为代码相似度检测结果的重要依据。

#### `src/similarity/report.py`

结果报告输出模块。该文件负责将实验指标、预测结果或分析结果保存为 JSON 文件，并可生成简单的 HTML 报告，方便查看代码相似度检测结果。

### 5. 实验与分析

#### `src/similarity/experiments.py`

模型对比实验模块。该文件用于在相同数据和特征条件下比较不同模型的分类效果，计算准确率、精确率、召回率、F1 值、ROC-AUC 等指标，并输出实验结果文件。

#### `src/similarity/strict_eval.py`

严格评估实验模块。该文件主要包含特征消融实验和按题目划分的泛化实验。特征消融实验用于观察不同特征组对模型效果的影响；按题目划分实验用于检查模型在未见过题目上的泛化能力。

#### `src/similarity/analysis.py`

论文表格与统计结果生成模块。该文件负责根据实验输出生成论文中使用的数据表格，例如数据集统计表、模型对比表、特征消融结果表和泛化实验结果表。

#### `src/similarity/review.py`

人工复核辅助分析模块。该文件用于从预测结果中抽取待人工检查的代码对样本，支持按分数区间分层抽样，并在人工标注后汇总复核结果、生成分析段落，帮助验证高相似代码对输出的合理性。

### 6. 测试文件

#### `tests/test_preprocess.py`

预处理模块测试文件，主要验证注释移除、标识符归一化和字面量归一化等逻辑是否符合预期。

#### `tests/test_features.py`

特征提取模块测试文件，主要验证相似代码能够得到较高基础相似度，不同代码的序列相似度相对较低。

#### `tests/test_pipeline.py`

训练流程与命令行相关测试文件，覆盖小样本训练流程、跨用户代码对过滤和人工复核命令参数等功能。

#### `tests/test_experiments.py`

模型对比实验测试文件，验证对比实验能够正常运行并生成预期指标文件。

#### `tests/test_analysis.py`

论文表格生成测试文件，验证统计表和实验结果表能够按照预期写入输出目录。

#### `tests/test_review.py`

人工复核模块测试文件，验证复核样本生成、分层抽样、复核结果汇总和分析文本生成等功能。

#### `tests/test_strict_eval.py`

严格评估模块测试文件，验证特征消融实验和按题目划分泛化实验能够输出对应的汇总结果和评估指标。

## 六、其他项目文件

### `pyproject.toml`

项目配置文件，定义项目名称、版本、Python 版本要求、依赖包、命令行入口和测试配置。安装项目后，可通过配置的命令行入口运行相似度检测程序。

### `.gitignore`

Git 忽略规则文件，用于排除虚拟环境、缓存、测试缓存、构建产物、输出结果和系统临时文件，避免把本地生成文件提交到版本库。
