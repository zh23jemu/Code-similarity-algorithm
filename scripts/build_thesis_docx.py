from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = PROJECT_ROOT / "2.1工科论文参考模板.docx"
DRAFT_PATH = PROJECT_ROOT / "thesis_draft.md"
OUTPUT_PATH = PROJECT_ROOT / "毕业论文_代码相似度检测.docx"
PAPER_TABLES_DIR = PROJECT_ROOT / "paper_tables"
ABLATION_TABLE_PATH = PROJECT_ROOT / "outputs_ablation" / "ablation_metrics.csv"
QUESTION_SPLIT_PATH = PROJECT_ROOT / "outputs_question_split" / "question_split_metrics.json"


def normalize_text(raw_text: str) -> str:
    """去除 Markdown 痕迹并做轻量文本清洗。"""

    return raw_text.replace("**", "").strip()


def markdown_to_sections(markdown_text: str) -> list[tuple[int, str]]:
    """把 Markdown 粗略转换成标题与正文段落序列。

    这里只处理当前草稿用到的一级、二级、三级标题和普通段落，
    目的是尽量把草稿内容按模板样式写入 Word，而不是做完整 Markdown 解析。
    """

    sections: list[tuple[int, str]] = []
    for raw_line in markdown_text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            sections.append((0, ""))
            continue
        if line.startswith("### "):
            sections.append((3, line[4:].strip()))
        elif line.startswith("## "):
            sections.append((2, line[3:].strip()))
        elif line.startswith("# "):
            sections.append((1, line[2:].strip()))
        else:
            sections.append((0, line))
    return sections


def clear_document_body(doc: Document) -> None:
    """清空模板中的正文元素，但保留最后的节属性定义。"""

    body = doc._element.body
    children = list(body)
    for child in children[:-1]:
        body.remove(child)


def add_paragraph_with_style(doc: Document, text: str, style_name: str) -> None:
    paragraph = doc.add_paragraph(style=style_name)
    if text:
        paragraph.add_run(text)


def add_reference_list(doc: Document) -> None:
    """添加当前论文的参考文献列表。"""

    references = [
        "[1] DSA Dataset[DB/OL]. Zenodo, 2026. Available: https://zenodo.org/records/18136250.",
        "[2] 吴贺. 前后端解耦模式及开发[J]. 计算机系统应用, 2017, 26(2): 217-221.",
        "[3] 李智慧. 大型网站技术架构[M]. 北京: 电子工业出版社, 2013.",
        "[4] 许令波. 深入分析Java Web技术内幕: 第2版[M]. 北京: 电子工业出版社, 2014.",
        "[5] Breiman L. Random Forests[J]. Machine Learning, 2001, 45(1): 5-32.",
        "[6] Ratner A, Bach S, Ehrenberg H, et al. Snorkel: Rapid Training Data Creation with Weak Supervision[J]. Proceedings of the VLDB Endowment, 2017, 11(3): 269-282.",
        "[7] Jurafsky D, Martin J H. Speech and Language Processing[M]. 3rd draft ed. 2023.",
        "[8] Manning C D, Raghavan P, Schutze H. Introduction to Information Retrieval[M]. Cambridge: Cambridge University Press, 2008.",
        "[9] 王斌, 张海军. 程序代码相似度检测研究综述[J]. 计算机工程与应用, 相关年份后续可按学校数据库检索补全.",
        "[10] 弱监督学习、代码相似度检测及程序查重相关中文文献，后续可继续按学校图书馆数据库补充与替换。",
    ]
    for item in references:
        add_paragraph_with_style(doc, item, "Normal")


def add_table_from_dataframe(doc: Document, dataframe: pd.DataFrame, caption: str, source: str) -> None:
    """把真实实验数据表插入到论文中。"""

    if dataframe.empty:
        return
    add_paragraph_with_style(doc, caption, "Caption")
    table = doc.add_table(rows=1, cols=len(dataframe.columns))
    table.style = "Table Grid"
    for idx, column in enumerate(dataframe.columns):
        table.rows[0].cells[idx].text = str(column)
    for row in dataframe.itertuples(index=False):
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = str(value)
    add_paragraph_with_style(doc, f"数据来源：{source}", "annotation text")


def add_toc_field(paragraph) -> None:
    """插入 Word 目录域，打开文档后可更新目录。"""

    run = paragraph.add_run()
    fld_char_begin = OxmlElement("w:fldChar")
    fld_char_begin.set(qn("w:fldCharType"), "begin")

    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = r'TOC \o "1-3" \h \z \u'

    fld_char_separate = OxmlElement("w:fldChar")
    fld_char_separate.set(qn("w:fldCharType"), "separate")

    placeholder = OxmlElement("w:t")
    placeholder.text = "目录将在 Word 中右键更新域后显示。"

    fld_char_end = OxmlElement("w:fldChar")
    fld_char_end.set(qn("w:fldCharType"), "end")

    run._r.append(fld_char_begin)
    run._r.append(instr_text)
    run._r.append(fld_char_separate)
    run._r.append(placeholder)
    run._r.append(fld_char_end)


def insert_tables_if_needed(doc: Document, text: str) -> None:
    """在对应段落后插入论文实验表。"""

    if text.endswith("题目提交数量分布见表1，训练样本标签分布见表2。"):
        question_table = PAPER_TABLES_DIR / "table_question_distribution.csv"
        label_table = PAPER_TABLES_DIR / "table_label_distribution.csv"
        if question_table.exists():
            df = pd.read_csv(question_table).head(10)
            add_table_from_dataframe(doc, df, "表1. 题目提交数量分布（前10项）", "Zenodo DSA Dataset 统计结果")
        if label_table.exists():
            df = pd.read_csv(label_table)
            add_table_from_dataframe(doc, df, "表2. 训练样本标签分布", "本系统跨用户训练样本统计")
    elif text.endswith("模型对比结果见表3。"):
        compare_table = PAPER_TABLES_DIR / "table_model_comparison.csv"
        if compare_table.exists():
            df = pd.read_csv(compare_table)
            columns = [column for column in ["method", "accuracy", "precision", "recall", "f1", "roc_auc"] if column in df.columns]
            add_table_from_dataframe(doc, df[columns], "表3. 模型对比实验结果", "本系统模型对比实验结果")
    elif text.endswith("消融实验结果见表4。"):
        if ABLATION_TABLE_PATH.exists():
            df = pd.read_csv(ABLATION_TABLE_PATH)
            columns = [column for column in ["ablation", "accuracy", "precision", "recall", "f1", "roc_auc"] if column in df.columns]
            add_table_from_dataframe(doc, df[columns], "表4. 特征消融实验结果", "本系统特征消融实验结果")
    elif text.endswith("严格泛化实验结果见表5。"):
        if QUESTION_SPLIT_PATH.exists():
            metrics = json.loads(QUESTION_SPLIT_PATH.read_text(encoding="utf-8"))
            df = pd.DataFrame(
                [
                    {
                        "model_type": metrics["model_type"],
                        "train_activities": metrics["train_activities"],
                        "test_activities": metrics["test_activities"],
                        "train_samples": metrics["train_samples"],
                        "test_samples": metrics["test_samples"],
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                    }
                ]
            )
            add_table_from_dataframe(doc, df, "表5. 按题目划分的严格泛化实验结果", "本系统严格泛化实验结果")
    elif text.endswith("人工辅助复核统计结果见表6。"):
        review_table = PAPER_TABLES_DIR / "table_manual_review_summary.csv"
        if review_table.exists():
            df = pd.read_csv(review_table)
            add_table_from_dataframe(doc, df, "表6. 高分样本人工辅助复核统计结果", "本系统高分样本人工辅助复核结果")
    elif "人工辅助复核统计结果见表6。" in text:
        review_table = PAPER_TABLES_DIR / "table_manual_review_summary.csv"
        if review_table.exists():
            df = pd.read_csv(review_table)
            add_table_from_dataframe(doc, df, "表6. 高分样本人工辅助复核统计结果", "本系统高分样本人工辅助复核结果")


def build_document() -> None:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"找不到模板文件: {TEMPLATE_PATH}")
    if not DRAFT_PATH.exists():
        raise FileNotFoundError(f"找不到论文草稿文件: {DRAFT_PATH}")

    shutil.copyfile(TEMPLATE_PATH, OUTPUT_PATH)
    doc = Document(OUTPUT_PATH)
    clear_document_body(doc)

    draft_text = DRAFT_PATH.read_text(encoding="utf-8")
    sections = markdown_to_sections(draft_text)

    title_cn = "基于弱监督训练建模的本科代码相似度检测算法设计与实现"
    title_en = "Design and Implementation of Undergraduate Code Similarity Detection Based on Weakly Supervised Training"

    filtered_sections = [(level, text) for level, text in sections if text not in {title_cn, title_en}]

    # 摘要部分
    add_paragraph_with_style(doc, title_cn, "摘要、参考文献、注释")
    for level, text in filtered_sections:
        if level == 2 and text == "摘要":
            continue
        if level == 2 and text == "Abstract":
            doc.add_paragraph("")
            add_paragraph_with_style(doc, title_en, "摘要、参考文献、注释")
            continue
        if level == 2 and text == "第一章 绪论":
            break
        if text:
            add_paragraph_with_style(doc, text, "Normal")
        else:
            doc.add_paragraph("")

    # 目录页
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    add_paragraph_with_style(doc, "目录", "一级标题")
    toc_paragraph = doc.add_paragraph(style="Normal")
    add_toc_field(toc_paragraph)

    # 新页开始正文
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    chapter_index = 0
    in_body = False
    for level, text in filtered_sections:
        if level == 2 and text == "摘要":
            continue
        if level == 2 and text == "Abstract":
            continue
        if level == 2 and text.startswith("第") and "章" in text:
            in_body = True
            chapter_index += 1
            chapter_title = text.split(" ", 1)[1] if " " in text else text
            chinese_number = ["一", "二", "三", "四", "五", "六", "七", "八", "九"][chapter_index - 1]
            add_paragraph_with_style(doc, f"{chinese_number}、{chapter_title}", "一级标题")
            continue
        if not in_body:
            continue
        if level == 3:
            add_paragraph_with_style(doc, text, "二级标题")
        elif level == 1:
            continue
        else:
            if text == "参考文献（待整理）":
                doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
                add_paragraph_with_style(doc, "参考文献", "一级标题")
                add_paragraph_with_style(
                    doc,
                    "放在新的一页，中文在前，英文在后，各自按照第一作者姓氏排列。正文中所列文献和这里列出文献要一一对应。",
                    "annotation text",
                )
                add_reference_list(doc)
                break
            style_name = "Normal" if text else "Body Text"
            add_paragraph_with_style(doc, text, style_name)
            if text:
                insert_tables_if_needed(doc, text)

    doc.save(OUTPUT_PATH)


if __name__ == "__main__":
    build_document()
