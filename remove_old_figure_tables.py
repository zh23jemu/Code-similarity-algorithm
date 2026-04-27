"""删除图1和图2前的旧表格式图示。

这些对象是原文已有的结构图/模块图，但在渲染器中被压成竖排文字。
新生成图片已经插入到同一图题前，因此只移除旧表格对象，不改正文段落。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DOCX_PATH = Path(r"C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx")
TMP_PATH = DOCX_PATH.with_name(DOCX_PATH.stem + "_old_tables_removed.docx")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def element_text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(qn("t"))).strip()


def is_old_table_figure(element: ET.Element) -> bool:
    text = element_text(element)
    markers = [
        "数据读取代码预处理特征提取弱监督标注模型训练相似度预测",
        "代码相似度检测系统数据加载模块预处理模块特征工程模块",
    ]
    return element.tag == qn("tbl") and any(marker in text for marker in markers)


def main() -> None:
    with zipfile.ZipFile(DOCX_PATH, "r") as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]

    output_entries: list[tuple[str, bytes]] = []
    removed = 0
    for name, data in entries:
        if name != "word/document.xml":
            output_entries.append((name, data))
            continue

        root = ET.fromstring(data)
        body = root.find(qn("body"))
        if body is None:
            raise RuntimeError("未找到正文。")
        for child in list(body):
            if is_old_table_figure(child):
                body.remove(child)
                removed += 1
        output_entries.append((name, ET.tostring(root, encoding="utf-8", xml_declaration=True)))

    with zipfile.ZipFile(TMP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in output_entries:
            target.writestr(name, data)
    shutil.copy2(TMP_PATH, DOCX_PATH)
    print(f"已删除旧表格式图示 {removed} 个。")


if __name__ == "__main__":
    main()
