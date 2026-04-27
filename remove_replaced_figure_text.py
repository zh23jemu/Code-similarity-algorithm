"""删除已由新图片替换的旧图示文字段落。

原文中部分图示不是普通图片，而是文本框/形状组合。渲染时这些对象会被拆成
竖排文字，影响版面。这里仅删除已经由新生成图片替换的图1、图3、图6前置
旧图示文字段落，保留原有图题和数据来源。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DOCX_PATH = Path(r"C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx")
TMP_PATH = DOCX_PATH.with_name(DOCX_PATH.stem + "_removed_replaced_text.docx")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

ET.register_namespace("w", W_NS)


def qn(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(qn("t"))).strip()


def main() -> None:
    """删除目标图题前的旧图示文字段落，并重建 DOCX。"""

    with zipfile.ZipFile(DOCX_PATH, "r") as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]

    updated_entries: list[tuple[str, bytes]] = []
    removed = 0

    for name, data in entries:
        if name != "word/document.xml":
            updated_entries.append((name, data))
            continue

        root = ET.fromstring(data)
        body = root.find(qn("body"))
        if body is None:
            raise RuntimeError("未找到 Word 文档正文。")

        captions = {
            "图1. 系统整体流程图",
            "图3. 代码相似度特征体系图",
            "图6. 实验验证思路图",
        }
        children = list(body)
        to_remove: set[ET.Element] = set()
        for index, child in enumerate(children):
            if child.tag != qn("p") or paragraph_text(child) not in captions:
                continue
            # 从图题向前寻找最近的非空、非图片段落。该段落是原图示被拆解出的文字。
            for previous in reversed(children[:index]):
                if previous.tag != qn("p"):
                    continue
                if previous.find(".//" + qn("drawing")) is not None:
                    continue
                text = paragraph_text(previous)
                if not text:
                    continue
                if text.startswith("为了") or text.startswith("在具体计算") or text.startswith("从实验验证"):
                    continue
                to_remove.add(previous)
                break

        for paragraph in to_remove:
            body.remove(paragraph)
            removed += 1

        updated_entries.append((name, ET.tostring(root, encoding="utf-8", xml_declaration=True)))

    with zipfile.ZipFile(TMP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in updated_entries:
            target.writestr(name, data)

    shutil.copy2(TMP_PATH, DOCX_PATH)
    print(f"已删除 {removed} 个被新图替换的旧图示文字段落。")


if __name__ == "__main__":
    main()
