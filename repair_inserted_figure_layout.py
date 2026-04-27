"""修复论文中插图附近的版面异常。

该脚本只处理图示相关段落：删除原有形状图被渲染成的残留文字，并在
“图2. 系统主要功能模块图”前补入新的位图版本，保留原图题和来源说明。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


DOCX_PATH = Path(r"C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx")
TMP_PATH = DOCX_PATH.with_name(DOCX_PATH.stem + "_layout_repaired.docx")
IMAGE_PATH = Path(r"C:\Coding\Code-similarity-algorithm\paper_figures\fig_system_function_modules.png")
MEDIA_NAME = "generated_system_function_modules.png"

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}

for prefix, uri in NS.items():
    if prefix != "pr":
        ET.register_namespace(prefix, uri)


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(qn("w", "t"))).strip()


def next_relationship_id(rels_root: ET.Element) -> int:
    max_id = 0
    for rel in rels_root:
        rid = rel.attrib.get("Id", "")
        if rid.startswith("rId") and rid[3:].isdigit():
            max_id = max(max_id, int(rid[3:]))
    return max_id + 1


def next_doc_pr_id(document_root: ET.Element) -> int:
    max_id = 0
    for node in document_root.iter(qn("wp", "docPr")):
        value = node.attrib.get("id", "0")
        if value.isdigit():
            max_id = max(max_id, int(value))
    return max_id + 1


def image_size_emu(image_path: Path, max_width_cm: float = 14.2) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width_px, height_px = img.size
    width_emu = int(max_width_cm / 2.54 * 914400)
    return width_emu, int(width_emu * height_px / width_px)


def make_image_paragraph(rel_id: str, doc_pr_id: int) -> ET.Element:
    cx, cy = image_size_emu(IMAGE_PATH)
    alt = "代码相似度检测系统主要功能模块图"
    xml = f"""
    <w:p xmlns:w="{NS['w']}" xmlns:r="{NS['r']}" xmlns:wp="{NS['wp']}" xmlns:a="{NS['a']}" xmlns:pic="{NS['pic']}">
      <w:pPr>
        <w:jc w:val="center"/>
        <w:spacing w:before="120" w:after="80"/>
      </w:pPr>
      <w:r>
        <w:drawing>
          <wp:inline distT="0" distB="0" distL="0" distR="0">
            <wp:extent cx="{cx}" cy="{cy}"/>
            <wp:effectExtent l="0" t="0" r="0" b="0"/>
            <wp:docPr id="{doc_pr_id}" name="{alt}" descr="{alt}"/>
            <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
            <a:graphic>
              <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic>
                  <pic:nvPicPr>
                    <pic:cNvPr id="0" name="{IMAGE_PATH.name}" descr="{alt}"/>
                    <pic:cNvPicPr/>
                  </pic:nvPicPr>
                  <pic:blipFill>
                    <a:blip r:embed="{rel_id}"/>
                    <a:stretch><a:fillRect/></a:stretch>
                  </pic:blipFill>
                  <pic:spPr>
                    <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                  </pic:spPr>
                </pic:pic>
              </a:graphicData>
            </a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>
    """
    return ET.fromstring(xml)


def is_old_figure_text(text: str) -> bool:
    """识别原形状图被渲染成的无标点长串文字。"""

    markers = [
        "数据读取代码预处理特征提取弱监督标注模型训练相似度预测报告与复核",
        "代码相似度检测系统数据加载模块预处理模块特征工程模块",
        "特征类别代表特征作用说明词法相似度",
        "验证层次实验内容主要目的伪标签一致性评估",
    ]
    return any(marker in text for marker in markers)


def main() -> None:
    with zipfile.ZipFile(DOCX_PATH, "r") as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]

    document_root = ET.fromstring(dict(entries)["word/document.xml"])
    rels_root = ET.fromstring(dict(entries)["word/_rels/document.xml.rels"])
    body = document_root.find(qn("w", "body"))
    if body is None:
        raise RuntimeError("未找到 Word 文档正文。")

    removed = 0
    for paragraph in list(body):
        if paragraph.tag != qn("w", "p"):
            continue
        text = paragraph_text(paragraph)
        if is_old_figure_text(text):
            body.remove(paragraph)
            removed += 1

    already_inserted = any(MEDIA_NAME in rel.attrib.get("Target", "") for rel in rels_root)
    if not already_inserted:
        rel_id = f"rId{next_relationship_id(rels_root)}"
        ET.SubElement(
            rels_root,
            qn("pr", "Relationship"),
            {
                "Id": rel_id,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": f"media/{MEDIA_NAME}",
            },
        )
        children = list(body)
        for index, paragraph in enumerate(children):
            if paragraph.tag == qn("w", "p") and paragraph_text(paragraph) == "图2. 系统主要功能模块图":
                body.insert(index, make_image_paragraph(rel_id, next_doc_pr_id(document_root)))
                break
        else:
            raise RuntimeError("未找到图2图题。")

    output_entries: list[tuple[str, bytes]] = []
    replaced_media = False
    for name, data in entries:
        if name == "word/document.xml":
            data = ET.tostring(document_root, encoding="utf-8", xml_declaration=True)
        elif name == "word/_rels/document.xml.rels":
            data = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)
        output_entries.append((name, data))

    if not any(name == f"word/media/{MEDIA_NAME}" for name, _ in output_entries):
        output_entries.append((f"word/media/{MEDIA_NAME}", IMAGE_PATH.read_bytes()))
        replaced_media = True

    with zipfile.ZipFile(TMP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in output_entries:
            target.writestr(name, data)

    shutil.copy2(TMP_PATH, DOCX_PATH)
    print(f"已清理旧图示文字段落 {removed} 个，图2新图片写入：{replaced_media}")


if __name__ == "__main__":
    main()
