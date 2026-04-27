"""向论文 DOCX 中局部插入结构图和必要衔接文字。

该脚本采用低层 OOXML 补丁方式处理 Word 文件，目的在于只新增图片段落
和少量上下文说明，尽量不重写全文结构，避免影响原有段落、表格、标题样式。
"""

from __future__ import annotations

import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(r"C:\Coding\Code-similarity-algorithm")
DOCX_PATH = ROOT / "毕业论文_代码相似度检测_格式修订版_改写.docx"
WORK_PATH = ROOT / "毕业论文_代码相似度检测_格式修订版_改写_插图临时.docx"
FIGURES = [
    {
        "anchor_caption": "图1. 系统整体流程图",
        "image": ROOT / "paper_figures" / "fig_ml_pipeline_architecture.png",
        "media_name": "generated_ml_pipeline_architecture.png",
        "context": (
            "为了更直观地说明系统中数据输入、代码预处理、特征工程、弱监督建模"
            "以及结果输出之间的关系，本文在总体设计部分补充系统流程结构图，"
            "将各模块之间的数据流向和功能边界进行集中展示。"
        ),
        "alt": "代码相似度检测总体流程结构图",
    },
    {
        "anchor_caption": "图3. 代码相似度特征体系图",
        "image": ROOT / "paper_figures" / "fig_feature_fusion_classifier.png",
        "media_name": "generated_feature_fusion_classifier.png",
        "context": (
            "在具体计算过程中，两份 Java 代码会经过相同的预处理流程得到 Token "
            "序列，再从词法重合、序列组织、结构统计和规模差异等角度形成数值特征，"
            "最后输入传统机器学习分类器进行相似概率预测。"
        ),
        "alt": "代码对特征融合与分类结构图",
    },
    {
        "anchor_caption": "图6. 实验验证思路图",
        "image": ROOT / "paper_figures" / "fig_weak_supervision_evaluation.png",
        "media_name": "generated_weak_supervision_evaluation.png",
        "context": (
            "从实验验证角度看，本文并不只依赖单一训练测试划分结果，而是把弱监督"
            "样本构造、模型对比、特征消融、按题目划分泛化实验和人工复核结合起来，"
            "以便从多个侧面说明检测结果的可靠性。"
        ),
        "alt": "弱监督建模与实验评估流程图",
    },
]

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
}

for prefix, uri in NS.items():
    if prefix not in {"pr", "ct"}:
        ET.register_namespace(prefix, uri)


def qn(prefix: str, tag: str) -> str:
    """生成带命名空间的 XML 标签名。"""

    return f"{{{NS[prefix]}}}{tag}"


def paragraph_text(paragraph: ET.Element) -> str:
    """提取一个 Word 段落中的可见文本，用于查找插图锚点。"""

    return "".join(node.text or "" for node in paragraph.iter(qn("w", "t"))).strip()


def make_text_paragraph(text: str) -> ET.Element:
    """创建普通正文段落，依赖文档 Normal 样式承接原有格式。"""

    paragraph = ET.Element(qn("w", "p"))
    run = ET.SubElement(paragraph, qn("w", "r"))
    text_node = ET.SubElement(run, qn("w", "t"))
    text_node.text = text
    return paragraph


def image_size_emu(image_path: Path, max_width_cm: float = 14.2) -> tuple[int, int]:
    """按页面正文宽度等比例计算图片在 Word 中的 EMU 尺寸。"""

    with Image.open(image_path) as img:
        width_px, height_px = img.size
    max_width_emu = int(max_width_cm / 2.54 * 914400)
    aspect = height_px / width_px
    return max_width_emu, int(max_width_emu * aspect)


def make_image_paragraph(rel_id: str, image_path: Path, doc_pr_id: int, alt: str) -> ET.Element:
    """创建居中显示的 Word 内嵌图片段落。"""

    cx, cy = image_size_emu(image_path)
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
            <wp:cNvGraphicFramePr>
              <a:graphicFrameLocks noChangeAspect="1"/>
            </wp:cNvGraphicFramePr>
            <a:graphic>
              <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                <pic:pic>
                  <pic:nvPicPr>
                    <pic:cNvPr id="0" name="{image_path.name}" descr="{alt}"/>
                    <pic:cNvPicPr/>
                  </pic:nvPicPr>
                  <pic:blipFill>
                    <a:blip r:embed="{rel_id}"/>
                    <a:stretch><a:fillRect/></a:stretch>
                  </pic:blipFill>
                  <pic:spPr>
                    <a:xfrm>
                      <a:off x="0" y="0"/>
                      <a:ext cx="{cx}" cy="{cy}"/>
                    </a:xfrm>
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


def next_relationship_id(rels_root: ET.Element) -> int:
    """获取下一个可用的 rId 数字，避免覆盖文档中已有关系。"""

    max_id = 0
    for rel in rels_root:
        rel_id = rel.attrib.get("Id", "")
        if rel_id.startswith("rId") and rel_id[3:].isdigit():
            max_id = max(max_id, int(rel_id[3:]))
    return max_id + 1


def next_doc_pr_id(document_root: ET.Element) -> int:
    """获取下一个可用的图片对象编号。"""

    max_id = 0
    for node in document_root.iter(qn("wp", "docPr")):
        value = node.attrib.get("id", "0")
        if value.isdigit():
            max_id = max(max_id, int(value))
    return max_id + 1


def ensure_png_content_type(content_types_root: ET.Element) -> None:
    """确保 DOCX 包内声明了 PNG 图片内容类型。"""

    for child in content_types_root:
        if child.attrib.get("Extension") == "png":
            return
    ET.SubElement(
        content_types_root,
        qn("ct", "Default"),
        {"Extension": "png", "ContentType": "image/png"},
    )


def patch_docx() -> None:
    """复制原文档并向副本中插入图片、上下文段落和图片关系。"""

    shutil.copy2(DOCX_PATH, WORK_PATH)

    with zipfile.ZipFile(WORK_PATH, "a") as archive:
        document_xml = archive.read("word/document.xml")
        rels_xml = archive.read("word/_rels/document.xml.rels")
        content_types_xml = archive.read("[Content_Types].xml")

        document_root = ET.fromstring(document_xml)
        rels_root = ET.fromstring(rels_xml)
        content_types_root = ET.fromstring(content_types_xml)
        body = document_root.find(qn("w", "body"))
        if body is None:
            raise RuntimeError("未找到 Word 文档正文。")

        next_rid = next_relationship_id(rels_root)
        next_pr_id = next_doc_pr_id(document_root)
        inserted_count = 0

        for figure in FIGURES:
            anchor_caption = figure["anchor_caption"]
            paragraphs = list(body)
            caption_index = None
            for index, paragraph in enumerate(paragraphs):
                if paragraph.tag == qn("w", "p") and paragraph_text(paragraph) == anchor_caption:
                    caption_index = index
                    break
            if caption_index is None:
                raise RuntimeError(f"未找到插图锚点：{anchor_caption}")

            # 如果图片已存在，说明脚本曾经执行过，直接跳过以保证幂等。
            if any(figure["media_name"] in rel.attrib.get("Target", "") for rel in rels_root):
                continue

            rel_id = f"rId{next_rid}"
            next_rid += 1
            ET.SubElement(
                rels_root,
                qn("pr", "Relationship"),
                {
                    "Id": rel_id,
                    "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                    "Target": f"media/{figure['media_name']}",
                },
            )

            archive.write(figure["image"], f"word/media/{figure['media_name']}")
            context_paragraph = make_text_paragraph(figure["context"])
            image_paragraph = make_image_paragraph(rel_id, figure["image"], next_pr_id, figure["alt"])
            next_pr_id += 1

            body.insert(caption_index, image_paragraph)
            body.insert(caption_index, context_paragraph)
            inserted_count += 1

        ensure_png_content_type(content_types_root)

        archive.writestr("word/document.xml", ET.tostring(document_root, encoding="utf-8", xml_declaration=True))
        archive.writestr("word/_rels/document.xml.rels", ET.tostring(rels_root, encoding="utf-8", xml_declaration=True))
        archive.writestr("[Content_Types].xml", ET.tostring(content_types_root, encoding="utf-8", xml_declaration=True))

    shutil.copy2(WORK_PATH, DOCX_PATH)
    print(f"已插入 {inserted_count} 张图片并更新：{DOCX_PATH}")


if __name__ == "__main__":
    patch_docx()
