"""规范化 DOCX 包内关键 XML 的命名空间写法。

部分渲染器对 `[Content_Types].xml` 和关系文件更偏好 Office 常见的默认
命名空间形式。本脚本只调整包结构 XML 的序列化方式，不改变正文内容。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DOCX_PATH = Path(r"C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx")
TMP_PATH = DOCX_PATH.with_name(DOCX_PATH.stem + "_normalized.docx")
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def serialize_default_namespace(xml_bytes: bytes, namespace: str) -> bytes:
    """用默认命名空间重新序列化 XML，避免生成 ns0 前缀。"""

    root = ET.fromstring(xml_bytes)
    ET.register_namespace("", namespace)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    """重建 DOCX，并规范化内容类型和关系 XML。"""

    with zipfile.ZipFile(DOCX_PATH, "r") as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]

    with zipfile.ZipFile(TMP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, data in entries:
            if name == "[Content_Types].xml":
                data = serialize_default_namespace(data, CT_NS)
            elif name.endswith(".rels"):
                data = serialize_default_namespace(data, RELS_NS)
            target.writestr(name, data)

    shutil.copy2(TMP_PATH, DOCX_PATH)
    print(f"已规范化 DOCX 包 XML：{DOCX_PATH}")


if __name__ == "__main__":
    main()
