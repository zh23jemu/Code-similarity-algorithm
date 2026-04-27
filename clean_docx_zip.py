"""清理 DOCX 压缩包中的重复条目。

Word 通常会读取最后一个同名条目，但为了交付文件更稳妥，本脚本会把每个
压缩条目的最后版本写入新的 DOCX，再覆盖目标文档。
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


DOCX_PATH = Path(r"C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx")
CLEAN_PATH = DOCX_PATH.with_name(DOCX_PATH.stem + "_clean.docx")


def main() -> None:
    """保留每个 DOCX 内部路径的最后版本，重建为无重复条目的压缩包。"""

    with zipfile.ZipFile(DOCX_PATH, "r") as source:
        names = source.namelist()
        latest_names = list(dict.fromkeys(reversed(names)))
        latest_names.reverse()
        payloads = {name: source.read(name) for name in latest_names}

    with zipfile.ZipFile(CLEAN_PATH, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name in latest_names:
            target.writestr(name, payloads[name])

    shutil.copy2(CLEAN_PATH, DOCX_PATH)
    print(f"已清理重复条目并更新：{DOCX_PATH}")


if __name__ == "__main__":
    main()
