
import zipfile
from pathlib import Path
p = Path(r'C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx')
with zipfile.ZipFile(p) as z:
    print(z.read('[Content_Types].xml')[:2000].decode('utf-8', errors='replace'))
