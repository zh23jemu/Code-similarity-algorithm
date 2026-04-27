
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
p=Path(r'C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx')
ns={'pr':'http://schemas.openxmlformats.org/package/2006/relationships'}
with zipfile.ZipFile(p) as z:
    root=ET.fromstring(z.read('word/_rels/document.xml.rels'))
    for rel in root:
        rid=rel.attrib.get('Id')
        target=rel.attrib.get('Target')
        typ=rel.attrib.get('Type')
        if target and ('media' in target or 'image' in typ):
            print(rid, target)
