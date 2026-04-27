
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
p=Path(r'C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def q(prefix,tag): return '{%s}%s'%(ns[prefix],tag)
def text(el): return ''.join(t.text or '' for t in el.iter(q('w','t'))).strip()
with zipfile.ZipFile(p) as z:
    root=ET.fromstring(z.read('word/document.xml'))
body=root.find(q('w','body'))
paras=list(body)
for i, para in enumerate(paras):
    t=text(para)
    embeds=[n.attrib.get(q('r','embed')) for n in para.iter(q('a','blip'))]
    has_draw=para.find('.//'+q('w','drawing')) is not None
    if t in ['图1. 系统整体流程图','图2. 系统主要功能模块图','图3. 代码相似度特征体系图','图6. 实验验证思路图'] or embeds or has_draw:
        print(i, 'text=',t[:60], 'draw=', has_draw, 'embeds=',embeds)
