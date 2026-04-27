
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
p=Path(r'C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def q(p,t): return '{%s}%s'%(ns[p],t)
def txt(el): return ''.join(t.text or '' for t in el.iter(q('w','t'))).strip()
with zipfile.ZipFile(p) as z:
 root=ET.fromstring(z.read('word/document.xml')); body=root.find(q('w','body')); children=list(body)
 for i in range(88,104):
  el=children[i]; embeds=[n.attrib.get(q('r','embed')) for n in el.iter(q('a','blip'))]
  print(i, el.tag.split('}')[-1], repr(txt(el)[:200]), 'drawing', el.find('.//'+q('w','drawing')) is not None, embeds)
