
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
p=Path(r'C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def q(p,t): return '{%s}%s'%(ns[p],t)
def text(el): return ''.join(t.text or '' for t in el.iter(q('w','t'))).strip()
with zipfile.ZipFile(p) as z:
 root=ET.fromstring(z.read('word/document.xml')); body=root.find(q('w','body')); paras=list(body)
 for i in range(86,101):
  para=paras[i]; embeds=[n.attrib.get(q('r','embed')) for n in para.iter(q('a','blip'))]
  print(i, repr(text(para)[:100]), embeds)
 print('--- 126-134')
 for i in range(126,134):
  para=paras[i]; embeds=[n.attrib.get(q('r','embed')) for n in para.iter(q('a','blip'))]
  print(i, repr(text(para)[:100]), embeds)
 print('--- 166-174')
 for i in range(166,174):
  para=paras[i]; embeds=[n.attrib.get(q('r','embed')) for n in para.iter(q('a','blip'))]
  print(i, repr(text(para)[:100]), embeds)
