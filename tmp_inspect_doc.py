
from docx import Document
from pathlib import Path
p = Path(r'C:\Coding\Code-similarity-algorithm\毕业论文_代码相似度检测_格式修订版_改写.docx')
doc = Document(p)
keys = ['流程','模型','特征','算法','实验','图','相似度','设计','实现','随机森林','逻辑回归','弱监督']
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip().replace('\t',' ')
    if text and (len(text) < 80 or any(k in text for k in keys)):
        style = para.style.name if para.style else ''
        print(f'{i:04d} [{style}] {text[:160]}')
