"""Render the English Markdown report with repeatable tables and embedded figure."""
import html
import re
from pathlib import Path
import fitz
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.pagesizes import A4

root=Path(__file__).resolve().parents[2]
source=root/'server_review/EXPERIMENT_REPORT_SEED0_2026-10-11_EN.md'
dest=root/'output/pdf/EXPERIMENT_REPORT_SEED0_2026-10-11_EN.pdf'
dest.parent.mkdir(parents=True,exist_ok=True)
qa=root/'tmp/pdfs/seed0_report'
qa.mkdir(parents=True,exist_ok=True)
for name,file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
styles={
 'body':ParagraphStyle('body',fontName='Arial',fontSize=9.3,leading=12.3,spaceAfter=6),
 'title':ParagraphStyle('title',fontName='Arial-Bold',fontSize=20,leading=24,spaceAfter=14),
 'h2':ParagraphStyle('h2',fontName='Arial-Bold',fontSize=13,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='Arial-Bold',fontSize=10.5,leading=14,spaceBefore=9,spaceAfter=6,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=8,leading=10.5,spaceAfter=0),
 'quote':ParagraphStyle('quote',fontName='Arial',fontSize=8.8,leading=12,backColor=colors.HexColor('#edf2f7'),borderPadding=8,spaceBefore=5,spaceAfter=12),
}
def inline(s):
    s=s.replace('—','-').replace('–','-').replace('‑','-')
    s=html.escape(s)
    s=s.replace('⁻⁵','<super>-5</super>').replace('⁻⁶','<super>-6</super>').replace('³','<super>3</super>')
    s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',lambda m:'<link href="'+html.escape(html.unescape(m[2]) if html.unescape(m[2]).startswith(('https://','http://')) else (source.parent/html.unescape(m[2])).resolve().as_uri(),quote=True)+'" color="#234e70">'+m[1]+'</link>',s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font color="#344054">\1</font>',s)
    return s
def para(s,kind='body'): return Paragraph(inline(s),styles[kind])
story=[]
lines=source.read_text(encoding='utf-8').splitlines()
i=0
width=A4[0]-84
while i<len(lines):
    line=lines[i].strip()
    if not line: i+=1; continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',x) for x in cells): rows.append(cells)
            i+=1
        n=len(rows[0])
        ratios={2:[.27,.73],3:[.53,.23,.24],4:[.29,.22,.20,.29],6:[.15,.17,.17,.17,.17,.17],7:[.19,.09,.09,.16,.17,.16,.14]}[n]
        if rows[0][0]=='Strength' and n==7: ratios=[.09,.12,.12,.16,.15,.11,.25]
        if rows[0][0]=='Strength' and n==6: ratios=[.10,.18,.17,.16,.19,.20]
        ratios=[r/sum(ratios)*width for r in ratios]
        data=[[para(('**'+c+'**') if j==0 else c,'cell') for c in row] for j,row in enumerate(rows)]
        table=Table(data,colWidths=ratios,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4ebf3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fb')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9aaabc'))]))
        story.extend([table,Spacer(1,10)])
        continue
    if line.startswith('!['):
        p=re.search(r'\]\(([^)]+)\)',line)[1]
        im=Image(str(source.parent/p))
        scale=min(width/im.imageWidth,260/im.imageHeight)
        im.drawWidth=im.imageWidth*scale; im.drawHeight=im.imageHeight*scale
        story.extend([im,Spacer(1,8)])
    elif line.startswith('### '): story.append(para(line[4:],'h3'))
    elif line.startswith('## '): story.append(para(line[3:],'h2'))
    elif line.startswith('# '): story.append(para(line[2:],'title'))
    elif line.startswith('> '): story.append(para(line[2:],'quote'))
    else: story.append(para(line))
    i+=1
def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#c5ced8'));canvas.line(42,36,A4[0]-42,36)
    canvas.setFont('Arial',7.5);canvas.setFillColor(colors.HexColor('#526173'))
    canvas.drawString(42,24,'SERUM / CtrlRegen+ - exploratory seed-0 report | updated 12 October 2026')
    canvas.drawRightString(A4[0]-42,24,str(doc.page))
doc=SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=49,title='Output Laundering Before Downstream Adaptation - Seed 0',author='Research experiment report')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
pdf=fitz.open(dest)
texts=[]
for page in pdf:
    texts.append(page.get_text())
    page.get_pixmap(matrix=fitz.Matrix(1.15,1.15)).save(qa/f'page_{page.number+1:02d}.png')
text='\n'.join(texts)
for expected in ['4.42%','3.60%','97.96%','93.14%','52.30%','19,500','gradient','Supported statement']:
    assert expected in text, expected
assert '\ufffd' not in text
print(f'{dest}\nPages: {len(pdf)}\nText validation passed')
