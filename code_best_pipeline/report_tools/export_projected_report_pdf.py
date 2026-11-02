"""Export the Markdown report with rendered math and embedded original images."""
from pathlib import Path
import base64
import html
import io
import json
import re
import tarfile
import urllib.request
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'t2ifing_radioactive/result_vast/Report.md'
TEMP=ROOT/'tmp/pdfs/projected_report'
OUTPUT=ROOT/'report/Report_Projected_native_20261026.pdf'
TEMP.mkdir(parents=True,exist_ok=True);OUTPUT.parent.mkdir(parents=True,exist_ok=True)

assets=TEMP/'katex'
if not (assets/'dist/katex.min.js').exists():
    data=urllib.request.urlopen('https://registry.npmjs.org/katex/-/katex-0.16.22.tgz',timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith('package/dist/'):
                relative=Path(member.name).relative_to('package')
                destination=(assets/relative).resolve()
                if not destination.is_relative_to(assets.resolve()):raise RuntimeError('Unsafe asset path')
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())

text=SOURCE.read_text(encoding='utf-8-sig')
# Replace the large image tables with one full comparison page per sample.
start=text.index('### 6.1.')
end=text.index('## 7.')
folders=[
    ('Clean SD2.1 teacher','projected_native_teacher_preview_20261025/clean'),
    ('Baseline k=1 teacher','projected_native_teacher_preview_20261025/baseline_k1'),
    ('Projected native teacher','projected_native_teacher_preview_20261025/projected_native'),
    ('SD1.4 chưa fine-tune','projected_native_student_preview_20261025/unadapted'),
    ('Projected native student 7.500','projected_native_student_preview_20261025/student_projected_native'),
    ('Projected native student 10.000','projected_native_student10000_preview_20261026/student10000'),
]
manifest=json.loads((ROOT/'output_artifacts/previews/projected_native_teacher_preview_20261025/manifest.json').read_text())
figures=[]
for i in range(3):
    prompt=manifest['samples'][i]['prompt']
    first=' first' if i==0 else ''
    figures.append(f'<section class="figure-page{first}"><h2>6.{i+1}. Mẫu test {i:06d}</h2><p class="prompt">{html.escape(prompt)}</p><h3>Teacher - cùng prompt/seed</h3><div class="image-grid">')
    for k,(label,folder) in enumerate(folders):
        if k==3:figures.append('</div><h3>Student - cùng prompt/seed</h3><div class="image-grid">')
        image=ROOT/'output_artifacts/previews'/folder/f'sample_{i:06d}.png'
        url='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()
        figures.append(f'<figure><img src="{url}"/><figcaption>{html.escape(label)}</figcaption></figure>')
    figures.append('</div><p class="figure-note">PNG gốc 512 × 512. Mẫu lấy theo thứ tự test, không chọn theo độ đẹp. Bản student 10.000 là seeded rerun, không phải exact resume của bản 7.500.</p></section>')
prefix=text[:start]
# Every comparison page carries its prompt; omit the duplicate prompt table.
prefix=prefix[:prefix.index('| Sample | Prompt |')]
text=prefix+'\n\n'+''.join(figures)+'\n\n'+text[end:]

def block(match):
    return '\n\n<div class="math-display" data-tex="'+html.escape(match.group(1).strip(),quote=True)+'"></div>\n\n'
text=re.sub(r'\$\$(.*?)\$\$',block,text,flags=re.S)
text=re.sub(r'(?<!\$)\$(?!\$)([^\n$]+)\$(?!\$)',lambda m:'<span class="math-inline" data-tex="'+html.escape(m.group(1),quote=True)+'"></span>',text)
rendered=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
rendered=rendered.replace('<h2>5. ', '<h2 class="major-section">5. ')
rendered=rendered.replace('<h2>6. ', '<h2 class="major-section">6. ')
rendered=rendered.replace('<h2>7.', '<h2 class="closing">7.')
# Local source artifacts are named in the PDF, without machine-specific file links.
rendered=re.sub(r'<a href="(?!https?://)([^"]*)">(.*?)</a>',r'<span class="source-ref">\2</span>',rendered,flags=re.S)

css='''
@page{size:A4;margin:16mm 14mm 18mm}
*{box-sizing:border-box}
body{margin:0;color:#192b3d;font-family:Arial,sans-serif;font-size:12.7px;line-height:1.5}
h1{font-size:25px;line-height:1.2;color:#123c5e;margin:0 0 18px;padding-bottom:12px;border-bottom:3px solid #287c9d}
h2{font-size:18px;line-height:1.3;color:#123c5e;margin:24px 0 10px;break-after:avoid}
h3{font-size:14px;line-height:1.35;color:#17647d;margin:18px 0 8px;break-after:avoid}
p{margin:7px 0 10px;orphans:3;widows:3}
ul{padding-left:19px}li{margin:5px 0;orphans:3;widows:3}
strong{color:#102f48}
code{font-family:Consolas,monospace;font-size:10.5px;overflow-wrap:anywhere;background:#f0f4f7;padding:1px 3px}
a{color:#17647d;text-decoration:none}
table{border-collapse:collapse;width:100%;font-size:10.3px;line-height:1.35;margin:12px 0 16px;table-layout:auto}
thead{display:table-header-group}
tr{break-inside:avoid}
th{background:#163f5a;color:white;text-align:left;font-weight:600;padding:7px 5px}
td{border-bottom:1px solid #d6e1e8;padding:6px 5px;vertical-align:top}
tbody tr:nth-child(even){background:#f2f6f9}
td:first-child,th:first-child{max-width:160px;overflow-wrap:anywhere}
.math-display{margin:14px 0;text-align:center;break-inside:avoid}
.katex{font-size:1.06em!important}
.katex-display{margin:0!important;overflow:visible}
.source-ref{color:#526677}
.figure-page{break-before:page;break-inside:avoid}
.figure-page.first{break-before:auto}
.closing{break-before:page}
.major-section{break-before:page}
.figure-page h2{margin-top:0}
.figure-page h3{margin-top:20px}
.image-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
figure{margin:0;break-inside:avoid}
figure img{width:100%;height:auto;display:block;border:1px solid #d6e1e8}
figcaption{font-size:10.8px;line-height:1.35;margin-top:6px;color:#28475d;min-height:28px}
.prompt{font-size:13px;padding:10px 12px;background:#edf4f8;border-left:3px solid #287c9d}
.figure-note{font-size:11px;color:#5d6f7e;margin-top:20px}
'''
document='<!doctype html><html lang="vi"><meta charset="utf-8"><title>Report - Projected native</title>'
document+=f'<link rel="stylesheet" href="{(assets/"dist/katex.min.css").as_uri()}"><style>{css}</style><body>{rendered}'
document+=f'<script src="{(assets/"dist/katex.min.js").as_uri()}"></script>'
document+='''<script>window.mathErrors=[];document.querySelectorAll('[data-tex]').forEach(el=>{try{katex.render(el.dataset.tex,el,{displayMode:el.classList.contains('math-display'),throwOnError:true})}catch(e){window.mathErrors.push(e.message)}})</script></body></html>'''
html_file=TEMP/'report_print.html';html_file.write_text(document,encoding='utf-8')
with sync_playwright() as playwright:
    browser=playwright.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
    page=browser.new_page(viewport={'width':1000,'height':1400})
    page.goto(html_file.as_uri(),wait_until='networkidle')
    page.evaluate('document.fonts.ready')
    errors=page.evaluate('window.mathErrors')
    if errors:raise RuntimeError(errors)
    images=page.locator('img').evaluate_all('(images)=>images.map(i=>({ok:i.complete&&i.naturalWidth===512,src:i.src.slice(0,30)}))')
    if len(images)!=18 or not all(i['ok'] for i in images):raise RuntimeError(images)
    page.pdf(path=str(OUTPUT),format='A4',print_background=True,prefer_css_page_size=True,
        display_header_footer=True,header_template='<div></div>',
        footer_template='<div style="width:100%;font-family:Arial;font-size:9px;color:#63778a;padding:0 14mm;display:flex;justify-content:space-between"><span>Semantic fingerprint | Projected native | 26/10/2026</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>')
    browser.close()
print('PDF',OUTPUT,'BYTES',OUTPUT.stat().st_size)

# Render every page for visual QA, plus a compact contact sheet.
import fitz
from PIL import Image,ImageOps,ImageDraw
pdf=fitz.open(OUTPUT);thumbs=[]
for i,page in enumerate(pdf):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False)
    path=TEMP/f'page_{i+1:02d}.png';pix.save(path)
    im=Image.open(path).convert('RGB');im.thumbnail((310,440))
    card=Image.new('RGB',(330,470),'#dce3e8');card.paste(im,((330-im.width)//2,12))
    ImageDraw.Draw(card).text((12,450),f'Page {i+1}',fill='#123c5e');thumbs.append(card)
columns=3;rows=(len(thumbs)+columns-1)//columns
sheet=Image.new('RGB',(columns*330,rows*470),'white')
for i,im in enumerate(thumbs):sheet.paste(im,((i%columns)*330,(i//columns)*470))
sheet.save(TEMP/'contact_sheet.png')
texts='\n'.join(p.get_text() for p in pdf)
assert len(pdf)>=6 and 'Projected native' in texts and '53,35774' in texts
print('PAGES',len(pdf),'IMAGE_OCCURRENCES',sum(len(p.get_images()) for p in pdf),'MATH_ERRORS',errors)
print('CONTACT_SHEET',TEMP/'contact_sheet.png')
