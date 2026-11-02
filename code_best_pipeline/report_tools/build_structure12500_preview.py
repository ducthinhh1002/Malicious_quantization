from pathlib import Path, PurePosixPath
import hashlib
import html
import json
import tarfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output_artifacts/previews/structure12500_student_preview_20261027'
OUT.mkdir(parents=True,exist_ok=True)
with tarfile.open(ROOT/'output_artifacts/archives/structure12500_student_preview_20261027.tar') as archive:
    for entry in archive:
        rel=PurePosixPath(entry.name)
        if rel.is_absolute() or '..' in rel.parts or not entry.isfile():
            raise RuntimeError('Unexpected archive member')
        target=OUT.joinpath(*rel.parts)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.extractfile(entry).read())
manifest=json.loads((OUT/'manifest.json').read_text())
for row in manifest['files']:
    if hashlib.sha256((OUT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
        raise RuntimeError('Image hash mismatch')
labels={'student12500':'Structure12.500 — TPR91,1%',
        'student7500':'Structure7.500 — TPR87,4%',
        'unadapted':'SD1.4 gốc — chưa fine-tune'}
sections=[]
for sample in manifest['samples']:
    cards=[]
    for key,title in labels.items():
        entry=sample['images'][key]
        status='Detect' if entry['detected'] else 'Dưới ngưỡng'
        cards.append(f'''<figure><figcaption><strong>{title}</strong><br>
        Score {entry['score']:.4f} · {status}</figcaption>
        <a href="{entry['path']}" target="_blank"><img src="{entry['path']}" loading="lazy" alt="{html.escape(title)}"></a></figure>''')
    sections.append(f'''<section><h2>Ảnh {sample['index']+1} · seed {sample['generation_seed']}</h2>
    <p>{html.escape(sample['prompt'])}</p><div class="grid">{''.join(cards)}</div></section>''')
document=f'''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Structure12.500 — ảnh student</title>
<style>body{{font:16px/1.5 Arial,sans-serif;background:#f3f5f8;color:#162235;margin:0;padding:28px}}
main{{max-width:1580px;margin:auto}}h1{{margin:0 0 10px}}h2{{font-size:20px;margin:0}}section{{background:white;border-radius:12px;padding:18px;margin:24px 0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}figure{{margin:0}}figcaption{{padding:8px 0;min-height:52px}}
img{{display:block;width:100%;height:auto;border-radius:6px}}a{{color:#174fb0}}.meta{{color:#4b596d}}
@media(max-width:800px){{body{{padding:12px}}.grid{{grid-template-columns:1fr}}}}</style>
<main><h1>Ảnh output student Structure12.500</h1>
<p>8 ảnh đầu tiên của tập test, cùng prompt/seed và PNDM50, CFG7,5, 512×512.
Ảnh PNG gốc; bấm vào ảnh để mở riêng.</p>
<p class="meta">Cột trái: Structure12.500 · giữa: Structure7.500 · phải: SD1.4 gốc.
Ngưỡng detection cố định {manifest['threshold']:.6f}. Score là điểm watermark, không phải điểm chất lượng ảnh.
LPIPS trung bình:12.500 =0,52882;7.500 =0,51656. Thấp hơn nghĩa là gần ảnh SD1.4 tham chiếu hơn.</p>
{''.join(sections)}<p><a href="manifest.json">Manifest prompt/seed và SHA256</a></p></main></html>'''
(OUT/'index.html').write_text(document,encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps({'verified_files':len(manifest['files']),
    'all_sha256_match':True,'samples':len(manifest['samples']),'generation_settings_paired':True},indent=2))
print('VERIFIED',len(manifest['files']),'images; gallery:',OUT/'index.html')
