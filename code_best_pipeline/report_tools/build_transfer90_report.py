"""Build a source-linked English report, 20-sample gallery and embedded PDF."""
from pathlib import Path, PurePosixPath
import hashlib, html, io, json, tarfile, re
import numpy as np
from PIL import Image as PILImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'output_artifacts/previews/transfer90_report_20261027'
TEMP=BUNDLE;TEMP.mkdir(parents=True,exist_ok=True)
PDF=ROOT/'report/EXPERIMENT_REPORT_TRANSFER90_20261027_EN.pdf'
MD=ROOT/'t2ifing_radioactive/result_vast/Report_Transfer90_20261027_EN.md'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def extract():
    with tarfile.open(ROOT/'output_artifacts/archives/transfer90_report_20261027.tar') as tar:
        for entry in tar:
            rel=PurePosixPath(entry.name)
            if rel.is_absolute() or '..' in rel.parts or not entry.isfile():raise ValueError(entry.name)
            target=BUNDLE.joinpath(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tar.extractfile(entry).read())
    m=read(BUNDLE/'manifest.json');assert len(m['samples'])==20
    for f in m['files']:
        assert hashlib.sha256((BUNDLE/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
    (BUNDLE/'verification.json').write_text(json.dumps({'all_sha256_verified':True,'files':len(m['files']),'images':120,'samples':20},indent=2))
    return m

def normalize(name,x):
    u=x['utility'];return {'name':name,'step':x.get('selected_checkpoint',{}).get('selected_step',7500),
        'tpr':x['metrics']['detection_rate'],'lpips':x['lpips'],'fid':u['frechet_inception_distance'],
        'kid':u['kernel_inception_distance_mean'],'clip':u['clip_score']}

def collect():
    evidence=BUNDLE/'evidence';new=read(evidence/'student_summary.json');baseline=read(evidence/'baseline_k1_summary.json')
    structure=read(ROOT/'output_artifacts/results/historical_summaries/structure_student12500_final_20261027.json')
    # Equality of the full generation protocol establishes pairing of student images.
    gens=[read(evidence/f'generation_{n}.json') for n in ['transfer90_student','structure12500','baseline_k1','unadapted']]
    identities=[{k:v for k,v in g.items() if k!='checkpoint'} for g in gens]
    assert all(g==identities[0] for g in identities)
    assert new['utility']['image_hashes']['reference']==structure['utility']['image_hashes']['reference']==baseline['utility']['image_hashes']['reference']
    runs=[normalize('Baseline k=1',baseline)]
    raw={}
    mapping=[('Quality','quality_student_final_20261024.json'),('Continuation','continuation_student_final_20261025.json'),
        ('Native full','native_full_student_final_20261025.json'),('Projected native 7500','projected_native_student_final_20261025.json'),
        ('Projected native 10000','projected_native_student10000_final_20261026.json'),
        ('Structure7500','structure_margin_student_final_20261026.json'),('Structure10000 warm restart','structure_margin_student10000_final_20261026.json'),
        ('Structure12500','structure_student12500_final_20261027.json')]
    for name,file in mapping:
        x=read(ROOT/'output_artifacts/results/historical_summaries'/file);raw[name]=x;runs.append(normalize(name,x))
    recovery=read(ROOT/'output_artifacts/results/historical_summaries/scale085_recovery_final_20261024.json')['results']
    for r in recovery:
        if r['name']=='scale100_7500':continue
        u=r['utility'];runs.append({'name':r['name'],'step':r['selected_step'],'tpr':r['tpr'],'lpips':r['lpips'],
            'fid':u['frechet_inception_distance'],'kid':u['kernel_inception_distance_mean'],'clip':u['clip_score']})
    other=read(ROOT/'output_artifacts/results/historical_summaries/minhnqn_final_comparison_20261025.json')
    for name in ['quality_gradient','quality_cka']:
        r=other['results'][name];runs.append({'name':name,'step':r['selected_step'],'tpr':r['detection'],'lpips':r['lpips'],
            'fid':r['frechet_inception_distance'],'kid':r['kernel_inception_distance_mean'],'clip':r['clip_score']})
    for name,r in other.get('preceding_campaign',{}).get('results',{}).items():
        if name=='quality_transfer':runs.append({'name':name,'step':r['selected_step'],'tpr':r['detection'],'lpips':r['lpips'],
            'fid':r['frechet_inception_distance'],'kid':r['kernel_inception_distance_mean'],'clip':r['clip_score']})
    runs.append(normalize('Transfer90',new))
    tau=new['metrics']['threshold'];pairs={};rng=np.random.default_rng(20261027);idx=rng.integers(0,1000,(5000,1000))
    for name,x in [('Baseline k=1',baseline),('Structure12500',structure),('Structure7500',raw['Structure7500']),('Projected native 7500',raw['Projected native 7500'])]:
        a=np.array(new['scores'])>=tau;b=np.array(x['scores'])>=tau
        ds=a.astype(float)-b;dl=np.array(new['lpips_per_image'])-x['lpips_per_image']
        dc=np.array(new['utility']['clip_per_image'])-x['utility']['clip_per_image']
        pairs[name]={'tpr_difference_pp':float(ds.mean()*100),'tpr_ci95_pp':(np.quantile(ds[idx].mean(1),[.025,.975])*100).tolist(),
            'lpips_difference':float(dl.mean()),'lpips_ci95':np.quantile(dl[idx].mean(1),[.025,.975]).tolist(),
            'clip_difference':float(dc.mean()),'clip_ci95':np.quantile(dc[idx].mean(1),[.025,.975]).tolist(),
            'gained':int((a&~b).sum()),'lost':int((~a&b).sum()),'lpips_improved_fraction':float((dl<0).mean())}
    result={'runs':runs,'paired':pairs,'bootstrap':{'resamples':5000,'seed':20261027,'unit':'prompt index; one trained model per branch'},
            'generation_pairing_checked':True,'fid_reference_hashes_match':True}
    (BUNDLE/'comparison.json').write_text(json.dumps(result,indent=2))
    return new,baseline,structure,read(evidence/'verification.json'),result

def gallery(m):
    names=[('baseline_k1','Baseline k=1 student'),('structure12500','Structure12500 student'),('transfer90_student','Transfer90 student'),
        ('clean_teacher','Clean SD2.1 teacher'),('transfer90_teacher','Transfer90 teacher'),('unadapted','Unadapted SD1.4')]
    sections=[]
    for s in m['samples']:
        cards=[]
        for key,label in names:
            x=s['images'][key];score=x.get('score');caption=label+(f' | score {score:.4f}' if score is not None else '')
            cards.append(f'<figure><figcaption>{html.escape(caption)}</figcaption><a href="{x["path"]}"><img src="{x["path"]}" loading="lazy"></a></figure>')
        sections.append(f'<section><h2>Sample {s["index"]:02d}</h2><p>{html.escape(s["prompt"])}</p><p>Student seed {500000+s["index"]}; teacher seed = image ID {s["image_id"]}. Paired within model family.</p><div class="grid">'+''.join(cards)+'</div></section>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transfer90:20 paired samples</title><style>body{font:16px/1.5 Arial;background:#edf2f6;color:#172c41;margin:24px}main{max-width:1600px;margin:auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}section{background:white;padding:20px;margin:24px 0;border-radius:10px}figure{margin:0}img{width:100%;display:block}figcaption{min-height:48px}h2{margin:0}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style><main><h1>Transfer90 -20 paired test samples</h1><p>First20 test samples, no visual filtering. Original512x512 PNGs; click to open. Student threshold0.8541307449. Watermark score is not an image-quality score. Teacher and student have different noise seeds; compare seeds only within each family.</p>'''+''.join(sections)+'</main></html>'
    (BUNDLE/'index.html').write_text(doc,encoding='utf-8')

def plots(data,verification):
    names=['Baseline k=1','Projected native 7500','Structure7500','Structure12500','Transfer90']
    fig,ax=plt.subplots(1,2,figsize=(10.2,3.6),layout='constrained')
    palette=['#8796a5','#bc7b27','#159780','#3270ab','#bc3041']
    for name,color in zip(names,palette):
        r=next(x for x in data['runs'] if x['name']==name)
        for a,key in zip(ax,['lpips','fid']):
            a.scatter(r[key],100*r['tpr'],s=65,c=color,label=name,zorder=3)
    for a,title in zip(ax,['Paired LPIPS (lower is closer)','FID to clean SD2.1 (lower)']):
        a.set(xlabel=title,ylabel='Student detection (%)',ylim=(70,101));a.grid(alpha=.2)
    ax[0].legend(fontsize=7,loc='lower left')
    fig.savefig(TEMP/'pareto.png',dpi=220);plt.close(fig)
    ms=verification['validation_milestones'];fig,ax=plt.subplots(1,2,figsize=(10.2,3.4),layout='constrained')
    steps=[r['step'] for r in ms]
    ax[0].plot(steps,[100*r['metrics']['tpr'] for r in ms],'-o',color='#bc3041')
    ax[0].axhline(90,color='gray',ls='--',lw=1);ax[0].set(xlabel='Transfer90 student updates',ylabel='Validation detection (%)')
    ax[1].plot(steps,[r['denoising_mse'] for r in ms],'-o',color='#3270ab');ax[1].set(xlabel='Transfer90 student updates',ylabel='Validation denoising MSE')
    for a in ax:a.grid(alpha=.2);a.ticklabel_format(axis='x',style='plain')
    fig.savefig(TEMP/'validation.png',dpi=220);plt.close(fig)

def report(m,new,baseline,structure,v,data):
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyX',fontName='Helvetica',fontSize=9.1,leading=12.6,spaceAfter=7))
    styles.add(ParagraphStyle(name='SmallX',fontName='Helvetica',fontSize=7.4,leading=10,spaceAfter=5))
    styles.add(ParagraphStyle(name='CellX',fontName='Helvetica',fontSize=7.6,leading=10))
    styles.add(ParagraphStyle(name='CodeX',fontName='Courier',fontSize=7.5,leading=10,spaceAfter=8,backColor=colors.HexColor('#edf2f6'),borderPadding=6))
    styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=21;styles['Title'].leading=25
    styles['Heading2'].fontSize=12;styles['Heading2'].leading=15;styles['Heading2'].spaceBefore=10
    story=[];md=[];width=511
    def tidy(t):
        t=re.sub(r',(?=[A-Za-z])',', ',t)
        t=re.sub(r'(?<=[0-9%])(?=(?:student|teacher|images|training|test|validation|updates|steps|prompt|additional|and|to|for|with|per|image|times|native|clean|denoiser|original|matched)\b)',' ',t)
        t=re.sub(r'(?<=[a-z])(?=(?:5000|1000|15000|12500|10000|7500|600|800|200|50|20|10)\b)',' ',t)
        t=re.sub(r'(?<=[;:])(?=[A-Za-z0-9])',' ',t)
        return t
    def p(t,small=False):
        t=tidy(t)
        story.append(Paragraph(t,styles['SmallX' if small else 'BodyX']));md.append(t.replace('<b>','**').replace('</b>','**').replace('<br/>','\n'))
    def h(t):story.append(Paragraph(t,styles['Heading2']));md.append('## '+t)
    def formula(t):story.append(Paragraph(html.escape(t).replace('\n','<br/>'),styles['CodeX']));md.append('```text\n'+t+'\n```')
    def table(headers,rows,widths=None):
        parsed=[[Paragraph(tidy(html.escape(str(c))),styles['CellX']) for c in row] for row in [headers,*rows]]
        t=Table(parsed,colWidths=widths or [width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfe8f0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9aaebf'))]))
        story.extend([t,Spacer(1,9)]);md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
    def page():story.append(PageBreak());md.append('\n---\n')
    def image(path,w):
        im=PILImage.open(path);hh=w*im.height/im.width;story.append(Image(str(path),width=w,height=hh));md.append(f'![Figure](../../{path.relative_to(ROOT).as_posix()})')
    rnew=normalize('Transfer90',new);rstruct=normalize('Structure12500',structure);rbase=normalize('Baseline k=1',baseline)
    ptitle='Experimental Report: Transfer90 Semantic Watermark Inheritance'
    story.append(Paragraph(ptitle,styles['Title']));md.append('# '+ptitle)
    p('<b>Updated:</b>27 October2026. <b>Scope:</b>seed-0 SD2.1 teacher to ordinary SD1.4 LoRA student. Transfer90 completed at10:33 ICT. This report follows the experimental structure of the supplied October11 English report; its SERUM/CtrlRegen numbers belong to a different experiment and are not pooled here.')
    h('1. Main findings')
    p('<b>Transfer90 reaches97.8% single-image detection (978/1000), Wilson95%CI[96.69,98.54]%.</b> Both the estimate and its lower confidence bound exceed90%. The score threshold,8-bit key and semantic verifier are unchanged from the comparable k=1 campaign. This is the highest detection among the comparable completed branches listed here.')
    p('<b>Against Structure12500:</b>detection increases6.7 percentage points, FID improves52.89584 to52.56332, and CLIP increases0.786258 to0.786725. LPIPS worsens0.528817 to0.539145; KID is effectively unchanged at the displayed scale. Transfer90 therefore does not improve every quality metric.')
    p('<b>Against the matched-panel k=1 baseline:</b>detection increases22.8 percentage points, LPIPS falls0.550447 to0.539145, and FID/KID improve. CLIP is slightly lower. Longer training, a different teacher and a different LR schedule change together; the result is a system-level comparison, not an isolated effect of the three-view loss.')
    h('2. Experimental protocol')
    table(['Component','Completed setting'],[
        ['Teacher','Manojb/stable-diffusion-2-1-base; warm-start completed Projected native(k=1), then800additional refinement updates; teacher LoRA rank32.'],
        ['Student','Pinned CompVis/stable-diffusion-v1-4, newly initialized LoRA rank32 attention+convolution; frozen VAE/text encoder; ordinary denoising MSE.'],
        ['Data','5000teacher training images;1000validation;1000test;5000clean calibration prompts. Same test prompt panel as Structure12500 and k=1baseline.'],
        ['Teacher optimization','AdamW LR5e-6, batch4; auxiliary native objective every2updates; microbatch1; full50-step PNDM forward, derivatives through last10denoiser calls.'],
        ['Student optimization','15000updates, batch4,fp16,seed0;LR1e-4through10000,cosine to2e-5by15000.'],
        ['Selection / generation','Minimum validation denoising MSE at7500/10000/12500/15000; selected15000. PNDM50,CFG7.5,512x512,batch16; student seed500000+index.']], [96,415])
    page()
    h('3. Method: native preservation and transfer-aware teacher refinement')
    p('<b>Lineage.</b>The original k=1 teacher checkpoint3000 was refined for600updates to obtain Projected native. Transfer90 warm-starts that adapter and adds800updates. Structure12500 uses a separate Structure/margin teacher refined600updates directly from the original k=1 checkpoint. Transfer90 is not a continuation of the Structure12500 student.')
    p('<b>Inherited watermark objective.</b>The original threshold_cfg_v1 loss preserves clean conditional/unconditional teacher predictions and applies calibrated semantic shifts to predicted clean latents and their teacher-VAE cycle. The semantic term operates on timesteps up to500. The downstream student receives only images/captions and ordinary denoising supervision.')
    p('<b>Native image preservation.</b>For each training prompt/seed, generate the full PNDM trajectory and compare the final RGB image against its clean SD2.1 paired image. RGB quantization is represented differentiably. Only the last10denoiser calls carry gradients; the forward pass and PNDM history remain complete. This is inspired by DRaFT-K[1], not a shortened10-step sampler.')
    formula('L_quality = LPIPS(x_marked,x_clean) +0.1*MSE_RGB(x_marked,x_clean)\n            +0.25*(1-MS_SSIM(x_marked,x_clean))')
    p('<b>Three teacher-side views.</b>(1)Native marked RGB;(2)stochastic SD1.4 VAE encode/decode;(3)one noising/denoising reconstruction through a frozen historical SD1.4 proxy, cycling timesteps50,150,300 and using CFG7.5. The proxy is the Projected native10000 student trained on historical watermarked images. Its weights are frozen; gradients pass through it to the teacher image. The final student is separately initialized and is never used as this proxy.')
    formula('score_v = mean_j(key_j * standardized_coordinate_vj)\nd_v = max(target_v-score_v,0)/sigma_clean\nL_margin =0.5*mean_v(d_v^2) +0.5*max_v(d_v^2)\nL_bits = mean_v,j[max(0,0.25-key_j*coordinate_vj)^2]\nL_owner = L_margin +0.1*L_bits')
    p('<b>Score floors.</b>Let tau=max(frozen teacher threshold,frozen student threshold) and r=min(1,(update+1)/200). Native/cycle floor=tau+(1+r)*sigma_clean; denoised floor=tau+(0.5+0.5r)*sigma_clean. The native target also preserves the previous Projected native image score when that score exceeds the floor. Deficits are penalized per image before averaging; the worst of the three views receives extra weight. This finite-view objective is not a population CVaR guarantee.')
    p('<b>Gradient balancing and projection.</b>Compute base,quality and ownership gradients separately. Scale auxiliary gradients to target1.0and1.5times the base-gradient norm, with coefficient cap100. AdamW proposes an update; on auxiliary steps, project its actual parameter displacement onto the intersection of the three local halfspaces g_i dot delta&lt;=0. This is inspired by gradient surgery[2], but operates on the realized AdamW displacement. It provides local truncated-gradient constraints, not a monotonic detection/quality guarantee.')
    page()
    h('4. Final student results and paired comparisons')
    table(['Branch','Steps','Detected','TPR /95%CI','LPIPS','FID','KID x1000','CLIP'],[
        [r['name'],r['step'],f'{int(round(r["tpr"]*1000))}/1000',f'{100*r["tpr"]:.1f}%'+(' /96.69-98.54' if r['name']=='Transfer90' else ' /89.17-92.71' if r['name']=='Structure12500' else ''),f'{r["lpips"]:.6f}',f'{r["fid"]:.3f}',f'{1000*r["kid"]:.4f}',f'{r["clip"]:.6f}'] for r in [rbase,rstruct,rnew]], [92,32,49,95,57,48,65,73])
    p('<b>Metric definitions.</b>LPIPS(AlexNet0.1)[3]compares each student image with unadapted SD1.4 at the same prompt/seed; lower means closer, not necessarily more attractive. FID[4]/KID[5]compare1000student outputs with clean SD2.1 teacher outputs, not real COCO. CLIPScore=2.5*max(cosine(image,text),0),CLIPViT-B/32. Teacher LPIPS uses a different reference(clean SD2.1), so its magnitude must not be ranked against student LPIPS.')
    rows=[]
    for name in ['Baseline k=1','Structure12500','Structure7500','Projected native 7500']:
        x=data['paired'][name];lo,hi=x['tpr_ci95_pp'];ll,lh=x['lpips_ci95']
        rows.append([name,f'{x["tpr_difference_pp"]:+.1f} [{lo:+.1f},{hi:+.1f}]',f'{x["lpips_difference"]:+.5f} [{ll:+.5f},{lh:+.5f}]',f'{x["gained"]}/{x["lost"]}',f'{100*x["lpips_improved_fraction"]:.1f}%'])
    table(['Transfer90 minus','TPR delta pp /95%CI','LPIPS delta /95%CI','Gained/lost detection','Images LPIPS improved'],rows,[96,120,145,75,75])
    p('Paired percentile bootstrap:5000prompt-index resamples,seed20261027. Generation settings and clean-reference hashes match for the main three-way comparison. These intervals describe variation across sampled prompts for fixed trained models; they do not measure retraining-seed variation or correct adaptive test reuse.')
    h('5. Frozen detector, clean controls and bit recovery')
    p('Student threshold tau=0.8541307449340821,calibration target FPR1% on5000clean prompts. The empirical held-out FPR is0.8%(8/1000)for unadapted SD1.4 and0.2%(2/1000)for the historical clean-trained7500student. Their Wilson95%CIs are[0.406,1.571]%and[0.055,0.726]%. A matched clean-trained15000control is missing.')
    p('Transfer90 per-image bit accuracy is91.9125%; exactly all8bits are recovered in51.6%of images. Signs of the mean coordinates recover8/8bits(100%aggregate key accuracy). <b>These three quantities differ from97.8%single-image threshold detection.</b> A100%aggregate bit figure must not be presented as100%image-level detection.')
    page()
    h('6. Comparison with completed historical branches')
    p('Comparable-panel k=1 experiments:1000test images,the same student score threshold and metric reference conventions. Budgets and teacher recipes differ; rows are descriptive operating points. No attribution to one component follows from ranking this table.')
    table(['Student branch','Updates','TPR%','LPIPS','FID','KID x1000','CLIP'],[
        [r['name'],r['step'],f'{100*r["tpr"]:.1f}',f'{r["lpips"]:.6f}',f'{r["fid"]:.3f}',f'{1000*r["kid"]:.4f}',f'{r["clip"]:.6f}'] for r in data['runs']], [151,45,43,68,60,72,72])
    p('<b>Earlier99.70%result.</b>The historical lora_r32_conv_wm row in result_vast used a selected kappa=2teacher and a different artifact/panel configuration:TPR99.70%,FID62.676,KID0.0069348,CLIP0.79334. It is preserved as historical context, not the matched k=1baseline. Student LPIPS is not supplied in that row. Therefore we cannot claim Transfer90 improves its LPIPS or reproduces its setup.')
    p('<b>Lessons supported by the runs.</b>Uniform scale0.85reduced detection sharply; longer student training did not consistently repair it. Native image objectives improved paired fidelity more reliably than the earlier proxy-only refinements. Structure7500 retains better LPIPS than Transfer90; Transfer90 prioritizes inheritance and reaches a much stronger detection operating point. Neither outcome proves that all visual artifacts have been removed.')
    page()
    h('7. Teacher evidence and student learning curve')
    tv=v['teacher_validation'];tq=tv['quality'];tm=tv['metrics'];tt=v['teacher_test']
    table(['Measure','Transfer90 teacher'],[
        ['Validation detection',f'{tm["detected"]}/1000 ({100*tm["detection_rate"]:.1f}%)'],
        ['Test detection',f'{tt["detected"]}/1000 ({100*tt["tpr"]:.1f}%); teacher threshold0.8509314060'],
        ['Validation LPIPS / CLIP',f'{tq["paired_lpips"]:.6f} / {tq["fp_clip"]:.6f}; clean teacher CLIP {tq["clean_clip"]:.6f}'],
        ['Interpretation','Teacher detection is strong, but is not a substitute for downstream student detection; teacher FID/KID not included in this exported completed summary.']], [140,371])
    table(['Student checkpoint','Validation MSE','Validation TPR','Selected'],[[str(x['step']),f'{x["denoising_mse"]:.9f}',f'{100*x["metrics"]["tpr"]:.1f}%','Yes' if x['step']==15000 else 'No'] for x in v['validation_milestones']],[125,130,130,126])
    image(TEMP/'validation.png',width)
    p('At12500updates, Transfer90 validation detection is96.3%,versus91.3%for Structure12500 on validation. This reduces the concern that the entire difference appears only after the extra2500updates, but remains a teacher-and-training-recipe comparison. A fixed12500 Transfer90 test endpoint was not evaluated by this report; do not substitute validation96.3%for its test TPR.',small=True)
    page()
    h('8. Quality/detection trade-offs and limitations')
    image(TEMP/'pareto.png',width)
    p('Transfer90 is strongest in detection among the displayed comparable branches; Structure7500 lies farther left in LPIPS. FID values are close at this1000-image sample size. Small KID/FID gaps are point estimates, not evidence of a statistically established ranking. KID subset standard deviation is not a confidence interval across training seeds.')
    p('<b>Visual audit scope.</b>The following20pages show the first20test prompts with no selection by attractiveness,score or detection. Each page includes clean/marked SD2.1teachers and unadapted SD1.4,then baseline k=1,Structure12500and Transfer90students. Teacher seeds use image IDs; student seeds are500000+index. Pairing holds within teacher and student families; teacher-vs-student noise is not identical. The gallery retains all120original512x512PNGs; PDF images use high-quality JPEG previews for portability.')
    p('<b>Limits.</b>One training seed and repeatedly inspected test prompts; no fresh confirmatory holdout. Teacher source,refinement duration,student budget and LR schedule change together. The frozen proxy is a surrogate rather than differentiated full student training. No matched15000clean control,robustness-to-distortion sweep or systematic human preference/anatomy evaluation is completed. Per-image LPIPS is reference distance; altered composition can raise it even if an image is plausible.')
    p('<b>Supported statement.</b>Under the current seed-0 protocol,Transfer90 preserves strong image-level semantic detection through ordinary LoRA adaptation while improving LPIPS/FID/KID relative to the matched k=1baseline. It raises detection above Structure12500at the cost of worse paired LPIPS. The evidence supports a stronger inheritance operating point,not a universal image-quality improvement or a reproduction of the historical kappa=2result.')
    h('9. Evidence, reproducibility and next checks')
    p('The evidence bundle includes completed student/teacher JSONs,verifier report,baseline summary,source code,all generation manifests,20matched prompts and SHA256checks for120images. Student adapter SHA256:960564f91f2d132a139ff5beb7c3b202f8955fe18f0f488d57dc3473c8b7adb9. The snapshot was selected only by validation MSE0.1465669530. Current server location:/var/lib/wmq/logs/project_native_transfer90/runs/run_20261026_114758_661241/transfer90. Archived manifests retain the old/root/logs paths; relocation requires path repair before rerunning.')
    p('Priority validation:repeat seeds and a fresh holdout;add matched15000clean adaptation;compare fixed budgets and leave-one-component-out teacher ablations;perform blinded human comparisons of the same prompt/seed panel. Checkpoint averaging was cancelled and is not part of these results.')
    p('Primary methodological references:<br/>[1]Clark et al.,Directly Fine-Tuning Diffusion Models on Differentiable Rewards(ICLR2024),arxiv.org/abs/2309.17400.<br/>[2]Yu et al.,Gradient Surgery for Multi-Task Learning(NeurIPS2020),arxiv.org/abs/2001.06782.<br/>[3]Zhang et al.,The Unreasonable Effectiveness of Deep Features as a Perceptual Metric(CVPR2018),arxiv.org/abs/1801.03924.<br/>[4]Heusel et al.,GANs Trained by a Two Time-Scale Update Rule...(NeurIPS2017),arxiv.org/abs/1706.08500.<br/>[5]Binkowski et al.,Demystifying MMD GANs(ICLR2018),arxiv.org/abs/1801.01401.<br/>Channel motivation:HMARK(arxiv.org/abs/2512.00094)and Radioactive Watermarks in Diffusion and Autoregressive Image Generative Models(arxiv.org/abs/2506.23731). Their reported metrics are not directly comparable to this experiment.',small=True)
    # All20 comparisons: six images per page, with family-specific seeds.
    for sample in m['samples']:
        page();h(f'10.{sample["index"]+1}. Paired visual comparison: sample{sample["index"]:02d}')
        p('<b>Prompt:</b>'+html.escape(sample['prompt']))
        p(f'Teacher seed(image ID):{sample["image_id"]};student seed:{500000+sample["index"]}. Teacher-vs-student seeds differ. First20test samples;no visual filtering.',small=True)
        families=[('clean_teacher','Clean SD2.1 teacher'),('transfer90_teacher','Transfer90 teacher'),('unadapted','Unadapted SD1.4'),
                  ('baseline_k1','Baseline k=1 student'),('structure12500','Structure12500 student'),('transfer90_student','Transfer90 student')]
        cells=[]
        for key,label in families:
            x=sample['images'][key];src=BUNDLE/x['path'];im=PILImage.open(src).convert('RGB');buf=io.BytesIO();im.save(buf,format='JPEG',quality=92);buf.seek(0)
            score=x.get('score');caption=label
            if score is not None:
                threshold=.8509314060211183 if key in ['transfer90_teacher','clean_teacher'] else .8541307449340821
                caption+=f'<br/>Score{score:.4f}'+(' | detect' if score>=threshold else ' | below threshold')
            cells.append([Paragraph(caption,styles['SmallX']),Image(buf,width=163,height=163)])
            md.append(f'**{label}**\n\n![{label}](../../output_artifacts/previews/transfer90_report_20261027/{x["path"]})')
        grid=Table([cells[:3],cells[3:]],colWidths=[170.33]*3)
        grid.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),14)]));story.append(grid)
        p('Score measures watermark evidence,not image quality. Inspect composition,faces,hands,textures and prompt fidelity in the paired images. Full-resolution original PNGs are available in the accompanying gallery.',small=True)
    MD.write_text('\n\n'.join(md),encoding='utf-8')
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#afc1d2'));c.line(42,31,553,31)
        c.setFont('Helvetica',7);c.setFillColor(colors.HexColor('#536b80'));c.drawString(42,20,'Transfer90 / SD2.1 -> SD1.4 | exploratory seed-0 report |27October2026');c.drawRightString(553,20,str(doc.page))
    doc=SimpleDocTemplate(str(PDF),pagesize=(595.28,841.89),rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=45,title=ptitle,author='Watermark Research',pageCompression=1)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('PDF',PDF,'BYTES',PDF.stat().st_size,'MD',MD)

def main():
    m=extract();new,base,structure,v,data=collect();gallery(m);plots(data,v);report(m,new,base,structure,v,data)
    print('PAIRED',json.dumps(data['paired'],indent=2))

if __name__=='__main__':main()
