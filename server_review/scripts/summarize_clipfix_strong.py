"""Summarize only the four corrected raw/strong conditions, never pre-fix rows."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

r=Path('/root/workspace/serum')
names=['C00_seed0_clipfix','C10_seed0_clipfix','C01_step1.00_seed0_clipfix','C11_step1.00_seed0_clipfix']
threshold=np.float32(json.loads((r/'metrics/serum_threshold_1pct.json').read_text())['threshold'])
rng=np.random.default_rng(42000)
def ci(v):
    return np.quantile([v[rng.integers(0,len(v),len(v))].mean() for _ in range(1000)],[.025,.975]).tolist()
score={};det={};clip={};rows=[]
for n in names:
    status=json.loads((r/f'metrics/{n}_status.json').read_text());assert status['stage']=='all' and status['state']=='completed'
    score[n]=np.load(r/f'metrics/{n}_scores.npy').reshape(-1);clip[n]=np.load(r/f'metrics/{n}_clipscore.npy').reshape(-1)
    assert len(score[n])==len(clip[n])==5000 and np.isfinite(score[n]).all() and np.isfinite(clip[n]).all()
    det[n]=(score[n]>=threshold).astype(float)
    utility=json.loads((r/f'metrics/{n}_utility.json').read_text())
    best=json.loads((r/f'students/{n}/validation_selection.json').read_text())['best']
    rows.append({'name':n,'positive_count':int(det[n].sum()),'detection_rate':float(det[n].mean()),'detection_ci95':ci(det[n]),'score_mean':float(score[n].mean()),'score_median':float(np.median(score[n])),'clipscore':float(clip[n].mean()),'clipscore_ci95':ci(clip[n]),'best':best,**utility['distribution_metrics']})
c00,c10,c01,c11=names
effects={}
for label,values in {'C10_minus_C11_tpr':det[c10]-det[c11],'clean_adjusted_tpr_reduction':(det[c10]-det[c00])-(det[c11]-det[c01]),'delta_launder_mean_score':(score[c11]-score[c01])-(score[c10]-score[c00]),'C11_minus_C10_clipscore':clip[c11]-clip[c10],'C01_minus_C00_clipscore':clip[c01]-clip[c00]}.items():
    effects[label]={'mean':float(values.mean()),'ci95':ci(values)}
roc={}
fig,axes=plt.subplots(1,2,figsize=(11,4))
for tag,clean,wm in [('raw',c00,c10),('strong',c01,c11)]:
    pos,neg=score[wm],score[clean]
    ranks=rankdata(np.concatenate([pos,neg]));auc=float((ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
    roc[tag]={'auc':auc}
    x=np.concatenate([neg,pos]);labels=np.concatenate([np.zeros(len(neg)),np.ones(len(pos))]);order=np.argsort(-x,kind='stable');xs=x[order];ys=labels[order]
    endpoints=np.r_[np.flatnonzero(np.diff(xs)),len(xs)-1]
    tp=np.r_[0,np.cumsum(ys)[endpoints]]/len(pos);fp=np.r_[0,np.cumsum(1-ys)[endpoints]]/len(neg)
    axes[0].plot(fp,tp,label=f'{tag}: AUC {auc:.3f}')
for n in names:axes[1].hist(score[n],bins=np.linspace(0,1,51),histtype='step',label=n.replace('_seed0_clipfix',''))
axes[0].plot([0,1],[0,1],'k--',alpha=.4);axes[0].set(xlabel='False positive rate',ylabel='True positive rate');axes[0].legend()
axes[1].set(yscale='log',xlabel='SERUM score',ylabel='Image count');axes[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(r/'metrics/clipfix_strong_roc_hist.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5))
for row in rows:
    if row['name'] in [c10,c11]:
        x,y=row['clipscore'],row['detection_rate']
        ax.errorbar(x,100*y,xerr=[[x-row['clipscore_ci95'][0]],[row['clipscore_ci95'][1]-x]],yerr=[[100*(y-row['detection_ci95'][0])],[100*(row['detection_ci95'][1]-y)]],fmt='o',label=row['name'])
ax.set(xlabel='Prompt-image CLIPScore',ylabel='Detection (%)',title='Corrected seed 0: raw vs strong');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(r/'metrics/clipfix_strong_frontier.png',dpi=180);plt.close(fig)
# Fixed prompt outputs for qualitative comparison; no selection by detector score.
for page in range(2):
    sheet=Image.new('RGB',(1024,1060),'white');d=ImageDraw.Draw(sheet)
    for col,n in enumerate(names):d.text((col*256+4,4),n.replace('_seed0_clipfix',''),fill='black')
    for row in range(4):
        idx=page*4+row
        for col,n in enumerate(names):
            with Image.open(r/f'student_outputs/{n}/sample_{idx:06d}.png') as im:sheet.paste(im.resize((256,256)),(col*256,30+row*256))
    sheet.save(r/f'metrics/clipfix_strong_qualitative_{page}.png')
result={'scope':'Corrected seed-0 2x2, strong only by user instruction; no mild reruns. Baseline adequacy and attack success require review; no automatic pass.','threshold':float(threshold),'n_per_condition':5000,'bootstrap_replicates':1000,'rows':rows,'paired_effects':effects,'roc':roc,'reference':'5000 clean SD2.1 teacher outputs; not COCO FID','remaining':'Other adaptation seeds, mechanism and defense conditional on credible baseline and stable effect'}
(r/'metrics/clipfix_strong_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
