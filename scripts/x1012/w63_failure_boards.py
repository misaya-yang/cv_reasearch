"""Post-hoc inspection of all 19 W63 tasks whose exact SH-field oracle is <.5.

No model calls or method selection. All original pair/task identities retained.
"""
from pathlib import Path
import gzip
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/x1012'),str(ROOT/'scripts/x1013'),str(ROOT/'src')]
import common as c
import plan_common as pc
from ics.official_data import decoded_rgb

OUT=c.task_path('W63')


def oracle_mask(field,gt,valid):
    values=field[valid];labels=gt[valid]
    order=np.argsort(-values,kind='stable');scores=values[order]
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    intersection=np.cumsum(labels[order],dtype=np.int64)[ends]
    scores_iou=intersection/np.maximum(int(labels.sum())+ends+1-intersection,1)
    j=int(np.argmax(scores_iou));end=int(ends[j]);cut=values[order[end+1]] if end+1<len(order) else -np.inf
    pred=(field>cut)&valid;iu=c.iu(pred,gt,valid)
    assert abs(iu[0]/max(iu[1],1)-scores_iou[j])<1e-12
    return pred,float(scores_iou[j]),float(cut)


def contour(ax,mask,color):
    if mask.any() and not mask.all():ax.contour(mask,levels=[.5],colors=[color],linewidths=.6)


def tint(ax,mask):
    overlay=np.zeros((*mask.shape,4),np.float32)
    overlay[...,1]=1.;overlay[...,2]=.5;overlay[...,3]=mask*.23
    ax.imshow(overlay)


def main():
    assert (OUT/'complete.json').exists() and (OUT/'scored.json').exists()
    meta={r['pair_id']:r for r in json.loads((OUT/'manifest.json').read_text())['episodes']}
    with gzip.open(ROOT/'evidence/research/w63_task_switch_S.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
    cases=[]
    for r in rows:
        for i in range(2):
            if r['oracle'][i][i]<.5:cases.append((r,i))
    assert len(cases)==19
    dest=OUT/'failure_boards';dest.mkdir(exist_ok=True)
    manifest=[]
    for page,start in enumerate(range(0,len(cases),4),1):
        chunk=cases[start:start+4];fig,axes=plt.subplots(len(chunk),4,figsize=(13,3.8*len(chunk)),squeeze=False)
        for line,(r,i) in enumerate(chunk):
            label=('A','B')[i];b=meta[r['pair_id']]['bindings'][i];gt,valid=pc.truth(b)
            with np.load(r['prediction'],allow_pickle=False) as z:
                field=z['field_'+label];post=c.unpack(z[label+'+']);best,j,cut=oracle_mask(field,gt,valid)
            assert abs(j-r['oracle'][i][i])<1e-12
            reference=np.asarray(decoded_rgb(c.ASSETS/b['row']['reference_path'],b['row'].get('reference_crop')).resize((1024,1024)))
            query=np.asarray(decoded_rgb(c.ASSETS/b['row']['query_path'],b['row'].get('query_crop')).resize((1024,1024)))
            reference_mask=pc.lab.mask1024(b['row']['reference_mask_path']).numpy()
            ax=axes[line,0];ax.imshow(reference);tint(ax,reference_mask);contour(ax,reference_mask,'#00ff80')
            ax.set_title(f'{r["pair_id"]} / {label}: class {r["classes"][i]}\nReference label (green)',fontsize=9)
            ax=axes[line,1];ax.imshow(query);tint(ax,gt);contour(ax,gt,'#00ff80');contour(ax,post,'#ff4455')
            iu=r['post']['iu'][i][i];natural=iu[0]/max(iu[1],1)
            ax.set_title(f'Query: GT green, SH+ red\nIoU={natural:.3f}',fontsize=9)
            ax=axes[line,2];ax.imshow(field,cmap='magma',vmin=0,vmax=1);contour(ax,gt,'#00ff80');contour(ax,field>.5,'white')
            ax.set_title('Pre-CRF field [0,1]\nGT green, natural .5 cut white',fontsize=9)
            ax=axes[line,3];ax.imshow(query);tint(ax,gt);contour(ax,gt,'#00ff80');contour(ax,best,'#ffd84d')
            ax.set_title(f'GT-selected exact cut (yellow)\nIoU={j:.6f}; diagnostic only',fontsize=9)
            for ax in axes[line]:ax.set_axis_off()
            manifest.append(dict(pair_id=r['pair_id'],task=label,dataset=r['dataset'],class_id=r['classes'][i],
                                 natural_post_iou=natural,oracle_iou=j,oracle_cut=cut,gt_fraction=float(gt.mean()),
                                 page=f'page_{page:02d}.png',row=line+1,source_indices=r['source_indices'][i]))
        fig.suptitle('W63: all fixed-field failures | Post-hoc inspection, no parameter changes',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.97));fig.savefig(dest/f'page_{page:02d}.png',dpi=130);plt.close(fig)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved all19 task examples across5 pages; oracle masks reproduce sealed exact scores')


if __name__=='__main__':main()
