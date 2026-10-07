"""Deterministic diagnostic plots from copied images and sealed token masks."""
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
FONT=font_manager.findfont('DejaVu Sans')
F=ImageFont.truetype(FONT,15);BOLD=ImageFont.truetype(FONT,18)
CW,CH=320,282


def overlay(im,regions):
    a=np.array(im).astype(float)
    for mask,color in regions:
        m=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize(im.size,Image.Resampling.NEAREST))>0
        a[m]=.42*a[m]+.58*np.array(color)
    return Image.fromarray(a.clip(0,255).astype(np.uint8))


def fit(im):
    cell=Image.new('RGB',(CW,CH),'#f5f5f1');im=ImageOps.contain(im,(CW-8,CH-8),Image.Resampling.LANCZOS)
    cell.paste(im,((CW-im.width)//2,(CH-im.height)//2));return cell


def main():
    entries=json.loads((ROOT/'manifest.json').read_text())['cases'];panels=[]
    for meta in entries:
        d=ROOT/f"case{meta['case']:02d}"
        ref=Image.open(d/'reference.png').convert('RGB');q=Image.open(d/'query.png').convert('RGB')
        rm=np.asarray(Image.open(d/'reference_mask.png'))>0;gt=np.asarray(Image.open(d/'query_mask.png'))>0
        z=np.load(d/'masks.npz');b,t,m=z['base'],z['truth'],z['candidate'];deleted=b&~m
        imgs=[overlay(ref,[(rm,(255,220,20))]),overlay(q,[(gt,(0,235,70))]),
              overlay(q,[(b&t,(0,220,80)),(b&~t,(255,40,20)),(~b&t,(35,100,255))]),
              overlay(q,[(deleted&~t,(0,220,240)),(deleted&t,(245,0,220))]),
              overlay(q,[(m&t,(0,220,80)),(m&~t,(255,40,20)),(~m&t,(35,100,255))])]
        panel=Image.new('RGB',(CW*5,CH+96),'white');draw=ImageDraw.Draw(panel)
        title=f"{meta['case']:02d}  {meta['dataset']}  {meta['pack']}  class {meta['c']}  |  B IoU {meta['base_iou']*100:.1f} -> C1 {meta['candidate_iou']*100:.1f}"
        draw.text((8,4),title,font=BOLD,fill='black')
        for j,(im,name) in enumerate(zip(imgs,['Reference + mask (yellow)','Query + original GT (green)','B: TP green / FP red / FN blue','Deleted: correct cyan / wrong magenta','C1: TP green / FP red / FN blue'])):
            draw.text((j*CW+8,33),name,font=F,fill='black');panel.paste(fit(im),(j*CW,58))
        dt=meta['record']['edits']['joint_mask'][2];df=meta['record']['edits']['joint_mask'][3]
        draw.text((8,CH+67),f"Deleted true {dt} / false {df} tokens | B area / truth {meta['pred_to_truth']:.2f} | Metric masks: sealed 64x64, nearest-projected for display.",font=F,fill='#333333')
        panel.save(d/'comparison.png');panels.append(panel)
    for start in (0,5):
        h=panels[0].height;sheet=Image.new('RGB',(CW*5,h*5),'white')
        for k,im in enumerate(panels[start:start+5]):sheet.paste(im,(0,k*h))
        sheet.save(ROOT/f'contact_{start+1:02d}_{start+5:02d}.png')
    print('Rendered 10 diagnostic comparisons and 2 contact sheets.')


if __name__=='__main__':main()
