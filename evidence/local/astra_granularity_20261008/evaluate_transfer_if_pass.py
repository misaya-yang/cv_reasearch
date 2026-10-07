"""Only after C1 passes: one streaming read of PASCAL, SUIM and LVIS; no feature cache."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from candidate import predict
from run_candidate import ARMS, summarize

ROOT=Path(__file__).resolve().parent


def main():
    receipt=json.loads((ROOT/'candidate_C1/receipt.json').read_text())
    if not receipt['passed']:
        (ROOT/'transfer_skipped.json').write_text(json.dumps({'reason':'C1 did not pass fixed gates; no PASCAL/SUIM/LVIS inference run'})+'\n')
        print('TRANSFER_NOT_RUN: fixed candidate failed gate',flush=True)
        return
    # Importing the encoder is delayed until the actual measured gate passes.
    from cache_features import enc, coverage, O
    result={}
    start=time.monotonic()
    out=ROOT/'transfer_C1'
    out.mkdir(exist_ok=False)
    for dataset,packs in [('PASCAL-Part',['pascal_part']),('SUIM',['suim']),('LVIS-92i',[f'lvis_f{i}' for i in range(4)])]:
        records=[]
        for pack in packs:
            rows=json.loads((O/('claude_rcg2_'+pack)/'rows.json').read_text())
            field=np.load(O/('claude_rcg2_'+pack)/'counts.npz')['fields']
            data=O/'claude_packs'/pack
            with (out/(pack+'.jsonl')).open('x') as stream:
                for i,row in enumerate(rows):
                    q=torch.tensor(enc(data/'data'/row['query']).astype(np.float32),device='cuda')
                    r=torch.tensor(enc(data/'data'/row['support']).astype(np.float32),device='cuda')
                    rm=np.asarray(Image.open(data/'ann'/row['support']).convert('L'))>0
                    cov=torch.tensor(coverage(rm.astype(np.float32)),device='cuda')
                    b=field[i].reshape(-1)>.5
                    pred={k:v.cpu().numpy() for k,v in predict(q,r,cov,torch.tensor(b,device='cuda')).items()}
                    tt=coverage((np.asarray(Image.open(data/'ann'/row['query']).convert('L'))>0).astype(np.float32))>.5
                    iu={k:[int((m&tt).sum()),int((m|tt).sum())] for k,m in dict(B=b,**pred).items()}
                    edits={}
                    for k,m in pred.items():
                        ad,de=m&~b,b&~m
                        edits[k]=[int((ad&tt).sum()),int((ad&~tt).sum()),int((de&tt).sum()),int((de&~tt).sum())]
                    label=pack+':'+str(row['c']) if dataset=='LVIS-92i' else str(row['c'])
                    rec=dict(key=row['key'],pack=pack,**{'class':label},iu=iu,edits=edits)
                    records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush()
                    if (i+1)%25==0 or i+1==len(rows):
                        print(json.dumps(dict(dataset=dataset,pack=pack,n=i+1,total=len(rows),seconds=time.monotonic()-start)),flush=True)
        result[dataset]=summarize(records)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('TRANSFER_DATASET_DONE',dataset,json.dumps(result[dataset]),flush=True)
    text=(ROOT/'REPORT.md').read_text()
    marker='\n## 上一轮泛化研究（本轮候选前）'
    lines=['','扩展单次读数（只报告）：']
    for name,r in result.items():
        m=r['arms']['joint_mask'];ci=m['delta_ci95']
        actions='/'.join(map(str,m['edits_addTP_addFP_delTP_delFP']))
        lines.append(f"{name}: B {r['B']:.2f} → C1 {m['miou']:.2f}, Δ{m['delta']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]；四动作{actions}。")
    (ROOT/'REPORT.md').write_text(text.replace(marker,'\n'.join(lines)+'\n'+marker))
    (out/'receipt.json').write_text(json.dumps(dict(state='COMPLETE',seconds=time.monotonic()-start),indent=2)+'\n')


if __name__=='__main__':
    main()
