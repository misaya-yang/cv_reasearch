"""A4 continuation: label-free J chooses strength c * novelty**s, s in [0,1]."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import lab
import plan_common as pc
import plan_report as pr
FIELDS=lab.ASSETS/'a/x1012_P0/fields'
OUT=lab.ASSETS/'a/x1012_A4/modulation'
STRENGTHS=tuple(float(x) for x in np.linspace(0,1,9))


def quality(mask,c,back):
    native=mask.reshape(64,16,64,16).mean(axis=(1,3)).reshape(-1)>.5
    if not native.any():return 0.
    precision=float(np.mean(c[native]));capture=float(np.sum(back[native]))
    return precision*capture/max(precision+capture-precision*capture,1e-12)


def produce():
    pc.setup('cpu');OUT.mkdir(parents=True,exist_ok=True)
    for name in ('predictions','records'):(OUT/name).mkdir(exist_ok=True)
    tasks=pc.seed_rows(150);config=OUT/'config.json'
    if not config.exists():
        pc.write(config,{'state':'FROZEN','s_grid':STRENGTHS,'method':'maximise label-free J over c*novelty**s; lower s wins ties',
                         'n':len(tasks),'query_GT_in_inference':False,'new_encoder_calls':0,
                         'source_sha256':{str(p):pc.sha(p) for p in (Path(__file__),Path(pc.__file__),Path(lab.__file__))}})
        pc.write(OUT/'manifest.json',tasks)
    assert all(pc.sha(p)==h for p,h in pc.read(config)['source_sha256'].items())
    start=time.monotonic();receipts=[]
    for i,b in enumerate(tasks,1):
        name=pc.key(b);rec=OUT/'records'/(name+'.json');path=OUT/'predictions'/(name+'.npz')
        if rec.exists():
            r=pc.read(rec);assert pc.sha(r['path'])==r['sha256'];receipts.append(r);continue
        with pc.inference_only(b):
            f=pc.read(FIELDS/'records'/(name+'.json'));assert pc.sha(f['path'])==f['sha256']
            with np.load(f['path'],allow_pickle=False) as z:
                c=z['c32'].copy();back=z['back_f'].astype(np.float64)
                novelty=np.asarray(z['sim_max'],np.float32);novelty=(novelty-novelty.min())/max(float(novelty.max()-novelty.min()),1e-6)
                labels=torch.from_numpy(z['regions64'].astype(np.int64));oh=torch.nn.functional.one_hot(labels,64).float()
                ind0=z['mask/IND0'].copy();indn=z['mask/INDN'].copy()
            masks=[];values=[];continuous=[]
            for s in STRENGTHS:
                v=torch.from_numpy(c*np.power(novelty,s));field=oh@((oh.T@v)/oh.sum(0).clamp_min(1));continuous.append(field.numpy())
                mask=pc.unpack(ind0 if s==0 else indn) if s in (0,1) else pc.readout(field)
                masks.append(pc.pack(mask));values.append(quality(mask,c,back))
            chosen=int(np.argmax(values));endpoint=8 if values[8]>values[0] else 0
            pred={'IND0':ind0,'INDN':indn,'J_endpoint':masks[endpoint],'J_continuous':masks[chosen]}
        assert not path.exists();np.savez_compressed(path,**pred,selected_field32=continuous[chosen])
        r={'index':name,'path':str(path),'sha256':pc.sha(path),'selected_s':STRENGTHS[chosen],
           'J_grid':values,'endpoint_s':STRENGTHS[endpoint],'field_source_sha256':f['sha256'],'new_encoder_calls':0,'query_GT_reads':0}
        pc.write(rec,r);receipts.append(r)
        if i%100==0:print({'completed':i,'n':len(tasks),'seconds':round(time.monotonic()-start,1)},flush=True)
    pc.write(OUT/'sealed.json',{'state':'ALL_J_MODULATION_PREDICTIONS_SEALED','n':len(tasks),'records':receipts,
                              'seconds':time.monotonic()-start,'new_encoder_calls':0,'config_sha256':pc.sha(config)})


def score():
    seal=pc.read(OUT/'sealed.json');indexed={r['index']:r for r in seal['records']};records=[]
    arms=('FoRIS','IND0','INDN','J_endpoint','J_continuous')
    for b in pc.read(OUT/'manifest.json'):
        r=indexed[pc.key(b)];assert pc.sha(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as z:pred={a:pc.unpack(z[a]) for a in arms if a!='FoRIS'}
        pred['FoRIS']=lab.foris_cli(b).numpy();records.append(pr.score_record(b,pred))
    result=pr.report('A4_modulation',OUT,records,arms,['A4 J差值的报告半weighted AUROC>.75后，按计划进入逐例连续调制检验。',
        's固定0..1九点只是标签无关目标J的数值搜索，未用GT选s/阈值/域；并列选更小s。c/novelty/region均复用原缓存。',
        'J_endpoint用J直接在原两端点选择；J_continuous可选择中间强度。无FoRIS输出进入推断。',
        '首轮仍是已曝光原开发1360，不是confirm；无新增encoder、view、训练。'])
    pc.write(OUT/'completion.json',{'state':'COMPLETE_FIXED_J_MODULATION','n':len(records),'new_encoder_calls':0,
              'uniform_nondecreasing_vs_IND0':all(s['J_continuous']['class_miou']>=s['IND0']['class_miou'] for s in result['summary'].values()),
              'uniform_over_FoRIS':all(s['J_continuous']['delta_to_foris_pp']>0 for s in result['summary'].values()),'confirm_scored':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('produce','score'));a=p.parse_args();produce() if a.mode=='produce' else score()
