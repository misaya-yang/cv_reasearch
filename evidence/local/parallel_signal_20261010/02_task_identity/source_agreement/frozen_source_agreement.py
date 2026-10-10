"""Frozen single candidate: whole flips require source-pre agreement, all900."""
from analyze_whole import *

def main():
    out=ROOT/'source_agreement';out.mkdir(exist_ok=True);assert not(out/'sealed.json').exists()
    cfg=dict(method='Pixelwise strict majority of original final FoRIS, original final whole-role candidate, and actual source pre-CRF mask rendered to original resolution.',source='Existing actual64 score; official minmax with1e-6 floor, bilinear1024 align_corners=False, strict>.5, bilinear original>0.5.',rule='When whole and FoRIS disagree, source_pre decides. When they agree retain their decision. No fitted coefficients or new thresholds.',scope=['paco600','coco200','deep100'],query_GT_in_inference=False,script_sha256=sha(__file__),helper_sha256=sha(ROOT/'analyze_whole.py'))
    write(out/'config.json',cfg);(out/'frozen_source_agreement.py').write_bytes(Path(__file__).read_bytes())
    started=time.monotonic();receipts=[];manifests={}
    for ds,name in [('paco','paco_role_competition600_20261010'),('coco','coco_role_competition200_20261010'),('deep','deepglobe_role_competition100_20261010')]:
        src=DATA/name;rows=read(src/'manifest.json');manifests[ds]=dict(rows=rows,manifest_sha256=sha(src/'manifest.json'),root=str(src))
        inf={r['episode_id']:r for r in lines(src/'inference.jsonl')};sources=read((DATA/'paco_anchored_task600_20261010' if ds=='paco' else src)/'source_score_index.json')
        dest=out/ds;dest.mkdir(exist_ok=True);(dest/'predictions').mkdir(exist_ok=True)
        for row in rows:
            rec=inf[row['episode_id']];shape=tuple(row['query_size_hw']);pp=src/'predictions'/rec['filename'];assert sha(pp)==rec['prediction_sha256']
            with np.load(pp,allow_pickle=False) as z:b=unpack(z,'original/foris.crf',shape);c=unpack(z,'original/task.competitive',shape)
            sr=sources[row['episode_id']];assert sha(sr['path'])==sr['sha256']
            with np.load(sr['path'],allow_pickle=False) as z:s=z[sr.get('score_key','score')].copy()
            sn=(s-s.min())/max(float(s.max()-s.min()),1e-6);pre=render(resize(sn,(1024,1024))>.5,shape)
            candidate=np.where(b==c,b,pre);assert np.array_equal(candidate,(b&c)|(b&pre)|(c&pre))
            # If the whole candidate abstains, agreement is exactly baseline.
            assert np.array_equal(np.where(b==b,b,pre),b)
            target=dest/'predictions'/rec['filename'];np.savez_compressed(target,original_hw=shape,source_pre=np.packbits(pre),source_agreement=np.packbits(candidate))
            receipts.append(dict(dataset=ds,episode_id=row['episode_id'],filename=rec['filename'],prediction_sha256=sha(target),old_prediction_sha256=rec['prediction_sha256'],score_sha256=sr['sha256'],changed_from_whole=int((candidate!=c).sum()),changed_from_foris=int((candidate!=b).sum())))
        print(json.dumps(dict(stage='inference',dataset=ds,n=len(rows))),flush=True)
    # No query image/GT has been opened at this point, all datasets frozen together.
    write(out/'sealed.json',dict(state='ALL900_PREDICTIONS_SEALED',n=len(receipts),config_sha256=sha(out/'config.json'),manifests={ds:v['manifest_sha256'] for ds,v in manifests.items()},receipts=receipts,query_GT_reads=0,raw_reads=0,new_encoder_calls=0,CRF_calls=0,source_head_calls=0,inference_seconds=time.monotonic()-started,threads=1,processes=1))
    summary={};index={r['episode_id']:r for r in receipts}
    for ds,meta in manifests.items():
        src=Path(meta['root']);rows=meta['rows'];previous={r['episode_id']:r for r in lines(ROOT/ds/'episodes.jsonl')};records=[]
        oldindex={r['episode_id']:r for r in lines(src/'inference.jsonl')}
        for row in rows:
            eid=row['episode_id'];rec=index[eid];shape=tuple(row['query_size_hw']);pp=out/ds/'predictions'/rec['filename'];assert sha(pp)==rec['prediction_sha256']
            with np.load(src/'predictions'/oldindex[eid]['filename']) as z:b=unpack(z,'original/foris.crf',shape);c=unpack(z,'original/task.competitive',shape)
            with np.load(pp) as z:pre=unpack(z,'source_pre',shape);pred=unpack(z,'source_agreement',shape)
            with Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
            assert array_sha(raw)==row['query_mask_hash'];gt=resize(raw,shape,'nearest')>.5
            iu={**previous[eid]['iu'],'source_pre':counts(pre,gt),'source_agreement':counts(pred,gt)};es={**previous[eid]['edits'],'source_pre':edits(pre,b,gt),'source_agreement':edits(pred,b,gt)}
            records.append(dict(episode_id=eid,fold=row['fold'],class_id=row['loader_class_id'],iu=iu,edits=es,edits_vs_whole=edits(pred,c,gt),query_photo_id=row.get('query_photo_id')))
        expected=read(src/'config.json').get('expected_classes') if ds!='deep' else {'0':[0]}
        ag=aggregate(records,expected,'foris.crf','source_agreement');cs=ag.pop('classes');ag['n']=len(rows);ag['edits_vs_foris']=np.sum([r['edits']['source_agreement'] for r in records],0).tolist();ag['edits_vs_whole']=np.sum([r['edits_vs_whole'] for r in records],0).tolist();summary[ds]=ag
        (out/ds/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(out/ds/'classes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cs))
    write(out/'summary.json',summary);write(out/'COMPLETE.json',dict(state='COMPLETE',n=900,all900_sealed_before_GT=True,seconds=time.monotonic()-started,miou={ds:s['miou'] for ds,s in summary.items()}))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
