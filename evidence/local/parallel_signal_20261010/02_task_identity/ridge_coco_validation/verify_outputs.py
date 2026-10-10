"""Read-only full-output reconstruction and post-seal difficulty diagnostics."""
from run import *


def main():
    started=time.monotonic();cfg=read(ROOT/'config.json');seal=read(ROOT/'sealed.json')
    assert seal['state']=='ALL200_PREDICTIONS_SEALED' and sha(ROOT/'config.json')==seal['config_sha256']
    manifest=read(ROOT/'manifest.json');index=read(ROOT/'source_index.json');rows=[];checks=defaultdict(int)
    for row,rec in zip(manifest,seal['receipts']):
        assert row['episode_id']==rec['episode_id'];source=index[row['episode_id']];shape=tuple(row['query_size_hw'])
        fp=ROOT/'fields'/rec['filename'];pp=ROOT/'predictions'/rec['filename']
        assert sha(fp)==rec['fields_sha256'] and sha(pp)==rec['prediction_sha256'];checks['new_file_sha']+=2
        with np.load(fp) as z:score=z['ridge_global'].copy();h=z['evidence_global'].copy()
        scale=rec['calibration']['separation'];threshold=rec['calibration']['threshold']
        expected=np.clip((score-threshold)/scale,-1,1) if scale>np.finfo(float).eps else np.zeros_like(score)
        assert np.array_equal(expected,h);checks['evidence_formula']+=1
        sr=source['source_score'];assert sha(sr['path'])==sr['sha256']
        with np.load(sr['path']) as z:s=z['score'].reshape(64,64).astype(float);oldh=z['h'].copy() if bool(z['valid']) else np.zeros((64,64))
        assert sha(source['baseline_path'])==source['baseline_sha256']
        with np.load(source['baseline_path']) as z:b=np.unpackbits(z[source['baseline_key']],count=np.prod(shape)).reshape(shape).astype(bool)
        with np.load(pp) as z:p=np.unpackbits(z['anchor.global'],count=np.prod(shape)).reshape(shape).astype(bool)
        span=float(s.max()-s.min());norm=(s-s.min())/max(span,1e-6)
        amp=np.abs(2*resize(norm,shape)-1) if span else np.zeros(shape)
        g=(2*b.astype(float)-1)*amp;d=g+.5*resize(h,shape);rebuilt=(d>0)|((d==0)&b)
        assert np.array_equal(p,rebuilt);checks['complete_original_mask']+=1
        assert np.array_equal((g>0)|((g==0)&b),b);checks['zero_h_identity']+=1
        # Query labels are only used after all outputs were sealed and already scored.
        with c.math.Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert c.math.array_sha(raw)==row['query_mask_hash'];t=resize(raw,(1024,1024),'nearest')>.5
        bc=resize(b,(1024,1024))>.5;pc=resize(p,(1024,1024))>.5
        mutable=np.abs(2*resize(norm,(1024,1024))-1)<=.5 if span else np.ones((1024,1024),bool)
        rois={'baseline_fg':bc,'mutable_fg':bc&mutable,'mutable_bg':~bc&mutable,'new_added':pc&~bc,'new_deleted':bc&~pc}
        signals={'new_ridge':score.ravel(),'new_h':h.ravel(),'old_h':resize(oldh,(128,128)).ravel(),'source':resize(s,(128,128)).ravel()}
        ranked={name:{signal:c.math.weighted_rank(values,c.math.mass128(mask&t),c.math.mass128(mask&~t)) for signal,values in signals.items()} for name,mask in rois.items()}
        hc=resize(h,(1024,1024));roles={name:dict(pixels=int(mask.sum()),positive=int((mask&(hc>0)).sum()),negative=int((mask&(hc<0)).sum())) for name,mask in {'TP':bc&t,'FP':bc&~t,'FN':~bc&t,'TN':~bc&~t}.items()}
        rows.append(dict(episode_id=row['episode_id'],ranks=ranked,roles=roles))
    summary={roi:{signal:{metric:dict(n=len(v:=[r['ranks'][roi][signal][metric] for r in rows if r['ranks'][roi][signal][metric] is not None]),mean=float(np.mean(v)) if v else None) for metric in ['auc','ap','tpr_at_fpr05']} for signal in rows[0]['ranks'][roi]} for roi in rows[0]['ranks']}
    roles={role:{metric:int(sum(r['roles'][role][metric] for r in rows)) for metric in ['pixels','positive','negative']} for role in rows[0]['roles']}
    (ROOT/'diagnostic_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(ROOT/'verification.json',dict(state='VERIFIED',n=len(rows),checks=dict(checks),ranks=summary,canvas_role_totals=roles,seconds=time.monotonic()-started,
          interpretation='Post-seal GT diagnosis. canonical1024 from original final guides; exact ROI FG/BG pixel mass per128 node. New score/h are128 fields; old score/h are64 bilinear128. Not original mIoU or independent confirmation.',source_seal_sha256=sha(ROOT/'sealed.json'),script_sha256=sha(__file__),generator_sha256=sha(ROOT/'run.py'),raw_reads=0,model_calls=0))
    print(json.dumps(dict(checks=dict(checks),ranks={roi:{s:v['auc'] for s,v in values.items()} for roi,values in summary.items()},roles=roles,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
