"""Reconstruct existing LVIS window/component decisions, no raw or inference."""
from analyze_whole import *
from scipy.ndimage import label

def main():
    started=time.monotonic();out=ROOT/'lvis';out.mkdir(exist_ok=True);assert not(out/'sealed.json').exists()
    roots={k:DATA/v for k,v in {'source':'lvis_fixedwindows200_20261009','reference':'lvis_window_presence200_20261009','context':'lvis_context_presence200_20261009','component':'lvis_component_presence200_20261009'}.items()}
    manifest=read(roots['source']/'manifest.json');indices={k:{r['episode_id']:r for r in lines(p/'inference.jsonl')} for k,p in roots.items()}
    known={}
    for k in ['context','component']:
        for r in lines(roots[k]/'score/scored_episodes.jsonl'):known.setdefault(r['episode_id'],{}).update(r['iu'])
    for k in ['reference','context','component']:assert read(roots[k]/'manifest.json')==manifest
    records=[];windows=[];components=[];checks=defaultdict(int);receipts=[]
    for idx,row in enumerate(manifest):
        eid=row['episode_id'];shape=tuple(row['query_size_hw']);recs={k:ix[eid] for k,ix in indices.items()};fn=recs['source']['filename']
        assert all(r['filename']==fn for r in recs.values())
        assert recs['source']['raw_keys']==recs['context']['query_window_raw_keys']==recs['component']['query_window_raw_keys']==recs['reference']['query_raw_keys'];checks['window_key_identity']+=1
        masks={};canvases={};paths=[]
        for k,p in roots.items():
            pp=p/'predictions'/fn;paths.append(dict(path=str(pp),sha256=sha(pp)))
            with np.load(pp,allow_pickle=False) as z:
                for key in z.files:
                    if key.startswith('original/'):
                        arm=key.split('/',1)[1];masks[arm]=unpack(z,key,shape)
                    elif key.startswith('canvas/'):canvases[key.split('/',1)[1]]=unpack(z,key,(1024,1024))
        b=masks['foris.crf'];guide=render(b,(1024,1024))
        with Image.open(row['query_mask_path']) as im:raw=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert array_sha(raw)==row['query_mask_hash'];checks['query_mask_hash']+=1
        gt=resize(raw,shape,'nearest')>.5;t=resize(raw,(1024,1024),'nearest')>.5
        iu={a:counts(m,gt) for a,m in masks.items() if a in known[eid]}
        for a,value in iu.items():assert value==known[eid][a],(eid,a,value,known[eid][a]);checks['original_IU']+=1
        masks={a:masks[a] for a in iu};es={a:edits(m,b,gt) for a,m in masks.items()}
        votes={a:np.zeros((1024,1024),np.uint8) for a in ['window.native','presence.reference','presence.query_context','presence.reference_query','component.query_context','component.global_agreement_budget']};den=np.zeros((1024,1024),np.uint8)
        wp=roots['source']/'window_masks'/fn;paths.append(dict(path=str(wp),sha256=sha(wp)));wr=[];cr=[]
        with np.load(wp,allow_pickle=False) as z:
            for j,box in enumerate(recs['source']['window_boxes']):
                x0,y0,x1,y1=box;local=unpack(z,'window.native/'+str(j),(512,512));tl=t[y0:y1,x0:x1];bl=guide[y0:y1,x0:x1]
                tp=int((local&tl).sum());fp=int((local&~tl).sum());den[y0:y1,x0:x1]+=1
                take={'window.native':True,'presence.reference':recs['reference']['accepted'][j],**{a:recs['context']['selections'][a][j] for a in ['presence.query_context','presence.reference_query']}}
                for a,v in take.items():
                    if v:votes[a][y0:y1,x0:x1]+=local
                wr.append(dict(episode_id=eid,window=j,tp=tp,fp=fp,gt_pixels=int(tl.sum()),reference_margin=recs['context']['reference_margin'][j],context_margin=recs['context']['query_context_margin'][j],take=take))
                ids,n=label(local,np.ones((3,3),bool));detail=recs['component']['regions'][j];assert detail['components']==n
                ar=np.bincount(ids.ravel(),minlength=n+1)[1:];assert ar.tolist()==detail['component_sizes'];checks['component_partition_sizes']+=1
                tps=np.bincount(ids.ravel(),weights=tl.ravel(),minlength=n+1)[1:];overlap=np.bincount(ids.ravel(),weights=bl.ravel(),minlength=n+1)[1:]
                for a,takes in detail['selections'].items():votes[a][y0:y1,x0:x1]+=np.asarray([False]+takes,bool)[ids]
                for k in range(n):
                    cr.append(dict(episode_id=eid,window=j,component=k+1,area=int(ar[k]),tp=int(tps[k]),fp=int(ar[k]-tps[k]),guide_overlap=int(overlap[k]),margin=detail['context_margin'][k],agreement=detail['global_agreement'][k],accept=detail['selections']['component.query_context'][k],geometry_accept=detail['selections']['component.global_agreement_budget'][k],window_accept=take['presence.query_context']))
        for a,v in votes.items():
            canvas=2*v>den
            if a in canvases:assert np.array_equal(canvas,canvases[a]);checks['canvas_reconstruction']+=1
            assert np.array_equal(render(canvas,shape),masks[a]);checks['original_reconstruction']+=1
        # Ranking only within the existing proposal foreground; area weights overlap across windows.
        wrank={key:rank(np.array([w[key] if w[key] is not None else 0 for w in wr]),np.array([w['tp'] for w in wr]),np.array([w['fp'] for w in wr])) for key in ['reference_margin','context_margin']}
        crank={key:rank(np.array([c[key] if c[key] is not None else 0 for c in cr]),np.array([c['tp'] for c in cr]),np.array([c['fp'] for c in cr])) for key in ['margin','agreement']}
        gate={a:{typ:int(sum(w[typ] for w in wr if w['take'][a])) for typ in ['tp','fp']} for a in ['window.native','presence.reference','presence.query_context','presence.reference_query']}
        gate['component.query_context']={typ:int(sum(c[typ] for c in cr if c['accept'])) for typ in ['tp','fp']}
        gate['component.global_agreement_budget']={typ:int(sum(c[typ] for c in cr if c['geometry_accept'])) for typ in ['tp','fp']}
        rec=dict(episode_id=eid,fold=row['fold'],class_id=row['loader_class_id'],iu=iu,edits=es,query_guide_recall=float((guide&t).sum()/max(t.sum(),1)),query_guide_precision=float((guide&t).sum()/max(guide.sum(),1)),window_rank=wrank,component_rank=crank,gate=gate,gt_pixels=int(gt.sum()),canvas_gt_pixels=int(t.sum()))
        records.append(rec);windows.extend(wr);components.extend(cr);receipts.append(dict(episode_id=eid,files=paths,query_mask_hash=row['query_mask_hash']))
        if (idx+1)%50==0:print(json.dumps(dict(dataset='lvis',n=idx+1,seconds=time.monotonic()-started)),flush=True)
    comparisons={}
    for base,cand in [('foris.crf','presence.query_context'),('foris.crf','component.query_context'),('presence.reference','presence.query_context'),('presence.query_context','component.query_context'),('component.global_agreement_budget','component.query_context')]:
        rs=[]
        for r in records:
            # Edits for non-FoRIS comparisons are reconstructed from current final masks.
            eid=r['episode_id'];row=next(x for x in manifest if x['episode_id']==eid);shape=tuple(row['query_size_hw']);fn=indices['source'][eid]['filename'];ps={}
            for p in roots.values():
                with np.load(p/'predictions'/fn) as z:
                    for a in [base,cand]:
                        if 'original/'+a in z:ps[a]=unpack(z,'original/'+a,shape)
            with Image.open(row['query_mask_path']) as im:gt=resize(np.asarray(im.convert('L'))>0,shape,'nearest')>.5
            rs.append(dict(episode_id=eid,fold=r['fold'],class_id=r['class_id'],iu={base:r['iu'][base],cand:r['iu'][cand]},edits={cand:edits(ps[cand],ps[base],gt)}))
        ag=aggregate(rs,None,base,cand);cs=ag.pop('classes');ag['edits']=np.sum([r['edits'][cand] for r in rs],0).tolist();comparisons[base+' -> '+cand]=ag
        (out/(base+'__'+cand+'.classes.jsonl')).write_text(''.join(json.dumps(c)+'\n' for c in cs))
    groups={}
    for group,test in [('guide_recall_lt20',lambda r:r['query_guide_recall']<.2),('guide_recall_20to50',lambda r:.2<=r['query_guide_recall']<.5),('guide_recall_ge50',lambda r:r['query_guide_recall']>=.5)]:
        rs=[r for r in records if test(r)]
        if rs:
            ag=aggregate(rs,None,'foris.crf','component.query_context');ag.pop('classes');ag['n']=len(rs);ag['gate']={a:{typ:int(sum(r['gate'][a][typ] for r in rs)) for typ in ['tp','fp']} for a in rs[0]['gate']};groups[group]=ag
    transition={}
    for rk in [False,True]:
        for qk in [False,True]:
            ws=[w for w in windows if w['take']['presence.reference']==rk and w['take']['presence.query_context']==qk]
            transition[f'reference={rk},query={qk}']=dict(n=len(ws),tp=sum(w['tp'] for w in ws),fp=sum(w['fp'] for w in ws),empty_gt_windows=sum(w['gt_pixels']==0 for w in ws),zero_tp_windows=sum(w['tp']==0 for w in ws))
    compdiag={}
    for name,test in [('no_guide_overlap',lambda c:c['guide_overlap']==0),('has_guide_overlap',lambda c:c['guide_overlap']>0),('window_rejected',lambda c:not c['window_accept']),('window_accepted',lambda c:c['window_accept'])]:
        cs=[c for c in components if test(c)];compdiag[name]=dict(n=len(cs),tp=sum(c['tp'] for c in cs),fp=sum(c['fp'] for c in cs),accepted_n=sum(c['accept'] for c in cs),accepted_tp=sum(c['tp'] for c in cs if c['accept']),accepted_fp=sum(c['fp'] for c in cs if c['accept']))
    ranks={family:{signal:{metric:dict(n=len(v:=[r[family][signal][metric] for r in records if r[family][signal][metric] is not None]),mean=float(np.mean(v)) if v else None) for metric in ['auc','ap','tpr_fpr05']} for signal in records[0][family]} for family in ['window_rank','component_rank']}
    summary=dict(n=len(records),comparisons=comparisons,ranks=ranks,window_transition=transition,component_conditions=compdiag,guide_recall_groups=groups,checks=dict(checks),seconds=time.monotonic()-started,limits='Saved source masks/decisions reconstructed, no raw descriptors recomputed. Component/window TP/FP are overlapping proposal occurrences. GT groups only diagnose existing decisions; they do not select inference.')
    for name,rs in [('episodes',records),('windows',windows),('components',components)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rs))
    write(out/'summary.json',summary);write(out/'sealed.json',dict(state='COMPLETE',script_sha256=sha(__file__),helper_sha256=sha(ROOT/'analyze_whole.py'),source_manifests={k:sha(p/'manifest.json') for k,p in roots.items()},source_ledgers={k:sha(p/'inference.jsonl') for k,p in roots.items()},receipts=receipts,checks=dict(checks),raw_reads=0,new_encoder_calls=0,processes=1,threads=1,seconds=time.monotonic()-started))
    print(json.dumps(dict(n=len(records),comparisons=comparisons,ranks=ranks)),flush=True)
if __name__=='__main__':main()
