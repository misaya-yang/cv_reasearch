#!/usr/bin/env python3
"""Index only existing sealed complete masks; no inference, rerendering or GT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows_of(path):
    d=read(path)
    return d if isinstance(d,list) else d.get('episodes',d.get('rows',[]))


def identity(r):
    return (int(r['c']),Path(r['support']).name,Path(r['query']).name)


def draw(r):
    return (int(r['fold']),int(r['e']),*identity(r))


def source(root):
    s=read(root/'sealed.json');rs=rows_of(root/'manifest.json')
    if not rs or not isinstance(s.get('predictions'),dict) or sha(root/'manifest.json')!=s.get('manifest_sha256'):
        return None
    if s.get('state') not in (None,'ALL_PREDICTIONS_SEALED'):
        return None
    if len({r['key'] for r in rs})!=len(rs) or not {r['key'] for r in rs}<=set(s['predictions']):
        return None
    if not all(isinstance(s['predictions'][r['key']],str) for r in rs):
        return None
    fp=root/'predictions'/(rs[0]['key']+'.npz')
    if not fp.exists():return None
    with np.load(fp,allow_pickle=False) as z:
        arms=[k for k in z.files if z[k].dtype==np.uint8 and z[k].shape==(131072,)]
    if not arms:return None
    return dict(root=root,seal=s,rows=rs,arms=arms,seal_sha256=sha(root/'sealed.json'),
                manifest_sha256=sha(root/'manifest.json'),gt_metadata={k:v for k,v in s.items() if 'gt' in k.lower() or 'label' in k.lower()})


def match(src, canonical, exact_draws):
    bykey={r['key']:r for r in src['rows']};byid={};bydraw={}
    for r in src['rows']:
        byid.setdefault(identity(r),[]).append(r);bydraw.setdefault(draw(r),[]).append(r)
    out=[]
    for r in canonical:
        candidate=bykey.get(r['key'])
        if candidate is not None and identity(candidate)==identity(r) and int(candidate['fold'])==int(r['fold']):
            if exact_draws and draw(candidate)!=draw(r):return None
        else:
            candidates=bydraw.get(draw(r),[]) if exact_draws else byid.get(identity(r),[])
            if len(candidates)>1 and not exact_draws:
                candidates=bydraw.get(draw(r),[])
            if len(candidates)!=1:return None
            candidate=candidates[0]
        if int(candidate['fold'])!=int(r['fold']):return None
        out.append(candidate)
    return out


def validate_file(job):
    record,arms=job
    if sha(record['prediction_path'])!=record['prediction_sha256']:
        raise ValueError('Changed sealed prediction '+record['prediction_path'])
    arrays={}
    with np.load(record['prediction_path'],allow_pickle=False) as z:
        for arm in arms:
            a=z[arm]
            if a.dtype!=np.uint8 or a.shape!=(131072,):
                raise ValueError('Not a complete packed1024 mask: '+arm)
            arrays[arm]=hashlib.sha256(a.tobytes()).hexdigest()
    return arrays


def catalog_source(src, matched, canonical):
    records=[dict(episode_key=r['key'],source_key=s['key'],prediction_path=str(src['root']/'predictions'/(s['key']+'.npz')),
                  prediction_sha256=src['seal']['predictions'][s['key']]) for r,s in zip(canonical,matched)]
    return dict(id=src['root'].name,source_run=str(src['root']),source_n=len(src['rows']),
                source_seal_sha256=src['seal_sha256'],source_manifest_sha256=src['manifest_sha256'],
                source_seal_state=src['seal'].get('state'),legacy_without_state=src['seal'].get('state') is None,
                gt_metadata=src['gt_metadata'],arms=src['arms'],records=records)


def verify_catalog(cat,workers):
    digests={a:hashlib.sha256() for a in cat['arms']}
    with ThreadPoolExecutor(workers) as pool:
        results=pool.map(validate_file,[(r,cat['arms']) for r in cat['records']])
        for r,masks in zip(cat['records'],results):
            r['packed_array_sha256']=masks
            for arm,digest in masks.items():digests[arm].update(bytes.fromhex(digest))
    cat['ordered_mask_sequence_sha256']={a:h.hexdigest() for a,h in digests.items()}
    cat['all_files_and_mask_shapes_verified']=True
    return cat


def build(a):
    outputs=a.workspace/'outputs';a.out.mkdir(parents=True,exist_ok=False)
    available=[];excluded=[];started=time.monotonic()
    for root in sorted(outputs.iterdir()):
        if not root.is_dir() or 'graft' in root.name:continue
        if not (root/'sealed.json').is_file() or not (root/'manifest.json').is_file():continue
        try:s=source(root)
        except (ValueError,KeyError,OSError):s=None
        if s is not None and len(s['rows'])>=241:available.append(s)
    byname={s['root'].name:s for s in available}
    for cohort,canonical_name,exact in [('PUBLIC4000','frozen_subtoken4000_v1',True),('DEV241','frozen_fine_raw_dev241_v1',False)]:
        canonical_src=byname[canonical_name];canonical=canonical_src['rows'];n=len(canonical)
        cats=[]
        for src in available:
            if exact and len(src['rows'])!=n:continue
            matched=match(src,canonical,exact)
            if matched is None:
                if not exact:excluded.append(dict(cohort=cohort,source=src['root'].name,reason='does_not_cover_all_canonical_identities_unambiguously'))
                continue
            cat=verify_catalog(catalog_source(src,matched,canonical),a.workers)
            cats.append(cat)
            print(json.dumps(dict(cohort=cohort,source=cat['id'],arms=len(cat['arms']),n=n,seconds=time.monotonic()-started)),flush=True)
        if exact:
            arms=None;records=[];providers={}
            for r in canonical:
                root=outputs/'claude_official'/('run'+str(r['public_batch']))
                name=str(root)
                if name not in providers:
                    s=source(root)
                    if s is None:raise ValueError('Unsealed official block')
                    providers[name]=s
                s=providers[name];original=next(x for x in s['rows'] if x['key']==r['source_key'])
                if draw(original)!=draw(r):raise ValueError('Official draw changed')
                if arms is None:arms=s['arms']
                if arms!=s['arms']:raise ValueError('Official arm mismatch')
                records.append(dict(episode_key=r['key'],source_key=original['key'],prediction_path=str(root/'predictions'/(original['key']+'.npz')),
                                    prediction_sha256=s['seal']['predictions'][original['key']]))
            cat=dict(id='claude_official_4000_existing_masks',source_run='seven existing sealed blocks; no new mask creation',source_n=n,
                     arms=arms,records=records,providers={k:dict(seal_sha256=v['seal_sha256'],manifest_sha256=v['manifest_sha256']) for k,v in providers.items()},
                     gt_metadata={'query_gt_in_inference':False})
            cats.append(verify_catalog(cat,a.workers))
        else:
            # Project existing official4000 masks onto the original241 identities; no rerendering.
            providers={};selected=[];arms=None
            for r in canonical:
                candidates=[]
                for b in range(7):
                    root=outputs/'claude_official'/('run'+str(b));name=str(root)
                    if name not in providers:providers[name]=source(root)
                    s=providers[name]
                    if s is not None:candidates.extend((s,x) for x in s['rows'] if identity(x)==identity(r) and int(x['fold'])==int(r['fold']))
                if len(candidates)>1:candidates=[(s,x) for s,x in candidates if draw(x)==draw(r)]
                if len(candidates)!=1:raise ValueError('Ambiguous official DEV projection')
                s,x=candidates[0]
                if arms is None:arms=s['arms']
                selected.append(dict(episode_key=r['key'],source_key=x['key'],prediction_path=str(s['root']/'predictions'/(x['key']+'.npz')),
                                     prediction_sha256=s['seal']['predictions'][x['key']]))
            cats.append(verify_catalog(dict(id='claude_official_projected_DEV241',source_run='existing official block masks matched by class/photos',source_n=4000,
               arms=arms,records=selected,providers={k:dict(seal_sha256=v['seal_sha256'],manifest_sha256=v['manifest_sha256']) for k,v in providers.items() if v},gt_metadata={'query_gt_in_inference':False}),a.workers))
        aliases={}
        for cat in cats:
            for arm,digest in cat['ordered_mask_sequence_sha256'].items():aliases.setdefault(digest,[]).append(dict(source=cat['id'],arm=arm))
        consumer_rows=[dict(r,masks={}) for r in canonical];consumer_arms=[]
        for cat in cats:
            gt=cat.get('gt_metadata',{})
            per_query=any(bool(v) for k,v in gt.items() if 'per_query_gt_routing' in k or k in ['query_gt_in_inference','query_labels_opened','query_truth_opened'])
            fitted=bool(gt.get('query_gt_used_for_DEV_fold_fitting',False))
            for arm in cat['arms']:
                arm_id=cat['id']+'::'+arm;digest=cat['ordered_mask_sequence_sha256'][arm]
                eligible=not per_query and not fitted and 'oracle' not in arm.lower()
                consumer_arms.append(dict(id=arm_id,producer_id='mask_method:'+digest,source_id=cat['id'],npz_key=arm,
                    source_run=cat['source_run'],same_information_eligible=eligible,
                    eligibility_reason='explicit query GT routing/oracle or DEV-fold fitted readout excluded' if not eligible else 'existing same-cohort complete mask; no extra resource recorded in source seal; protocol qualification retained',
                    exposure='reused DEV; not fresh confirmation',gt_metadata=gt))
                for r,record in zip(consumer_rows,cat['records']):
                    r['masks'][arm_id]=dict(path=record['prediction_path'],key=arm,sha256=record['prediction_sha256'],
                       packed_array_sha256=record['packed_array_sha256'][arm])
        result=dict(state='EXISTING_COMPLETE_MASK_LIBRARY_VERIFIED',cohort=cohort,n=n,resolution=1024,encoding='np.packbits uint8[131072]',
                    common_origin='unchanged model.raw_nn on DEV241; no rawNN4000 is invented',canonical_source=canonical_name,
                    canonical_manifest_sha256=canonical_src['manifest_sha256'],rows=consumer_rows,arms=consumer_arms,sources=cats,
                    source_arm_count=sum(len(c['arms']) for c in cats),distinct_mask_sequences=len(aliases),exact_sequence_alias_groups=list(aliases.values()),
                    deduplication='no episode/draw deduplication; aliases only identify exact ordered packed-mask sequences',
                    no_GT_opened=True,no_features_opened=True,no_new_masks=True,excluded_running_graft=True,
                    missing={'frozen_public4000.delete_p':'count-only report/episodes/receipt; no sealed full-mask file, not fabricated'},
                    seconds=time.monotonic()-started)
        p=a.out/('manifest_'+cohort.lower()+'.json');p.write_text(json.dumps(result,separators=(',',':'))+'\n')
        print(json.dumps(dict(cohort=cohort,manifest=str(p),sources=len(cats),source_arms=result['source_arm_count'],distinct_sequences=len(aliases),sha256=sha(p))),flush=True)
    for s in available:
        if sha(s['root']/'sealed.json')!=s['seal_sha256'] or sha(s['root']/'manifest.json')!=s['manifest_sha256']:
            raise ValueError('Source metadata changed during indexing')
    receipt=dict(state='MASK_LIBRARIES_COMPLETE',source_code_sha256=sha(__file__),seconds=time.monotonic()-started,excluded=excluded,
                 manifests={p.name:sha(p) for p in a.out.glob('manifest_*.json')},no_GPU=True,no_GT=True,no_features=True)
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if not 1<=args.workers<=6:p.error('one to six read-only CPU readers')
    build(args)
