#!/usr/bin/env python3
"""Gated C2 minimum-context fields from raw features; no query labels or encoder.

Requires the fixed C2 representation and all five reference self checks. This
does not deploy C2 or enable its full mechanism. score_c2_context_premise.py
opens query labels only after all 6000 development fields have been sealed.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from ics.foris import observe
from ics.official_data import file_hash,load_inputs
from ics.representations import source_reference_roles,unit_tokens
from cached_dino import CachedDINO,cache_host,dino_profile
from raw_feature_cache import RawFeatureCache,canonical_hash
from check_c2_self_reference import qualified_choice
from frozen_dino_plan import locked_run,prediction_file,read_rows,write
from frozen_unary_candidates import first_context_roles,minimal_context

DATASETS=('coco','pascal_part','paco_part')
BRANCHES=('Q/16','K/16','Q/24','K/24')


def self_qualified(report,choice):
    datasets=report.get('datasets',{})
    return (report.get('passed') is True and report.get('choice')==choice and
            set(datasets)==set(('coco','lvis','pascal_part','paco_part','suim')) and
            all(d.get('passed') is True and d.get('n')==5 and
                d.get('means',{}).get('real',0)>=.98 and
                d['means']['real']-d['means'].get('wrong',1)>=.05 for d in datasets.values()))


def infer(host,row,assets,bases,choice):
    image,mask,query=load_inputs(dict(row,query_mask_path='/query-label-forbidden'),assets)
    host.set_reference(image,mask);host.set_target(query)
    try:
        ref_mask=host._ref_masks.unsqueeze(1)
        inputs=torch.cat([host._ref_images,host._tgt_image[None]],dim=0)
        raw=host.encoder.raw(inputs,BRANCHES)
        maps=raw['O/24'].reshape(2,64,64,1024).permute(0,3,1,2).contiguous()
        pair=F.normalize(maps,dim=1)[None]
        processed=host._part1_positional_debias(pair,ref_mask,1)
        with observe(host) as got:
            host._part2_background_suppression(processed,ref_mask,1,64,64)
        foreground,background=source_reference_roles(got['reference_features'][0,0],got['reference_fg'][0])
        if not len(background):
            return {k:np.full(4096,.5,dtype=np.float32) for k in ('probability','direct_probability','wrong_probability')},dict(reference_valid=False)
        app=F.normalize(processed[0].flatten(2).transpose(1,2).half().float(),dim=-1)
        coverage=F.interpolate(ref_mask.float(),(64,64),mode='area')[0,0].numpy()
        roles=first_context_roles(app[0],foreground.numpy(),background.numpy(),coverage)
        qk={key:unit_tokens(raw[key],bases[key] if choice.endswith('/deb') else None) for key in BRANCHES}
        result=minimal_context(app[0],app[1],{k:v[0] for k,v in qk.items()},
                               {k:v[1] for k,v in qk.items()},roles,num_heads=16)
        fields={k:result[k].numpy() for k in ('probability','direct_probability','wrong_probability')}
        return fields,dict(reference_valid=True,roles=roles,query_anchor=result['query_anchor'],wrong_anchor=result['wrong_anchor'])
    finally:
        host._ref_images=host._ref_masks=host._tgt_image=host._orig_tgt_size=None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=REPO.parent/'cv_data')
    p.add_argument('--parent-run',type=Path,required=True)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--self-run',type=Path,required=True)
    p.add_argument('--basis',type=Path,required=True)
    p.add_argument('--raw-cache-profile',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    choice=qualified_choice(json.loads(a.decision.read_text()))
    if choice is None:
        print('C2 representation did not qualify; no cache/input work',flush=True);return
    self_report=json.loads((a.self_run/'report.json').read_text())
    if not self_qualified(self_report,choice):
        print('C2 reference self check did not qualify; no cache/input work',flush=True);return
    self_config=json.loads((a.self_run/'config.json').read_text())
    self_seal=json.loads((a.self_run/'sealed.json').read_text())
    if self_seal.get('n')!=25 or self_seal.get('config_sha256')!=file_hash(a.self_run/'config.json'):
        raise ValueError('Require the sealed25 reference self checks')
    if self_config['decision_sha256']!=file_hash(a.decision) or self_config['scorer_sha256']!=file_hash(REPO/'scripts/frozen_unary_candidates.py'):
        raise ValueError('Self check belongs to different choice or C2 formula')
    seal=json.loads((a.parent_run/'sealed.json').read_text())
    parent_config=json.loads((a.parent_run/'config.json').read_text())
    if seal['config_sha256']!=file_hash(a.parent_run/'config.json') or seal['inference_index_sha256']!=file_hash(a.parent_run/'inference.jsonl'):
        raise ValueError('Sealed parent configuration/index changed')
    for origin,digest in parent_config['source_sha256'].items():
        if file_hash(origin)!=digest:
            raise ValueError('C2 prerequisite must use the frozen parent implementation: '+origin)
    if seal['state']!='ALL_PREDICTIONS_SEALED' or parent_config['split_role']!='dev' or parent_config['prepared_protocol'].get('seed')!=1:
        raise ValueError('Require the sealed fresh seed1 parent')
    if seal['manifest_sha256']!=file_hash(a.parent_run/'manifest.json'):
        raise ValueError('Parent manifest changed')
    rows=read_rows(a.parent_run/'manifest.json')
    if len(rows)!=6000 or {r['dataset'] for r in rows}!=set(DATASETS) or any(sum(r['dataset']==name for r in rows)!=2000 for name in DATASETS):
        raise ValueError('Require all three complete2000 development cohorts')
    if self_config['manifest_sha256']!=parent_config['manifest_sha256']:
        raise ValueError('Self references and parent originate from different frozen manifests')
    profile=json.loads(a.raw_cache_profile.read_text())
    if a.raw_cache_profile.parent.name!=canonical_hash(profile) or profile!=dino_profile(a.assets,profile['producer_device']):
        raise ValueError('Raw cache identity differs from current encoder/preprocessing')
    if profile['weights_sha256']!=parent_config['weights_sha256']:
        raise ValueError('Parent and raw features use different weights')
    if self_config['weights_sha256']!=profile['weights_sha256'] or self_config['choice']!=choice:
        raise ValueError('Self check and raw features use different weights/choice')
    if self_config['native_basis_sha256']!=file_hash(a.assets/'native_assets/positional_basis.pt'):
        raise ValueError('Native basis differs from self check')
    if parent_config['basis_sha256']!=self_config['native_basis_sha256']:
        raise ValueError('Parent and self check use different native bases')
    if self_config['own_bases_sha256']!=file_hash(a.basis/'complete.json'):
        raise ValueError('Own branch bases differ from self check')
    bases={}
    if choice.endswith('/deb'):
        complete=json.loads((a.basis/'complete.json').read_text())
        for key in BRANCHES:
            entry=complete['bases'][key];path=a.basis/entry['path']
            if file_hash(path)!=entry['sha256']:raise ValueError('Own basis changed')
            bases[key]=torch.load(path,map_location='cpu',weights_only=True)['basis']
    code=[Path(__file__),REPO/'scripts/cached_dino.py',REPO/'scripts/raw_feature_cache.py',REPO/'scripts/frozen_unary_candidates.py']
    config=dict(role='C2 minimum context prerequisite, not segmentation gain',choice=choice,
        decision_sha256=file_hash(a.decision),self_report_sha256=file_hash(a.self_run/'report.json'),
        parent_run=str(a.parent_run.resolve()),parent_seal_sha256=file_hash(a.parent_run/'sealed.json'),
        parent_manifest_sha256=seal['manifest_sha256'],raw_cache_profile_sha256=file_hash(a.raw_cache_profile),
        code_sha256={str(f):file_hash(f) for f in code},query_labels_opened=False,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    torch.set_num_threads(2)
    with locked_run(a.out):
        if any(f.name!='run.lock' for f in a.out.iterdir()):raise FileExistsError('Use a fresh output; no mixed attempts')
        write(a.out/'config.json',config);write(a.out/'manifest.json',rows)
        (a.out/'source').mkdir();(a.out/'fields').mkdir()
        for file in code:shutil.copyfile(file,a.out/'source'/file.name)
        cache=RawFeatureCache(a.raw_cache_profile.parent.parent,profile)
        adapter=CachedDINO(cache,pair_branches=BRANCHES)
        host=cache_host(a.assets,adapter)
        with torch.inference_mode(),(a.out/'inference.jsonl').open('w',buffering=1) as log:
            for n,row in enumerate(rows,1):
                began=time.monotonic();fields,details=infer(host,row,a.assets,bases,choice)
                if adapter.encoder is not None or adapter.encoder_calls:raise ValueError('Cache replay attempted encoding')
                file=prediction_file(row);path=a.out/'fields'/file;np.savez_compressed(path,**fields)
                record=dict(episode_id=row['episode_id'],field_file=file,field_sha256=file_hash(path),
                            replay_encoder_calls=0,seconds=time.monotonic()-began,**details)
                log.write(json.dumps(record)+'\n');print(json.dumps(dict(n=n,total=len(rows),episode_id=row['episode_id'])),flush=True)
        write(a.out/'sealed.json',dict(state='ALL_CONTEXT_FIELDS_SEALED',n=len(rows),
              config_sha256=file_hash(a.out/'config.json'),manifest_sha256=file_hash(a.out/'manifest.json'),
              inference_index_sha256=file_hash(a.out/'inference.jsonl'),query_labels_opened=False))


if __name__=='__main__':main()
