#!/usr/bin/env python3
"""Small, reference-conditioned full-token candidate ranker pilot.

Frozen final DINO tokens, fixed candidate masks, fixed scalar/RQ/RQdonor arms.
Train/dev/test category splits and all-role photo purging come from --manifest.
Only dev chooses checkpoints; test labels are evaluated once after fitting.
This pilot is a fit/protocol diagnostic, not independent evidence of a method gain.
No encoder, download, GPU launch orchestration or candidate mask modification.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SEED = 2044
ARMS = ('scalar', 'rq', 'rqdonor')
MIN_TRAIN = 15
BYTE_CAP = 2_000_000_000


def photo_id(value):
    text = str(value)
    m = re.search(r'(\d+)(?:\.[A-Za-z0-9]+)?$', Path(text).name)
    return str(int(m.group(1))) if m else text


def safe_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as ex:
        if 'mmap' not in str(ex):
            raise
        return torch.load(path, map_location='cpu', weights_only=False)


def resolve_data_path(value, index_path, allowed_roots):
    p = Path(value)
    resolved = p.resolve() if p.is_absolute() or p.exists() else (index_path.parent / p).resolve()
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError('data file outside explicitly declared preparation directories')
    return resolved


def merge_controls_overlay(base, overlay, base_path):
    """Tokens immutable; source core masks/counts/provenance remain exact prefix."""
    immutable = {'reference_tokens','query_tokens','reference_mask','donor_tokens','donor_masks',
                 'token_context','donor_token_context'}
    if immutable & set(overlay):
        raise ValueError('controls overlay attempts to overwrite token/annotation evidence')
    for key in ('e','c','split'):
        if overlay.get(key) != base.get(key):
            raise ValueError('overlay episode/class/split mismatch')
    if list(overlay.get('photo_ids', [])) != list(base['photo_ids']):
        raise ValueError('overlay all-role ordered photo IDs changed')
    if Path(overlay['source_episode_path']).resolve() != base_path.resolve():
        raise ValueError('overlay source episode path mismatch')
    core = len(base['candidate_masks'])
    if int(overlay['source_candidate_count']) != core or len(overlay['candidate_masks']) < core:
        raise ValueError('overlay core candidate count mismatch')
    for key in ('candidate_masks','candidate_iu','candidate_iou','candidate_original_iu','candidate_original_iou'):
        if key in base:
            if key not in overlay or not torch.equal(torch.as_tensor(base[key]),torch.as_tensor(overlay[key][:core])):
                raise ValueError('overlay changed original candidate prefix: '+key)
    if base['candidate_provenance'] != overlay['candidate_provenance'][:core]:
        raise ValueError('overlay changed original candidate provenance prefix')
    if 'candidate_native_mask_bits' in base and not np.array_equal(base['candidate_native_mask_bits'],overlay['candidate_native_mask_bits'][:core]):
        raise ValueError('overlay changed original native mask prefix')
    audit = overlay.get('audit', {})
    if audit.get('core_native_iu_exact') is not True:
        raise ValueError('overlay lacks passing native-prefix audit')
    if 'candidate_original_iu' in base and audit.get('core_original_iu_exact') is not True:
        raise ValueError('overlay lacks passing original-resolution prefix audit')
    merged = dict(base);merged.update(overlay)
    merged['_overlay_prefix_verified'] = True
    return merged


def load_records(index_path, max_bytes=BYTE_CAP):
    """Preparation index with optional small overlays; no feature cache copies."""
    if index_path.suffix.lower() == '.json':
        document = json.loads(index_path.read_text())
        entries = document.get('episodes', document.get('records', document.get('episode_files', document.get('files'))))
        if not isinstance(entries,list):raise ValueError('index needs an episode list')
        roots = [index_path.parent.resolve()]
        if document.get('source_report'):
            source = Path(document['source_report'])
            source = source.resolve() if source.is_absolute() or source.exists() else (index_path.parent/source).resolve()
            if not source.exists():raise ValueError('declared overlay source report is missing')
            roots.append(source.parent)
        paths=[];overlays=[]
        for entry in entries:
            value = entry if isinstance(entry,str) else next((entry[k] for k in ('episode_pt','path','file','pt') if k in entry),None)
            if value is None:raise ValueError('index entry has no episode path')
            paths.append(resolve_data_path(value,index_path,roots))
            overlay_value = entry.get('overlay_path') if isinstance(entry,dict) else None
            overlays.append(resolve_data_path(overlay_value,index_path,[index_path.parent.resolve()]) if overlay_value else None)
        assets=set(paths)|{p for p in overlays if p is not None}
        if sum(p.stat().st_size for p in assets)>max_bytes:raise ValueError('pilot inputs exceed configured --max-bytes')
        records=[]
        for path,overlay_path in zip(paths,overlays):
            row=safe_load(path)
            if overlay_path is not None:row=merge_controls_overlay(row,safe_load(overlay_path),path)
            records.append(row)
        return records,document,[{'path':str(p),'bytes':p.stat().st_size} for p in sorted(assets)]
    if index_path.stat().st_size>max_bytes:raise ValueError('pilot input exceeds configured --max-bytes')
    document=safe_load(index_path);records=document['records'] if 'records' in document else [document]
    return records,{k:v for k,v in document.items() if k!='records' and not isinstance(v,torch.Tensor)},[dict(path=str(index_path),bytes=index_path.stat().st_size)]


def append_counterfactual_training(base_rows, report_path, manifest_path, max_bytes, assets, strict_tokens=True):
    """Frozen training-only CF tasks; shared tensors loaded once, no GT subset search."""
    document=json.loads(report_path.read_text())
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if document.get('state')!='COMPLETED' or document.get('source_manifest_sha256')!=manifest_sha:
        raise ValueError('CF availability state/new-manifest identity mismatch')
    if not document.get('summary',{}).get('continuation_eligible'):
        raise ValueError('Frozen CF availability gate did not pass')
    frozen=document.get('frozen_valid_training_records',[])
    if len(frozen)!=8 or len({int(r['dataset_index']) for r in frozen})!=8:
        raise ValueError('This matched card requires exactly the frozen eight unique tasks')
    expected=[r for r in document['records'] if r.get('usable_training_pair')]
    if frozen!=expected:raise ValueError('Frozen CF task subset/order changed')
    base_groups,_=purge_split(base_rows)
    train_photos=frozenset().union(*(r['_photos'] for r in base_groups['train']))
    held_photos=frozenset().union(*(r['_photos'] for r in base_rows if r['_split']!='train'))
    train_classes={int(r['c']) for r in base_groups['train']}
    roots=[report_path.parent.resolve()]
    if document.get('reused_first_report'):
        reused=Path(document['reused_first_report']).resolve()
        roots.append(reused if reused.is_dir() else reused.parent)
    items=[];physical={Path(r['path']).resolve():r for r in assets}
    for entry in frozen:
        if not entry.get('counterfactual') or not entry.get('usable_training_pair') or entry.get('split')!='train':
            raise ValueError('CF task eligibility metadata changed')
        shared_path=resolve_data_path(entry['path'],report_path,roots)
        task_path=resolve_data_path(entry['overlay_path'],report_path,roots)
        items.append((entry,shared_path,task_path))
        for path in (shared_path,task_path):physical.setdefault(path,dict(path=str(path),bytes=path.stat().st_size))
    if sum(p.stat().st_size for p in physical)>max_bytes:
        raise ValueError('Unique base/shared-CF/overlay physical inputs exceed --max-bytes')
    pair_reports={int(r['pair_index']):r for r in document['pairs']}
    cached={};added=[];pair_tasks={};base_ids={int(r['e']) for r in base_rows}
    immutable={'reference_tokens','query_tokens','donor_tokens','donor_masks','donor_masks_components',
        'candidate_masks','candidate_provenance','candidate_native_mask_bits','direct_index','photo_ids','token_context','donor_token_context'}
    for entry,path,task_path in items:
        pair_index=int(entry['counterfactual_pair_index']);pair_meta=pair_reports[pair_index]
        audit=pair_meta['audit']
        flags=('same_rgb_patch_features_exact','candidate_union_shared','p1_union_shared','provenance_shared','donors_shared','support_masks_different')
        if not all(audit.get(key) is True for key in flags) or audit.get('direct_index_both')!=0:
            raise ValueError('CF same-union/different-gold interface audit failed')
        if path not in cached:
            shared=safe_load(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
            cached[path]=(shared,digest)
        shared,digest=cached[path];task=safe_load(task_path)
        if digest!=entry['source_shared_sha256'] or task.get('source_shared_sha256')!=digest:
            raise ValueError('CF shared tensor SHA mismatch')
        if Path(task['source_shared_path']).resolve()!=path or immutable&set(task):
            raise ValueError('CF task overwrote physical shared evidence')
        if task.get('split')!='train' or int(task['c'])!=int(entry['c']) or int(task['c']) not in train_classes:
            raise ValueError('CF labels outside existing training classes')
        photos=frozenset(map(photo_id,shared['photo_ids']))
        if not photos<=train_photos or photos&held_photos:
            raise ValueError('CF role photographs escape base TRAIN or overlap heldout')
        fixed=document['fixed_pairs'][pair_index]
        if shared['photo_ids']!=[fixed['support'],fixed['query']]+fixed['donors']:
            raise ValueError('CF physical role/provenance pair changed')
        if set(pair_meta['classes'])!=set(fixed['classes']) or len(shared['candidate_masks'])!=8 or shared['direct_index']!=0:
            raise ValueError('CF fixed K8 union/direct/provenance identity changed')
        row=dict(shared);row.update(task)
        row['e']=10000+int(entry['dataset_index'])
        if row['e'] in base_ids or row['e'] in {int(r['e']) for r in added}:
            raise ValueError('CF global episode ID collision')
        row['_split']='train';row['_photos']=photos;row['_counterfactual']=True
        validate_record(row,strict_tokens=strict_tokens)
        pair_tasks.setdefault(path,[]).append(row);added.append(row)
    for path,tasks in pair_tasks.items():
        if len(tasks)!=2 or tasks[0]['c']==tasks[1]['c'] or torch.equal(tasks[0]['reference_mask'],tasks[1]['reference_mask']):
            raise ValueError('CF pair lacks two different legal gold conditions')
        for key in ('reference_tokens','query_tokens','donor_tokens','donor_masks','candidate_masks'):
            if tasks[0][key].data_ptr()!=tasks[1][key].data_ptr():
                raise ValueError('CF shared tensors duplicated or task evidence changed')
    return base_rows+added,dict(tasks=len(added),pairs=len(cached),new_episode_ids=[int(r['e']) for r in added],
        shared_tensor_files=len(cached),physical_bytes=sum(p.stat().st_size for p in physical),
        source_manifest_sha256=manifest_sha,report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        same_shared_tensors=True,only_frozen_subset=True,all_role_photos_in_base_train_only=True,
        noninput_metadata=['c','e','pair_id','task_tag','candidate_provenance.hypothesis_id']),list(physical.values())


def apply_manifest(records, path):
    document = json.loads(path.read_text())
    entries = document if isinstance(document, list) else document['records']
    by_e = {int(r['e']): r for r in entries}
    if len(by_e) != len(entries):
        raise ValueError('duplicate manifest episode IDs')
    for row in records:
        entry = by_e[int(row['e'])]
        if int(entry['c']) != int(row['c']):
            raise ValueError('manifest class mismatch')
        split = entry['split']
        if split not in ('train', 'dev', 'test'):
            raise ValueError('manifest.split must be train/dev/test')
        ids = entry.get('image_ids', entry.get('photo_ids'))
        if ids is None:
            ids = [entry['support_id'], entry['query_id']] + entry['pool_ids']
        full_ids = frozenset(map(photo_id, ids))
        stored = frozenset(map(photo_id, row['photo_ids']))
        if stored != full_ids:
            raise ValueError('all-role photo IDs disagree with immutable manifest')
        if 'split' in row and row['split'] != split:
            raise ValueError('prepared split disagrees with manifest')
        row['_split'] = split
        row['_photos'] = full_ids
    return document


def purge_split(records):
    groups = {split: [r for r in records if r['_split'] == split] for split in ('train', 'dev', 'test')}
    category_sets = {split: {int(r['c']) for r in group} for split, group in groups.items()}
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        if category_sets[a] & category_sets[b]:
            raise ValueError(f'categories overlap between {a}/{b}')
    test_ids = frozenset().union(*(r['_photos'] for r in groups['test']))
    dev_all_ids = frozenset().union(*(r['_photos'] for r in groups['dev']))
    kept = dict(test=groups['test'],
        dev=[r for r in groups['dev'] if not (r['_photos'] & test_ids)],
        train=[r for r in groups['train'] if not (r['_photos'] & (test_ids | dev_all_ids))])
    retained_ids = {s: frozenset().union(*(r['_photos'] for r in g)) for s, g in kept.items()}
    assert all(not (retained_ids[a] & retained_ids[b]) for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')))
    return kept, dict(before={k: len(v) for k, v in groups.items()}, after={k: len(v) for k, v in kept.items()},
        categories={k: sorted(v) for k, v in category_sets.items()}, all_role_photo_intersections_zero=True,
        retained_episode_ids={k: [int(r['e']) for r in v] for k, v in kept.items()})


def fallback_scalar_features(masks, direct_index):
    masks = masks.bool().flatten(1)
    direct = masks[direct_index]
    area = masks.float().mean(1)
    intersection = (masks & direct).sum(1).float()
    union = (masks | direct).sum(1).float().clamp_min(1)
    flag = torch.zeros(len(masks)); flag[direct_index] = 1
    return torch.stack([area, intersection / union, (masks != direct).float().mean(1), flag], dim=1)


def validate_record(row, strict_tokens=True):
    if 'candidate_state' in row and row['candidate_state'] != 'COMPLETE':
        raise ValueError('candidate preparation is incomplete')
    if 'donor_tokens' in row and row.get('donor_token_state', 'COMPLETE') != 'COMPLETE':
        raise ValueError('donor token preparation is incomplete')
    masks = row['candidate_masks'].bool()
    if masks.ndim != 3 or masks.shape[1] != masks.shape[2]:
        raise ValueError('candidate masks need [K,h,h]')
    k, h, _ = masks.shape
    if k < 1 or int(row.get('direct_index', 0)) not in range(k):
        raise ValueError('missing valid direct candidate')
    row['direct_index'] = int(row.get('direct_index', 0))
    row['candidate_masks'] = masks
    expected = (h*h, 1024) if strict_tokens else (h*h, row['reference_tokens'].shape[-1])
    if strict_tokens and h != 64:
        raise ValueError('real pilot requires 64x64 masks')
    for key in ('reference_tokens', 'query_tokens'):
        token = row[key]
        if tuple(token.shape) != expected:
            raise ValueError(f'{key} must contain full spatial final-DINO tokens {expected}')
        if strict_tokens and token.dtype != torch.float16:
            raise ValueError('prepared full tokens must be FP16')
    if tuple(row['reference_mask'].shape) != (h, h):
        raise ValueError('reference annotation has incorrect shape')
    row['reference_mask'] = row['reference_mask'].bool()
    labels = torch.as_tensor(row['candidate_iou']).float()
    if labels.shape != (k,) or not torch.isfinite(labels).all() or (labels < 0).any() or (labels > 1).any():
        raise ValueError('candidate IoU labels must be [K] in [0,1]')
    row['candidate_iou'] = labels
    metric = row.get('candidate_iou_metric', 'native_iou' if 'candidate_iu' in row else None)
    if metric not in ('native_iou', 'patch_iou'):
        raise ValueError('declare candidate_iou_metric or supply native candidate_iu')
    row['candidate_iou_metric'] = metric
    if 'candidate_iu' in row:
        iu = torch.as_tensor(row['candidate_iu']).double()
        if iu.shape != (k,2) or not torch.isfinite(iu).all() or (iu < 0).any() or (iu[:,0] > iu[:,1]).any():
            raise ValueError('invalid candidate native I/U')
        if metric == 'native_iou' and not torch.allclose(labels.double(), iu[:,0]/iu[:,1].clamp_min(1), atol=1e-6):
            raise ValueError('native labels disagree with candidate I/U')
        row['candidate_iu'] = iu
    if len(row.get('candidate_provenance', [])) != k:
        raise ValueError('all candidates require frozen provenance')
    if 'scalar_features' in row:
        features = torch.as_tensor(row['scalar_features']).float()
        if features.ndim != 2 or features.shape[0] != k or not torch.isfinite(features).all():
            raise ValueError('invalid scalar feature table')
        row['_scalar_source'] = 'prepared_label_free_scores'
    else:
        features = fallback_scalar_features(masks, row['direct_index'])
        row['_scalar_source'] = 'mask_geometry_only_fallback_not_a_strong_scalar_control'
    row['scalar_features'] = features
    if 'donor_tokens' in row and len(row['donor_tokens']):
        token = row['donor_tokens']
        if token.ndim != 3 or tuple(token.shape[1:]) != expected:
            raise ValueError('donor tokens need [D,P,C] in the same full token space')
        for source in row['candidate_provenance']:
            if source.get('kind')=='donor_two_hop' and not 0<=int(source.get('donor_slot',-1))<len(token):
                raise ValueError('candidate source donor_slot missing/out of range')
        if 'donor_masks' in row and tuple(row['donor_masks'].shape) != (len(token),h,h):
            raise ValueError('donor P1 masks have incorrect shape')
    return row


class ScalarRanker(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 64), nn.GELU(), nn.Linear(64,64), nn.GELU(), nn.Linear(64,1))
    def forward(self, row):
        return self.net(row['scalar_features']).flatten()


class ConditionalRanker(nn.Module):
    """RQ and RQdonor instantiate IDENTICAL modules and parameter counts."""
    def __init__(self, token_dim, feature_dim, use_donors, width=64, queries=4):
        super().__init__()
        self.use_donors = use_donors
        self.project = nn.Linear(token_dim, width)
        self.position = nn.Sequential(nn.Linear(2,width), nn.GELU(), nn.Linear(width,width))
        self.role = nn.Embedding(3,width)
        self.annotation = nn.Embedding(3,width)  # background/foreground/unknown donor P1
        self.mask_state = nn.Embedding(4,width)  # inside/outside x disagreement with direct
        self.queries = nn.Parameter(torch.randn(queries,width) * .02)
        self.reference_attention = nn.MultiheadAttention(width,4,batch_first=True)
        self.query_attention = nn.MultiheadAttention(width,4,batch_first=True)
        self.donor_attention = nn.MultiheadAttention(width,4,batch_first=True)
        self.null_donor = nn.Parameter(torch.zeros(width))
        self.head = nn.Sequential(nn.Linear(3*width+feature_dim,128), nn.GELU(), nn.Linear(128,1))
    def spatial(self, h, device):
        axes = torch.linspace(-1,1,h,device=device)
        y,x = torch.meshgrid(axes,axes,indexing='ij')
        return self.position(torch.stack([x,y],dim=-1).reshape(-1,2))
    def forward(self, row):
        masks = row['candidate_masks'].flatten(1).long()
        k = len(masks); h = row['candidate_masks'].shape[-1]
        pos = self.spatial(h, masks.device)
        ref_tokens = F.normalize(row['reference_tokens'].float(),dim=-1)
        qry_tokens = F.normalize(row['query_tokens'].float(),dim=-1)
        ref = self.project(ref_tokens) + pos + self.role.weight[0]
        ref = ref + self.annotation(row['reference_mask'].flatten().long())
        rq,_ = self.reference_attention(self.queries[None],ref[None],ref[None],need_weights=False)
        conditioned = self.queries[None] + rq
        query_base = self.project(qry_tokens) + pos + self.role.weight[1]
        disagreement = masks != masks[row['direct_index']]
        states = masks + 2*disagreement.long()
        query = query_base[None] + self.mask_state(states)
        qq,_ = self.query_attention(conditioned.expand(k,-1,-1),query,query,need_weights=False)
        reference_summary = rq.mean(1).expand(k,-1)
        donor_summary = self.null_donor[None].expand(k,-1)
        if self.use_donors:
            tokens = row['donor_tokens']
            donor = self.project(F.normalize(tokens.float(),dim=-1)) + pos[None] + self.role.weight[2]
            if 'donor_masks' in row:
                donor = donor + self.annotation(row['donor_masks'].flatten(1).long())
            else:
                donor = donor + self.annotation.weight[2]
            # Each donor has its own attended summary. Provenance selects the
            # actual source donor for two-hop hypotheses; pooled/direct use bank mean.
            dq,_ = self.donor_attention(conditioned.expand(len(donor),-1,-1),donor,donor,need_weights=False)
            by_donor = dq.mean(1)
            aggregate = by_donor.mean(0)[None].expand(k,-1)
            slots = row['candidate_donor_slots']
            donor_summary = torch.where((slots>=0)[:,None],by_donor[slots.clamp_min(0)],aggregate)
        features = torch.cat([reference_summary,qq.mean(1),donor_summary,row['scalar_features']],dim=1)
        return self.head(features).flatten()


def ranking_loss(scores, labels):
    gaps = labels[:,None]-labels[None,:]
    positive = gaps > 0
    if positive.any():
        difference = scores[:,None]-scores[None,:]
        pair = (F.softplus(-difference[positive])*gaps[positive]).sum()/gaps[positive].sum()
    else:
        pair = scores.sum()*0
    regression = F.mse_loss(scores.sigmoid(),labels)
    return pair + .1*regression, pair, regression


def model_input(row, device, mean, std, include_donors):
    # Labels, class IDs, episode IDs and photos NEVER enter the scorer.
    keys = ['reference_tokens','query_tokens','reference_mask','candidate_masks']
    if include_donors:
        keys += ['donor_tokens'] + (['donor_masks'] if 'donor_masks' in row else [])
    result = {key: row[key].to(device,non_blocking=True) for key in keys}
    result['direct_index'] = row['direct_index']
    result['candidate_donor_slots'] = torch.tensor([int(p['donor_slot']) if p.get('kind')=='donor_two_hop' else -1
        for p in row['candidate_provenance']],device=device,dtype=torch.long)
    result['scalar_features'] = (row['scalar_features'].to(device)-mean)/std
    return result


def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


@torch.no_grad()
def evaluate(model, rows, device, mean, std, arm):
    model.eval(); records=[]
    for row in rows:
        with autocast(device):
            score = model(model_input(row,device,mean,std,arm=='rqdonor')).float()
            loss,_,_ = ranking_loss(score,row['candidate_iou'].to(device))
        chosen = int(score.argmax()); direct=row['direct_index']
        record = dict(e=int(row['e']),c=int(row['c']),chosen=chosen,direct_index=direct,
            scores=score.cpu().tolist(),chosen_iou=float(row['candidate_iou'][chosen]),
            direct_iou=float(row['candidate_iou'][direct]),oracle_iou=float(row['candidate_iou'].max()),loss=float(loss))
        if 'candidate_iu' in row:
            record['chosen_iu']=row['candidate_iu'][chosen].tolist()
            record['direct_iu']=row['candidate_iu'][direct].tolist()
            record['oracle_iu']=row['candidate_iu'][int(row['candidate_iou'].argmax())].tolist()
        records.append(record)
    summary=dict(episodes=len(records),mean_candidate_iou=float(np.mean([r['chosen_iou'] for r in records])),
        direct_mean_iou=float(np.mean([r['direct_iou'] for r in records])),
        candidate_oracle_mean_iou=float(np.mean([r['oracle_iou'] for r in records])),
        loss=float(np.mean([r['loss'] for r in records])))
    if all('chosen_iu' in r for r in records):
        def miou(key):
            total={}
            for r in records:
                v=total.setdefault(r['c'],[0.,0.]); v[0]+=r[key][0];v[1]+=r[key][1]
            return 100*float(np.mean([i/max(u,1) for i,u in total.values()]))
        summary.update(native_class_miou=miou('chosen_iu'),direct_native_class_miou=miou('direct_iu'),
            candidate_episode_oracle_native_class_miou=miou('oracle_iu'))
    return summary,records


def cpu_smoke():
    torch.set_num_threads(1); torch.manual_seed(SEED)
    rows=[]
    for e in range(10):
        masks=torch.rand(7,4,4)>.5
        row=dict(e=e,c=e//2,photo_ids=[f'photo_{e*3}',f'photo_{e*3+1}',f'photo_{e*3+2}'],
            reference_tokens=torch.randn(16,32).half(),query_tokens=torch.randn(16,32).half(),
            reference_mask=torch.rand(4,4)>.5,candidate_masks=masks,candidate_iou=torch.tensor([.2,.8,.4,.6,.3,.9,.1]),
            candidate_iou_metric='patch_iou',candidate_provenance=[{'kind':'direct_1shot'}]+[{'kind':'donor_two_hop','donor_slot':j} for j in range(3)]+[{'kind':name} for name in ('naive_native_multi','naive_cached_native_p1','current_ag4_cached_native_p1')],
            donor_tokens=torch.randn(3,16,32).half(),donor_masks=torch.rand(3,4,4)>.5)
        rows.append(validate_record(row,strict_tokens=False))
    dim=rows[0]['scalar_features'].shape[1];mean=torch.zeros(dim);std=torch.ones(dim)
    counts={};losses={}
    for arm in ARMS:
        torch.manual_seed(SEED)
        model=ScalarRanker(dim) if arm=='scalar' else ConditionalRanker(32,dim,arm=='rqdonor')
        counts[arm]=sum(p.numel() for p in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
        for row in rows:
            opt.zero_grad();scores=model(model_input(row,torch.device('cpu'),mean,std,arm=='rqdonor'))
            loss,_,_=ranking_loss(scores,row['candidate_iou']);assert torch.isfinite(loss)
            loss.backward();opt.step()
        before=model(model_input(rows[0],torch.device('cpu'),mean,std,arm=='rqdonor')).detach()
        altered=dict(rows[0]);altered['candidate_iou']=1-altered['candidate_iou'];altered['c']=999;altered['e']=999
        after=model(model_input(altered,torch.device('cpu'),mean,std,arm=='rqdonor')).detach()
        assert torch.equal(before,after),'labels or IDs entered model'
        if arm=='rqdonor':
            provenance=[dict(p) for p in rows[0]['candidate_provenance']]
            provenance[1]['donor_slot']=2
            swapped=dict(rows[0],candidate_provenance=provenance)
            changed=model(model_input(swapped,torch.device('cpu'),mean,std,True)).detach()
            assert not torch.equal(before[1],changed[1]),'associated donor provenance ignored'
            assert torch.equal(before[0],changed[0]),'direct aggregate should not depend on another candidate slot'
        summary,_=evaluate(model,rows,torch.device('cpu'),mean,std,arm);losses[arm]=summary['loss']
    assert counts['rq']==counts['rqdonor']
    # Strict all-role purge includes donor photographs, not only support/query.
    splits=[]
    for i,split in enumerate(('train','train','dev','test')):
        splits.append(dict(e=i,c=i,_split=split,_photos=frozenset({str(i), 'shared_donor'} if i in (0,3) else {str(i)})))
    kept,audit=purge_split(splits);assert [r['e'] for r in kept['train']]==[1]
    print(json.dumps(dict(cpu_smoke='PASSED',episodes=10,three_arms_backprop=True,
        equal_RQ_RQD_parameters=counts['rq']==counts['rqdonor'],parameter_counts=counts,
        mixed_candidates=7,associated_donor_slot_changes_only_source_candidate=True,
        GT_and_ID_input_invariance=True,donor_photo_purge=True,final_losses=losses)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--file',type=Path)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--counterfactual-report',type=Path)
    ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--max-bytes',type=int,default=BYTE_CAP)
    ap.add_argument('--arms',default=','.join(ARMS))
    ap.add_argument('--cpu-smoke',action='store_true')
    a=ap.parse_args()
    if a.cpu_smoke:
        cpu_smoke();return
    if not all((a.file,a.manifest,a.out_dir)) or not 1<=a.epochs<=50 or not 1<=a.max_bytes<=6_000_000_000:
        ap.error('--file --manifest --out-dir required, epochs 1..50, max-bytes <=6000000000')
    requested=a.arms.split(',')
    if len(set(requested))!=len(requested) or any(x not in ARMS for x in requested):
        ap.error('arms must be unique scalar,rq,rqdonor choices')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device)
    if device.type=='cuda':
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.25')))
    a.out_dir.mkdir(parents=True,exist_ok=True)
    report=dict(state='RUNNING',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        seed=SEED,method_claim='Small extra-supervised fit/protocol pilot only; not a blind generalization or novelty result.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.file,a.manifest,Path(__file__))},
        architecture='4 learned attention queries, full normalized frozen final-DINO tokens, learned 2D positions, reference FG/BG, candidate inside/outside/disagreement, width64',
        train_rule='AdamW lr3e-4 wd1e-3, accumulation4 episodes; gap-weighted pair logistic loss + .1 sigmoid-IoU MSE; <=50 epochs',
        checkpoint_rule='Fixed final epoch for tiny dev; dev loss selection only if >=20 dev episodes and >=2 dev classes; never test selection',
        arms={})
    started=time.monotonic();log=a.out_dir/'epochs.jsonl'
    def write():
        report['elapsed_s']=time.monotonic()-started
        tmp=a.out_dir/'status.tmp';tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(a.out_dir/'status.json')
    def emit(event):
        line=json.dumps(event,allow_nan=False);print(line,flush=True)
        with log.open('a') as stream:stream.write(line+'\n')
    if a.counterfactual_report:
        report['source_sha256'][str(a.counterfactual_report)]=hashlib.sha256(a.counterfactual_report.read_bytes()).hexdigest()
    write()
    try:
        records,provenance,assets=load_records(a.file,a.max_bytes)
        if not records or len({int(r['e']) for r in records})!=len(records):raise ValueError('empty/duplicate episodes')
        apply_manifest(records,a.manifest)
        for r in records:validate_record(r)
        if a.counterfactual_report:
            records,cf_audit,assets=append_counterfactual_training(records,a.counterfactual_report,a.manifest,a.max_bytes,assets)
            report['counterfactual_append']=cf_audit
        if len({r['candidate_iou_metric'] for r in records})!=1:raise ValueError('mixed target metrics')
        if len({r['scalar_features'].shape[1] for r in records})!=1:raise ValueError('inconsistent scalar feature dimensions')
        groups,audit=purge_split(records);report['split_audit']=audit;report['data_assets']=assets
        report['data_metric']=records[0]['candidate_iou_metric']
        report['native_resolution']='Prepared candidate_iu native host scoring (1024-square when specified by preparation); original-photo-size IU is not substituted'
        report['scalar_feature_sources']=sorted({r['_scalar_source'] for r in records})
        report['candidate_masks_frozen_identical_across_arms']=True
        report['overlay_prefix_audits_verified']=sum(bool(r.get('_overlay_prefix_verified')) for r in records)
        report['donor_conditioning']='Per-donor attended summaries, candidate_provenance donor_slot for two-hop; aggregate for direct/multi-source; no class IDs'
        report['scalar_fallback_active']=any(r['_scalar_source']!='prepared_label_free_scores' for r in records)
        emit(dict(event='split_audit',**audit))
        if len(groups['train'])<MIN_TRAIN or not groups['dev'] or not groups['test']:
            report.update(state='NOT_EVALUABLE',reason='Need >=15 train plus nonempty dev/test after strict photo purge; redesign sampling, never relax isolation')
            write();return
        train,dev,test=groups['train'],groups['dev'],groups['test']
        dev_selection = len(dev)>=20 and len({int(r['c']) for r in dev})>=2
        report['dev_checkpoint_selection_enabled']=dev_selection
        report['checkpoint_epoch_if_tiny_dev']=a.epochs if not dev_selection else None
        allfeatures=torch.cat([r['scalar_features'] for r in train])
        mean=allfeatures.mean(0).to(device);std=allfeatures.std(0,unbiased=False).clamp_min(1e-6).to(device)
        completed={}
        for arm in requested:
            ar=dict(state='RUNNING',best_dev_epoch=None);report['arms'][arm]=ar
            if arm=='rqdonor' and not all('donor_tokens' in r and len(r['donor_tokens'])>0 for r in train+dev+test):
                ar.update(state='NOT_EVALUABLE',reason='Actual donor full-token evidence absent in at least one retained record; null-input repeat is not an RQdonor comparison')
                write();emit(dict(event='arm_skip',arm=arm,**ar));continue
            random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
            if device.type=='cuda':torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats()
            dim=allfeatures.shape[1]
            model=(ScalarRanker(dim) if arm=='scalar' else ConditionalRanker(1024,dim,arm=='rqdonor')).to(device)
            ar['parameters']=sum(p.numel() for p in model.parameters())
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
            best=float('inf');best_path=a.out_dir/(arm+'_best.pt');order_rng=np.random.default_rng(SEED)
            for epoch in range(1,a.epochs+1):
                tick=time.monotonic();model.train();order=order_rng.permutation(len(train));losses=[]
                for begin in range(0,len(order),4):
                    batch=order[begin:begin+4];opt.zero_grad(set_to_none=True)
                    for i in batch:
                        r=train[int(i)]
                        with autocast(device):
                            score=model(model_input(r,device,mean,std,arm=='rqdonor')).float()
                            loss,_,_=ranking_loss(score,r['candidate_iou'].to(device))
                        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                        (loss/len(batch)).backward();losses.append(float(loss.detach()))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                dev_summary,_=evaluate(model,dev,device,mean,std,arm)
                if (dev_selection and dev_summary['loss']<best) or (not dev_selection and epoch==a.epochs):
                    best=dev_summary['loss'];ar['best_dev_epoch']=epoch
                    torch.save(dict(model=model.state_dict(),scalar_mean=mean.cpu(),scalar_std=std.cpu(),
                        arm=arm,epoch=epoch,dev=dev_summary,source_sha256=report['source_sha256']),best_path)
                ar['last_epoch']=epoch;ar['best_dev_loss']=best if math.isfinite(best) else None
                event=dict(event='epoch',arm=arm,epoch=epoch,train_loss=float(np.mean(losses)),
                    dev=dev_summary,best_dev_epoch=ar['best_dev_epoch'],epoch_concurrent_s=time.monotonic()-tick)
                emit(event);write()
            checkpoint=safe_load(best_path);model.load_state_dict(checkpoint['model'])
            # Only now, after dev freezes the checkpoint, evaluate this arm on test.
            test_summary,test_records=evaluate(model,test,device,mean,std,arm)
            ar.update(state='COMPLETED',test=test_summary,test_records=test_records,checkpoint=str(best_path),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
            completed[arm]=ar['parameters'];emit(dict(event='arm_done',arm=arm,test=test_summary));write()
            del model,opt,checkpoint
        if 'rq' in completed and 'rqdonor' in completed:
            assert completed['rq']==completed['rqdonor'],'RQ/RQdonor parameter mismatch'
        report['state']='COMPLETED' if all(v['state']=='COMPLETED' for v in report['arms'].values()) else 'PARTIALLY_EVALUABLE'
        write()
    except BaseException as ex:
        report.update(state='ERROR',error=repr(ex));write();raise


if __name__=='__main__':main()
