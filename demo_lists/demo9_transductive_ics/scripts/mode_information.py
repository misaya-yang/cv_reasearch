#!/usr/bin/env python3
"""M1 A1--A3 information probe; no repaired-mask method or A4 evaluation.

The clustering and support readout see no donor/query truth. PRIMARY is normalized
s.Pd (debiased region prototypes), s.Po is a prespecified spatial-feature control.
Both use INSID3 agglomerative_clustering at tau=.6. Standard seed-0, M=15 pools
exclude the query/support names and repeated donor images. All truths remain in
an evaluator outside the inference ImageSet. A1 is diagnostic and uses labels.

Example: python scripts/mode_information.py --file CACHE.pt --limit 30 \
    --out results/mode_information_f1_30.json --prepared-root /root/autodl-tmp/demo9
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


PRIMARY = 'debiased'


def pooled_regions(s, space, cluster, tau=.6):
    """No labels or pseudo labels enter mode construction."""
    prototypes = s.Pd if space == PRIMARY else s.Po
    features = F.normalize(torch.cat(prototypes), dim=1)
    labels = torch.as_tensor(cluster(features, tau), device=features.device).long().reshape(-1)
    if len(labels) != len(features):
        raise ValueError('Official clustering returned the wrong number of region labels')
    # Do not depend on official labels being consecutive or zero-based.
    _, labels = torch.unique(labels, sorted=True, return_inverse=True)
    region_mode = list(labels.split(s.K))
    pixel_mode = torch.stack([m[l] for m, l in zip(region_mode, s.lab)])
    return dict(region_mode=region_mode, pixel_mode=pixel_mode, K=int(labels.max())+1)


def support_labels(modes, gold):
    """Support patch-area majority; ties and unseen modes are unknown (-1).

    Pure-label control: modes containing both FG and BG support patches are also
    unknown. No GT outside the sole annotated support enters either readout.
    """
    pm = modes['pixel_mode'][0]; k = modes['K']
    fg = torch.bincount(pm[gold], minlength=k)
    bg = torch.bincount(pm[~gold], minlength=k)
    state = torch.full_like(fg, -1)
    state[fg > bg] = 1; state[bg > fg] = 0
    pure = torch.full_like(fg, -1)
    pure[(fg > 0) & (bg == 0)] = 1
    pure[(bg > 0) & (fg == 0)] = 0
    return dict(fg=fg, bg=bg, state=state, pure=pure,
                absent=(fg+bg == 0), conflict=(fg > 0) & (bg > 0))


def unknown_evidence(modes, support, h):
    """Label-free evidence, one occurrence per (mode, non-support image).

    Adjacent means any patch lies in the eight-neighbour dilation of a known
    foreground-mode patch in the SAME image. The reference mask only labels
    modes; neither the query prediction nor donor presence truth is consulted.
    Any-adjacent aggregates multiple disjoint instances of one mode in an image.
    Every image participates, including images with no known foreground modes.
    """
    pm = modes['pixel_mode']; k = modes['K']; state = support['state']
    evidence = []
    n_images = pm.shape[0]-1
    fg_present = []
    presence = torch.zeros(k, device=pm.device, dtype=torch.long)
    adjacency = torch.zeros_like(presence)
    presence_with_fg = torch.zeros_like(presence)
    region_seen = torch.zeros_like(presence)
    region_adjacent = torch.zeros_like(presence)
    for j in range(1, len(pm)):
        image_modes = pm[j]
        foreground = (state[image_modes] == 1).reshape(1, 1, h, h).float()
        near = F.max_pool2d(foreground, kernel_size=3, stride=1, padding=1).reshape(-1) > 0
        fg_present.append(bool(foreground.any()))
        present_modes = torch.unique(image_modes)
        adjacent_modes = torch.unique(image_modes[near])
        presence[present_modes] += 1; adjacency[adjacent_modes] += 1
        if bool(foreground.any()): presence_with_fg[present_modes] += 1
        # Region-granularity control without one GPU synchronization per region.
        region_modes = modes['region_mode'][j]
        touched = torch.bincount(modes['local_lab'][j][near], minlength=len(region_modes)) > 0
        region_seen.scatter_add_(0, region_modes, torch.ones_like(region_modes))
        region_adjacent.scatter_add_(0, region_modes, touched.long())
    target_images = sum(fg_present)
    for m in torch.where(support['absent'])[0].tolist():
        n = int(presence[m]); a = int(adjacency[m]); rn = int(region_seen[m]); ra = int(region_adjacent[m])
        evidence.append(dict(mode=m, appearances=n, adjacent_appearances=a,
                             detachability=(n-a)/n if n else None,
                             adjacency_rate=a/n if n else None,
                             prevalence=n/max(n_images, 1),
                             prevalence_given_known_fg=int(presence_with_fg[m])/max(target_images, 1),
                             region_appearances=rn,
                             region_detachability=(rn-ra)/rn if rn else None))
    return dict(unknown_modes=evidence, non_support_images=n_images,
                images_with_known_fg_mode=target_images,
                occurrence_unit='mode,image; adjacent if any instance touches an 8-neighbour known-FG mode',
                interpretation='Candidate negative evidence. No known-FG anchor does not establish target absence.')


def area_truth_counts(pixel_labels, truth, k, h):
    """Evaluator only: assign native GT pixels to nearest-resized region labels.

    Majority uses native pixel AREA, not a vote per region or per image. Final
    scoring still uses ImageSet's standard bilinear boolean-mask upsampling.
    """
    size = truth.shape[-1]
    if size % h == 0:
        factor = size//h
        # Exactly matches nearest resizing for integer grid scaling, using native
        # GT pixel counts per patch. Avoid allocating a full-res int64 mode map.
        patch_fg = truth.reshape(len(truth), h, factor, h, factor).sum((2, 4)).reshape(-1)
        fg = torch.zeros(k, dtype=torch.long, device=truth.device)
        fg.scatter_add_(0, pixel_labels.reshape(-1), patch_fg)
        area = torch.bincount(pixel_labels.reshape(-1), minlength=k)*(factor*factor)
    else:
        native = F.interpolate(pixel_labels.reshape(-1, 1, h, h).float(), size=(size, size), mode='nearest').long().reshape(len(truth), -1)
        fg = torch.bincount(native[truth.reshape(len(truth), -1)], minlength=k)
        area = torch.bincount(native.reshape(-1), minlength=k)
    return fg, area-fg


def binary_counts(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    return [tp, fp, fn]


def precision_recall(counts):
    tp, fp, fn = counts
    return dict(precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                tp=tp, fp=fp, fn=fn)


def auc(scores, labels, weights=None):
    """Mann--Whitney AUC with exact tie half-credit; None for one-class input."""
    if not len(scores): return None
    x = np.asarray(scores, dtype=float); y = np.asarray(labels, dtype=bool)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    total_p, total_n = w[y].sum(), w[~y].sum()
    if total_p <= 0 or total_n <= 0: return None
    order = np.argsort(x, kind='stable'); x, y, w = x[order], y[order], w[order]
    numerator = 0.; prior_n = 0.; start = 0
    while start < len(x):
        end = start+1
        while end < len(x) and x[end] == x[start]: end += 1
        pos = w[start:end][y[start:end]].sum(); neg = w[start:end][~y[start:end]].sum()
        numerator += pos*(prior_n+neg*.5); prior_n += neg; start = end
    return float(numerator/(total_p*total_n))


def error_pair_counts(occurrences, background_only=False):
    """Ordered same-mode/image pairs; descriptive, not independent trials."""
    eligible = [r for r in occurrences if r['gt_pure_label'] is not None
                and (not background_only or r['gt_pure_label'] == 0)]
    n = len(eligible); errors = sum(r['diagnostic_error'] for r in eligible)
    return dict(ordered_pairs=n*(n-1), source_error_pairs=errors*(n-1),
                destination_error_pairs=errors*(n-1), joint_error_pairs=errors*(errors-1),
                pure_occurrences=n, error_occurrences=errors)


def finalize_error_statistics(counts):
    result = dict(counts)
    pairs = counts['ordered_pairs']; source = counts['source_error_pairs']
    base = counts['destination_error_pairs']/pairs if pairs else None
    conditional = counts['joint_error_pairs']/source if source else None
    result.update(p_error_destination=base, p_error_destination_given_source_error=conditional,
                  conditional_error_excess=conditional-base if conditional is not None and base is not None else None)
    return result


def pseudo_error_diagnostic(modes, p1, truth, h):
    """Evaluator only: direct shared-mode P1 errors and available negative votes.

    Presence is label-independent mode membership. P1 FG is fraction >.5 (ties
    BG); GT pure FG >=.8, pure BG <.2, other occurrences excluded from error-pair
    diagnostics. Ground truth never filters peers for negative-vote availability.
    Shared modes are a correspondence proxy, not verified triangle-cycle edges.
    """
    pixel_mode = modes['pixel_mode']; k = modes['K']; by_mode = [[] for _ in range(k)]
    for j in range(1, len(pixel_mode)):
        pm = pixel_mode[j]
        area = torch.bincount(pm, minlength=k).cpu().tolist()
        pseudo_fg = torch.bincount(pm[p1[j]], minlength=k).cpu().tolist()
        gtfg, gtbg = area_truth_counts(pm[None], truth[j:j+1], k, h)
        gtfg, gtbg = gtfg.cpu().tolist(), gtbg.cpu().tolist()
        for m in range(k):
            if not area[m]: continue
            pf = pseudo_fg[m]/area[m]; native_area = gtfg[m]+gtbg[m]
            gf = gtfg[m]/max(native_area, 1); pure = 1 if gf >= .8 else (0 if gf < .2 else None)
            pred = int(pf > .5)
            by_mode[m].append(dict(image=j, area_patches=area[m], area_native_pixels=native_area,
                p1_fg_fraction=pf, gt_fg_fraction=gf, gt_fg_native_pixels=gtfg[m],
                p1_mode_label=pred, gt_pure_label=pure,
                diagnostic_error=(pred != pure) if pure is not None else None))
    stats = dict(pure_bg_multi_image_modes=0, pure_bg_multi_image_native_area=0,
                 all_fg_pure_bg_multi_image_modes=0, all_fg_pure_bg_multi_image_native_area=0,
                 strict_all_patches_fg_pure_bg_multi_image_modes=0, strict_all_patches_fg_pure_bg_multi_image_native_area=0,
                 fp_occurrences_with_peers=0, fp_occurrences_with_negative_peer=0,
                 fp_occurrence_native_area=0, fp_native_area_with_negative_peer=0,
                 fp_correspondence_pairs=0, fp_negative_vote_pairs=0)
    pair_totals = {name:{key:0 for key in ('ordered_pairs','source_error_pairs','destination_error_pairs',
                    'joint_error_pairs','pure_occurrences','error_occurrences')} for name in ('all_pure','pure_bg')}
    mode_rows = []
    for m, rows in enumerate(by_mode):
        if not rows: continue
        native_area = sum(r['area_native_pixels'] for r in rows)
        fg_area = sum(r['gt_fg_native_pixels'] for r in rows)
        pure_bg_mode = fg_area/max(native_area,1) < .2
        all_fg = all(r['p1_mode_label'] == 1 for r in rows)
        all_patches_fg = all(r['p1_fg_fraction'] == 1 for r in rows)
        if pure_bg_mode and len(rows) >= 2:
            stats['pure_bg_multi_image_modes'] += 1; stats['pure_bg_multi_image_native_area'] += native_area
            if all_fg:
                stats['all_fg_pure_bg_multi_image_modes'] += 1
                stats['all_fg_pure_bg_multi_image_native_area'] += native_area
            if all_patches_fg:
                stats['strict_all_patches_fg_pure_bg_multi_image_modes'] += 1
                stats['strict_all_patches_fg_pure_bg_multi_image_native_area'] += native_area
        for source in rows:
            if source['gt_pure_label'] != 0 or source['p1_mode_label'] != 1 or len(rows)<2: continue
            peers = [r for r in rows if r['image'] != source['image']]
            negative = sum(r['p1_mode_label'] == 0 for r in peers)
            stats['fp_occurrences_with_peers'] += 1; stats['fp_occurrences_with_negative_peer'] += int(negative>0)
            stats['fp_occurrence_native_area'] += source['area_native_pixels']
            stats['fp_native_area_with_negative_peer'] += source['area_native_pixels']*int(negative>0)
            stats['fp_correspondence_pairs'] += len(peers); stats['fp_negative_vote_pairs'] += negative
        for name, bg_only in [('all_pure',False),('pure_bg',True)]:
            counts=error_pair_counts(rows,bg_only)
            for key,value in counts.items(): pair_totals[name][key] += value
        mode_rows.append(dict(mode=m, appearances=len(rows), native_area=native_area,
                             gt_fg_fraction=fg_area/max(native_area,1), diagnostic_pure_bg=pure_bg_mode,
                             all_present_images_p1_fg=all_fg,
                             all_present_images_p1_all_patches_fg=all_patches_fg, occurrences=rows))
    denominators = [('all_fg_bg_mode_fraction','all_fg_pure_bg_multi_image_modes','pure_bg_multi_image_modes'),
                    ('all_fg_bg_native_area_fraction','all_fg_pure_bg_multi_image_native_area','pure_bg_multi_image_native_area'),
                    ('fp_occurrence_negative_peer_fraction','fp_occurrences_with_negative_peer','fp_occurrences_with_peers'),
                    ('fp_native_area_negative_peer_fraction','fp_native_area_with_negative_peer','fp_occurrence_native_area'),
                    ('fp_correspondence_negative_vote_fraction','fp_negative_vote_pairs','fp_correspondence_pairs')]
    for name,numerator,denominator in denominators:
        stats[name] = stats[numerator]/stats[denominator] if stats[denominator] else None
    return dict(summary=stats, error_pairs={name:finalize_error_statistics(counts) for name,counts in pair_totals.items()},
                modes=mode_rows, contract='Shared-mode proxy; GT evaluator only; >.5 P1 FG; GT pure FG >=.8/BG <.2; all non-support images; ordered pair counts retain dependence; mode-majority unanimity does not establish region/patch unanimity or cycle-edge correctness.')


def diagnostic_modes(s, modes, support, evidence, truth, patch_truth):
    """A1 and A2/A3 evaluations only, after all legal readouts were frozen."""
    fg, bg = area_truth_counts(modes['pixel_mode'], truth, modes['K'], s.h)
    majority = fg > bg   # Diagnostic majority ties fixed as background.
    global_majority = majority[modes['pixel_mode'][1]]
    lfg, lbg = area_truth_counts(s.lab[1:2], truth[1:2], s.K[1], s.h)
    local_majority = (lfg > lbg)[s.lab[1]]
    local_all = []
    for j in range(s.n):
        xfg, xbg = area_truth_counts(s.lab[j:j+1], truth[j:j+1], s.K[j], s.h)
        local_all.append((xfg > xbg)[s.lab[j]])
    area = fg+bg; purity = torch.maximum(fg, bg).float()/area.clamp_min(1)
    purity_np = purity.cpu().numpy()
    # A2 is scored both on query alone and on the complete unlabeled collection.
    a2 = {}
    for name, state in [('majority', support['state']), ('pure_no_conflict', support['pure'])]:
        pred = (state[modes['pixel_mode']] == 1)
        a2[name] = dict(query_patch=precision_recall(binary_counts(pred[1], patch_truth[1])),
                        collection_patch=precision_recall(binary_counts(pred[1:], patch_truth[1:])))
        known = state >= 0
        a2[name]['mode_label_precision'] = float(majority[state == 1].float().mean()) if (state == 1).any() else None
        a2[name]['known_mode_label_accuracy'] = float((state[known] == majority[known].long()).float().mean()) if known.any() else None
        a2[name]['known_mode_fraction'] = float(known.float().mean())
    # Unknown truth is NON-SUPPORT AREA only; no support contribution to the label.
    ufg, ubg = area_truth_counts(modes['pixel_mode'][1:], truth[1:], modes['K'], s.h)
    unknown = []
    for item in evidence['unknown_modes']:
        row = dict(item); m = row['mode']; af, ab = int(ufg[m]), int(ubg[m]); total = af+ab
        row.update(diagnostic_fg_pixels=af, diagnostic_bg_pixels=ab,
                   diagnostic_fg_fraction=af/max(total, 1), diagnostic_fg_majority=af > ab,
                   diagnostic_purity=max(af, ab)/max(total, 1), diagnostic_majority_tie=af == ab)
        unknown.append(row)
    valid = [x for x in unknown if x['appearances'] and not x['diagnostic_majority_tie']]
    scores = [1-x['detachability'] for x in valid]; labels = [x['diagnostic_fg_majority'] for x in valid]
    weights = [x['diagnostic_fg_pixels']+x['diagnostic_bg_pixels'] for x in valid]
    return dict(A1_gt_majority_diagnostic=global_majority, A1_local_gt_majority_diagnostic=local_majority,
                A1_local_all_gt_majority=local_all, A2=a2,
                mode_diagnostics=dict(n_modes=modes['K'],
                    support_conflict=dict(n_modes=int(support['conflict'].sum()),
                        mode_fraction=float(support['conflict'].float().mean()),
                        support_patch_fraction=float((support['fg']+support['bg'])[support['conflict']].sum()/s.P),
                        collection_native_area_fraction=float(area[support['conflict']].sum()/area.sum().clamp_min(1))),
                    foreground_majority_modes=int(majority.sum()),
                    purity_quantiles=dict(zip(['min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'],
                                             np.quantile(purity_np, [0,.1,.25,.5,.75,.9,1]).tolist())),
                    mode_fraction_purity_ge_90=float((purity >= .9).float().mean()),
                    area_fraction_purity_ge_90=float(area[purity >= .9].sum()/area.sum().clamp_min(1)),
                    modes=[dict(mode=m, foreground_pixels=int(fg[m]), background_pixels=int(bg[m]),
                                purity=float(purity[m]), support_fg_patches=int(support['fg'][m]),
                                support_bg_patches=int(support['bg'][m]), support_label=int(support['state'][m]),
                                support_pure_label=int(support['pure'][m])) for m in range(modes['K'])]),
                A3=dict(evidence, unknown_modes=unknown, n_auc_modes=len(valid),
                        n_fg_modes=sum(labels), n_bg_modes=len(labels)-sum(labels),
                        auc_fg_from_attached=auc(scores, labels),
                        area_weighted_auc_fg_from_attached=auc(scores, labels, weights)))


def standard_pool(d, episode, rng, M):
    cls, names = d['cls'], d['names']; n = len(names)//2
    ref, query = 2*episode, 2*episode+1; c = int(cls[ref])
    candidates = [2*o+1 for o in range(n) if o != episode and cls[2*o] == c
                  and names[2*o+1] not in (names[ref], names[query])]
    seen = set(); candidates = [x for x in candidates if not (names[x] in seen or seen.add(names[x]))]
    return [ref, query]+[int(x) for x in rng.permutation(candidates)[:M]]


def official_cluster(prepared_root, paths):
    path = Path(paths.DEMO4)/'INSID3/utils/clustering.py'
    if not path.is_file():
        raise FileNotFoundError(f'Official INSID3 clustering not found: {path}')
    sys.path.insert(0, str(path.parent.parent))
    spec = importlib.util.spec_from_file_location('mode_information_official_clustering', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.agglomerative_clustering, path


def summarize(records, seed=0):
    acc = {}
    for r in records:
        for key, (i, u) in r['iu'].items():
            pair = acc.setdefault(key, {}).setdefault(r['c'], [0.,0.]); pair[0] += i; pair[1] += u
    miou = {key:100*float(np.mean([i/max(u,1) for i,u in rows.values()])) for key,rows in acc.items()}
    spaces = {}
    for space in (PRIMARY, 'original'):
        rows = [r['spaces'][space] for r in records]
        cts = np.sum([x['A2']['majority']['collection_patch']['tp:fp:fn'] for x in rows], axis=0).tolist() if rows else [0,0,0]
        unknown = [m for x in rows for m in x['A3']['unknown_modes'] if m['appearances'] and not m['diagnostic_majority_tie']]
        scores = [1-m['detachability'] for m in unknown]; labs = [m['diagnostic_fg_majority'] for m in unknown]
        weights = [m['diagnostic_fg_pixels']+m['diagnostic_bg_pixels'] for m in unknown]
        a2p = precision_recall(cts)['precision']; a3 = auc(scores,labs)
        spaces[space] = dict(A2_collection_patch=precision_recall(cts), A3_pooled_mode_auc=a3,
                            A3_pooled_area_weighted_auc=auc(scores,labs,weights),
                            A3_unknown_modes=len(unknown), A3_fg_modes=sum(labs), A3_bg_modes=len(labs)-sum(labs),
                            gates=dict(A1_miou_ge_72=miou.get(space+':A1_gt_majority_diagnostic',0)>=72,
                                       A2_precision_ge_80=a2p is not None and a2p>=.8,
                                       A3_auc_ge_75=a3 is not None and a3>=.75))
        spaces[space]['all_gates_pass']=all(spaces[space]['gates'].values())
        corr_rows = [x['P1_error_correlation'] for x in rows]
        additive = ('pure_bg_multi_image_modes','pure_bg_multi_image_native_area',
                    'all_fg_pure_bg_multi_image_modes','all_fg_pure_bg_multi_image_native_area',
                    'strict_all_patches_fg_pure_bg_multi_image_modes','strict_all_patches_fg_pure_bg_multi_image_native_area',
                    'fp_occurrences_with_peers','fp_occurrences_with_negative_peer',
                    'fp_occurrence_native_area','fp_native_area_with_negative_peer',
                    'fp_correspondence_pairs','fp_negative_vote_pairs')
        corr = {key:sum(x['summary'][key] for x in corr_rows) for key in additive}
        for name,num,den in [('all_fg_bg_mode_fraction','all_fg_pure_bg_multi_image_modes','pure_bg_multi_image_modes'),
                             ('all_fg_bg_native_area_fraction','all_fg_pure_bg_multi_image_native_area','pure_bg_multi_image_native_area'),
                             ('fp_occurrence_negative_peer_fraction','fp_occurrences_with_negative_peer','fp_occurrences_with_peers'),
                             ('fp_native_area_negative_peer_fraction','fp_native_area_with_negative_peer','fp_occurrence_native_area'),
                             ('fp_correspondence_negative_vote_fraction','fp_negative_vote_pairs','fp_correspondence_pairs')]:
            corr[name] = corr[num]/corr[den] if corr[den] else None
        corr['error_pairs'] = {name:finalize_error_statistics({key:sum(x['error_pairs'][name][key] for x in corr_rows)
                                for key in ('ordered_pairs','source_error_pairs','destination_error_pairs',
                                            'joint_error_pairs','pure_occurrences','error_occurrences')})
                               for name in ('all_pure','pure_bg')}
        spaces[space]['P1_error_correlation']=corr
    # Paired exploratory interval uses episode resampling within each observed class.
    # On a single fold this is neither independent generalization nor a final result.
    intervals = {}
    if len(records) >= 2:
        rng = np.random.default_rng(seed); keys=list(miou); draws={key:[] for key in keys if key!='1shot'}
        classes=sorted(set(r['c'] for r in records))
        arrays={c:{key:np.array([r['iu'][key] for r in records if r['c']==c]) for key in keys} for c in classes}
        for _ in range(1000):
            totals={key:[] for key in keys}
            for c in classes:
                ids=rng.integers(0,len(arrays[c]['1shot']),len(arrays[c]['1shot']))
                for key in keys:
                    i,u=arrays[c][key][ids].sum(axis=0); totals[key].append(i/max(u,1))
            base=np.mean(totals['1shot'])
            for key in draws: draws[key].append(100*(np.mean(totals[key])-base))
        intervals={key:dict(delta_pp=miou[key]-miou['1shot'], ci95_pp=np.quantile(v,[.025,.975]).tolist()) for key,v in draws.items()}
    return dict(miou=miou, spaces=spaces, paired_exploratory_vs_1shot=intervals)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--file'); ap.add_argument('--out')
    ap.add_argument('--prepared-root', default='/root/autodl-tmp/demo9')
    ap.add_argument('--limit', type=int, default=30); ap.add_argument('--M', type=int, default=15)
    ap.add_argument('--pool-seed', type=int, default=0); ap.add_argument('--tau', type=float, default=.6)
    ap.add_argument('--every', type=int, default=10); ap.add_argument('--self-check', action='store_true')
    a=ap.parse_args()
    if a.self_check: self_check(); return
    if not a.file or not a.out: ap.error('--file and --out are required without --self-check')
    if a.limit < 1 or a.M < 0 or a.every < 1: ap.error('limit/every must be positive; M must be nonnegative')
    sys.path.insert(0,str(Path(a.prepared_root)/'scripts'))
    import _paths
    from tics import ImageSet, one_shot
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
    cluster, cluster_path=official_cluster(a.prepared_root,_paths)
    d=torch.load(a.file,weights_only=False); rng=np.random.default_rng(a.pool_seed)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); records=[]; start=time.time()
    source_paths=[Path(__file__),cluster_path,Path(a.prepared_root)/'tics/imageset.py']
    report=dict(state='RUNNING',file=a.file,fold=d.get('fold'),args=vars(a),dataset='COCO-20i',
        scope='Single-fold development information probe; no method score or independent generalization claim.',
        contract=dict(primary=PRIMARY, spatial_control='original normalized s.Po',tau=a.tau,
          mode_clustering='Official INSID3 agglomerative_clustering; unweighted regions; no masks',
          support_labels='patch-area majority; equal area or absent=unknown; unknown predicted BG',
          A1_labels='Evaluator native-pixel-area majority across all images; ties=BG; accuracy-optimal diagnostic, not a strict mIoU bound',
          A3_labels='Evaluator native-pixel-area majority across non-support images; ties excluded from AUC',
          gates='PRIMARY A1 query class-mIoU >=72, A2 collection patch microprecision >=.8, A3 pooled mode AUC >=.75',
          dependence='Modes share image/episode context and reused gallery images; pooled mode AUC and error-pair conditional excess are descriptive, not independent-image generalization.',
          predictions=['A1 >=72: modes potentially adequate; below fails frozen granularity',
                       'A2 precision >=.8: sole support may label modes; below refutes reliable anchoring at frozen rule',
                       'A3 AUC >=.75: detachability may carry evidence; below fails prespecified target',
                       'Failure stops before A4; diagnostics retained, no mask repair or parameter search']),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},records=records)
    def write(final=False):
        report.update(episodes=len(records),elapsed_s=time.time()-start)
        if final or len(records)%a.every==0: report.update(summarize(records))
        tmp=out.with_suffix(out.suffix+'.tmp'); tmp.write_text(json.dumps(report,allow_nan=False)); tmp.replace(out)
    write()
    try:
        for e in range(min(a.limit,len(d['names'])//2)):
            idx=standard_pool(d,e,rng,a.M); ref,q=idx[:2]; c=int(d['cls'][ref])
            legal64=torch.zeros_like(d['gt64'][idx]); legal64[0]=d['gt64'][ref]
            legalbits=np.zeros_like(d['gt_bits'][idx]); legalbits[0]=d['gt_bits'][ref]
            s=ImageSet(dict(c=c,names=[d['names'][x] for x in idx],fq=d['fq'][idx],lab=d['lab'][idx],
                           Po=[d['Po'][x] for x in idx],gt64=legal64,gt_bits=legalbits,S=d['S']))
            if bool(s.gt64[1:].any()) or bool(s.gt[1:].any()): raise AssertionError('Non-support GT entered inference')
            p1=one_shot(s,list(range(1,s.n))); donors=list(range(2,s.n)); gold=s.gt64[0]
            predictions={'1shot':p1[1], 'naive':s.predict(1,[0]+donors,[gold]+[p1[j] for j in donors],backward='majority')}
            prepared={}; timings={}
            for space in (PRIMARY,'original'):
                ts=time.time(); modes=pooled_regions(s,space,cluster,a.tau); modes['local_lab']=s.lab
                support=support_labels(modes,gold); evidence=unknown_evidence(modes,support,s.h)
                predictions[space+':A2_support_majority']=(support['state'][modes['pixel_mode'][1]]==1)
                predictions[space+':A2_support_pure_no_conflict']=(support['pure'][modes['pixel_mode'][1]]==1)
                prepared[space]=(modes,support,evidence); timings[space]=time.time()-ts
            # Evaluation truth first becomes available after every legal inference is frozen.
            patch_truth=d['gt64'][idx].to(s.fd.device).bool().reshape(s.n,-1)
            truth=torch.from_numpy(np.unpackbits(d['gt_bits'][idx],axis=1)).to(s.fd.device).bool().reshape(s.n,s.S,s.S)
            predictions['true']=s.predict(1,[0]+donors,[gold]+[patch_truth[j] for j in donors],backward='pooled',k=5)
            space_records={}
            for space,(modes,support,evidence) in prepared.items():
                diag=diagnostic_modes(s,modes,support,evidence,truth,patch_truth)
                predictions[space+':A1_gt_majority_diagnostic']=diag.pop('A1_gt_majority_diagnostic')
                predictions[space+':A1_local_region_gt_majority_diagnostic']=diag.pop('A1_local_gt_majority_diagnostic')
                local_all=diag.pop('A1_local_all_gt_majority'); diag['A1_local_all_gt_majority_native_counts']=[binary_counts(s.up(m),truth[j]) for j,m in enumerate(local_all)]
                for rule in ('majority','pure_no_conflict'):
                    cc=diag['A2'][rule]['collection_patch']; cc['tp:fp:fn']=[cc['tp'],cc['fp'],cc['fn']]
                diag['P1_error_correlation']=pseudo_error_diagnostic(modes,p1,truth,s.h)
                space_records[space]=diag
            iu={}; native_pr={}
            for key,m in predictions.items():
                full=s.up(m); iu[key]=[int((full & truth[1]).sum()),int((full | truth[1]).sum())]
                native_pr[key]=precision_recall(binary_counts(full,truth[1]))
            records.append(dict(e=e,c=c,support=d['names'][ref],query=d['names'][q],
                           donor_ids=[d['names'][x] for x in idx[2:]],pool=len(donors),iu=iu,
                           query_native_pr=native_pr,spaces=space_records,mode_s=timings))
            write()
            if (e+1)%a.every==0: print(e+1,round(time.time()-start,1),report['miou'],report['spaces'],flush=True)
            del s
        report['state']='COMPLETED'; write(final=True)
        print(json.dumps({k:report[k] for k in ('episodes','elapsed_s','miou','spaces')}),flush=True)
    except BaseException as exc:
        report.update(state='ERROR',error=repr(exc)); write(final=True); raise


def self_check():
    from types import SimpleNamespace
    device=torch.device('cpu'); h=4
    lab=torch.tensor([[0,0,1,1]*4,[0,0,1,1]*4,[0,0,1,1]*4])
    pd=[torch.tensor([[1.,0.,0.],[0.,1.,0.]]),torch.tensor([[1.,0.,0.],[0.,0.,1.]]),torch.tensor([[0.,0.,1.],[0.,1.,0.]])]
    fake=SimpleNamespace(Pd=pd,Po=pd,K=[2]*3,lab=lab,h=h,n=3,P=h*h)
    def cluster(x,tau): return x.argmax(1)
    m=pooled_regions(fake,PRIMARY,cluster); m['local_lab']=lab
    gold=lab[0]==0; support=support_labels(m,gold)
    assert support['state'].tolist()==[1,0,-1]
    ev=unknown_evidence(m,support,h)
    row=ev['unknown_modes'][0]
    assert row['appearances']==2 and row['adjacent_appearances']==1 and row['detachability']==.5
    truth=torch.stack([gold,gold,~gold])
    fg,bg=area_truth_counts(m['pixel_mode'],truth.reshape(3,h,h),3,h)
    diag=diagnostic_modes(fake,m,support,ev,truth.reshape(3,h,h),truth)
    assert diag['A2']['majority']['collection_patch']['precision']==1
    assert 'support_conflict' in diag['mode_diagnostics']
    zero_truth=torch.zeros(3,h,h,dtype=torch.bool)
    ps={j:torch.ones(h*h,dtype=torch.bool) for j in range(1,3)}
    corr=pseudo_error_diagnostic(m,ps,zero_truth,h)
    assert corr['summary']['pure_bg_multi_image_modes']==1
    assert corr['summary']['all_fg_pure_bg_multi_image_modes']==1
    assert corr['summary']['fp_correspondence_negative_vote_fraction']==0
    ps[2].zero_(); mixedcorr=pseudo_error_diagnostic(m,ps,zero_truth,h)
    assert mixedcorr['summary']['all_fg_pure_bg_multi_image_modes']==0
    assert mixedcorr['summary']['fp_correspondence_negative_vote_fraction']==1
    assert mixedcorr['error_pairs']['pure_bg']['joint_error_pairs']==0
    assert mixedcorr['error_pairs']['pure_bg']['ordered_pairs']==2
    assert int((fg+bg).sum())==3*h*h
    mixed=gold.clone(); mixed[:4]=~mixed[:4]
    st=support_labels(m,mixed)
    assert bool(st['conflict'][0]) and bool(st['conflict'][1])
    assert st['pure'].tolist()==[-1,-1,-1]
    assert auc([0,1],[False,True])==1 and auc([1,0],[False,True])==0
    assert auc([1,1],[False,True])==.5 and auc([0],[True]) is None
    assert auc([0,1,1],[False,True,False],[2,3,2])==.75
    d=dict(cls=[0]*8,names=['support','query','x','query','x','donor','x','donor'])
    assert standard_pool(d,0,np.random.default_rng(0),15)==[0,1,5]
    print('CPU self-check passed: label-free modes, unknown support labels, conflict control, 8-neighbour detachability, native area totals, tie-aware AUC, deduplicated standard pool.')


if __name__=='__main__': main()
