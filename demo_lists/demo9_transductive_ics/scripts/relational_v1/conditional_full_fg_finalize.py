"""Conditional, GT-free continuation of the source-truncation AUDIT.

Not a registered replacement method: all-source FG vs old top8 BG is asymmetric.
No encoder, training, search, query labels, threshold tuning or live runner here.

Four lines (conditional on a completed full-FG audit):
1. Assumption: fixed source stars omitted by top8 FG reverse at least one accepted U comparison.
2. Prediction: exact rescored UF_all <= UF_top8 for every audited root; only UF_all<=UB_old can withdraw a deletion; zero such roots implies zero continuation.
3. Match: omitted fixed-star explanation exists in this measured source library; it does not establish query semantics or fair method gain.
4. Mismatch: no rescued root ends this branch; a larger UF_all invalidates same-cost/superset provenance; fewer patch deletions alone never establish a better final mask.
"""
import math
import numpy as np
from relational_exclusion_cpu import finalize_deletion


def pure_background_source_control(source_stars, source_coverage):
    """Cause-control only. Boundary BG with FG companions can be legitimate.

    Keep ALL original FG-centered stars unchanged, including mixed companions.
    Keep BG only if every existing star position has known reference cov==0.
    This function neither selects banks nor creates a prediction.
    """
    cov = np.asarray(source_coverage).reshape(-1)
    kept = {root:star for root,star in source_stars.items()
            if cov[star.root] > 0 or all(cov[i] == 0 for i in star.ids)}
    return kept, dict(scope='cause-control only, not assumed correction',
                      FG_stars_unchanged=True,
                      rejected_BG_stars=len(source_stars)-len(kept))


def finalize_audited_full_fg(*, original_records, full_fg_U_by_root,
                             completed_roots, native_pre_final, native_final,
                             finalizer, feature_grid_hw=(64,64),
                             mapping_contract=None, deletion_lifter=None):
    """Consume explicit GT-free exact full-FG costs, never infer missing costs.

    Caller must map the audit schema explicitly; completed_roots must equal all
    original accepted roots for this pair. Full-FG retains the ORIGINAL maximin
    stars, endpoint weights and cost primitive. Query GT is not an argument.
    Caller owns the original RGB/static public-finalizer callback and verified
    working-grid mapping. No-op rescore reuses saved native when no deletion remains.
    """
    h,w=feature_grid_hw
    accepted={int(r['query_root']):r for r in original_records
              if r.get('status')=='ok' and r['joint_flip_predicate']['accepted']}
    ids=set(accepted)
    if set(map(int,completed_roots)) != ids or set(map(int,full_fg_U_by_root)) != ids:
        raise ValueError('full-FG audit must cover every accepted root; no guessed background or partial implicit fallback')
    original=np.zeros((h,w),bool);candidate=original.copy();rescued=[];ledger=[]
    for root,r in accepted.items():
        if not 0 <= root < h*w:
            raise ValueError('query root is outside verified full grid')
        p=r['joint_flip_predicate'];LF,LB,UF,UB=(float(p[k]) for k in ('LF','LB','UF','UB'))
        u=float(full_fg_U_by_root[root])
        if not all(math.isfinite(x) for x in (LF,LB,UF,UB,u)) or not (LF<LB and UB<UF):
            raise ValueError('original accepted witness is inconsistent')
        if u>UF:
            raise ValueError('full-FG superset cost exceeds old top8; reject audit primitive/provenance')
        original.reshape(-1)[root]=True
        withdraw=u<=UB  # original strict BG advantage has vanished; ties retain FG
        candidate.reshape(-1)[root]=not withdraw
        if withdraw:rescued.append(root)
        ledger.append(dict(query_root=root,UF_top8=UF,UB_old_top8=UB,UF_all=u,
                           withdrew_sampled_star_flip=withdraw))
    result=finalize_deletion(candidate,native_pre_final=native_pre_final,native_final=native_final,
                             finalizer=finalizer,mapping_contract=mapping_contract,
                             deletion_lifter=deletion_lifter)
    result['audit']=dict(scope='conditional asymmetric all-source-FG FIXED STAR FAMILY vs old top8-BG; NOT full constellation space or a fair new method',
                         original_accepted=len(ids),withdrawn_roots=rescued,
                         original_patch_deletions=int(original.sum()),candidate_patch_deletions=int(candidate.sum()),
                         query_GT_used=False,encoder_calls=0,new_acceptances=0,
                         final_deletion_monotonicity_across_arms_not_claimed=True,ledger=ledger)
    return result


def finalize_pair_from_audit_rows(*, pair_id, original_records, audited_rows, **finalizer_inputs):
    """Explicit bridge from source_truncation_audit.roots.jsonl, no file/GT IO.

    Incomplete roots are refused, including a budget-limited partial pair.
    Only the caller-frozen, wholly scanned maximin family is certified here.
    """
    costs={};completed=[]
    old={int(r['query_root']):r for r in original_records
         if r.get('status')=='ok' and r['joint_flip_predicate']['accepted']}
    for r in audited_rows:
        if r['pair_id']!=pair_id or not r.get('search_complete') or not r.get('full_family_includes_old_FG') or not r.get('top8_cost_exact'):
            raise ValueError('pair/search-complete/superset/exact-cost audit contract failed')
        root=int(r['query_root'])
        if root in costs or root not in old:
            raise ValueError('duplicate or non-accepted root in fixed-family audit')
        p=old[root]['joint_flip_predicate']
        if any(float(r['old_'+key])!=float(p[key]) for key in ('LF','LB','UF','UB')):
            raise ValueError('audit old cost does not exactly match the frozen witness')
        if int(r['all_source_FG_fixed_stars'])<len(old[root]['sampled_F_roots']):
            raise ValueError('claimed complete FG family is smaller than old sampled FG bank')
        if float(r['LF_all'])>float(p['LF']):
            raise ValueError('full-FG independent superset cost worsened: inconsistent primitive')
        costs[root]=float(r['UF_all']);completed.append(root)
    return finalize_audited_full_fg(original_records=original_records,full_fg_U_by_root=costs,
                                    completed_roots=completed,**finalizer_inputs)
