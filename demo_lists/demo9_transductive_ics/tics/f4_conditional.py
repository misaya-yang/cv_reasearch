"""F4 B/C/D/E: original-anchor conditional negative evidence, pair inference only.

The fixed public FoRIS Stage2 supplies both actual query streams and native
foreground prototypes. Stage1, SF, candidate votes, grouping, seed and CRF are
not replaced. No query annotation, new encoder call or learned parameter.
"""
from contextlib import contextmanager
import inspect

import torch
import torch.nn.functional as F


class UnsupportedF4Domain(ValueError):
    """Nonempty, nonunit/zero direction domains retain native explicitly."""


@contextmanager
def _owned_methods(host, replacements):
    old={name:(name in host.__dict__,host.__dict__.get(name)) for name in replacements}
    try:
        for name,value in replacements.items():setattr(host,name,value)
        yield
    finally:
        for name,(owned,value) in old.items():
            if owned:setattr(host,name,value)
            else:delattr(host,name)


def _check_unit(values,name):
    if values.ndim==1:values=values[None]
    if values.dtype not in (torch.float32,torch.float64) or not torch.isfinite(values).all():
        raise UnsupportedF4Domain(name+': nonfinite/non-FP32-or-FP64 domain')
    norms=values.norm(dim=1)
    tolerance=64*torch.finfo(values.dtype).eps
    if not torch.all((norms-1).abs()<=tolerance):
        raise UnsupportedF4Domain(name+': actual native vector is not unit; do not silently repair it')
    return dict(min=float(norms.min()),max=float(norms.max()),tolerance=tolerance)


def _unit_mean(tokens,name):
    if not len(tokens):raise UnsupportedF4Domain(name+': nonempty set required')
    mean=tokens.mean(0)
    if not torch.isfinite(mean).all():
        raise UnsupportedF4Domain(name+': nonempty mean is nonfinite')
    value=F.normalize(mean,dim=0)
    _check_unit(value,name)
    return value


def _full_average_tree(tokens):
    """One exact source-metric average-linkage tree; no query/data fitting."""
    if len(tokens)==1:return [],[]
    from sklearn.cluster import AgglomerativeClustering
    normalized=F.normalize(tokens,dim=1)
    distance=(1-(normalized@normalized.T).clamp(-1,1)).detach().cpu().numpy()
    tree=AgglomerativeClustering(n_clusters=None,distance_threshold=0.,metric='precomputed',
        linkage='average',compute_full_tree=True,compute_distances=True)
    tree.fit(distance)
    # Drop the quadratic matrix/model when this function returns.
    return tree.children_.copy(),tree.distances_.copy()


def _cut_average_tree(children,distances,size,count,device):
    """Exact first size-count merges; equal distances follow the frozen tree."""
    if not 1<=count<=size:raise UnsupportedF4Domain('Matched E cluster count outside global token budget')
    parent=list(range(size));node_member=list(range(size))+[-1]*(size-1)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    merges=size-count
    for step in range(merges):
        left,right=map(int,children[step])
        a,b=find(node_member[left]),find(node_member[right])
        lo,hi=min(a,b),max(a,b);parent[hi]=lo;node_member[size+step]=lo
    roots=[find(i) for i in range(size)]
    labels_by_root={root:i for i,root in enumerate(sorted(set(roots)))}
    if len(labels_by_root)!=count:raise RuntimeError('Shared average tree did not produce exact matched cluster count')
    labels=torch.tensor([labels_by_root[root] for root in roots],device=device,dtype=torch.long)
    return labels,(float(distances[merges-1]) if merges else None)


def _bank(tokens,source_scope,tau_c,*,name='bank',global_mean=None,partition=None):
    """Partition unit tokens; read means from original members, not pooled units."""
    mean=_unit_mean(tokens,name+'_global') if global_mean is None else global_mean
    _check_unit(mean,name+'_global')
    normalized=F.normalize(tokens,dim=1)
    if partition is None:
        labels=source_scope['agglomerative_clustering'](normalized,tau=tau_c)
        count=int(labels.max())+1
        effective_cut=float(1-tau_c)
    else:
        labels,effective_cut=partition
        count=int(labels.max())+1
    means=[];sizes=[]
    for cluster in range(count):
        selected=tokens[labels==cluster]
        if not len(selected):raise UnsupportedF4Domain(name+': empty assigned cluster')
        means.append(_unit_mean(selected,name+'_cluster_'+str(cluster)))
        sizes.append(len(selected))
    directions=torch.stack([mean,*means])
    return directions,dict(cluster_count=count,slots=1+count,token_count=len(tokens),
                           cluster_sizes=sizes,effective_distance_cut=effective_cut,
                           effective_unique_unit_directions=len(torch.unique(directions,dim=0)))


def source_evidence(ctx):
    """Observe real native Stage2, including its exact original hard top-k IDs.

    The returned source result is untouched. Exact native score/mask replay is
    required. The process-local topk tap is confined to the original helper's
    single Stage2 call and restores its original callable even after a failure.
    """
    cached=getattr(ctx,'_f4_source_evidence',None)
    if cached is not None:return cached
    host=ctx.host
    original_stage=host._part2_stage2_contrastive_score
    original_stats=host._reference_contrastive_prototypes
    stage_signature=inspect.signature(original_stage)
    stats_signature=inspect.signature(original_stats)
    captured={};active=[False]
    source_scope=type(host)._part2_stage2_contrastive_score.__globals__
    def stats(*args,**kwargs):
        if not active[0]:return original_stats(*args,**kwargs)
        if 'stats_result' in captured:raise RuntimeError('Expected exactly one original Stage2 prototype call')
        captured['stats_args']=dict(stats_signature.bind(*args,**kwargs).arguments)
        saved_topk=torch.topk;calls=[]
        def observed_topk(*values,**options):
            result=saved_topk(*values,**options)
            calls.append(dict(indices=result.indices.detach().clone(),
                              values=result.values.detach().clone(),
                              input=values[0].detach().clone()))
            return result
        torch.topk=observed_topk
        try:result=original_stats(*args,**kwargs)
        finally:torch.topk=saved_topk
        captured['stats_result']=result
        captured['native_topk_calls']=calls
        return result
    def stage(*args,**kwargs):
        if 'stage_result' in captured:raise RuntimeError('Expected exactly one original Stage2 call')
        captured['stage_args']=dict(stage_signature.bind(*args,**kwargs).arguments)
        active[0]=True
        try:result=original_stage(*args,**kwargs)
        finally:active[0]=False
        captured['stage_result']=result
        return result
    with _owned_methods(host,{'_part2_stage2_contrastive_score':stage,
                              '_reference_contrastive_prototypes':stats}):
        replay=ctx.replay()
    if not torch.equal(replay['mask'],ctx.native['mask']) or not torch.equal(replay['score'],ctx.native['score']):
        raise RuntimeError('Observing native F4 evidence changes complete source prediction')
    kw=captured['stage_args'];stats_kw=captured['stats_args']
    if kw['n_refs']!=1 or stats_kw['mu_fg_per_reference'] is not True:
        raise RuntimeError('F4 single-reference actual Stage2 role contract not met')
    if captured['stage_result'] is None or captured['stats_result'] is None:
        raise RuntimeError('Original task has no foreground; do not drop it or forge F4 evidence')
    mu_fg,mu_bg,prototypes=captured['stats_result']
    if prototypes is None or not len(prototypes):
        raise RuntimeError('Actual native foreground modes are required, not invented replacements')
    beta=float(host.dino_bg_weight);temperature=max(1e-4,float(host.cluster_logsumexp_temp))
    if beta!=.55 or temperature!=.07:
        raise RuntimeError('Document fixes beta=.55/tau=.07; do not silently run another policy')
    raw_F=host.use_raw_target_clustering and kw['tgt_feat_raw'] is not None
    raw_B=host.use_raw_target_scoring and kw['tgt_feat_raw'] is not None
    q_F=kw['tgt_feat_raw'] if raw_F else kw['tgt_feat']
    q_B=kw['tgt_feat_raw'] if raw_B else kw['tgt_feat']
    mu_bg_orth=mu_bg-(mu_bg*mu_fg).sum()*mu_fg
    residual=mu_bg_orth/mu_bg_orth.norm().clamp_min(1e-8)
    native_logits=torch.einsum('bchw,kc->bkhw',q_F,prototypes)
    fg_map=(temperature*torch.logsumexp(native_logits/temperature,dim=1)).squeeze(0)
    bg_map=torch.einsum('bchw,c->bhw',q_B,residual).squeeze(0)
    original_score,original_sf,original_sbn,_=captured['stage_result']
    if not torch.equal(fg_map-beta*bg_map,original_score):
        raise RuntimeError('Exact original two-stream Stage2 reconstruction failed')
    ref_feats=stats_kw['ref_feats']
    roles=stats_kw['ref_masks_bool'][0].reshape(-1).bool()
    tokens=ref_feats[0,0].flatten(1).T
    # Match the original helper's COLUMN-major boolean selection/cat/topk
    # reduction before checking mu_bg. A transposed row-major recreation can
    # choose a different floating reduction kernel despite identical members.
    native_bg_columns=ref_feats[0,0][:,~stats_kw['ref_masks_bool'][0]]
    background_columns=torch.cat([native_bg_columns],dim=1)
    background=background_columns.T
    global_ids=torch.nonzero(~roles,as_tuple=False).flatten()
    packet=dict(state='SUPPORTED',fg_prototypes=prototypes,residual_bg=residual,
                mu_fg=mu_fg,native_mu_bg=mu_bg,q_F=q_F,q_B=q_B,
                native_foreground_map=fg_map,native_fg_logits=native_logits[0],raw_bg_map=bg_map,
                original_sf=original_sf,original_sbn=original_sbn,
                all_bg_tokens=background,all_bg_columns=background_columns,all_bg_global_ids=global_ids,
                ref_role_mask=roles,source_scope=source_scope,
                metadata=dict(beta=beta,temperature=temperature,cluster_tau=float(host.tau),
                    actual_two_streams=True,query_streams_bitwise_equal=torch.equal(q_F,q_B),
                    q_F_source='raw' if raw_F else 'Stage1 gated',
                    q_B_source='raw' if raw_B else 'Stage1 gated',
                    source_mask_helper='actual native Stage2 downsample_mask roles',
                    source_native_replay_exact=True,encoder_calls=0,query_GT_used=False))
    if not len(background):
        if captured['native_topk_calls'] or not torch.equal(bg_map,torch.zeros_like(bg_map)):
            raise RuntimeError('Original truly-empty background does not follow zero-BG path')
        packet.update(state='NO_BACKGROUND',hard_bank=background,hard_token_ids=global_ids,
                      hard_local_ids=global_ids,hard_bank_metadata={})
    else:
        calls=captured['native_topk_calls']
        k_hard=max(1,int(.2*len(background)))
        if len(calls)!=1 or calls[0]['indices'].numel()!=k_hard or calls[0]['input'].numel()!=len(background):
            raise RuntimeError('Could not identify the exact original hard20 membership')
        indices=calls[0]['indices']
        hard=background[indices]
        actual_mean=F.normalize(background_columns[:,indices].mean(dim=1),p=2,dim=0)
        if not torch.equal(actual_mean,mu_bg):
            raise RuntimeError('Captured original hard-token members do not reproduce native mu_bg')
        packet.update(hard_local_ids=indices,hard_token_ids=global_ids[indices],
                      global_bg_similarity=calls[0]['input'])
        try:
            _check_unit(prototypes,'original p_k')
            _check_unit(mu_fg,'original mu_fg')
            _check_unit(residual,'original r0')
            bank,audit=_bank(hard,source_scope,float(host.tau),name='native_hard20',global_mean=mu_bg)
            packet.update(hard_bank=bank,hard_bank_metadata=audit)
        except UnsupportedF4Domain as exc:
            packet.update(state='UNSUPPORTED_DOMAIN',unsupported_reason=str(exc),
                          hard_bank=background.new_empty((0,background.shape[1])),hard_bank_metadata={})
    ctx._f4_source_evidence=packet
    return packet


def _stable_top_ids(scores,count):
    """New selections break exact similarity ties by original source row order."""
    return torch.argsort(scores,descending=True,stable=True)[:count]


def extension_banks(evidence,include_global=False):
    """D and E share actual D token union M and per-FG additional cluster slots."""
    old=evidence.get('_extension_banks')
    if old is not None:
        if include_global and not old['global_banks']:_materialize_global_banks(evidence,old)
        return old
    bg=evidence['all_bg_tokens'];fg=evidence['fg_prototypes'];mu=evidence['mu_fg']
    count=len(evidence['hard_local_ids']);scope=evidence['source_scope'];tau=evidence['metadata']['cluster_tau']
    conditional=[];audits=[];selected=[]
    inherited_key=tuple(sorted(evidence['hard_local_ids'].detach().cpu().tolist()))
    cache={inherited_key:(evidence['hard_bank'],evidence['hard_bank_metadata'])}
    exposure=torch.zeros(len(bg),dtype=torch.bool,device=bg.device)
    exposure[evidence['hard_local_ids']]=True
    for index,p in enumerate(fg):
        ids=_stable_top_ids(torch.einsum('cn,c->n',evidence['all_bg_columns'],p),count)
        key=tuple(sorted(ids.detach().cpu().tolist()))
        if key in cache:bank,audit=cache[key]
        else:
            # Canonical source-row order makes identical member sets share the
            # same exact tree, means and directions, irrespective of rank order.
            canonical=torch.tensor(key,device=bg.device,dtype=torch.long)
            bank,audit=_bank(bg[canonical],scope,tau,name='conditional_'+str(index))
            cache[key]=(bank,audit)
        conditional.append(bank);selected.append(ids);audits.append(audit);exposure[ids]=True
    M=int(exposure.sum())
    # The inherited native top-k may have implementation-defined tie members.
    # Preserve those exact members, then fill the global top-M value budget in
    # stable source-row order; only boundary ties can differ from naive top-M.
    inherited=evidence['hard_local_ids']
    native_set=torch.zeros(len(bg),dtype=torch.bool,device=bg.device);native_set[inherited]=True
    remaining=_stable_top_ids(evidence['global_bg_similarity'],len(bg))
    remaining=remaining[~native_set[remaining]][:M-len(inherited)]
    global_ids=torch.cat([inherited,remaining])
    if len(global_ids)!=M or len(torch.unique(global_ids))!=M:
        raise RuntimeError('Global control does not match independent source-token exposure')
    result=dict(conditional=conditional,conditional_audits=audits,conditional_ids=selected,
                global_banks=[],global_audits=[],global_local_ids=global_ids,
                unique_D_exposure=M,union_D_global_ids=evidence['all_bg_global_ids'][exposure],
                global_E_global_ids=evidence['all_bg_global_ids'][global_ids],
                unique_conditional_pools=len(cache),global_full_trees_built=0,
                unique_global_tree_cuts=0)
    evidence['_extension_banks']=result
    if include_global:_materialize_global_banks(evidence,result)
    return result


def _materialize_global_banks(evidence,result):
    """Only E pays/builds the shared global tree; its failure cannot cancel D."""
    bg=evidence['all_bg_tokens'];scope=evidence['source_scope'];tau=evidence['metadata']['cluster_tau']
    global_tokens=bg[result['global_local_ids']]
    M=len(global_tokens)
    children,distances=_full_average_tree(global_tokens)
    cut_cache={};global_banks=[];global_audits=[]
    for index,audit in enumerate(result['conditional_audits']):
        target_count=audit['cluster_count']
        if target_count in cut_cache:bank,global_audit=cut_cache[target_count]
        else:
            partition=_cut_average_tree(children,distances,M,target_count,bg.device)
            bank,global_audit=_bank(global_tokens,scope,tau,partition=partition,
                                    name='global_matched_'+str(index))
            cut_cache[target_count]=(bank,global_audit)
        if len(bank)!=len(result['conditional'][index]):
            raise RuntimeError('Global E control does not match additional per-branch prototype slots')
        global_banks.append(bank);global_audits.append(global_audit)
    result.update(global_banks=global_banks,global_audits=global_audits,
                  global_full_trees_built=1,unique_global_tree_cuts=len(cut_cache))


def conditional_score(evidence,policy):
    """Literal projection-max-LSE, streamed per FG without K×L×Q materialization."""
    if policy not in ('anchor','001','002','global_matched'):
        raise ValueError('Unknown frozen F4 policy')
    p=evidence['fg_prototypes'];r0=evidence['residual_bg'];base=evidence['hard_bank']
    qB=evidence['q_B'][0].flatten(1).T
    f_native=evidence['native_fg_logits'].flatten(1).T
    beta=evidence['metadata']['beta'];temperature=evidence['metadata']['temperature']
    extended=extension_banks(evidence,include_global=(policy=='global_matched')) if policy in ('002','global_matched') else None
    branches=[];direction_counts=[];effective_counts=[]
    for k,prototype in enumerate(p):
        pieces=[r0[None]]
        if policy!='anchor':pieces.append(base)
        if policy=='002':pieces.append(extended['conditional'][k])
        if policy=='global_matched':pieces.append(extended['global_banks'][k])
        bank=torch.cat(pieces)
        residuals=bank-(bank@prototype)[:,None]*prototype
        # Deliberately NO residual unit normalization and NO artificial zero
        # direction or ReLU; signed maximum follows the document exactly.
        penalty=(qB@residuals.T).max(dim=1).values
        branches.append(f_native[:,k]-beta*penalty)
        direction_counts.append(len(residuals))
        effective_counts.append(len(torch.unique(residuals,dim=0)))
    logits=torch.stack(branches,dim=1)
    Z=(temperature*torch.logsumexp(logits/temperature,dim=1)).reshape(evidence['raw_bg_map'].shape)
    effective=(evidence['native_foreground_map']-Z)/beta
    sbn=effective-effective.min();sbn=sbn/sbn.max().clamp_min(1e-6)
    audit=dict(policy=policy,beta=beta,temperature=temperature,
        conditional_projection='native r0 anchor plus inherited hard-bank; projection magnitude retained',
        projected_residuals_renormalized=False,negative_penalty_relu=False,
        foreground_modes=len(p),base_unit_slots=len(base),direction_slots_by_branch=direction_counts,
        effective_unique_directions_by_branch=effective_counts,
        foreground_map_and_sf_preserved=True,stage1_and_group_inputs_preserved=True,
        effectiveBG_definition='(native F - new Z)/beta; not an independent BG cosine',
        source_hard_token_ids=evidence['hard_token_ids'].detach().cpu().tolist(),
        query_streams_bitwise_equal=evidence['metadata']['query_streams_bitwise_equal'],
        extra_encoder_calls=0,no_query_GT=True)
    if extended is not None:
        audit.update(unique_D_exposure=extended['unique_D_exposure'],
            union_D_global_ids=extended['union_D_global_ids'].detach().cpu().tolist(),
            global_E_global_ids=extended['global_E_global_ids'].detach().cpu().tolist(),
            additional_cluster_slots_D=[v['cluster_count'] for v in extended['conditional_audits']],
            additional_cluster_slots_E=[v['cluster_count'] for v in extended['global_audits']],
            additional_cluster_slots_E_planned=[v['cluster_count'] for v in extended['conditional_audits']],
            conditional_bank_audits=extended['conditional_audits'],global_bank_audits=extended['global_audits'],
            matched_exposure_and_slots=True,
            unique_conditional_pools=extended['unique_conditional_pools'],
            global_full_trees_built=extended['global_full_trees_built'],
            unique_global_tree_cuts=extended['unique_global_tree_cuts'],
            global_tree_ties='exact frozen average-linkage merge order; canonical first-leaf label IDs',
            global_E_ties='inherit exact original hard membership, stable source-row ties for remaining top global-FG similarities',
            same_actual_aggregate_token_count_or_cut_claimed=False)
    return Z,sbn,effective,audit


def run_conditional(ctx,policy='001',evidence=None):
    evidence=source_evidence(ctx) if evidence is None else evidence
    if evidence['state']!='SUPPORTED':
        result=ctx.replay()
        result['f4_audit']=dict(policy=policy,state=evidence['state'],native_retained=True,
            unsupported_reason=evidence.get('unsupported_reason'),query_GT_used=False,extra_encoder_calls=0)
        return result
    try:Z,sbn,effective,audit=conditional_score(evidence,policy)
    except UnsupportedF4Domain as exc:
        result=ctx.replay()
        result['f4_audit']=dict(policy=policy,state='UNSUPPORTED_EXTENSION_DOMAIN',
                              native_retained=True,reason=str(exc),extra_encoder_calls=0,no_query_GT=True)
        return result
    def change(original_result,ctx,**kwargs):
        old,sf,old_sbn,mu,denoised=original_result
        if not torch.equal(sf,evidence['original_sf']):
            raise RuntimeError('Native SF changed before F4 override')
        return Z,sf,sbn,mu,denoised
    result=ctx.replay(part2_override=change)
    invariants=('part2_sf','part2_mu_fg','part2_tgt_denoised','candidate_hard','candidate_vote',
                'seed_prior','seed_cluster_labels','semantic_cluster_labels',
                'part3_candidate_vote','part3_seed_prior')
    checked=[]
    for name in invariants:
        if name in ctx.native['stages']:
            if name not in result['stages'] or not torch.equal(result['stages'][name],ctx.native['stages'][name]):
                raise RuntimeError('F4 changed a frozen source field: '+name)
            checked.append(name)
    audit.update(state='ACTIVE',frozen_stage_fields_exact=checked,
                 Part4_BG_coupling_recomputed=True,complete_public_pipeline=True)
    result['f4_audit']=audit
    result['effective_background_map']=effective
    return result


ARMS={'B':lambda ctx:run_conditional(ctx,'anchor'),
      'C':lambda ctx:run_conditional(ctx,'001'),
      'D':lambda ctx:run_conditional(ctx,'002'),
      'E':lambda ctx:run_conditional(ctx,'global_matched')}
