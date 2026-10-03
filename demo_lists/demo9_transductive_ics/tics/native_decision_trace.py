"""Read-only hooks for the original FoRIS membership decision path.

No labels, fitting, scoring replacement, model allocation or file writes.
Tensor outputs are copied to CPU; the actual original result is returned by
identity. Scope one prediction and restore pre-existing instance overrides.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import torch


def _reference_witness_function():
    spec=importlib.util.spec_from_file_location('_native_trace_witness',Path(__file__).with_name('reference_witness_audit.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.reference_witness_audit


STAGES = {
    '_part2_background_suppression': ('part2_score','part2_sf','part2_sbn'),
    '_locate_candidates': ('candidate_hard','candidate_vote'),
    '_build_seed_cluster_prior': ('seed_prior',),
    '_part3_clustering': ('part3_score','part3_candidate_vote','part3_seed_prior'),
    '_semantic_disagreement_penalty': ('part4_penalty',),
    '_semantic_cluster_reweight_map': ('part4_cluster_delta',),
    '_part4_semantic_consistency_correction': ('part4_score',),
    '_binarize_response': ('pre_refinement_mask',),
    '_finalize_mask': ('post_refinement_mask',),
}


@contextmanager
def trace_native_decisions(host):
    saved = {}; packet = dict(state='CAPTURING',maps={},binarization={},
                             query_labels_received=False,original_outputs_returned=True)
    def wrap(name, original, fields):
        def call(*args, **kwargs):
            if name == '_binarize_response':
                score = args[0] if args else kwargs['score_hw']
                packet['binarization'] = dict(minimum=float(score.min()),maximum=float(score.max()),
                    midpoint=float((score.min()+score.max())/2),range_floor=1e-6,
                    threshold_in_source_score_units=float(score.min())+.5*max(float(score.max()-score.min()),1e-6),
                    target_hw=list(kwargs['target_hw']),source_uses_query_global_extrema=True)
            # Observe the ORIGINAL target grouping, including RGB/position
            # in the public source path. No copied descriptors or new groups.
            cluster_field = {'_build_seed_cluster_prior':'seed_cluster_labels',
                '_semantic_cluster_reweight_map':'semantic_cluster_labels'}.get(name)
            scope = getattr(getattr(original,'__func__',original),'__globals__',{})
            old_cluster = scope.get('agglomerative_clustering') if cluster_field else None
            labels = []
            def observed_cluster(values,*cargs,**ckwargs):
                value=old_cluster(values,*cargs,**ckwargs)
                labels.append(value.detach().cpu().clone())
                return value
            if old_cluster is not None:scope['agglomerative_clustering']=observed_cluster
            try:
                result=original(*args,**kwargs)
            finally:
                if old_cluster is not None:scope['agglomerative_clustering']=old_cluster
            if labels:
                if len(labels)!=1 or not isinstance(result,torch.Tensor):
                    raise ValueError('Unexpected original target-clustering interface')
                packet['maps'][cluster_field]=labels[0].reshape(result.shape[-2:])
            if result is None:
                packet.setdefault('native_empty_stages',[]).append(name)
                return result
            if name=='_locate_candidates' and all(k in kwargs for k in ('ref_feats','tgt_feat','ref_masks','n_refs','h','w')):
                scope=getattr(getattr(original,'__func__',original),'__globals__',{})
                downsample=scope.get('downsample_mask')
                if not callable(downsample):raise ValueError('Original candidate mask-downsample operation missing')
                n,h,w=kwargs['n_refs'],kwargs['h'],kwargs['w']
                ref_masks=torch.stack([downsample(kwargs['ref_masks'][j:j+1],h,w).squeeze(0) for j in range(n)])
                # Same complete query matrix width as the original einsum;
                # do not introduce a different chunk-width GEMM into argmax
                # replay at near ties. Matrix is transient, never exported.
                witness=_reference_witness_function()(kwargs['ref_feats'][0],kwargs['tgt_feat'][0],ref_masks,chunk=h*w)
                if not torch.equal(witness['source_candidate'],result[0]):
                    raise ValueError('Computed source nearest-label witness does not replay actual native candidate')
                packet['witness_audit']=dict(source_state=witness['source_state_class'],cosine_state=witness['cosine_state_class'],
                    source_ref_normalization_axis=witness['source_ref_normalization_axis'],
                    cosine_ref_normalization_axis=witness['cosine_ref_normalization_axis'],
                    target_normalization_axis=witness['target_normalization_axis'],
                    source_candidate_exact=True,query_GT_used=False,
                    margin_definition='max across all reference foreground scores minus max across all background scores; multi-reference source candidates use per-reference label votes',
                    posterior_or_reliability_guarantee=False)
                for prefix,branch in [('witness','source'),('cosine','cosine')]:
                    for field in ('fg_max','bg_max','margin','nearest_label','per_ref_margin'):
                        value=witness[branch+'_'+field]
                        if value is not None:packet['maps'][prefix+'_'+field]=value.detach().cpu().clone()
                    field='witness_source_candidate' if branch=='source' else 'cosine_candidate'
                    packet['maps'][field]=witness[branch+'_candidate'].detach().cpu().clone()
            values=result if isinstance(result,tuple) else (result,)
            if len(values)<len(fields):raise ValueError('Native trace stage interface mismatch: '+name)
            for field,value in zip(fields,values):
                if not isinstance(value,torch.Tensor):raise TypeError('Native stage field is not a tensor: '+field)
                packet['maps'][field]=value.detach().cpu().clone()
            return result
        return call
    try:
        for name,fields in STAGES.items():
            if not callable(getattr(host,name,None)):raise ValueError('Native source stage missing: '+name)
            saved[name]=(name in host.__dict__,host.__dict__.get(name))
            original=getattr(host,name)
            setattr(host,name,wrap(name,original,fields))
        yield packet
        packet['state']='CAPTURED'
    except BaseException:
        packet['state']='ERROR';raise
    finally:
        for name,(existed,previous) in saved.items():
            if existed:setattr(host,name,previous)
            else:delattr(host,name)


def native_midrange_mask(score):
    """Exact grid-size FoRIS scalar rule, without upsampling/refinement.

    Degenerate range follows the source floor. This is an arithmetic helper
    for CPU counterexamples, never an alternative segmentation pipeline.
    """
    shifted=score-score.min()
    return shifted/shifted.max().clamp_min(1e-6)>.5
