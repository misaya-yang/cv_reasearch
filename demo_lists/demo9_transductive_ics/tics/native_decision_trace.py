"""Read-only hooks for the original FoRIS membership decision path.

No labels, fitting, scoring replacement, model allocation or file writes.
Tensor outputs are copied to CPU; the actual original result is returned by
identity. Scope one prediction and restore pre-existing instance overrides.
"""
from contextlib import contextmanager
import torch


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
            result=original(*args,**kwargs)
            if result is None:
                packet.setdefault('native_empty_stages',[]).append(name)
                return result
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
