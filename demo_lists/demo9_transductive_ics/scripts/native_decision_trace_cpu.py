#!/usr/bin/env python3
"""CPU hook invariants and exact extracted-source binarization counterexample."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();torch.set_num_threads(1)
    src=a.source.read_text();tree=ast.parse(src)
    host_class=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FoRIS')
    method=next(n for n in host_class.body if isinstance(n,ast.FunctionDef) and n.name=='_binarize_response')
    # Execute the exact inspected pure method only, not third-party module
    # imports, weights, constructors, encoder or CRF.
    scope={'torch':torch,'F':F}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(a.source),'exec'),scope)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('native_trace_cpu',root/'tics/native_decision_trace.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Fixture:
        _binarize_response=scope['_binarize_response']
        def _part2_background_suppression(self,x):return x,x,x,torch.ones(3),torch.ones(3)
        def _locate_candidates(self,x):return x>0,x
        def _build_seed_cluster_prior(self,x):return x
        def _part3_clustering(self,x):return x,x,x
        def _semantic_disagreement_penalty(self,x):return x*.1
        def _semantic_cluster_reweight_map(self,x):return x*.01
        def _part4_semantic_consistency_correction(self,x):return x
        def _finalize_mask(self,x):return x
    host=Fixture();records=[]
    for i in range(10):
        torch.manual_seed(2070+i);x=torch.randn(4,4)
        before=dict(host.__dict__)
        original=host._part2_background_suppression
        host._part2_background_suppression=original  # Existing override must survive.
        with module.trace_native_decisions(host) as trace:
            result=host._part2_background_suppression(x)
            assert result[0] is x
            host._locate_candidates(x);host._build_seed_cluster_prior(x);host._part3_clustering(x)
            host._semantic_disagreement_penalty(x);host._semantic_cluster_reweight_map(x)
            host._part4_semantic_consistency_correction(x)
            pred=host._binarize_response(x,target_hw=(4,4));out=host._finalize_mask(pred)
            assert out is pred
            assert torch.equal(pred,module.native_midrange_mask(x))
            assert trace['maps']['part2_score'].data_ptr()!=x.data_ptr()
        assert host.__dict__=={'_part2_background_suppression':original}
        assert trace['state']=='CAPTURED' and len(trace['maps'])==14
        try:
            with module.trace_native_decisions(host) as failed:raise RuntimeError('fixture error')
        except RuntimeError:pass
        assert host.__dict__=={'_part2_background_suppression':original} and failed['state']=='ERROR'
        del host._part2_background_suppression
        assert host.__dict__==before
        records.append(dict(index=i,score_and_output_identity_preserved=True,hooks_restored_success_and_error=True,exact_source_grid_binarization=True))
    # All old pixels/scores/ranks stay fixed. Add confident negative evidence:
    # the query-global minimum moves the boundary and flips old background.
    old=torch.tensor([[.1,.1],[.2,.2]])
    extended=torch.cat([old,torch.full((1,2),-.5)],dim=0)
    original=host._binarize_response(old,target_hw=tuple(old.shape))
    changed=host._binarize_response(extended,target_hw=tuple(extended.shape))
    assert torch.equal(original,torch.tensor([[False,False],[True,True]]))
    assert changed[:2].all() and not changed[2].any()
    result=dict(state='CPU_NATIVE_DECISION_TRACE_PASSED',cases=records,
                exact_source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
                source_hashes={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__),root/'tics/native_decision_trace.py',a.source]},
                counterexample=dict(old_grid=old.tolist(),new_grid=extended.tolist(),
                    old_mask=original.tolist(),new_mask=changed.tolist(),
                    previous_background_flips_to_foreground=int((changed[:2]&~original).sum()),
                    same_existing_scores_and_ranks=True,real_image_prevalence_or_task_gain_measured=False),
                real_encoder_or_CRF_executed=False,CUDA_initialized=torch.cuda.is_initialized(),
                qualification='Exact scalar source behavior and synthetic hook API, not actual installed full-source end-to-end replay or a new method')
    if a.out:
        if a.out.exists():raise ValueError('Preserve prior receipt')
        a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['state','CUDA_initialized','real_encoder_or_CRF_executed']}))


if __name__=='__main__':main()
