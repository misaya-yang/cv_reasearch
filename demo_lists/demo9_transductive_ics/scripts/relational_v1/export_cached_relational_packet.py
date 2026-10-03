"""Export existing pinned q/r cache + extent packet to a clean method packet.

Reads only cached CPU tensors, score, source cov and optionally packed native.
Never imports capture/model runners or reads query truth. No feature generation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

CLEAN_FIELDS={'query_features','source_features','query_xy','source_xy','query_valid','source_valid','source_coverage','native_patch_fg'}


def _hw(meta,key):
    value=meta.get(key)
    if not isinstance(value,list) or len(value)!=2 or any(type(x)!=int or x<1 for x in value):
        raise ValueError(f'{key} must be explicit positive [height,width]')
    return tuple(value)


def _coordinates(grid,working):
    h,w=grid;H,W=working
    y,x=np.meshgrid((np.arange(h)+.5)*H/h,(np.arange(w)+.5)*W/w,indexing='ij')
    return np.stack([x,y],axis=-1).astype(np.float64)


def export_packet(*,features_path,packet_path,metadata,native_patch_mode,out_path):
    """Two explicit native-region choices; neither is claimed as benchmark replay."""
    feature_path=Path(features_path);packet_path=Path(packet_path);out=Path(out_path)
    if native_patch_mode not in ('score_native','cached_native_coverage'):
        raise ValueError('choose score_native or cached_native_coverage explicitly')
    if out.suffix!='.npz' or out.exists() or out.with_suffix('.json').exists():
        raise ValueError('fresh .npz output and adjacent .json paths required')
    required=('pair_id','representation_id','feature_provenance','source_coverage_provenance','cache_alignment_provenance')
    if any(not isinstance(metadata.get(k),str) or not metadata[k].strip() for k in required):
        raise ValueError('nonempty provenance fields required: '+','.join(required))
    if metadata.get('mapping_contract')!='full_grid_regular_no_padding':
        raise ValueError('only caller-confirmed full regular source/query frames without padding are supported')
    if feature_path.stem!=packet_path.stem or feature_path.stem!=metadata['pair_id']:
        raise ValueError('feature/packet filename stems and explicit pair_id must agree')
    grid=_hw(metadata,'feature_grid_hw');source_hw=_hw(metadata,'source_work_hw')
    query_hw=_hw(metadata,'query_work_hw');_hw(metadata,'query_original_hw')
    # Never retry with unsafe pickle or non-CPU loading.
    feature=torch.load(feature_path,map_location='cpu',weights_only=True)
    if not isinstance(feature,dict) or set(feature)!={'q','r','debiased'} or type(feature['debiased'])!=bool:
        raise ValueError('expected pinned cache dictionary q,r,debiased; not a pool or arbitrary payload')
    arrays={}
    for name in ('q','r'):
        value=feature[name]
        if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float16 or value.device.type!='cpu'
                or value.ndim!=2 or value.shape[0]!=grid[0]*grid[1] or value.shape[1]<1):
            raise ValueError('q/r must be CPU float16 row-major [h*w,D] matching metadata')
        arrays[name]=value.float().numpy().reshape(*grid,value.shape[1])
    if arrays['q'].shape!=arrays['r'].shape:raise ValueError('pinned one-reference q/r shapes differ')
    with np.load(packet_path,allow_pickle=False) as packet:
        score=packet['score'];coverage=packet['cov']  # SOURCE annotation coverage
        if score.shape!=grid or score.dtype!=np.float32 or not np.isfinite(score).all():
            raise ValueError('score must be finite float32 feature grid, matching pinned writer')
        if coverage.shape!=grid or coverage.dtype.kind not in 'fiu' or not np.isfinite(coverage).all() or np.any((coverage<0)|(coverage>1)):
            raise ValueError('source cov must match feature grid and lie in [0,1]')
        if native_patch_mode=='score_native':
            score_tensor=torch.from_numpy(score.copy())
            shifted=score_tensor-score_tensor.min()
            normalized=shifted/shifted.max().clamp_min(1e-6)
            native=(normalized>.5).numpy()
            native_definition='cached raw score float32 -> native minmax floor1e-6 -> patch >.5; not finalized mask coverage'
        else:
            H,W=query_hw;h,w=grid
            if H%h or W%w:raise ValueError('coverage mode requires integer nonoverlapping grid cells; no guessed interpolation')
            bits=packet['native']
            if bits.dtype!=np.uint8 or bits.ndim!=1 or bits.size!=(H*W+7)//8:
                raise ValueError('packed native byte length/dtype must match explicit query_work_hw')
            unpacked=np.unpackbits(bits)
            if unpacked[H*W:].any():raise ValueError('nonzero packed native padding bits')
            mask=unpacked[:H*W].reshape(H,W)
            cov_native=mask.reshape(h,H//h,w,W//w).mean(axis=(1,3))
            native=cov_native>=.5
            native_definition='cached finalized native model mask, integer-cell patch foreground coverage >=.5'
    valid=lambda a:np.isfinite(a).all(-1) # zero/overflow rows also abstain inside core operator
    clean={'query_features':arrays['q'],'source_features':arrays['r'],
        'query_xy':_coordinates(grid,query_hw),'source_xy':_coordinates(grid,source_hw),
        'query_valid':valid(arrays['q']),'source_valid':valid(arrays['r']),
        'source_coverage':coverage.copy(),'native_patch_fg':np.asarray(native,dtype=bool)}
    assert set(clean)==CLEAN_FIELDS
    report={**metadata,'native_patch_mode':native_patch_mode,'native_patch_mask_definition':native_definition,
        'source_role_definition':'source cov>0 foreground, cov==0 background; not query evaluation coverage',
        'coordinate_system':'regular feature-cell centers in each supplied native working-frame pixel coordinates',
        'cache_tensor_format':'pinned q/r row-major fp16 -> float32; no extra encoding or preprocessing',
        'cache_debiased_flag':feature['debiased'],'query_GT_read':False,
        'cache_alignment_verified_by_adapter':False,'full_grid_mapping_confirmed_by_caller':True,
        'output_scope':'clean query-grid method input only; no original finalizer or full mask run',
        'baseline57_reproduction_claim':False,'model_calls':0,
        'input_files':{'features':str(feature_path),'packet':str(packet_path)}}
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**clean)
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',required=True);p.add_argument('--packet',required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--native-patch-mode',required=True,choices=('score_native','cached_native_coverage'))
    p.add_argument('--out',required=True);a=p.parse_args(argv)
    report=export_packet(features_path=a.features,packet_path=a.packet,
        metadata=json.loads(Path(a.metadata).read_text()),native_patch_mode=a.native_patch_mode,out_path=a.out)
    print(json.dumps({'output':a.out,'mode':report['native_patch_mode'],'model_calls':0,'query_GT_read':False,'full_mask_run':False}))

if __name__=='__main__':main()
