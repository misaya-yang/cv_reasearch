"""Execute prepared candidates on shared cached inputs, without GT.

This module is an inference backend of run_cpu_feature_candidates.py, not a
additional segmentation method. All individual candidate contracts remain fixed.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np


METHODS=('adjacency','huber','color_bottleneck','constellation','reference_shape','reference_covariance','query_recurrence','reference_prior_shift','reference_quadratic')
AVAILABLE_METHODS=METHODS+('pro_reference_relations','reference_hull','constellation_local','reference_triplet_relations','reference_absorption','reference_gaussian_density','mean_rgb_potts')
METHOD_METADATA={name:dict(kind='independent_candidate',independent_method_increment=1)
                 for name in AVAILABLE_METHODS}
METHOD_METADATA.update(
    constellation_local=dict(kind='revision',version_id='reference_constellation_local_v2',
                             revised_family='held_out_reference_landmark_consensus_v1',
                             independent_method_increment=0),
    reference_gaussian_density=dict(kind='revision',version_id='reference_density_full_feature_bounded_v2',
                                   revised_family='reference_score_density_query_prior_v1',
                                   independent_method_increment=0),
    mean_rgb_potts=dict(kind='strong_simple_control',version_id='locked_MEAN_RGB128_boundary_Potts_control_v1',
                       independent_method_increment=0))
INDEPENDENT_METHOD_INCREMENT={name:metadata['independent_method_increment']
                              for name,metadata in METHOD_METADATA.items()}


def mean_base(inputs,graph_dtype):
    """Preserve the normalized sparse graph's original dtype before the CG solve."""
    from scipy import sparse
    from scipy.sparse.linalg import cg
    y,a,left,right,weights=inputs
    n=y.size
    graph=sparse.coo_matrix((np.r_[weights,weights].astype(graph_dtype),
                            (np.r_[left,right],np.r_[right,left])),shape=(n,n)).tocsr()
    degree=np.asarray(graph.sum(1)).ravel()
    matrix=sparse.diags(a.ravel())+16*(sparse.diags(degree)-graph)
    field,status=cg(matrix,a.ravel()*y.ravel(),x0=y.ravel(),rtol=1e-7,atol=1e-9,maxiter=300)
    if status:raise RuntimeError(f'Original MEAN control did not converge: {status}')
    residual=float(np.linalg.norm(matrix@field-a.ravel()*y.ravel())/max(np.linalg.norm(a.ravel()*y.ravel()),1e-12))
    return field.reshape(y.shape).astype(np.float32),dict(relative_residual=residual,
                                                         graph_storage_dtype=str(graph.dtype),
                                                         source='locked_MEAN_original_sparse_dtype_then_CG_float64')


def load_rgb(row):
    from ..experiment import sha
    if row.get('query_rgb_export'):
        path=Path(row['query_rgb_export'])
        with np.load(path,allow_pickle=False) as data:rgb=data['rgb'].copy()
        origin=dict(source='existing_RGB128_packet',path=str(path),sha256=sha(path),keys_read=['rgb'])
    else:
        from PIL import Image
        path=Path(row['query_image_export'])
        with Image.open(path) as image:
            canonical=image.convert('RGB').resize((1024,1024),Image.Resampling.BILINEAR)
            rgb=np.asarray(canonical.resize((128,128),Image.Resampling.BILINEAR)).copy()
        origin=dict(source='query_RGB_to_canonical1024_to_RGB128_PIL_bilinear',path=str(path),sha256=sha(path))
    if rgb.shape!=(128,128,3):raise ValueError('Require the fixed RGB128 query cue')
    return rgb,origin


def one_episode(row,run,config):
    import torch
    import torch.nn.functional as F
    from ..experiment import load_inputs,render,sha
    from . import huber_graph
    started,cpu_started=time.perf_counter(),time.process_time()
    (q,r,cov,score),source=load_inputs(config['root'],row)
    # Cached features and reference labels are shared; target labels are not opened.
    graph_started=time.perf_counter()
    graph_inputs,origin=huber_graph.make_mean_inputs(q,r,cov,score)
    original,base_info=mean_base(graph_inputs,origin.get('graph_storage_dtype','float64'))
    base=original
    if row.get('base_field_export'):
        path=Path(row['base_field_export'])
        with np.load(path,allow_pickle=False) as data:base=data[row['base_field_key']].astype(np.float32)
        if base.shape!=(64,64) or not np.isfinite(base).all():raise ValueError('Invalid existing MEAN field')
        difference=float(np.max(np.abs(base-original)))
        if difference>1e-6:raise ValueError(f'Existing MEAN field not consistent with bound graph: maxdiff={difference}')
        base_info.update(existing_path=str(path),existing_sha256=sha(path),maximum_field_difference=difference)
    shared_seconds=time.perf_counter()-graph_started
    fields={'mean.control':base,'clipped_mean.control':np.clip(base,0,1)}
    masks={key:render(value) for key,value in fields.items()}
    audits={}
    for name in config['prepared_methods']:
        method_started=time.perf_counter()
        if name=='adjacency':
            from .reference_adjacency import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            fields['adjacency']=result['field']
            masks['adjacency']=F.interpolate(torch.from_numpy(result['token_mask'].astype(np.float32))[None,None],
                                             (1024,1024),mode='nearest')[0,0].numpy()>.5
            fields['adjacency_bilinear.control']=result['field']
            masks['adjacency_bilinear.control']=render(result['field'])
            masks['mean_nearest.control']=F.interpolate(torch.from_numpy((base>.5).astype(np.float32))[None,None],
                                                        (1024,1024),mode='nearest')[0,0].numpy()>.5
            info=result['info']
        elif name=='huber':
            method_config=huber_graph.Config(**config['method_configs'][name])
            field,info=huber_graph.predict(*graph_inputs,cfg=method_config)
            fields['huber']=field;masks['huber']=render(field)
            boxed,boxed_info=huber_graph.quadratic_control(*graph_inputs,graph_lambda=method_config.graph_lambda)
            fields['boxed_quadratic.control']=boxed;masks['boxed_quadratic.control']=render(boxed)
            fields['pregraph.control']=graph_inputs[0];masks['pregraph.control']=render(graph_inputs[0])
            fields['original_quadratic.control']=original;masks['original_quadratic.control']=render(original)
            info['boxed_control']=boxed_info
        elif name=='color_bottleneck':
            from .color_bottleneck import Config,predict
            rgb,rgb_origin=load_rgb(row)
            result=predict(rgb,base,Config(**config['method_configs'][name]))
            fields['color_bottleneck']=result['field'];masks['color_bottleneck']=render(result['field'])
            for control,value in (('color_same_resize.control',result['intermediate_base']),
                                  ('color_clipped_resize.control',np.clip(result['intermediate_base'],0,1)),
                                  ('color_only.control',result['color_field'])):
                fields[control]=value;masks[control]=render(value)
            info=result['info'];info['RGB_input']=rgb_origin
        elif name=='constellation':
            from .reference_constellation import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,value in (('constellation',result['field']),('constellation_bag.control',result['bag_field']),
                              ('constellation_prior.control',result['prior'])):
                fields[arm]=value;masks[arm]=render(value)
            info=result['info']
        elif name=='reference_shape':
            from .reference_shape import Config,predict
            result=predict(np.asarray(q),cov,base,Config(**config['method_configs'][name]),
                           reference_hw=row['support_image_hw'],query_hw=row['query_image_hw'])
            mask=result['token_mask'].astype(np.float32)
            generic=result['compactness_control'].astype(np.float32)
            fields['reference_shape']=mask
            masks['reference_shape']=F.interpolate(torch.from_numpy(mask)[None,None],(1024,1024),mode='nearest')[0,0].numpy()>.5
            fields['shape_generic_square.control']=generic
            masks['shape_generic_square.control']=F.interpolate(torch.from_numpy(generic)[None,None],(1024,1024),mode='nearest')[0,0].numpy()>.5
            fields['shape_bilinear.control']=mask;masks['shape_bilinear.control']=render(mask)
            masks['mean_nearest.control']=F.interpolate(torch.from_numpy((base>.5).astype(np.float32))[None,None],
                                                        (1024,1024),mode='nearest')[0,0].numpy()>.5
            info=result['info']
        elif name=='reference_covariance':
            from .reference_covariance import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            mask=result['token_mask'].astype(np.float32)
            scalar=result['trace_control'].astype(np.float32)
            fields['reference_covariance']=mask
            masks['reference_covariance']=F.interpolate(torch.from_numpy(mask)[None,None],(1024,1024),mode='nearest')[0,0].numpy()>.5
            fields['covariance_trace.control']=scalar
            masks['covariance_trace.control']=F.interpolate(torch.from_numpy(scalar)[None,None],(1024,1024),mode='nearest')[0,0].numpy()>.5
            fields['covariance_bilinear.control']=mask;masks['covariance_bilinear.control']=render(mask)
            fields['covariance_mode_margin.control']=result['mode_margin_field']
            masks['covariance_mode_margin.control']=F.interpolate(torch.from_numpy((result['mode_margin_field']>.5).astype(np.float32))[None,None],
                                                                 (1024,1024),mode='nearest')[0,0].numpy()>.5
            masks['mean_nearest.control']=F.interpolate(torch.from_numpy((base>.5).astype(np.float32))[None,None],
                                                        (1024,1024),mode='nearest')[0,0].numpy()>.5
            info=result['info']
        elif name=='query_recurrence':
            from .query_recurrence import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,value in (('query_recurrence',result['field']),
                              ('recurrence_all_seed.control',result['all_seed_control']),
                              ('recurrence_single_seed.control',result['single_seed_control'])):
                fields[arm]=value;masks[arm]=render(value)
            info=result['info']
        elif name=='reference_prior_shift':
            from .reference_prior_shift import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,value in (('reference_prior_shift',result['field']),
                              ('prior_balanced.control',result['balanced_control']),
                              ('prior_reference_fraction.control',result['reference_prior_control']),
                              ('prior_margin.control',result['margin_control'])):
                fields[arm]=value;masks[arm]=render(value)
            info=result['info']
        elif name=='reference_quadratic':
            from .reference_quadratic import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,key in (('reference_quadratic','field'),('quadratic_linear.control','linear_control'),
                            ('quadratic_homogeneous.control','homogeneous_control'),
                            ('quadratic_kernel_mean.control','kernel_mean_control'),
                            ('quadratic_nearest.control','nearest_control'),
                            ('quadratic_subspace.control','subspace_control')):
                fields[arm]=result[key];masks[arm]=render(result[key])
            info=result['info']
        elif name=='pro_reference_relations':
            from .pro_reference_relations import Config,predict
            # Pro M2's unary is the original packet score, never the MEAN field.
            result=predict(np.asarray(q),np.asarray(r),cov,score,Config(**config['method_configs'][name]))
            for arm,key in (('pro_reference_relations','field'),
                            ('pro_relations_zero.control','zero_control'),
                            ('pro_relations_positive.control','positive_control'),
                            ('pro_relations_absolute.control','absolute_control'),
                            ('pro_relations_pair_independent.control','pair_independent_control'),
                            ('pro_relations_block.control','block_control')):
                fields[arm]=result[key];masks[arm]=render(result[key])
            info=result['info']
        elif name=='reference_triplet_relations':
            from .reference_triplet_relations import Config,predict
            # Triplet's unary is the original packet score, never the MEAN field.
            result=predict(np.asarray(q),np.asarray(r),cov,score,Config(**config['method_configs'][name]),
                           include_pair_control=True)
            for arm,key in (('reference_triplet_relations','field'),
                            ('triplet_zero.control','zero_control'),
                            ('triplet_absolute.control','absolute_control'),
                            ('triplet_no_third.control','no_third_control'),
                            ('triplet_pro_m2.control','pair_m2_control')):
                fields[arm]=result[key];masks[arm]=render(result[key])
            info=result['info']
        elif name=='reference_hull':
            from .reference_hull import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,key in (('reference_hull','field'),('hull_nearest.control','nearest_control'),
                            ('hull_centroid.control','centroid_control'),('hull_subspace.control','subspace_control'),
                            ('hull_affine.control','affine_control')):
                fields[arm]=result[key];masks[arm]=render(result[key])
            # Bounds are full arrays; only the predictor's scalar info enters JSON.
            info=result['info']
        elif name=='constellation_local':
            from .reference_constellation_local import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,key in (('constellation_local','field'),
                            ('constellation_local_global_v1.control','global_field'),
                            ('constellation_local_bag.control','bag_field'),
                            ('constellation_local_prior.control','prior')):
                fields[arm]=result[key];masks[arm]=render(result[key])
            info=result['info']
        elif name=='reference_absorption':
            from .reference_absorption import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,key in (('reference_absorption','field'),
                            ('absorption_nearest.control','nearest_control'),
                            ('absorption_kernel.control','kernel_control'),
                            ('absorption_one_hop.control','one_hop_control'),
                            ('absorption_one_step.control','one_step_control'),
                            ('absorption_component.control','component_control'),
                            ('absorption_full_harmonic.control','full_harmonic_control')):
                fields[arm]=result[key];masks[arm]=render(result[key])
            info=result['info']
        elif name=='mean_rgb_potts':
            from .mean_rgb_potts import Config,predict
            rgb,rgb_info=load_rgb(row)
            result=predict(np.asarray(q),np.asarray(r),cov,base,rgb,
                           original_shape=row['query_image_hw'],
                           cfg=Config(**config['method_configs'][name]))
            # Keep the binary 128-grid cut: no 128->64 compression or second
            # continuous interpretation. These masks use the Pro renderer.
            fields['mean_rgb_potts']=result['field']
            masks['mean_rgb_potts']=result['mask_work']
            fields['mean_rgb_unary.control']=result['unary_control']
            masks['mean_rgb_unary.control']=result['unary_mask_work']
            info=result['info'];info['rgb_producer']=rgb_info
        elif name=='reference_gaussian_density':
            from .reference_gaussian_density import Config,predict
            result=predict(np.asarray(q),np.asarray(r),cov,base,Config(**config['method_configs'][name]))
            for arm,key in (('reference_gaussian_density','field'),
                            ('gaussian_density_polynomial.control','polynomial_control'),
                            ('gaussian_density_quadratic_ridge.control','quadratic_ridge_control'),
                            ('gaussian_density_uniform.control','uniform_control'),
                            ('gaussian_density_nearest.control','nearest_control'),
                            ('gaussian_density_centroid.control','centroid_control'),
                            ('gaussian_density_standalone.control','standalone_control')):
                fields[arm]=result[key];masks[arm]=render(result[key])
            info=result['info']
        else:raise ValueError('Unrecognized prepared method')
        info['candidate_and_controls_seconds']=time.perf_counter()-method_started
        audits[name]=info
    occurrence=row['occurrence_id']
    pred_path=run/'predictions'/f'{occurrence}.npz'
    field_path=run/'fields'/f'{occurrence}.npz'
    np.savez_compressed(pred_path,**{name:np.packbits(mask) for name,mask in masks.items()})
    np.savez_compressed(field_path,**fields)
    import json
    import resource
    import sys
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt=dict(occurrence_id=occurrence,source_key=row['key'],inputs=source,
                 base=base_info,pregraph=origin,shared_pregraph_and_base_seconds=shared_seconds,
                 methods=audits,prediction_sha256=sha(pred_path),field_sha256=sha(field_path),
                 wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,
                 process_peak_rss_bytes=int(rss if sys.platform=='darwin' else rss*1024),
                 query_gt_opened=False,new_encoder_forwards=0,pid=__import__('os').getpid(),
                 independent_candidate_methods=sum(INDEPENDENT_METHOD_INCREMENT.get(name,1) for name in config['prepared_methods']),
                 candidate_versions=len(config['prepared_methods']),
                 selected_algorithm_rows=len(config['prepared_methods']),
                 method_metadata={name:METHOD_METADATA[name] for name in config['prepared_methods']},
                 note='shared execution backend; not another segmentation method')
    (run/'receipts'/f'{occurrence}.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt
