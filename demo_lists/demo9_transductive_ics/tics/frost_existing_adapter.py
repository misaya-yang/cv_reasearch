"""Load the complete existing FROST control without a hub factory/download.

This is the unchanged source head with an injected existing timm DINO wrapper,
not official-hub numerical reproduction or a new method. Do not transplant
FoRIS rank500 features or batch2 cache into FROST's rank250/batch3 path.
"""
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import sys
import types


def existing_frost_modules(source_root):
    root=Path(source_root).resolve()/'frost'
    required=[root/name for name in ['model.py','density.py','data.py','encoder.py']]
    if any(not p.is_file() for p in required):raise FileNotFoundError('Existing complete FROST source required')
    digests={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in required}
    token=hashlib.sha256(''.join(digests.values()).encode()).hexdigest()[:16]
    name='_demo9_existing_frost_'+token
    if name not in sys.modules:
        package=types.ModuleType(name);package.__path__=[str(root)]
        sys.modules[name]=package
    return importlib.import_module(name+'.model'),importlib.import_module(name+'.encoder'),digests


def construct_complete_frost(source_root, frozen_encoder, *, image_size=1024, device='cuda'):
    """Call source constructor, including its own BLACK/SVD250 calculation.

    No model is loaded here; caller already owns the existing eval FP32 frozen
    DINO weights and a permitted compute phase. Existing images/masks only.
    Constructor's noise forward and predict's [S,flip(S),Q] are real costs.
    """
    import torch
    if frozen_encoder.training or any(p.requires_grad for p in frozen_encoder.parameters()):
        raise ValueError('Existing backbone must be eval and frozen')
    parameters=list(frozen_encoder.parameters())
    if not parameters or any(p.dtype!=torch.float32 for p in parameters):
        raise ValueError('Prepared source-default FP32 backbone required, not undeclared AMP')
    if not callable(getattr(frozen_encoder,'get_intermediate_layers',None)):
        raise ValueError('Existing wrapper must expose get_intermediate_layers')
    source,encoder,hashes=existing_frost_modules(source_root)
    extractor=encoder.DINOv3FeatureExtractor(frozen_encoder,image_size=image_size,patch_size=16)
    model=source.FROST(extractor,raw_encoder=frozen_encoder,image_size=image_size,
                       device=device,resize_to_orig_size=False).eval().requires_grad_(False)
    return model,dict(source_hashes=hashes,control='complete_existing_FROST',
        implementation='source_head_injected_existing_timm_backbone_adaptation',
        official_hub_numerical_parity_claimed=False,basis_rank=250,
        basis_computed_by_original_constructor=True,noise_encoder_cost_included=True,
        query_episode_batch='[support,flip_support,query] source B3',
        FoRIS_batch2_or_rank500_features_reused=False,source_hyperparameters_changed=False,
        pretrained_weights_downloaded=False)


def verify_frost_public_tensors(model,support,mask,query):
    """The production assertion, also executed by actual-source CPU preflight.

    Public FROST preserves query NCHW, unlike public FoRIS's CHW target.
    No squeezing or silent re-preprocessing; exact matched input values.
    """
    import torch
    expected={'_ref_images':support,'_ref_masks':mask,'_tgt_image':query}
    for field,value in expected.items():
        actual=getattr(model,field,None)
        if not isinstance(actual,torch.Tensor) or actual.shape!=value.shape or actual.dtype!=value.dtype or actual.device!=value.device or not torch.equal(actual,value):
            raise RuntimeError('Complete FROST public tensor mismatch: '+field)
    if support.ndim!=4 or query.ndim!=4 or query.shape[0]!=1 or mask.ndim!=3:
        raise RuntimeError('Complete FROST public NCHW/mask interface not satisfied')
    return True


@contextmanager
def capture_frost_finalization(model):
    """Copy continuous posterior/candidate before source clears its stash."""
    packet=dict(state='CAPTURING',query_GT_received=False)
    existed='_finalize_mask' in model.__dict__;previous=model.__dict__.get('_finalize_mask')
    original=model._finalize_mask
    def finalize(mask,image):
        ell=getattr(model,'_post_ell',None);cand=getattr(model,'_post_cand',None)
        packet['continuous_density_available']=ell is not None
        packet['pre_refinement_mask']=mask.detach().cpu().clone()
        packet['density_tau']=float(getattr(model,'_post_tau',0.))
        if ell is not None:packet['continuous_posterior']=ell.detach().cpu().clone()
        if cand is not None:packet['dilated_candidate']=cand.detach().cpu().clone()
        result=original(mask,image)
        packet['final_mask']=result.detach().cpu().clone()
        return result
    try:
        model._finalize_mask=finalize
        yield packet
        packet['state']='CAPTURED'
    except BaseException:
        packet['state']='ERROR';raise
    finally:
        if existed:model._finalize_mask=previous
        else:delattr(model,'_finalize_mask')
