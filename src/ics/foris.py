"""Complete public FoRIS entry, extracted from the previous extent runner.

The host configuration and stage observation are preserved; the failed extent
arms, zooms, oracle selection and synthetic experiment CLI are not included.
"""
import sys
from contextlib import contextmanager

HOOKS = dict(_extract_features="raw", _part1_positional_debias="deb", _reference_contrastive_prototypes="reference_prototypes", _part2_background_suppression="s2",
             _part3_clustering="s3", _binarize_response="pre")

@contextmanager
def observe(host):
    """Read-only taps on the source stages; every original output is returned unchanged."""
    got, saved = {}, {n: (n in host.__dict__, host.__dict__.get(n)) for n in HOOKS}

    def wrap(name, fn):
        def call(*a, **k):
            if name == '_reference_contrastive_prototypes':
                # The last call is source stage 2: retain its actual reference
                # token set and Boolean mask for the representation controls.
                got['reference_features'] = a[0]
                got['reference_fg'] = a[1]
            out = fn(*a, **k)
            if name == "_binarize_response":
                got["score"] = a[0] if a else k["score_hw"]
            got[HOOKS[name]] = out[0] if isinstance(out, tuple) else out
            return out
        return call
    try:
        for n in HOOKS:
            setattr(host, n, wrap(n, getattr(host, n)))
        yield got
    finally:
        for n, (had, old) in saved.items():
            setattr(host, n, old) if had else delattr(host, n)



def run_foris(host, ref_img, ref_mask, tgt_img):
    """The public entry: set_reference / set_target / segment. Returns the final mask and what was observed."""
    try:
        host.set_reference(ref_img, ref_mask)
        host.set_target(tgt_img)
        mask_model, tgt = host._ref_masks[0].clone(), host._tgt_image.clone()
        with observe(host) as got:
            out = host.segment()
        return out.reshape(tgt.shape[-2:]).bool().clone(), got, mask_model, tgt
    finally:  # a failed call must not leave a reference behind for the next one
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None



def build_host(man, device="cuda", foris_root=None, *, weights=None, mask_refiner="crf"):
    import torch
    sys.path.insert(0, str(foris_root or man["foris_root"]))
    import models.foris as foris
    from .data import TimmDINOv3
    from .native_basis import reuse_native_basis

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda":
        import os
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DEMO4_GPU_FRAC", 0.3)))
    encoder = TimmDINOv3(weights).to(device).eval().requires_grad_(False)
    with reuse_native_basis(foris.FoRIS, man.get("projection_basis")):
        return foris.FoRIS(encoder=encoder, image_size=1024, svd_components=500, tau=0.6, mask_refiner=mask_refiner,
                           resize_to_orig_size=False, device=device).eval().requires_grad_(False)
