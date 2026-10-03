"""Full-rank angular geometry inside original FoRIS Part2 scoring only.

No replacement likelihood, posterior, gain, threshold or downstream feature
geometry is introduced. The caller owns the reference-only geometry fit.
"""
import torch
from .reference_nuisance_geometry import transform_native_maps


@torch.no_grad()
def native_geometry_part2(host, native_part1, native_part2_tuple, geometry,
                          reference_mask, grid, *, native_callback=None):
    """Recompute native score/sf/sbn, retaining original mu_fg and target.

    native_part1 is the cached native [1,2,C,H,W] reference+query map.
    reference_mask is [1,image_H,image_W]; grid=(H,W). An exact identity
    geometry returns the SAME cached tuple object, without a scoring call.
    For a nonidentity geometry, the original Part2 is called once with the
    transformed maps and unchanged native settings/mask. Its final two tuple
    fields are replaced by the original cached objects for Stage3/4.

    Default callback is bound from the host CLASS, bypassing instance-level
    frozen-replay hooks. Alternatively provide the original bound callback
    saved before entering replay. This function installs or mutates no hooks.
    """
    if not isinstance(geometry, torch.Tensor) or geometry.ndim != 2 or geometry.shape[0] != geometry.shape[1]:
        raise ValueError('square full-rank angular geometry required')
    if not torch.isfinite(geometry).all():
        raise ValueError('finite angular geometry required')
    identity = torch.eye(len(geometry), device=geometry.device, dtype=geometry.dtype)
    if torch.equal(geometry, identity):
        return native_part2_tuple
    # A native failure has no score contract to replace; preserve its fallback.
    if native_part2_tuple is None:
        return None
    if not isinstance(native_part2_tuple, tuple) or len(native_part2_tuple) != 5:
        raise ValueError('original native Part2 five-tuple required')
    if not isinstance(native_part1, torch.Tensor) or native_part1.ndim != 5:
        raise ValueError('native Part1 [B,T,C,H,W] map required')
    if native_part1.shape[:2] != (1, 2) or native_part1.shape[2] != len(geometry):
        raise ValueError('single reference/query pair in common native descriptor coordinates required')
    if len(grid) != 2 or tuple(int(v) for v in grid) != tuple(native_part1.shape[-2:]):
        raise ValueError('native grid must match cached maps')
    if not isinstance(reference_mask, torch.Tensor) or reference_mask.ndim != 3 or reference_mask.shape[0] != 1:
        raise ValueError('reference mask must be [1,image_H,image_W]')
    transformed = transform_native_maps(native_part1, geometry)
    if native_callback is None:
        # Never read an instance's frozen Part2 replacement as the algorithm.
        native_callback = type(host)._part2_background_suppression.__get__(host, type(host))
    changed = native_callback(transformed, ref_masks=reference_mask.unsqueeze(1),
                              n_refs=1, h=int(grid[0]), w=int(grid[1]))
    if changed is None:
        raise ArithmeticError('transformed native Part2 failed after valid original Part2')
    if not isinstance(changed, tuple) or len(changed) != 5:
        raise ValueError('original scoring callback must return the native five-tuple')
    if any(v.shape != tuple(native_part1.shape[-2:]) or not torch.isfinite(v).all() for v in changed[:3]):
        raise ArithmeticError('native Part2 score/sf/sbn violated original grid/finite contract')
    return changed[:3] + (native_part2_tuple[3], native_part2_tuple[4])
