"""Original M29 fine128 -> work1024 threshold -> binary-original readout.

No64 compression, no geometry-crop before the work threshold, no third cut.
FP32 native score enters the exact original CPU100 NumPy bilinear arithmetic;
the source card specifies bilinear but no Torch kernel. Tile/operation grouping
is inherited read-only from render_exact_optim and sealed by the manifest.
"""
import time
import numpy as np
from .common import Result,readonly,array_hash
from .render_exact_optim import resize_exact,_axis,_sample_rows


def finish_m29(ep,score,method_id,info=None,*,block_rows=64):
    start=time.perf_counter();score=np.asarray(score,np.float32).copy()
    if score.shape!=(128,128) or not np.isfinite(score).all():raise ValueError('M29 needs a finite physical128 score field')
    # The fine field spans the recorded1024 encoder CANVAS. Reflected/padded
    # portions are already excluded by the inverse's physical validity operators.
    # Keep that canvas here; apply original-image geometry to the BINARY work
    # field only, rather than dropping padding or squeezing fine to64.
    work=resize_exact(score,(1024,1024),block_rows=block_rows,threshold=.5)
    g=ep.query_geometry
    if g:
        view=g['view_side'];sh,sw=g['resized_hw'];oy,ox=g['padding_top_left']
        if (view!=1024 or min(sh,sw)<=0 or min(oy,ox)<0 or oy+sh>view or ox+sw>view
                or any(float(x)!=int(x) for x in (sh,sw,oy,ox))):
            raise ValueError('Actual1024 physical resize/pad geometry required for M29 readout')
        if g.get('original_hw') is not None and tuple(g['original_hw'])!=tuple(ep.original_shape):
            raise ValueError('Original query H/W disagrees with physical geometry')
        # Literal continuous_original expression, retaining multiply/divide order.
        y=(oy+(np.arange(ep.original_shape[0])+.5)*sh/ep.original_shape[0])*work.shape[0]/view-.5
        x=(ox+(np.arange(ep.original_shape[1])+.5)*sw/ep.original_shape[1])*work.shape[1]/view-.5
        ya,xa=_axis(y,1024),_axis(x,1024);original=np.empty(ep.original_shape,bool);binary=work.astype(np.float32)
        for first in range(0,len(y),block_rows):
            last=min(len(y),first+block_rows);original[first:last]=_sample_rows(binary,ya,xa,first,last)>.5
    else:
        original=resize_exact(work.astype(np.float32),ep.original_shape,block_rows=block_rows,threshold=.5)
    metadata=dict(info or {},method_id=method_id,query_GT_read=False,complete_all_instances=True,
        renderer='M29 exact fine128 FP32-input NumPybilinear1024>.5 then physical binarybilinear-original>.5',
        method_renderer='original M29 two thresholds; never fine→64',
        source_renderer_parity='original M29 threshold order and physical padding preserved',
        source_threshold_stages=['continuous128 to canvas1024, strict>.5','binarycanvas1024 to original through recorded geometry, strict>.5'],
        renderer_input_dtype='float32',renderer_arithmetic='exact original NumPy CPU100 bilinear operation grouping; float64 coordinates/intermediates',
        field_hw=[128,128],work_hw=[1024,1024],original_hw=list(ep.original_shape),
        fine_field_array_sha256=array_hash(score),work_mask_array_sha256=array_hash(work),original_mask_array_sha256=array_hash(original),
        renderer_seconds=time.perf_counter()-start,no64_compression=True,physical_padding_preserved=True)
    return Result(score,.5,readonly(original),metadata)
