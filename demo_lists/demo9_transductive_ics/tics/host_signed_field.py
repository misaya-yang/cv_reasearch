"""Read-only exact signed-field adapters for existing bilinear host interfaces.

FoRIS source `_binarize_response`: min/max, native-dtype interpolation, >.5.
INSID3 source `_finalize_mask`: binary patch mask -> float interpolation >.5.
Requires NO CRF and NO original-size resize; full CRF remains a separate strong
baseline. This adapter never replaces native output, updates weights, or reads GT.
Real pretrained-host parity is a future E8 obligation, not established here.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
import torch
import torch.nn.functional as F


@dataclass
class SignedFieldCapture:
    fields: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    native_calls: int = 0
    restored: bool = False

    def only_field(self):
        if self.errors or len(self.fields) != 1:
            raise RuntimeError(f'Invalid host signed-field capture: {self.errors}, calls={self.native_calls}')
        return self.fields[0]


@contextmanager
def native_signed_field(host, kind):
    if kind not in ('insid3', 'foris'):
        raise ValueError('Explicit supported host required')
    if host.mask_refiner != 'bilinear' or host.resize_to_orig_size:
        raise ValueError('Signed-field correction only defined for the declared bilinear/model-size contract')
    capture=SignedFieldCapture()
    name='_binarize_response' if kind=='foris' else '_finalize_mask'
    original=getattr(host,name)
    owned=name in host.__dict__
    previous=host.__dict__.get(name)
    def wrapped(this,*args,**kwargs):
        result=original(*args,**kwargs)
        capture.native_calls+=1
        try:
            if kind=='foris':
                score=args[0] if args else kwargs['score_hw']
                target=kwargs['target_hw']
                normalized=score-score.min()
                normalized=normalized/normalized.max().clamp_min(1e-6)
                value=F.interpolate(normalized[None,None],size=target,mode='bilinear',align_corners=False)[0,0]-.5
            else:
                mask=args[0] if args else kwargs['mask']
                image=args[1] if len(args)>1 else kwargs['tgt_image']
                target=image.shape[-2:]
                value=F.interpolate(mask.reshape(1,1,*mask.shape[-2:]).float(),size=target,
                                    mode='bilinear',align_corners=False)[0,0]-.5
            if not torch.equal(value>0,result):
                raise RuntimeError('Signed field does not exactly reproduce unchanged native mask')
            capture.fields.append(value.detach().float().clone())
        except Exception as exc:
            # Side observation errors must not change the original host return.
            capture.errors.append(repr(exc))
        return result
    setattr(host,name,MethodType(wrapped,host))
    try:
        yield capture
    finally:
        if owned:setattr(host,name,previous)
        else:delattr(host,name)
        capture.restored=True
