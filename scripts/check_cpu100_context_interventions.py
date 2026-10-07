#!/usr/bin/env python3
"""Declared toy encoders and physical mapping audits; zero real DINO forwards."""
from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import label

from ics.cpu100.common import Episode, prototype_margin, unit, sha
from ics.cpu100.encoder import CachedCPUEncoder
from ics.cpu100 import context_interventions as module

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"evidence/local/cpu100_20261006/reports/context_interventions"


class ToyPixels:
    """Pixel-derived nuisance/interaction function; not a DINO impersonation.

    No query/reference label, mask, Episode or target coordinate is accepted.
    Red-versus-blue is a stipulated visible class trait. Brightness and scene
    arrangement are different observable features, not privileged identities.
    """
    def __init__(self,mode):self.mode=mode
    def __call__(self,image):
        colour=image.reshape(64,16,64,16,3).mean((1,3))
        trait=np.where(colour[...,0]>colour[...,2],1.,-1.)
        bright=(colour.sum(-1)-190)/50
        def stimulus(slot):
            window=image[768:1024,slot*512:(slot+1)*512].astype(float)
            difference=window[...,0]-window[...,2];selected=np.abs(difference)>30
            return float(np.sign(difference[selected].mean())) if selected.any() else 0.
        a,b=stimulus(0),stimulus(1)
        out=np.zeros((64,64,1024),np.float32);out[...,0]=1
        if self.mode.startswith("remote"):
            key=trait if self.mode=="remote" else np.sign(bright)
            out[...,1]=8*bright;out[...,2]=key*(a+b)
        elif self.mode.startswith("competition"):
            key=trait if self.mode!="competition_echo" else np.sign(bright)
            interaction=0 if self.mode=="competition_additive" else key*a*b
            out[...,1]=8*bright;out[...,2]=3*bright*(a+b)+interaction
        elif self.mode.startswith("external"):
            out[...,1]=100*bright
            quadrants=(bright[:32,:32].mean(),bright[:32,32:].mean(),
                       bright[32:,:32].mean(),bright[32:,32:].mean())
            for k,(y,x) in enumerate(((0,0),(0,32),(32,0),(32,32))):
                context=sum((j+1)*value for j,value in enumerate(quadrants) if j!=k)
                local=trait[y:y+32,x:x+32]
                if self.mode=="external_style_shift":
                    beta=np.where(bright[y:y+32,x:x+32]>0,.01,1.)
                else:beta=np.where(local>0,.01,1.)
                out[y:y+32,x:x+32,2]=beta*context
        elif self.mode=="orbit":
            yy,xx=np.indices((64,64));out[...,1]=trait
            out[...,2]=2*np.sin(2*np.pi*(xx+yy+1)/64)
        elif self.mode=="orbit_seam":
            # Entire object continuity, derived only from visible colour pixels.
            # A component split at an image boundary loses its semantic key.
            components,n=label(trait>0);key=trait.copy()
            for component in range(1,n+1):
                part=components==component
                if part[0].any() or part[-1].any() or part[:,0].any() or part[:,-1].any():key[part]=-1
            out[...,1]=key
        else:raise ValueError(self.mode)
        return out


def scene(mode):
    r=np.empty((1024,1024,3),np.uint8);q=r.copy();mask=np.zeros((1024,1024),bool);truth=mask.copy()
    r[:]=(10,10,80);q[:]=(30,30,230)
    if mode=="orbit":rb=(0,128,128,256);qb=(0,640,128,768)
    elif mode=="orbit_seam":rb=(256,256,384,384);qb=(480,480,544,544)
    elif mode.startswith("external"):rb=qb=(128,640,384,896)
    else:rb=(128,128,384,384);qb=(512,512,768,768)
    t,l,b,rr=rb;r[t:b,l:rr]=(230,30,30);mask[t:b,l:rr]=True
    t,l,b,rr=qb;q[t:b,l:rr]=(80,10,10);truth[t:b,l:rr]=True
    if mode=="external_style_shift":
        # Different legitimate background composition; target pixels stay fixed.
        q[512:1024,:512]=(15,15,100)
        q[512:1024,512:1024]=(25,25,180)
    return r,q,mask,truth


def episode(mode):
    r,q,mask,truth=scene(mode)
    binding=dict(synthetic_encoder=True,synthetic_function=mode,
                 native_output_dtype="float32",output_shape=[64,64,1024],
                 evidence="toy pixel-function only; no DINO weights or real forwards")
    encoder=CachedCPUEncoder(ToyPixels(mode),binding,max_cached_views=16)
    raw_r,raw_q=encoder(r),encoder(q)
    wf=mask.reshape(64,16,64,16).mean((1,3)).ravel()
    geometry=dict(view_side=1024,resized_hw=[1024,1024],padding_top_left=[0,0],original_hw=[1024,1024])
    ep=Episode(q=unit(raw_q.reshape(-1,1024)),r=unit(raw_r.reshape(-1,1024)),wf=wf,
               wvalid=np.ones(4096),q_hw=(64,64),r_hw=(64,64),q_valid=np.ones(4096),
               query_geometry=geometry,reference_geometry=geometry,original_shape=(1024,1024),
               r_rgb=r,q_rgb=q,reference_mask=mask,
               producer=dict(kind="cpu100_declared_synthetic_encoder",model_input_side=1024),
               source_id="toy_"+mode)
    for value in (ep.q,ep.r,ep.wf,ep.wvalid,ep.q_valid,ep.q_rgb,ep.r_rgb,ep.reference_mask):value.setflags(write=False)
    module.configure_encoder(encoder,encoder.binding)
    truth=truth.reshape(64,16,64,16).mean((1,3))>.5
    return ep,encoder,truth


def account(margin,truth):
    if not np.isfinite(margin).all():raise FloatingPointError("Nonfinite actual complete margin")
    pred=np.asarray(margin)>0
    tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth));fn=int(np.sum(~pred&truth))
    return dict(tp=tp,fp=fp,fn=fn,iou=tp/(tp+fp+fn) if tp+fp+fn else 0)


def evaluate(mode,primary,control_prefix):
    ep,encoder,truth=episode(mode);start=time.perf_counter()
    result=module.METHODS[primary](ep)
    entries={primary:dict(**account(result.margin,truth),info=result.info)}
    for key,fn in module.CONTROLS.items():
        if key.startswith(control_prefix):
            other=fn(ep);entries[key]=dict(**account(other.margin,truth),info=other.info)
    with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
        entries["native_prototype_control"]=account(prototype_margin(ep),truth)
    stats=encoder.stats()
    assert not stats["real_encoder_execution"] and stats["new_encoder_forwards"]==0
    return dict(mode=mode,results=entries,stats=stats,seconds=time.perf_counter()-start)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(schema="CPU100_CONTEXT_TOY_ONLY_V1",real_DINO_forwards=0,quality_scope="constructed possibilities and failures, not DINO gain")
    cases=(
        ("remote","context_remote_reference_response","context_panel_"),
        ("remote_echo","context_remote_reference_response","context_panel_"),
        ("external","context_external_shuffle_sensitivity","context_external_"),
        ("external_style_shift","context_external_shuffle_sensitivity","context_external_"),
        ("competition","context_probe_nonadditive_competition","context_competition_"),
        ("competition_additive","context_probe_nonadditive_competition","context_competition_"),
        ("competition_echo","context_probe_nonadditive_competition","context_competition_"),
        ("orbit","context_patch_position_orbit","context_orbit_"),
        ("orbit_seam","context_patch_position_orbit","context_orbit_"),
    )
    report["cases"]=[evaluate(*args) for args in cases]
    rng=np.random.default_rng(41);rgb=rng.integers(0,256,size=(1024,1024,3),dtype=np.uint8)
    positions=((0,0),(0,512),(512,0),(512,512))
    checks=[]
    for k,(y,x) in enumerate(positions):
        edited=module.preserved_quadrant_view(rgb,k)
        checks.append(np.array_equal(edited[y:y+512,x:x+512],rgb[y:y+512,x:x+512]))
    assert all(checks)
    assert all(np.array_equal(np.roll(np.roll(rgb,512,axis=a),-512,axis=a),rgb) for a in (0,1))
    ep,encoder,truth=episode("competition")
    body=module._mask(ep,768);coverage=module._coverage(body)
    assert abs(coverage.sum()*256-body.sum())<1e-10
    # Primary-plus-controls order is deliberately hostile to a 16-view LRU.
    # Small result caching must prevent re-encoding evicted earlier groups.
    before=encoder.stats()["callback_forward_attempts"]
    for fn in module.METHODS.values():fn(ep)
    for fn in module.CONTROLS.values():fn(ep)
    after=encoder.stats()["callback_forward_attempts"]
    assert after-before<=28
    report["audits"]=dict(all_four_local_quadrants_byte_identical=checks,
        exact_rgb_roll_inverse=True,known_mask_main_area_error=float(abs(coverage.sum()*256-body.sum())),
        all_methods_then_controls_new_toy_callback_attempts=after-before,max_allowed=28,
        cache_stats=encoder.stats(),real_DINO_forwards=0)
    report["code_sha256"]=sha(ROOT/"src/ics/cpu100/context_interventions.py")
    (OUT/"toy_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    for case in report["cases"]:
        print(json.dumps(dict(mode=case["mode"],results={k:{x:v[x] for x in ("tp","fp","fn","iou")} for k,v in case["results"].items()})))
    print(json.dumps(dict(report=str(OUT/"toy_report.json"),audits=report["audits"])))


if __name__=="__main__":main()
