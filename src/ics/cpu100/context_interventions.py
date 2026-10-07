"""Physical CPU DINO context probes; unchanged main pixels are scored only.

New observable candidates, not verified segmentation improvements.  Real
execution requires the bound existing CPU model; toy callbacks stay explicit.
"""
from __future__ import annotations

import hashlib
import threading
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .common import Result, prototype_margin, rgb_view, unit, validate
from .encoder import assert_matches_producer
from .cross_image_matching import _dot
from ics.methods.direct_dino_features import resize

ENCODER_REQUIRED = True
_ENCODER = None
_BINDING = None
_RESULT_EP = None
_RESULTS = {}
_RUN_LOCK = threading.RLock()
NEUTRAL = np.array((124, 116, 104), np.uint8)


def configure_encoder(callback, binding):
    global _ENCODER, _BINDING, _RESULT_EP, _RESULTS
    if not callable(callback) or not isinstance(binding, dict):
        raise ValueError("Explicit frozen CPU encoder callback and binding required")
    _ENCODER, _BINDING = callback, dict(binding)
    _RESULT_EP, _RESULTS = None, {}


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _ready(ep):
    validate(ep)
    if _ENCODER is None:
        raise RuntimeError("Context method requires configure_encoder; no guessed features")
    if ep.r_rgb is None or ep.q_rgb is None or ep.reference_mask is None:
        raise ValueError("Original legitimate RGB and complete known reference mask required")
    synthetic = ep.producer.get("kind") == "cpu100_declared_synthetic_encoder"
    if synthetic:
        if not _BINDING.get("synthetic_encoder"):
            raise ValueError("Synthetic Episode cannot be silently sent to a real encoder")
    else:
        assert_matches_producer(ep.producer, _BINDING)
        if _BINDING.get("synthetic_encoder"):
            raise ValueError("Toy callback is not real DINO evidence")
        if ep.producer.get("model_input_side") != 1024 or ep.q_hw != (64,64) or ep.r_hw != (64,64):
            raise ValueError("Original native1024 origin required; no native128/new1024 comparison")
    for geometry in (ep.query_geometry, ep.reference_geometry):
        if (geometry.get("view_side") != 1024 or geometry.get("resized_hw") != [1024,1024]
                or geometry.get("padding_top_left") != [0,0]):
            raise ValueError("Exact square native1024 origin geometry required")


def _encode(image):
    value = np.asarray(_ENCODER(np.ascontiguousarray(image, dtype=np.uint8)))
    if value.shape != (64,64,1024) or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Native raw FP32 final-LN64x64x1024 callback required")
    return value


def _stats():
    return _ENCODER.stats() if hasattr(_ENCODER, "stats") else dict(
        real_encoder_execution=False, synthetic_encoder=True, new_encoder_forwards=0)


def _native_rgb(ep, role):
    out = rgb_view(ep, role)
    if out.shape != (1024,1024,3) or out.dtype != np.uint8:
        raise ValueError("Recorded native1024 image transform required")
    return out


def _mask(ep, height=1024):
    return np.asarray(Image.fromarray(ep.reference_mask.astype(np.uint8)).resize(
        (1024,height), Image.Resampling.NEAREST), float)


def _coverage(mask):
    h,w = mask.shape
    return mask.reshape(h//16,16,w//16,16).mean((1,3)).ravel()


def _unit_map(a):
    return unit(np.asarray(a, float).reshape(-1,1024))


def _proto(r, q, wf):
    r,q = _unit_map(r),_unit_map(q)
    if wf.sum() <= 0: return np.full(len(q),-1.)
    if (1-wf).sum() <= 0: return np.full(len(q),1.)
    f=unit(np.average(r,axis=0,weights=wf));b=unit(np.average(r,axis=0,weights=1-wf))
    return _dot(q,f-b)


def _response(r, q, wf, fallback_r, fallback_q, guard=1e-10):
    guard=float(guard)
    r,q=np.asarray(r,float).reshape(-1,1024),np.asarray(q,float).reshape(-1,1024)
    if wf.sum()<=0 or (1-wf).sum()<=0:
        return _proto(fallback_r,fallback_q,wf),dict(response_fallback="missing reference class")
    f=np.average(r,axis=0,weights=wf);b=np.average(r,axis=0,weights=1-wf)
    gap=float(np.linalg.norm(f-b))
    if gap<=guard:
        return _proto(fallback_r,fallback_q,wf),dict(response_fallback="source response means indistinguishable",response_mean_gap=gap,guard=guard)
    ef=np.sum((q-f)**2,1);eb=np.sum((q-b)**2,1)
    margin=np.divide(eb-ef,eb+ef,out=np.zeros_like(ef),where=eb+ef>1e-20)
    return margin,dict(response_fallback=False,response_mean_gap=gap,guard=guard,
                       mean_reference_response_norm=float(np.linalg.norm(r,axis=1).mean()),
                       mean_query_response_norm=float(np.linalg.norm(q,axis=1).mean()))


def _result(ep, margin, source_hw, info, before):
    field=resize(np.asarray(margin).reshape(source_hw),ep.q_hw)
    field.ravel()[ep.q_valid<=0]=0
    after=_stats()
    keys=("cache_hits","cache_misses","new_encoder_forwards","callback_forward_attempts","forward_wall_seconds")
    delta={key:after.get(key,0)-before.get(key,0) for key in keys}
    return Result(field,dict(info,encoder_binding=_BINDING,encoder_delta=delta,
        real_encoder_execution=after.get("real_encoder_execution",False),
        query_gt_used=False,empirical_gain="unknown",new_forward_scope="same legitimate R/Q RGB only"))


def _probes(ep):
    r=_native_rgb(ep,"r");m=_mask(ep).astype(bool)
    if not m.any() or m.all(): return None
    y,x=np.nonzero(m)
    t,l=max(0,int(y.min())-16),max(0,int(x.min())-16)
    b,rr=min(1024,int(y.max())+17),min(1024,int(x.max())+17)
    foreground=r[t:b,l:rr].copy()
    nearest=distance_transform_edt(m,return_distances=False,return_indices=True)
    removed=r[nearest[0],nearest[1]]
    if not np.array_equal(removed[~m],r[~m]):
        raise RuntimeError("Natural reference background outside MR must remain identical")
    bg=removed[t:b,l:rr].copy()
    if np.any(m[nearest[0][m],nearest[1][m]]):
        raise RuntimeError("Negative probe must draw only known-background pixels")
    height,width=foreground.shape[:2];scale=192/max(height,width)
    sh,sw=max(1,round(height*scale)),max(1,round(width*scale))
    templates=[]
    for colour in (foreground,bg):
        templates.append(np.asarray(Image.fromarray(colour).resize((sw,sh),Image.Resampling.BILINEAR)).copy())
    area=np.asarray(Image.fromarray(m[t:b,l:rr].astype(np.float32)).resize((sw,sh),Image.Resampling.BILINEAR),float)
    return templates,dict(probe_hw=[sh,sw],probe_original_bbox=[int(t),int(l),int(b),int(rr)],
                           full_reference_mask_sha256=_digest(ep.reference_mask),
                           known_background_source_only=True,natural_outside_MR_identical=True,
                           probe_intervention="known-FG pixels replaced by nearest known-R-background pixels; semantic object absence not guaranteed",
                           fixed_context_padding_pixels=16,foreground_shape_alpha_not_used=True),area


def _main_canvas(ep,role):
    body=np.asarray(Image.fromarray(_native_rgb(ep,role)).resize((1024,768),Image.Resampling.BILINEAR))
    canvas=np.broadcast_to(NEUTRAL,(1024,1024,3)).copy();canvas[:768]=body
    return canvas


def _place(canvas,probe,slot):
    out=canvas.copy();h,w=probe.shape[:2];y=768+(256-h)//2;x=(256 if slot==0 else 768)-w//2
    out[y:y+h,x:x+w]=probe
    if not np.array_equal(out[:768],canvas[:768]):raise RuntimeError("Probe modified scored main pixels")
    return out


def _panel(ep,kind):
    _ready(ep);before=_stats();prepared=_probes(ep)
    if prepared is None:
        return Result(prototype_margin(ep),dict(mechanism=kind,encoder_delta={},response_fallback="missing known probe class"))
    probes,pinfo,area=prepared;response={};averages={};fg_only={};bg_only={};probe_atoms={};area_atoms={}
    view_hashes=[]
    for role in ("r","q"):
        canvas=_main_canvas(ep,role);d=np.zeros((48,64,1024));average=np.zeros_like(d)
        pos=np.zeros_like(d);neg=np.zeros_like(d);atoms=[];pooled=[]
        for slot in (0,1):
            maps=[]
            for label,probe in enumerate(probes):
                image=_place(canvas,probe,slot);h=_encode(image)
                maps.append(h[:48]);average+=h[:48]/4
                (pos if label==0 else neg)[:]+=h[:48]/2
                atoms.append(np.asarray(h[48:],float).mean((0,1)))
                sh,sw=probe.shape[:2];oy=768+(256-sh)//2;ox=(256 if slot==0 else 768)-sw//2
                weight=np.zeros((1024,1024));weight[oy:oy+sh,ox:ox+sw]=area
                token_weight=_coverage(weight)
                pooled.append(np.average(np.asarray(h,float).reshape(-1,1024),axis=0,weights=token_weight))
                view_hashes.append(dict(role=role,slot=slot,probe=label,view_sha256=_digest(image),
                    main_sha256=_digest(image[:768]),panel_rows_excluded=[48,64]))
            d+=(np.asarray(maps[0],float)-maps[1])/2
        response[role]=d;averages[role]=average;fg_only[role]=pos;bg_only[role]=neg;probe_atoms[role]=atoms;area_atoms[role]=pooled
    wf=_coverage(_mask(ep,768));extra={}
    if kind=="response":margin,extra=_response(response["r"],response["q"],wf,averages["r"],averages["q"])
    elif kind=="mean":margin=_proto(averages["r"],averages["q"],wf)
    elif kind=="foreground_view":margin=_proto(fg_only["r"],fg_only["q"],wf)
    elif kind=="background_view":margin=_proto(bg_only["r"],bg_only["q"],wf)
    elif kind=="inserted_probe":
        # Same views, but absolute stimulus embeddings: an uncounted rival readout.
        f=unit(np.mean([probe_atoms["q"][j] for j in (0,2)],axis=0))
        b=unit(np.mean([probe_atoms["q"][j] for j in (1,3)],axis=0))
        margin=_dot(_unit_map(averages["q"]),f-b)
        extra["control_scope"]="whole bottom16x64 panel average including neutral carrier; not original ProM4"
    elif kind=="probe_area":
        f=unit(np.mean([area_atoms["q"][j] for j in (0,2)],axis=0))
        b=unit(np.mean([area_atoms["q"][j] for j in (1,3)],axis=0))
        margin=_dot(_unit_map(averages["q"]),f-b)
        extra["control_scope"]="known source-MR area pooled stimulus matching in query context; same8views, not original ProM4 covariance pipeline"
    else:raise ValueError(kind)
    return _result(ep,margin,(48,64),dict(mechanism="remote_reference_"+kind,**pinfo,**extra,
        view_hashes=view_hashes,new_view_budget=8,main_window=[0,0,768,1024],
        native_resolution_reduction="main48rows, same-view controls identical; not native64 unchanged sampling"),before)


def remote_reference_response(ep):return _panel(ep,"response")
def panel_mean_control(ep):return _panel(ep,"mean")
def panel_foreground_view_control(ep):return _panel(ep,"foreground_view")
def panel_background_view_control(ep):return _panel(ep,"background_view")
def panel_inserted_probe_control(ep):return _panel(ep,"inserted_probe")


def preserved_quadrant_view(image,quadrant):
    out=image.copy();positions=((0,0),(0,512),(512,0),(512,512))
    other=[j for j in range(4) if j!=quadrant]
    for destination,source in zip(other,other[1:]+other[:1]):
        dy,dx=positions[destination];sy,sx=positions[source]
        out[dy:dy+512,dx:dx+512]=image[sy:sy+512,sx:sx+512]
    y,x=positions[quadrant]
    if not np.array_equal(out[y:y+512,x:x+512],image[y:y+512,x:x+512]):
        raise RuntimeError("Scored quadrant local pixels changed")
    return out


def _external(ep,kind):
    _ready(ep);before=_stats();changes={};means={};hashes=[]
    for role,origin in (("r",ep.r),("q",ep.q)):
        image=_native_rgb(ep,role);changed=np.zeros((64,64,1024));original=origin.reshape(64,64,1024)
        for k,(y,x) in enumerate(((0,0),(0,32),(32,0),(32,32))):
            view=preserved_quadrant_view(image,k);h=_encode(view)
            changed[y:y+32,x:x+32]=_unit_map(h[y:y+32,x:x+32]).reshape(32,32,1024)
            hashes.append(dict(role=role,quadrant=k,view_sha256=_digest(view),local_rgb_identical=True))
        changes[role]=original-changed;means[role]=(original+changed)/2
    wf=ep.wf;extra={}
    if kind=="sensitivity":
        sr=np.linalg.norm(changes["r"],axis=2).ravel();sq=np.linalg.norm(changes["q"],axis=2).ravel()
        if wf.sum()<=0 or ep.wb.sum()<=0:margin=prototype_margin(ep).ravel();extra["fallback"]="missing known class"
        else:
            f=float(np.average(sr,weights=wf));b=float(np.average(sr,weights=ep.wb))
            if abs(f-b)<=1e-10:margin=_proto(means["r"],means["q"],wf);extra["fallback"]="equal source sensitivity means"
            else:
                ef=(sq-f)**2;eb=(sq-b)**2; margin=np.divide(eb-ef,eb+ef,out=np.zeros_like(ef),where=eb+ef>1e-20)
            extra.update(source_fg_sensitivity=f,source_bg_sensitivity=b,mean_query_sensitivity=float(sq.mean()))
    elif kind=="mean":margin=_proto(means["r"],means["q"],wf)
    elif kind=="vector":margin,extra=_response(changes["r"],changes["q"],wf,means["r"],means["q"])
    else:raise ValueError(kind)
    return _result(ep,margin,(64,64),dict(mechanism="external_context_"+kind,**extra,
        view_hashes=hashes,new_view_budget=8,all_native_tokens_preserved_once=True),before)


def external_shuffle_sensitivity(ep):return _external(ep,"sensitivity")
def external_mean_feature_control(ep):return _external(ep,"mean")
def external_vector_change_control(ep):return _external(ep,"vector")


def _competition(ep,kind):
    _ready(ep);before=_stats();prepared=_probes(ep)
    if prepared is None:return Result(prototype_margin(ep),dict(mechanism=kind,response_fallback="missing probe class"))
    probes,pinfo,_=prepared;banks={};hashes=[]
    for role in ("r","q"):
        base=_main_canvas(ep,role);views=[]
        for a,b in ((0,0),(0,1),(1,0),(1,1)):
            image=_place(_place(base,probes[a],0),probes[b],1);h=_encode(image)
            views.append(h[:48]);hashes.append(dict(role=role,assignments=[a,b],view_sha256=_digest(image),
                                                   main_sha256=_digest(image[:768]),panel_rows_excluded=[48,64]))
        banks[role]=views
    means={role:sum(np.asarray(h,float)/4 for h in banks[role]) for role in banks}
    wf=_coverage(_mask(ep,768));extra={}
    def quantity(role):
        ff,fb,bf,bb=(np.asarray(h,float) for h in banks[role])
        if kind=="mixed":return ff-fb-bf+bb
        if kind=="first":return ff-bb
        if kind=="slot_left_fg":return ff-bf
        if kind=="slot_left_bg":return fb-bb
        if kind=="slot_right_fg":return ff-fb
        if kind=="slot_right_bg":return bf-bb
        raise ValueError(kind)
    if kind=="mean":margin=_proto(means["r"],means["q"],wf)
    elif kind=="stacked":
        # Exact squared-distance prototype on all4 views without a huge stack.
        ef=np.zeros(48*64);eb=np.zeros(48*64)
        if wf.sum()<=0 or (1-wf).sum()<=0:margin=_proto(means["r"],means["q"],wf)
        else:
            for r,q in zip(banks["r"],banks["q"]):
                r=np.asarray(r,float).reshape(-1,1024);q=np.asarray(q,float).reshape(-1,1024)
                f=np.average(r,axis=0,weights=wf);b=np.average(r,axis=0,weights=1-wf)
                ef+=np.sum((q-f)**2,1);eb+=np.sum((q-b)**2,1)
            margin=np.divide(eb-ef,eb+ef,out=np.zeros_like(ef),where=eb+ef>1e-20)
    else:
        r,q=quantity("r"),quantity("q")
        arithmetic_guard=max(1e-6,8*np.finfo(np.float32).eps*sum(float(np.linalg.norm(
            np.asarray(h,float),axis=2).mean()) for h in banks["r"])) if kind=="mixed" else 1e-10
        margin,extra=_response(r,q,wf,means["r"],means["q"],arithmetic_guard)
        extra["guard_scope"]="nominal subtraction guard only; not a model-forward error certificate"
    return _result(ep,margin,(48,64),dict(mechanism="probe_competition_"+kind,**pinfo,**extra,
        view_hashes=hashes,new_view_budget=8,raw_LN_difference_before_unit=True),before)


def probe_nonadditive_competition(ep):return _competition(ep,"mixed")
def competition_first_difference_control(ep):return _competition(ep,"first")
def competition_mean_control(ep):return _competition(ep,"mean")
def competition_stacked_control(ep):return _competition(ep,"stacked")
def competition_left_fg_control(ep):return _competition(ep,"slot_left_fg")
def competition_left_bg_control(ep):return _competition(ep,"slot_left_bg")
def competition_right_fg_control(ep):return _competition(ep,"slot_right_fg")
def competition_right_bg_control(ep):return _competition(ep,"slot_right_bg")


def _orbit(ep,kind):
    _ready(ep);before=_stats();maps={};hashes=[]
    for role,original in (("r",ep.r),("q",ep.q)):
        image=_native_rgb(ep,role);variants=[original.reshape(64,64,1024)]
        for axis in (1,0):
            view=np.roll(image,512,axis=axis);h=_encode(view)
            inverse=np.roll(_unit_map(h).reshape(64,64,1024),-32,axis=axis)
            if not np.array_equal(np.roll(view,-512,axis=axis),image):raise RuntimeError("RGB roll inverse failed")
            variants.append(inverse);hashes.append(dict(role=role,axis=axis,view_sha256=_digest(view),
                                                       patch_pixels_identical=True,inverse_exact=True))
        maps[role]=variants
    if kind=="features":margin=_proto(np.mean(maps["r"],axis=0),np.mean(maps["q"],axis=0),ep.wf)
    elif kind=="scalar":margin=np.mean([_proto(r,q,ep.wf) for r,q in zip(maps["r"],maps["q"])],axis=0)
    elif kind=="horizontal":margin=_proto(maps["r"][1],maps["q"][1],ep.wf)
    elif kind=="vertical":margin=_proto(maps["r"][2],maps["q"][2],ep.wf)
    else:raise ValueError(kind)
    return _result(ep,margin,(64,64),dict(mechanism="patch_position_orbit_"+kind,view_hashes=hashes,
        new_view_budget=4,position_and_context_not_causally_separated=True),before)


def patch_position_orbit(ep):return _orbit(ep,"features")
def orbit_scalar_average_control(ep):return _orbit(ep,"scalar")
def orbit_horizontal_control(ep):return _orbit(ep,"horizontal")
def orbit_vertical_control(ep):return _orbit(ep,"vertical")


def _group_cache(group, selectors, compute):
    """Store only small complete fields, never a second unbounded view cache.

    Compute all same-view controls while those views remain in the adapter LRU.
    Calling the four main methods before their controls therefore still needs
    at most28 distinct new views for one Episode, rather than re-encoding an
    evicted group.  The result cache retains only the current Episode.
    """
    def wrapper(selector):
        def run(ep):
            global _RESULT_EP, _RESULTS
            with _RUN_LOCK:
                _ready(ep)
                if _RESULT_EP is not ep:
                    _RESULT_EP, _RESULTS = ep, {}
                if group not in _RESULTS:
                    _RESULTS[group] = {name: compute(ep,name) for name in selectors}
                result = _RESULTS[group][selector]
                return Result(result.margin.copy(),dict(result.info))
        return run
    return wrapper


_P=_group_cache("panel",("response","mean","foreground_view","background_view","inserted_probe","probe_area"),_panel)
_E=_group_cache("external",("sensitivity","mean","vector"),_external)
_C=_group_cache("competition",("mixed","first","mean","stacked","slot_left_fg","slot_left_bg","slot_right_fg","slot_right_bg"),_competition)
_O=_group_cache("orbit",("features","scalar","horizontal","vertical"),_orbit)
METHODS={
    "context_remote_reference_response":_P("response"),
    "context_external_shuffle_sensitivity":_E("sensitivity"),
    "context_probe_nonadditive_competition":_C("mixed"),
    "context_patch_position_orbit":_O("features"),
}
CONTROLS={
    "context_panel_mean_control":_P("mean"),
    "context_panel_foreground_view_control":_P("foreground_view"),
    "context_panel_background_view_control":_P("background_view"),
    "context_panel_inserted_probe_control":_P("inserted_probe"),
    "context_panel_probe_area_control":_P("probe_area"),
    "context_external_mean_feature_control":_E("mean"),
    "context_external_vector_change_control":_E("vector"),
    "context_competition_first_difference_control":_C("first"),
    "context_competition_mean_control":_C("mean"),
    "context_competition_stacked_control":_C("stacked"),
    "context_competition_left_fg_control":_C("slot_left_fg"),
    "context_competition_left_bg_control":_C("slot_left_bg"),
    "context_competition_right_fg_control":_C("slot_right_fg"),
    "context_competition_right_bg_control":_C("slot_right_bg"),
    "context_orbit_scalar_average_control":_O("scalar"),
    "context_orbit_horizontal_control":_O("horizontal"),
    "context_orbit_vertical_control":_O("vertical"),
}
