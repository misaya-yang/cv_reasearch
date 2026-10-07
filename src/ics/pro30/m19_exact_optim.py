"""Same frozen Pro30 M19, exact ordered loops and sort reuse, no new method.

Original source/renderer are untouched and retained as explicit controls.
Cold functions never use a previous episode's fit. Startup compilation is
reported separately AND included if it happens within a method call.
"""
from __future__ import annotations
import ctypes
from functools import partial
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import numpy as np
from . import group_17_23 as original
from . import helpers_17_23 as H
from .common import validate,degenerate_margin,source_contract,unit,array_hash,br_margin
from .render_exact_optim import finish_exact

_LIB=None;_FAILURE=None;_LOCK=threading.Lock();_EPISODE=None;_CACHE={}
_COMPILE_RECEIPT={'compile_seconds':0.,'library_load_seconds':0.,'compiled_this_process':False}


def _library():
    global _LIB,_FAILURE
    if _LIB is not None:return _LIB
    if _FAILURE is not None:return None
    with _LOCK:
        if _LIB is not None:return _LIB
        source=Path(__file__).with_suffix('.cpp').with_name('m19_ordered_sum.cpp')
        compiler=shutil.which('c++') or shutil.which('g++')
        if compiler is None:_FAILURE='local compiler absent; exact NumPy source loop used';return None
        flags=['-O3','-std=c++17','-shared','-fPIC','-fno-fast-math','-ffp-contract=off']
        key=hashlib.sha256(source.read_bytes()+(' '.join(flags)).encode()).hexdigest()
        destination=Path(tempfile.gettempdir())/'pro30_m19_ordered_sum'/ (key+('.dylib' if os.uname().sysname=='Darwin' else '.so'))
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.is_file():
            temporary=destination.with_name(destination.name+'.'+str(os.getpid())+'.tmp');start=time.perf_counter()
            result=subprocess.run([compiler,*flags,str(source),'-o',str(temporary)],capture_output=True,text=True)
            _COMPILE_RECEIPT['compile_seconds']+=time.perf_counter()-start
            if result.returncode:_FAILURE='local compilation failed: '+result.stderr[-1000:];return None
            os.replace(temporary,destination);_COMPILE_RECEIPT['compiled_this_process']=True
        start=time.perf_counter();library=ctypes.CDLL(str(destination));fn=library.m19_ordered_sum
        fn.argtypes=[ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)];fn.restype=ctypes.c_int
        _COMPILE_RECEIPT['library_load_seconds']+=time.perf_counter()-start;_LIB=library
        return library


def reset_episode_cache():
    """Clear fitted banks/calibration; not a reset of static loaded software."""
    global _EPISODE
    _CACHE.clear();_EPISODE=None


def prepare_static_library():
    """Explicit one-time asset preparation, never hidden as cold inference."""
    library=_library()
    return dict(_COMPILE_RECEIPT,implementation='ordered double C++' if library else 'source NumPy add.at',fallback=_FAILURE)


def _ordered_sum(weighted,labels,k,*,compiled=True):
    weighted=np.ascontiguousarray(weighted,dtype=np.float64);labels=np.ascontiguousarray(labels,dtype=np.int64);sums=np.zeros((k,weighted.shape[1]),dtype=np.float64)
    library=_library() if compiled else None
    if library is None:np.add.at(sums,labels,weighted)
    else:
        double=ctypes.POINTER(ctypes.c_double);integer=ctypes.POINTER(ctypes.c_int64)
        status=library.m19_ordered_sum(len(labels),weighted.shape[1],k,labels.ctypes.data_as(integer),weighted.ctypes.data_as(double),sums.ctypes.data_as(double))
        if status:raise ValueError('Ordered accumulation invalid input/status '+str(status))
    if not np.isfinite(sums).all():raise FloatingPointError('Nonfinite ordered accumulation')
    return sums


def fps_lloyd_exact(x,weights,k=32,rounds=10):
    # Literal original initialization, unit arithmetic, ten updates, empty-mode
    # removal and final assignment. Only ordered scatter loop is compiled.
    x=np.asarray(x,float);w=np.asarray(weights,float);ids=np.flatnonzero(w>0)
    if not len(ids):return np.empty((0,x.shape[1])),np.empty(0),np.full(len(x),-1,int)
    xx=unit(x[ids]);ww=w[ids];k=min(int(k),len(ids));chosen=[int(np.argmax(ww))];distance=1-H.mm(xx,xx[chosen[0]])
    for _ in range(1,k):
        nxt=int(np.argmax(distance*ww))
        if distance[nxt]<=1e-12:break
        chosen.append(nxt);distance=np.minimum(distance,1-H.mm(xx,xx[nxt]))
    centers=xx[chosen].copy();weighted=xx*ww[:,None]
    for _ in range(rounds):
        labels=np.argmax(H.mm(xx,centers.T),axis=1);mass=np.bincount(labels,weights=ww,minlength=len(centers))
        sums=_ordered_sum(weighted,labels,len(centers));centers=unit(sums[mass>0])
    labels=np.argmax(H.mm(xx,centers.T),axis=1);mass=np.bincount(labels,weights=ww,minlength=len(centers));full=np.full(len(x),-1,int);full[ids]=labels
    return centers,mass,full


def _hyperplanes(ep,source='query'):
    x,v=(ep.q,ep.q_valid) if source=='query' else (ep.r,ep.wvalid)
    centers,mass,assignment=fps_lloyd_exact(x,v,32,10)
    if len(centers)<2:return None
    sim=H.mm(centers,centers.T);pairs=set()
    for i in range(len(centers)):
        remaining=np.array([j for j in range(len(centers)) if j!=i]);order=remaining[np.argsort(-sim[i,remaining],kind='stable')]
        for j in np.r_[order[:2],order[-2:]]:
            if np.linalg.norm(centers[i]-centers[j])>1e-10:pairs.add(tuple(sorted((i,int(j)))))
    pairs=sorted(pairs)
    if not pairs:return None
    norm=np.array([np.linalg.norm(centers[a]-centers[b]) for a,b in pairs]);rp=H.mm(ep.r,centers.T);qp=H.mm(ep.q,centers.T)
    R=np.column_stack([(rp[:,a]-rp[:,b])/n for (a,b),n in zip(pairs,norm)]);Q=np.column_stack([(qp[:,a]-qp[:,b])/n for (a,b),n in zip(pairs,norm)])
    R=np.c_[R,-R];Q=np.c_[Q,-Q];directions=np.vstack([(centers[a]-centers[b])/n for (a,b),n in zip(pairs,norm)]);directions=np.r_[directions,-directions]
    return centers,assignment,pairs,directions,R,Q


def _best_cut_ordered(score,order,fg,bg,sumfg,sumbg):
    """Exact source formula on a stable globally sorted/subset-filtered order."""
    if not len(order) or sumfg<=0 or sumbg<=0:return 0.,np.nan
    s=score[order];f=fg[order];b=bg[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
    cuts=np.r_[np.nextafter(s[0],-np.inf),s[ends]]
    loss=.5*np.r_[1.,np.cumsum(f)[ends]/sumfg+(sumbg-np.cumsum(b)[ends])/sumbg]
    idx=np.lexsort((np.arange(len(cuts)),np.abs(cuts),loss))[0]
    return float(cuts[idx]),float(loss[idx])


def calibration_exact(ep,R):
    groups=H.groups4(ep.r_hw);four_valid=all(ep.wf[groups==k].sum()>0 and ep.wb[groups==k].sum()>0 for k in range(4))
    # Source's H.best_cut casts to FP64 then masks nonzero total role mass.
    fg=np.asarray(ep.wf,float);bg=np.asarray(ep.wb,float);positive=(fg+bg)>0
    folds=[]
    for k in range(4):
        train=(groups!=k)&(ep.wvalid>0) if four_valid else ep.wvalid>0;held=(groups==k)&(ep.wvalid>0) if four_valid else train
        eligible=train&positive
        folds.append((eligible,held,fg[eligible].sum(),bg[eligible].sum(),ep.wf[held],ep.wb[held]))
    losses=np.empty((R.shape[1],4));thresholds=np.empty_like(losses)
    for c in range(R.shape[1]):
        order=np.argsort(R[:,c],kind='stable')
        for k,(train,held,sumfg,sumbg,heldfg,heldbg) in enumerate(folds):
            sorted_train=order[train[order]];cut,_=_best_cut_ordered(R[:,c],sorted_train,fg,bg,sumfg,sumbg);thresholds[c,k]=cut
            # Preserve source NumPy subset-sum order for held risk as well.
            losses[c,k]=H.risk(R[held,c],heldfg,heldbg,cut)
    chosen=int(np.lexsort((np.arange(R.shape[1]),np.ptp(losses,axis=1),np.mean(losses,axis=1)))[0]);cut,trainrisk=H.best_cut(R[:,chosen],ep.wf,ep.wb)
    return chosen,cut,trainrisk,four_valid,losses,thresholds


def _entry_cache(ep):
    global _EPISODE
    # Reference-label hash prevents label-change diagnostics reusing a fit.
    identity=(ep.source_id,id(ep.q),id(ep.r),array_hash(ep.wf),array_hash(ep.wvalid),array_hash(ep.q_valid))
    if identity!=_EPISODE:_CACHE.clear();_EPISODE=identity
    return _CACHE


def _end(ep,z,mid,info,start):
    elapsed=time.perf_counter()-start
    info=dict(info,**source_contract(19),quality='unmeasured identical frozen Pro30 M19 engineering optimization',
        total_method_wall_seconds_before_renderer=elapsed,cached_CPU_budget_seconds=2.,cost_gate_pass=elapsed<=2.,
        software_complete=True,query_GT_read=False,authorize_fixed600=False,engineering_revision='M19_exact_optim_v1',
        static_software_receipt=dict(_COMPILE_RECEIPT),ordered_sum_fallback=_FAILURE,
        unchanged_algorithm_budget={'K_max':32,'Lloyd_rounds':10,'source_groups':4,'pair_cap':128,'both_signs':True,'feature_dimension':ep.q.shape[1]},
        wall_cost_includes_compilation_if_first_method_call=True)
    output=finish_exact(ep,z,mid,info,invalid_value=-1.)
    whole=time.perf_counter()-start;output.info.update(total_method_wall_seconds=whole,
        cost_gate_pass=whole<=2.,cost_gate_includes_renderer_and_cold_initialization=True)
    return output


def m19_exact(ep,*,variant='query_continuous'):
    start=time.perf_counter();validate(ep);mid='PRO30_M19'+('' if variant=='query_continuous' else '__'+variant)
    # Original source called validate twice through degenerate_margin. Validated
    # nondegenerate episodes need no repeated full descriptor scan.
    if ep.wf.sum()<=0 or ep.wb.sum()<=0:
        z,info=degenerate_margin(ep);return _end(ep,z,mid,info,start)
    source='reference' if variant=='reference_pairs' else 'query';cache=_entry_cache(ep);bank_key=source+'_bank';fit_key=source+'_calibration'
    bank_hit=bank_key in cache;fit_hit=fit_key in cache;section=time.perf_counter()
    if not bank_hit:cache[bank_key]=_hyperplanes(ep,source)
    bank=cache[bank_key];bank_seconds=time.perf_counter()-section
    if bank is None:return _end(ep,H.proto(ep),mid,dict(active=False,inactive_reason='fewer two distinct query centers'),start)
    centers,assignment,pairs,directions,R,Q=bank;section=time.perf_counter()
    if not fit_hit:cache[fit_key]=calibration_exact(ep,R)
    chosen,cut,trainrisk,four_valid,losses,thresholds=cache[fit_key];calibration_seconds=time.perf_counter()-section
    z=Q[:,chosen]-cut
    if variant=='nearest_query_center':z=H.mm(centers,directions[chosen])[assignment]-cut
    split=0
    for group in range(len(centers)):
        points=z[assignment==group];split+=bool(np.any(points>0) and np.any(points<=0))
    return _end(ep,z,mid,dict(active=True,centers=centers.tolist(),unordered_pairs=pairs,candidate_directions=len(directions),
        candidate_reference_label_exposure=False,selected_index=chosen,selected_direction=directions[chosen].tolist(),threshold=cut,
        four_group_risks=losses[chosen].tolist(),four_group_thresholds=thresholds[chosen].tolist(),full_source_balanced_error=trainrisk,
        low_evidence_selection=not four_valid,split_query_cluster_fraction=split/len(centers),
        implementation_assumption='unchanged original FPS/spherical10Lloyd, feature-cosine pair selection; stable-sort filtering and same-order sums only',
        bank_cache_hit=bank_hit,calibration_cache_hit=fit_hit,bank_seconds_this_call=bank_seconds,source_calibration_seconds_this_call=calibration_seconds),start)


def original_control(ep):
    out=original.m19(ep);out.info.update(method_id='PRO30_M19__original_fixed_source',control_only=True,identical_inference_object=True);return out


METHODS={'PRO30_M19':m19_exact}
CONTROLS={'PRO30_M19__reference_pairs':partial(m19_exact,variant='reference_pairs'),
          'PRO30_M19__nearest_query_center':partial(m19_exact,variant='nearest_query_center'),
          'PRO30_M19__QP02_original_fixed':original._qp02_control,
          'PRO30_B_R':original.br_result,
          'PRO30_M19__original_fixed_source':original_control}
REQUIREMENTS={'PRO30_M19':list(original.REQUIREMENTS['PRO30_M19'])}
CONTRACTS={'PRO30_M19':dict(original.CONTRACTS['PRO30_M19'],
    engineering_revision='M19_exact_optim_v1',new_method_count=0,
    changes='same-order compiled scatter; stable-sort filtering; exact calibration reuse; exact blocked renderer',
    controls=list(CONTROLS),solver_budget_lowered=False)}
