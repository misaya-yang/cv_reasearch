"""Exact double Dinic compiled loop port, with the old Python solver fallback.

No objective, precision, resolution, tie order or numerical certificate changes.
A local system C++ compiler is optional; there is no download or dependency install.
"""
from __future__ import annotations
from pathlib import Path
import ctypes,hashlib,os,shutil,subprocess,tempfile,threading
import numpy as np
_LOCK=threading.Lock();_LIB=None;_FAILURE=None


def _library():
    global _LIB,_FAILURE
    if _LIB is not None:return _LIB
    if _FAILURE is not None:return None
    with _LOCK:
        if _LIB is not None:return _LIB
        source=Path(__file__).with_suffix('.cpp');compiler=shutil.which('c++') or shutil.which('g++')
        if not compiler:_FAILURE='local_C++_compiler_unavailable';return None
        key=hashlib.sha256(source.read_bytes()).hexdigest()
        folder=Path(tempfile.gettempdir())/'astra300_e_float_cut';folder.mkdir(exist_ok=True)
        destination=folder/(key+('.dylib' if os.uname().sysname=='Darwin' else '.so'))
        if not destination.exists():
            temporary=folder/(key+'.'+str(os.getpid())+'.tmp')
            options=['-O3','-std=c++17','-shared','-fPIC']
            result=subprocess.run([compiler,*options,str(source),'-o',str(temporary)],capture_output=True,text=True)
            if result.returncode:
                _FAILURE='local_C++_compile_failed:'+result.stderr[-1000:];return None
            os.replace(temporary,destination)
        library=ctypes.CDLL(str(destination));fn=library.e_float_cut
        double=ctypes.POINTER(ctypes.c_double);integer=ctypes.POINTER(ctypes.c_int64)
        fn.argtypes=[ctypes.c_int64,ctypes.c_int64,double,double,integer,double,ctypes.POINTER(ctypes.c_ubyte),double,integer];fn.restype=ctypes.c_int
        _LIB=library;return library


def exact_potts_cut(logits,edges,capacity):
    logits=np.ascontiguousarray(np.asarray(logits,np.float64).ravel());capacity=np.ascontiguousarray(np.asarray(capacity,np.float64))
    edges=np.ascontiguousarray(np.asarray(edges,np.int64).reshape(-1,2));n=len(logits)
    if not n or capacity.shape!=(len(edges),) or not np.isfinite(logits).all() or not np.isfinite(capacity).all() or np.any(capacity<0) or np.any(edges<0) or np.any(edges>=n):
        raise ValueError('Finite unary costs and nonnegative valid Potts edges required')
    library=_library()
    if library is None:
        from ics.methods.pro_paired_environment import exact_potts_cut as old
        labels,receipt=old(logits,edges,capacity);receipt.update(solver_implementation='original_Python_float_Dinic',compiled_unavailable=_FAILURE);return labels,receipt
    # Unary costs are computed by the original NumPy expression, so compiling
    # the loop does not substitute a different exp/log implementation.
    source=np.logaddexp(0,logits);sink=np.logaddexp(0,-logits);labels=np.empty(n,np.uint8)
    flow=ctypes.c_double();phases=ctypes.c_int64();double=ctypes.POINTER(ctypes.c_double);integer=ctypes.POINTER(ctypes.c_int64)
    status=library.e_float_cut(n,len(edges),source.ctypes.data_as(double),sink.ctypes.data_as(double),edges.ctypes.data_as(integer),
         capacity.ctypes.data_as(double),labels.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),ctypes.byref(flow),ctypes.byref(phases))
    if status:raise RuntimeError('Compiled floating Dinic allocation/execution failure')
    labels=labels.astype(bool)
    from ics.methods.pro_paired_environment import potts_energy
    energy=potts_energy(labels,logits,edges,capacity);gap=abs(energy-flow.value)
    if gap>1e-8*max(1.,abs(energy)):raise RuntimeError('Cut/maxflow energy certificate failed: '+str(gap))
    return labels,dict(energy=energy,maximum_flow=flow.value,certificate_gap=gap,phases=phases.value,
        capacity_semantics='floating point, no integer quantization',solver_implementation='same_order_double_Dinic_C++_port',
        compiled_source_sha256=hashlib.sha256(Path(__file__).with_suffix('.cpp').read_bytes()).hexdigest())
