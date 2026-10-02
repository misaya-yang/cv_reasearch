"""Build the official FoRIS CRF dependency for this GPU, without pip downloads.

Source algorithms untouched; architecture selection replaces upstream sm70
build flags, which CUDA13 no longer supports. Builds cached in this experiment.
"""
import json
import os
from pathlib import Path
import sys
import sysconfig
import traceback

ROOT=Path(__file__).resolve().parent
os.environ['CUDA_HOME']='/usr/local/cuda'
os.environ['TORCH_CUDA_ARCH_LIST']='8.9'
os.environ['MAX_JOBS']='2'
os.environ['TORCH_EXTENSIONS_DIR']=str(ROOT/'runtime/extensions')
sys.path[:0]=[str(ROOT/'crf_source/src'),'/root/demo4_cache/env']
import torch
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from setuptools import setup


def load_crf():
    base=ROOT/'crf_source/src/PermutohedralFiltering/source'
    path=ROOT/'runtime/extensions';path.mkdir(parents=True,exist_ok=True)
    suffix=sysconfig.get_config_var('EXT_SUFFIX')
    if not all((path/(name+suffix)).exists() for name in ['Permutohedral','Permutohedral_gpu']):
        # The host has no ninja binary. Use installed setuptools' ordinary compiler
        # rather than downloading another runtime or changing CRF algorithms.
        setup(name='cvpr_foris_crf_runtime',ext_modules=[
            CppExtension('Permutohedral',[str(base/'cpu/LatticeFilterKernel.cpp')],extra_compile_args=['-O2','-DNDEBUG']),
            CUDAExtension('Permutohedral_gpu',[str(base/'gpu/LatticeFilter.cu')],extra_compile_args={'cxx':['-O2','-DNDEBUG'],'nvcc':['-O2']})],
            cmdclass={'build_ext':BuildExtension.with_options(use_ninja=False)},
            script_args=['build_ext','--build-lib',str(path),'--build-temp',str(ROOT/'runtime/temp')])
    sys.path.insert(0,str(path))
    import CRF
    return CRF


if __name__=='__main__':
    out=ROOT/'results/crf_runtime.json';out.parent.mkdir(parents=True,exist_ok=True)
    try:
        c=load_crf()
        p=c.FrankWolfeParams(scheme='fixed',stepsize=1.,regularizer='l2',lambda_=1.,lambda_learnable=False,x0_weight=0.,x0_weight_learnable=False)
        m=c.DenseGaussianCRF(classes=2,alpha=12.,beta=.03,gamma=4.,spatial_weight=3.,bilateral_weight=20.,compatibility=1.,init='potts',solver='fw',iterations=10,params=p).cuda()
        image=torch.rand(1,3,64,64,device='cuda');logits=torch.rand(1,2,64,64,device='cuda')
        with torch.inference_mode():y=m(image,logits)
        assert y.shape==logits.shape and torch.isfinite(y).all()
        out.write_text(json.dumps({'state':'COMPLETED','shape':list(y.shape),'arch':'8.9','cuda':torch.version.cuda}))
    except Exception:
        out.write_text(json.dumps({'state':'ERROR','traceback':traceback.format_exc()}));raise
