"""Compiled complete-decoder comparison; shared-GPU diagnostics are explicit."""
import json
import argparse
import os
from pathlib import Path
import sys
import time
import subprocess
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/compute_structure'))
from pretrained_decoder_round import load_decoder,load_inputs
from baselines import PositionCache
from prototype import DensePhaseCache,dense_phase_predict,shape_plan
from benchmark import measure
import experimental_methods as experimental
from takeover_compact_phase import predict
from takeover_compact_head_infer import load_student
OUT=ROOT/'results/takeover_20261001_v1/local_student_head_v1/perf_p128'


def other_pids():
    return [int(x) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split() if int(x)!=os.getpid()]


def snapshot():
    return {'other_compute_pids':other_pids(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader'],text=True).strip()}


def main(concurrent_diagnostic=False,phase_fallback=False,minimum_free_mib=12000):
    OUT.mkdir(exist_ok=False)
    report={'status':'WAITING_VRAM_FOR_CONCURRENT_COMPILER_PREPARATION','pid':os.getpid(),'protocol':'FP32 TF32off, same real128 distinct original prompts, MB128, same dense-associated transformer/cache; strong dense phase and real compact head5/10%, all4 masks and4 IQ scores; Inductor fullgraph True/dynamic False, no fallback compiler. Compilation and untimed checks may prepare concurrently; formal timing waits for exclusive GPU. Complete warm lowresolution decoder excludes encoder/fullresolution resize/model load/GT diagnostics. Approximate candidate, not a relaxed original5e-5 equivalence pass.','records':[]}
    report['concurrent_diagnostic']=concurrent_diagnostic
    report['minimum_free_mib']=minimum_free_mib
    if phase_fallback:
        report['protocol']+=' Add phase5/10 local original-function fallback using identical frozen student, selector and budgets, same cached phase matrices as the dense control. Six arms; comparison retains fastest implicit-phase baseline.'
    if concurrent_diagnostic:
        report['protocol']+=' OVERRIDE: shared-GPU diagnostic only; balanced rotating/reversed rounds, every arm in each position twice, ten repetitions per arm/block. GPU contention is uncontrolled; these samples cannot establish exclusive speedup or publication latency. Formal exclusive queue remains separate.'
    def save():
        tmp=OUT/'report.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'report.json')
    save()
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<minimum_free_mib:time.sleep(10)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,_=load_decoder(ROOT/'results/takeover_20261001_v1/real_original/mask_decoder_state.pt',torch.device('cuda:0'))
    model.requires_grad_(False)
    image,pe,dense,sparse,info=load_inputs(ROOT/'results/takeover_20261001_v1/real_p128_inputs/encoded_inputs.npz',torch.device('cuda:0'))
    sp=sparse['perf'];assert len(sp)==128 and len(torch.unique(sp,dim=0))==128
    plans=shape_plan(model,4096,sp.shape[1]+5,'dense_assoc','auto','auto')
    cache=DensePhaseCache.build(model,image,dense,PositionCache.build(model,pe,plans))
    factor_cache=experimental.CacheFactory('factor_implicit_phase',None).build(model,image,dense,PositionCache.build(model,pe,None))
    student,center,basis=load_student(OUT.parent/'deployment.pt','cuda:0')
    student.requires_grad_(False)
    def original(s,ca):return dense_phase_predict(model,ca,s,'explicit',plans=plans)
    def implicit(s,ca):return experimental.forward('factor_implicit_phase',model,factor_cache,s,'explicit',None)
    def compact5(s,ca):return predict(model,ca,s,student,center,basis,.05,plans)
    def compact10(s,ca):return predict(model,ca,s,student,center,basis,.1,plans)
    def phase5(s,ca):return predict(model,ca,s,student,center,basis,.05,plans,phase_fallback=True)
    def phase10(s,ca):return predict(model,ca,s,student,center,basis,.1,plans,phase_fallback=True)
    report['status']='COMPILING';save()
    telemetry_log=(OUT/'gpu_telemetry.csv').open('x')
    telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv','--loop-ms=1000'],stdout=telemetry_log,stderr=subprocess.STDOUT)
    try:
        compiled={}
        with torch.inference_mode():
            arms=[('dense_phase',original),('implicit_phase',implicit),('compact5',compact5),('compact10',compact10)]
            if phase_fallback:arms += [('phase5',phase5),('phase10',phase10)]
            for name,fn in arms:
                report['current_arm']=name;save();start=time.monotonic()
                compiled[name]=torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False)
                result=compiled[name](sp,cache);torch.cuda.synchronize()
                assert result[0].shape==(128,4,256,256) and result[1].shape==(128,4)
                report.setdefault('compile_seconds',{})[name]=time.monotonic()-start;save();del result
            reference=compiled['dense_phase'](sp,cache)
            for name in ('compact5','compact10') + (('phase5','phase10') if phase_fallback else ()):
                value=compiled[name](sp,cache)
                report.setdefault('approximation_diagnostics',{})[name]={'max_logit_difference':float((value[0]-reference[0]).abs().max()),'lowres_mask_flip_fraction':float(((value[0]>0)!=(reference[0]>0)).float().mean()),'max_IQ_drift':float((value[1]-reference[1]).abs().max())}
                del value
            del reference;save()
            if phase_fallback:
                for budget in (5,10):
                    a=compiled['compact'+str(budget)](sp,cache);b=compiled['phase'+str(budget)](sp,cache)
                    report.setdefault('phase_vs_cudnn_diagnostics',{})[str(budget)]={'max_logit_difference':float((a[0]-b[0]).abs().max()),'lowres_pixel_flip_fraction':float(((a[0]>0)!=(b[0]>0)).float().mean()),'max_IQ_difference':float((a[1]-b[1]).abs().max())}
                    del a,b
                save()
            torch.cuda.empty_cache()
            names=tuple(name for name,_ in arms)
            if concurrent_diagnostic:
                orders=[names[i:]+names[:i] for i in range(len(names))]
                orders+= [tuple(reversed(order)) for order in orders]
                repetitions=10
                report['status']='SHARED_GPU_DIAGNOSTIC_TIMING'
            else:
                report['status']='WAITING_EXCLUSIVE_GPU_FOR_FORMAL_TIMING';save()
                while True:
                    others=other_pids()
                    if not others:break
                    report['waiting_other_compute_pids']=others;save();time.sleep(5)
                report['status']='EXCLUSIVE_TIMING'
                orders=[names,tuple(reversed(names))]
                repetitions=30
            save()
            for round_number,order in enumerate(orders,1):
                for name in order:
                    before=snapshot()
                    if before['other_compute_pids'] and not concurrent_diagnostic:raise RuntimeError('Concurrent GPU process appeared; timing invalid: '+str(before['other_compute_pids']))
                    for _ in range(10):del_result=compiled[name](sp,cache);del del_result
                    timing=measure(lambda:compiled[name](sp,cache),torch.device('cuda:0'),repetitions)
                    after=snapshot()
                    record={'round':round_number,'arm':name,'timing':timing,'before':before,'after':after}
                    report['records'].append(record);save()
                    if after['other_compute_pids'] and not concurrent_diagnostic:raise RuntimeError('Concurrent GPU process appeared during timing; samples invalid: '+str(after['other_compute_pids']))
                    print(json.dumps({'round':round_number,'arm':name,'median_ms':timing['median_ms']}),flush=True)
        report['status']='COMPLETED_SHARED_GPU_DIAGNOSTIC_NOT_EXCLUSIVE' if concurrent_diagnostic else 'COMPLETED_EXCLUSIVE_COMPILED_DECODER_TIMING'
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:
        telemetry.terminate();telemetry.wait(timeout=5);telemetry_log.close();save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path)
    p.add_argument('--concurrent-diagnostic',action='store_true',help='Balanced shared-GPU screening only, never formal exclusive timing.')
    p.add_argument('--phase-fallback',action='store_true')
    p.add_argument('--minimum-free-mib',type=int,default=12000)
    args=p.parse_args()
    if args.output_dir is not None:OUT=args.output_dir
    try:
        main(args.concurrent_diagnostic,args.phase_fallback,args.minimum_free_mib)
    except BaseException as exc:
        # Preserve loader/OOM evidence too, before the inner telemetry guard.
        path=OUT/'report.json'
        if path.exists():
            report=json.loads(path.read_text())
            if report.get('pid')==os.getpid() and report.get('status')!='ERROR':
                report.update(status='ERROR',error=repr(exc));tmp=OUT/'report.failure.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
        raise
