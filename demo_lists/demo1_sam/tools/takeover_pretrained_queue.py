"""Sequential real-P128 compiled diagnostics; numeric failures remain failures."""
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.input_dir/'prompt_manifest.json').read_text())
    assert manifest['prompts']==manifest['distinct_coordinates']==128
    report={'status':'RUNNING','started_at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
            'protocol':manifest['protocol'],'scientific_status':'Fixed 5e-5 numeric failures retained; all timings are diagnostic when failed; no end-to-end or SOTA claim',
            'prompt_manifest':str(a.input_dir/'prompt_manifest.json'),'runs':[]}
    status_path=a.output_dir/'queue_status.json';save(status_path,report)
    base=['dense_phase:explicit','cached_merged_phase:explicit','factor_implicit_phase:explicit']
    telemetry_log=(a.output_dir/'gpu_telemetry.csv').open('x')
    telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw','--format=csv','--loop-ms=1000'],stdout=telemetry_log,stderr=subprocess.STDOUT)
    try:
        for mb in (8,128):
            for rnd in (1,2):
                name=f'mb{mb}_round{rnd}';out=a.output_dir/name
                variants=base if rnd==1 else list(reversed(base))
                command=[sys.executable,'research/compute_structure/pretrained_decoder_round.py',
                         '--encoded-inputs',str(a.input_dir/'encoded_inputs.npz'),'--decoder-state',str(a.source_run/'mask_decoder_state.pt'),
                         '--output-dir',str(out),'--device','cuda:0','--regimes','perf','--microbatch',str(mb),
                         '--variants',','.join(variants),'--mode','compile','--timings','--diagnostic-on-failure','--warmup','10','--repetitions','30']
                row={'name':name,'status':'RUNNING','command':command};report['runs'].append(row);save(status_path,report)
                print('START '+name,flush=True);start=time.monotonic()
                with (a.output_dir/(name+'.log')).open('x') as log:
                    proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    row['pid']=proc.pid;save(status_path,report)
                    code=proc.wait()
                d=json.loads((out/'report.json').read_text())
                if not d.get('execution_completed') or len(d['records'])!=3 or any('warm_complete_prompt_batch' not in r for r in d['records']):
                    raise RuntimeError('Incomplete real-P128 performance group: '+name)
                if code not in (0,2):raise RuntimeError('Unexpected exit: '+str(code))
                row.update(status='DONE',returncode=code,numeric_status=d['status'],wall_seconds=time.monotonic()-start,
                           timings=[{'method':r['method'],'numeric_status':r['numeric_status'],'warm_ms':r['warm_complete_prompt_batch']['median_ms']} for r in d['records']])
                save(status_path,report);print('DONE '+name+' '+d['status'],flush=True)
        report['status']='GPU_DIAGNOSTIC_COMPLETE';report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        report.update(status='ERROR',error=str(error));raise
    finally:
        save(status_path,report);telemetry.terminate();telemetry.wait(timeout=5);telemetry_log.close()


if __name__=='__main__':main()
