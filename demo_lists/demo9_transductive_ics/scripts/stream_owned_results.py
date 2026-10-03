#!/usr/bin/env python3
"""Stream small owned science outputs over an existing SSH session.

CPU-only observer. Never starts/restarts a stage, waits for a client ACK,
touches CUDA or delays the resource guard's terminal shutdown. The caller
writes JSONL locally; a transport cutoff is handled as incomplete transfer,
not as evidence that the science job failed.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import zlib


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True,type=Path)
    p.add_argument('--analysis',required=True,type=Path)
    p.add_argument('--guard',required=True,type=Path)
    p.add_argument('--stream-cases',action='store_true',help='Send frozen rows/small traces incrementally, never wait for ACK')
    a=p.parse_args();sent={};sent_cases=set();begin=time.monotonic()
    def compressed(name,value):
        payload=zlib.compress(json.dumps(value,allow_nan=False).encode(),6)
        print(json.dumps(dict(artifact=name,payload_zlib_base64=base64.b64encode(payload).decode())),flush=True)
    while time.monotonic()-begin<(1300 if a.stream_cases else 980):
        for name,path in [('report',a.report),('analysis',a.analysis),('guard',a.guard)]:
            if name in sent or not path.is_file():continue
            try:value=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):continue
            state=str(value.get('state',''))
            if name=='report' and a.stream_cases:
                for row in value.get('records',[]):
                    key=tuple(row[k] for k in ['fold','e','c'])
                    if key in sent_cases:continue
                    trace=Path(row['trace_archive'])
                    expected='trace_%d_%d_%d.npz'%key
                    trace=trace if trace.is_absolute() else path.parent/trace
                    if trace.name!=expected or trace.resolve().parent!=path.parent.resolve():raise ValueError('Trace escapes owned report folder')
                    if not trace.is_file():continue
                    data=trace.read_bytes()
                    if len(data)>16*1024*1024:raise ValueError('Unexpected large trace, not a small science output')
                    if hashlib.sha256(data).hexdigest()!=row['trace_sha256']:raise ValueError('Frozen trace checksum differs')
                    compressed('case',dict(record=row,trace_base64=base64.b64encode(data).decode()))
                    sent_cases.add(key)
                if state in ('COMPLETED','ERROR'):
                    compressed('report_metadata',{**{k:v for k,v in value.items() if k!='records'},
                        'record_count':len(value.get('records',[])),'exported_case_count':len(sent_cases)})
                    sent[name]=state
                continue
            ready=(name=='report' and state in ('COMPLETED','ERROR') or
                   name=='analysis' and state in ('CPU_FROZEN_OUTPUT_ANALYSIS','CPU_NATIVE_MEMBERSHIP_ANALYSIS') or
                   name=='guard' and (state.startswith('SHUTDOWN_') or state=='DEFER_FOREIGN_GPU'))
            if ready:
                if a.stream_cases:compressed(name,value)
                else:print(json.dumps(dict(artifact=name,value=value)),flush=True)
                sent[name]=state
        if 'guard' in sent:return
        time.sleep(.1)
    print(json.dumps(dict(artifact='observer',value={'state':'TRANSPORT_OBSERVER_TIMEOUT','sent':sent})),flush=True)


if __name__=='__main__':main()
