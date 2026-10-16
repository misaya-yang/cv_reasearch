"""One read-only status/resource snapshot for the native W59/W60 heartbeat."""
from pathlib import Path
import json
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / 'cv_data/a'


def snapshot():
    processes, states = {}, {}
    window = json.loads((DATA/'x1013_W59/window.json').read_text())
    names = [job['run'] for job in window['jobs']]
    for name in names:
        out = DATA / ('x1013_' + name)
        ep = out / 'execution.json'
        if not ep.exists():
            states[name] = {'status': 'not_started'}
            continue
        e = json.loads(ep.read_text())
        state = dict(status=e['status'], pid=e['pid'],
                     records=sum(1 for _ in (out/'episodes.jsonl').open()),
                     complete=(out/'complete.json').exists(), scored=(out/'scored.json').exists())
        try:
            p = psutil.Process(e['pid'])
            same = abs(p.create_time()-e['create_time']) < .01 and any(
                Path(arg).name == Path(e['script']).name for arg in p.cmdline())
            state['identity_verified_running'] = same and p.is_running()
            if same:
                for child in [p] + p.children(recursive=True):
                    try:
                        child.cpu_percent(None)
                        processes[child.pid] = child
                    except psutil.Error:
                        pass
        except psutil.Error:
            state['identity_verified_running'] = False
        states[name] = state
    time.sleep(1)
    cpu, rss = 0., 0
    for p in processes.values():
        try:
            cpu += p.cpu_percent(None)
            rss += p.memory_info().rss
        except psutil.Error:
            pass
    result = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  sample_seconds=1, cpu_percent_of_machine=cpu/psutil.cpu_count(),
                  rss_gib=rss/2**30, mps_used=False, runs=states)
    with (DATA/'x1013_W59/resource_snapshots.jsonl').open('a') as f:
        f.write(json.dumps(result)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    snapshot()
