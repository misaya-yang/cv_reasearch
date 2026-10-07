import os,sys,time,signal,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEADLINE=1791397770.0
if '--now' not in sys.argv:
    while time.time()<DEADLINE:
        time.sleep(min(30,DEADLINE-time.time()))
(ROOT/'shutdown_event.json').write_text(json.dumps({'time':time.time(),'action':'terminate supervisord; user authorized two-hour server shutdown'})+'\n')
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: args=(proc/'cmdline').read_bytes().split(b'\0')
    except OSError: continue
    if any(a.endswith(b'/supervisord') for a in args):
        os.kill(int(proc.name),signal.SIGTERM)
