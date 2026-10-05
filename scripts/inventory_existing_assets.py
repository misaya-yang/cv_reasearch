#!/usr/bin/env python3
"""Read-only inventory; stdlib only, no model loading or downloads."""
import json
import os
from pathlib import Path
import subprocess


def text(path):
    try: return Path(path).read_text().strip()
    except OSError: return None


def command(argv):
    try:
        p=subprocess.run(argv,text=True,capture_output=True,timeout=15)
        return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as e: return {'error':str(e)}


def stat(path):
    p=Path(path)
    return {'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.is_file() else None}


root=Path('/root/autodl-tmp/demo9_extent')
mp=root/'results/extent_v1/episodes.json'
manifest=json.loads(mp.read_text()) if mp.exists() else {}
rows=manifest.get('episodes',[])
clean=[]
seen=set(); duplicates=[]; missing=[]
for r in rows:
    row={k:r[k] for k in ('fold','e','c','support','query')}
    key=f"{r['fold']}_{r['e']}_{r['c']}"
    ident=(r['c'],Path(r['support']).name,Path(r['query']).name)
    if ident in seen: duplicates.append(key)
    seen.add(ident); clean.append(row)
    checks={'feature':root/f'cache/evidence_v1/feat/{key}.pt',
            'packet':root/f'results/extent_v1/run/packets/{key}.npz',
            'support':Path(manifest['data_root'])/r['support'],
            'query':Path(manifest['data_root'])/r['query'],
            'support_mask':Path(manifest['annotation_root'])/Path(r['support']).with_suffix('.png'),
            'query_mask':Path(manifest['annotation_root'])/Path(r['query']).with_suffix('.png')}
    absent=[k for k,p in checks.items() if not p.is_file()]
    if absent: missing.append({'key':key,'missing':absent})
assets=[stat('/root/demo4_cache/models/dinov3-vitl16-timm/'+name) for name in ('config.json','model.safetensors')]
for key in ('foris_root','projection_basis','data_root','annotation_root'):
    if manifest.get(key): assets.append(stat(manifest[key]))
result={'mode':'read_only_no_tensor_runtime', 'cpu_max':text('/sys/fs/cgroup/cpu.max'),
        'memory_max':text('/sys/fs/cgroup/memory.max'),
        'nvidia':command(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader']),
        'processes':command(['ps','-eo','pid,ppid,pcpu,pmem,etime,args','--sort=-pcpu']),
        'manifest_path':str(mp),'episode_count':len(rows),'duplicate_episode_keys':duplicates,
        'missing_by_episode':missing,'assets':assets,
        'manifest':{**{k:manifest[k] for k in ('data_root','annotation_root','foris_root','projection_basis','seed') if k in manifest},'episodes':clean},
        'note':'241 is historical DEV; cloud220 membership is not inferred. File existence is not content/parity verification.'}
print(json.dumps(result,indent=2))
