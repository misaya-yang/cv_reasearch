"""Finite queue of fixed evidence diagnostics on existing complete raw/output.

No encoder or raw producer is invoked. Read-only source views preserve all
existing predictions; the stopped Chest run contributes only its complete
record prefix, explicitly not its unfinished600 benchmark.
"""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / 'cv_data'
ROOT = REPO / 'evidence/local/false_alarm_floor_20261010'
OUT = ROOT / 'execution'
PROFILE = DATA / 'a/paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158/profile.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def view(source, dataset, name, stopped=False):
    source = DATA / 'a' / source
    rows = read(source / 'manifest.json')
    if stopped:
        records = []
        for i, row in enumerate(rows):
            path = source / 'records' / f'{i:06d}.json'
            if not path.exists():
                break
            record = read(path)
            assert record['index'] == i and record['episode_id'] == row['episode_id']
            assert record['query_GT_reads'] == record['query_label_attempts'] == 0
            assert sha(source / 'predictions' / record['filename']) == record['prediction_sha256']
            assert sha(source / 'fields' / record['filename']) == record['field_sha256']
            records.append(record)
        assert records and len(records) < len(rows)
        selected = rows[:len(records)]
        scope = f'Previously stopped run: first{len(records)}/{len(rows)} completed records; no restart, new input or encoding'
    else:
        seal = read(source / 'sealed.json')
        records = read(source / 'inference.json')
        assert len(records) == len(rows) == seal['n']
        assert sha(source / 'inference.json') == seal['inference_sha256']
        by_id = {r['episode_id']: r for r in records}
        selected = [r for r in rows if r['dataset'] == dataset]
        records = [by_id[r['episode_id']] for r in selected]
        scope = 'All existing sealed records of this dataset; original order and legal duplicate draws retained'
    assert selected and all(r['dataset'] == dataset for r in selected)
    target = OUT / 'source_views' / name
    target.mkdir(parents=True, exist_ok=True)
    profile = read(PROFILE)
    frozen = dict(dataset=dataset, n=len(selected), scope=scope,
        source=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
        source_config_sha256=sha(source / 'config.json'),
        source_seal_sha256=sha(source / 'sealed.json') if (source / 'sealed.json').exists() else None,
        record_ids=[r['episode_id'] for r in selected],
        source_record_sha256={r['episode_id']: sha(source / 'records' / f"{r['index']:06d}.json") for r in records})
    if (target / 'VIEW.json').exists():
        assert read(target / 'VIEW.json') == frozen
    else:
        write(target / 'VIEW.json', frozen)
        write(target / 'manifest.json', selected)
        write(target / 'inference.json', records)
        write(target / 'config.json', dict(n=len(selected), profile_path=str(PROFILE),
            profile_sha256=sha(PROFILE), weights_sha256=profile['weights_sha256'],
            datasets=dict(Counter(r['dataset'] for r in selected)), source_view=frozen))
        for leaf in ('predictions', 'fields'):
            (target / leaf).symlink_to(source / leaf, target_is_directory=True)
    assert read(target / 'manifest.json') == selected
    return target, len(selected), scope


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / 'cached_queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = [
            ('extended_foris_seed0_20261010', 'isaid', 'existing_isaid60', False),
            ('extended_foris_seed0_20261010', 'isic', 'existing_isic600', False),
            ('fundus_foris_seed0_20261010', 'fundus', 'existing_fundus200', False),
            ('chest_foris_seed0_20261010', 'lung', 'existing_chest167', True)]
        jobs = []
        for source, dataset, name, stopped in sources:
            folder, n, scope = view(source, dataset, name, stopped)
            jobs.append(dict(name=name, cohort=str(folder), n=n, scope=scope))
        jobs.append(dict(name='existing_deep100', cohort=str(DATA / 'a/deepglobe_role_competition100_20261010'),
                         n=100, scope='All existing custom DeepGlobe road100; exploratory, public exact pairing unavailable'))
        plan = dict(jobs=jobs, source_sha256=sha(__file__), instrument_sha256=sha(ROOT / 'inmask_evidence.py'),
                    shell_sha256=sha(ROOT / 'run_source.sh'), threads_per_worker=2, cpu_workers=4,
                    new_encoder_calls=0, new_raw_writes=0,
                    purpose='Test floor-law mechanisms on already available datasets; do not regenerate assets to fill nominal sample counts')
        if (OUT / 'cached_queue_plan.json').exists():
            assert read(OUT / 'cached_queue_plan.json') == plan
        else:
            write(OUT / 'cached_queue_plan.json', plan)
        states = {}
        for job in jobs:
            folder = ROOT / 'inmask' / job['name']
            if (folder / 'report.json').exists():
                states[job['name']] = dict(state='ALREADY_SCORED', n=job['n'])
                continue
            command = ['bash', str(ROOT / 'run_source.sh'), '4', '--cohort', job['cohort'], '--name', job['name']]
            with (OUT / f"{job['name']}.log").open('ab') as log:
                child = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                         env=dict(os.environ, PYTHON=sys.executable))
                states[job['name']] = dict(state='RUNNING', pid=child.pid, argv=command, n=job['n'])
                write(OUT / 'cached_queue_activity.json', dict(state='RUNNING', pid=os.getpid(),
                    updated_utc=datetime.now(timezone.utc).isoformat(), current=job['name'], jobs=states))
                result = child.wait()
            states[job['name']].update(state='SCORED' if result == 0 and (folder / 'report.json').exists() else 'FAILED',
                                      returncode=result)
            write(OUT / 'cached_queue_activity.json', dict(state='ADVANCING', pid=os.getpid(),
                updated_utc=datetime.now(timezone.utc).isoformat(), jobs=states))
            print(json.dumps(dict(name=job['name'], **states[job['name']])), flush=True)
        write(OUT / 'cached_queue_activity.json', dict(state='QUEUE_FINISHED', pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(), jobs=states,
            all_scored=all(v['state'] in ('SCORED', 'ALREADY_SCORED') for v in states.values())))


if __name__ == '__main__':
    main()
