"""Verify completed autonomous-window artifacts and protected scoring sources."""
from pathlib import Path
import ast
import hashlib
import json
import re
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT.parent/'cv_data/a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old=json.loads((DATA/'x1013_W58_score_repair/inputs.json').read_text())['files']
    assert all(sha(ROOT/p)==digest for p,digest in old.items())
    result={'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'protected_source_and_old_evidence_hashes':len(old),'runs':{}}
    for name,n in [('W59',1285),('W60',1285),('W61',400),('W62',1285),('W63',63)]:
        out=DATA/('x1013_'+name)
        config=json.loads((out/'config.json').read_text());seal=json.loads((out/'complete.json').read_text())
        scored=json.loads((out/'scored.json').read_text())
        rows=list(map(json.loads,(out/'episodes.jsonl').read_text().splitlines()))
        assert len(rows)==config['n']==seal['n']==scored['n']==n
        assert sha(out/'config.json')==seal['config_sha256'] and sha(out/'episodes.jsonl')==seal['episodes_sha256']
        assert all(sha(r['prediction'])==r['prediction_sha256'] for r in rows)
        assert all(r['query_GT_reads']==0 for r in rows)
        for path,digest in config['sources'].items():
            candidate=ROOT/path
            if name=='W63' and path=='scripts/x1012/w63_task_switch.py': candidate=out/'producer_source.py'
            assert sha(candidate)==digest,(name,path)
        result['runs'][name]=dict(n=n,hashes_verified=True,prediction_GT_reads=0,
                                 config_sha256=sha(out/'config.json'),complete_sha256=sha(out/'complete.json'),
                                 scored_sha256=sha(out/'scored.json'))
    scripts=['diagnostic_metrics','test_diagnostic_metrics','w45_solvability','w49_token_resolution',
             'w58_score_audit','w59_normalization_causal','w59_window_analysis','w59_window_status',
             'w60_pure_background','w60_auc_support_audit','w61_reference_stability','w62_background_channels',
             'w63_task_switch','w63_failure_boards','w63_failure_source_audit','w58_window_verify']
    for name in scripts: ast.parse((ROOT/'scripts/x1012'/(name+'.py')).read_text())
    result['python_parse_files']=len(scripts)
    reports=list((ROOT/'docs/research/x1012_tasks').glob('W58*REPORT.md'))
    reports += [ROOT/'docs/research/x1012_tasks'/p for p in
                ('W59_normalization_causal_REPORT.md','W60_pure_background_REPORT.md',
                 'W61_reference_stability_REPORT.md','W62_background_channels_REPORT.md','W63_task_switch_REPORT.md',
                 'W63_failure_source_audit_REPORT.md')]
    broken=[];link_count=0
    for p in reports:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#')[0]
            if not target:continue
            link_count+=1
            if not (p.parent/target).exists():broken.append([str(p),target])
    assert not broken,broken
    result['local_report_links_verified']=link_count
    snapshots=list(map(json.loads,(DATA/'x1013_W59/resource_snapshots.jsonl').read_text().splitlines()))
    result['resource_snapshots']=dict(n=len(snapshots),max_observed_experiment_CPU_percent=max(r['cpu_percent_of_machine'] for r in snapshots),
                                     max_observed_RSS_GiB=max(r['rss_gib'] for r in snapshots),MPS_used=any(r['mps_used'] for r in snapshots),
                                     limitation='One-second snapshots of producer process trees, not continuous whole-machine telemetry. Four single-thread inference workers maximum.')
    assert result['resource_snapshots']['max_observed_experiment_CPU_percent']<=50
    assert not result['resource_snapshots']['MPS_used']
    result['score_repair_regression']=json.loads((DATA/'x1013_W58_score_repair/verification.json').read_text())
    result['task_switch_metric_checks']=json.loads((DATA/'x1013_W63/metric_checks.json').read_text())
    result['status']='PASS'
    path=ROOT/'evidence/research/w58_autonomous_window_verification.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
