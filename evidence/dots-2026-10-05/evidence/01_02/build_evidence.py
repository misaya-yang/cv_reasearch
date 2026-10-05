from pathlib import Path
import json,hashlib,shutil
R=Path(__file__).resolve().parents[3];O=Path(__file__).parent
S={
'prior120_locked':'strong_method_controls_20261005/validation121_locked/prior120_frozen_comparisons.json',
'new100_freeze':'cpu_method/additional121/arrived100_frozen.json',
'new100_field_audit':'cpu_method/additional121/arrived100_field_audit.json',
'cloud_families':'strong_method_controls_20261005/photo_group_statistics_pooled120.json',
'hypothesis_sensitivity':'cpu_method/hypothesis_selector120/statistics_compatibility.json',
'hypothesis_capacity':'cpu_method/hypothesis_selector120_identity_diagnostic/report.json',
'mmd120':'cpu_method/mmd_hypotheses120/report.json',
'mmd_native':'cpu_method/mmd_hypotheses120/native_stage_statistics.json',
'mmd_selection':'cpu_method/mmd_hypotheses120/selection_diagnostic.json',
'anchor_oracle20':'cpu_method/reference_anchor_identifiability20/report.json',
'anchor_oracle20_plan':'cpu_method/reference_anchor_identifiability20/plan.json',
'qda120':'cpu_method/source_qda120/report.json',
'context_global_control':'cpu_method/hypothesis_selector120/additional_context_controls_randomstate.json',
}
DROP={'audit','records','keys','group_episode_keys','per_class','oracle_selections','output_files','source_sha256'}
def compact(x):
 if isinstance(x,dict):return {k:compact(v)for k,v in x.items()if k not in DROP}
 if isinstance(x,list):return [compact(v)for v in x]
 return x
manifest=[]
for name,p in S.items():
 path=R/p
 if not path.exists():print('MISSING',p);continue
 raw=path.read_bytes();data=json.loads(raw);out=dict(source_report=p,source_sha256=hashlib.sha256(raw).hexdigest(),extracted_utc='2026-10-05T07:45:00Z',extraction='Only large per-episode/per-class arrays, file-hash maps, and raw audit arrays omitted; metric values unchanged.',report=compact(data));dest=O/(name+'.json');dest.write_text(json.dumps(out,ensure_ascii=False,indent=2));manifest.append(dict(source_report=p,source_sha256=out['source_sha256'],evidence_file=dest.name))
shutil.copy2(R/'running_idea_ledger.csv',O/'running_idea_ledger.csv')
(O/'manifest.json').write_text(json.dumps(manifest,indent=2));print('copied',len(manifest),'reports')
