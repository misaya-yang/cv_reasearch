"""Automatic paired analysis; preserve both supplied and independent reports."""
import argparse,json,time
from pathlib import Path
from analyze import analyze
from pose_eval import auc,ARMS

def run(reference,extension):
 ref=Path(reference);ext=Path(extension)
 while not (ext/'report.json').exists():time.sleep(10)
 a=json.loads((ref/'report.json').read_text());b=json.loads((ext/'report.json').read_text())
 lookup={r['key']:r for r in b['results']};combined=[]
 for r in a['results']:
  e=lookup[r['key']];merged=dict(r);arms=dict(r['arms'])
  for own,other in [('diagonal','extension_diagonal'),('gic','extension_gic'),('shuffled','extension_shuffled')]:
   v=dict(e['arms'][other]);v['details']=dict(rho=e.get('rho',0),status=v['status'],source='unmodified user supplied algebraic dual-endpoint core/pose')
   if 'FALLBACK' in v['status'] or v['status']=='FATAL':v['details']['fallback']=v.get('reason',v['status'])
   arms[own]=v
  merged['arms']=arms;combined.append(merged)
 summary={arm:{f'AUC@{t}':auc([r['arms'][arm]['error']['pose'] for r in combined],t) for t in [5,10,20]} for arm in ARMS}
 d=ext/'paired_common';d.mkdir(exist_ok=True)
 (d/'report.json').write_text(json.dumps(dict(results=combined,summary=summary,fatal_failures=b['fatal_failures'])))
 analyze(d)
 v=json.loads((d/'paired_analysis.json').read_text());v['implementation_scope']='gic/diagonal/shuffled supplied algebraic solver; native/full native_extra/density controls frozen independent report. Same matches/inliers/native init. Bbox basis because author cache omits image dimensions; NOT unmodified NPZ loader nor anisotropic RoMaV2.'
 (d/'paired_analysis.json').write_text(json.dumps(v,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);ap.add_argument('--extension',required=True);args=ap.parse_args();run(args.reference,args.extension)
