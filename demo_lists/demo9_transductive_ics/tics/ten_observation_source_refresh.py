"""Current-source public-FoRIS tiny CPU composition for m05/m10 only."""
import argparse,json,hashlib,sys,time
from pathlib import Path
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
 own=Path(__file__).resolve().parents[1];sys.path[:0]=[str(own/'scripts'),str(own)]
 from ten_direction_context_cpu import make_fixture
 from tics.ten_observation import METHODS
 start=time.monotonic();ctx,encoder,fixture_audit=make_fixture(Path('/root/autodl-tmp/demo8_local_verification/foris_source'),seed=6033,image_size=128)
 records=[]
 with torch.inference_mode():
  for m in METHODS:
   for role in ('naive','method'):
    before=dict(extra=ctx.budget.extra_b2_calls,actual=ctx.budget.actual_b2_calls)
    z=m['naive' if role=='naive' else 'function'](ctx)
    assert z['mask'].shape==tuple(ctx.work_hw) and z['mask'].dtype==torch.bool
    assert torch.isfinite(z['score']).all()
    records.append(dict(id=m['id'],role=role,state='PASSED',before=before,
                        after=dict(extra=ctx.budget.extra_b2_calls,actual=ctx.budget.actual_b2_calls),
                        abstain=z.get('audit',{}).get('abstain'),audit=z.get('audit',{})))
 source=Path(__file__).with_name('ten_observation.py')
 out=dict(state='CPU_SOURCE_REFRESH_PASSED',actual_source_execution=True,CUDA_initialized=torch.cuda.is_initialized(),no_query_GT=True,
          source_hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},records=records,
          factory_scope='real public FoRIS source + fake/color CPU encoder; bilinear refiner; not production DINO/CRF',
          fixture_audit=fixture_audit,extra_view_requests=ctx.budget.extra_b2_calls,actual_B2_counter=ctx.budget.actual_b2_calls,
          elapsed_seconds=time.monotonic()-start,production_accuracy_unmeasured=True)
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=1));print(json.dumps(out))
if __name__=='__main__':main()
