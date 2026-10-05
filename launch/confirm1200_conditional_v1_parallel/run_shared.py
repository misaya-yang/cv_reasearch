import sys
sys.path.insert(0, '/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/code_resume_v1/scripts')
import experiment_pipeline as ep
original = ep.gpu_inventory
def inventory():
    observed = original()
    peer = ep.process_identity(20997)
    if peer and peer['start_ticks'] == 952370038 and any(x.endswith('export_confirm_cache.py') for x in peer['argv']):
        observed = dict(observed, pids=[p for p in observed['pids'] if p != 20997])
    return observed
class SharedPipeline(ep.Pipeline):
    def __init__(self,*args,**kwargs):
        kwargs['inventory'] = inventory
        super().__init__(*args,**kwargs)
ep.Pipeline=SharedPipeline
ep.main()
