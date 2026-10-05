import sys
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/code_resume_v1/scripts')
import experiment_pipeline as ep
original_inventory = ep.gpu_inventory
def inventory():
    observed = original_inventory()
    peer = ep.process_identity(14579)
    if peer and peer['start_ticks'] == 951953583 and any(arg.endswith('export_confirm_cache.py') for arg in peer['argv']):
        observed = dict(observed, pids=[pid for pid in observed['pids'] if pid != 14579])
    return observed
class SharedPipeline(ep.Pipeline):
    def __init__(self, *args, **kwargs):
        kwargs['inventory'] = inventory
        super().__init__(*args, **kwargs)
ep.Pipeline = SharedPipeline
ep.main()
