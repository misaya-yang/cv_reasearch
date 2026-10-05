import sys
sys.path.insert(0, '/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/code_resume_v1/scripts')
import experiment_pipeline as ep
PEERS = {23852: 952542795}
original_inventory = ep.gpu_inventory
def inventory():
    observed=original_inventory()
    allowed=set()
    for pid,start in PEERS.items():
        current=ep.process_identity(pid)
        if current and current['start_ticks']==start:
            allowed.add(pid)
    return dict(observed,pids=[pid for pid in observed['pids'] if pid not in allowed])
class SharedPipeline(ep.Pipeline):
    def __init__(self,*args,**kwargs):
        kwargs['inventory']=inventory
        super().__init__(*args,**kwargs)
ep.Pipeline=SharedPipeline
ep.main()
