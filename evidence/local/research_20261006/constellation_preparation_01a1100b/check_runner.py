"""Four full-dimensional synthetic entries through two workers; no GT key reads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='constellation_runner_check_') as folder:
        folder=Path(folder);source=folder/'input.npz';output=folder/'predictions'
        q=np.zeros((4096,1024),dtype=np.float32);r=np.zeros_like(q);q[:,4]=1;r[:,4]=1
        cov=np.zeros((64,64));base=np.full((64,64),.2)
        anchors=np.array([10+10j,20+10j,10+20j,20+20j])
        for k,point in enumerate(anchors):
            i=int(point.imag)*64+int(point.real);query=point+28+28j;j=int(query.imag)*64+int(query.real)
            r[i]=0;r[i,k]=1;cov[int(point.imag),int(point.real)]=1
            q[j]=0;q[j,k]=1
        np.savez_compressed(source,q=q,r=r,cov=cov,base=base,
                            query_gt=np.array([{'forbidden':True}],dtype=object))
        cmd=[sys.executable,str(ROOT/'scripts/run_reference_constellation.py'),'--inputs',
             *([str(source)]*4),'--out',str(output),'--workers','2']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        if result.returncode:raise RuntimeError(result.stderr)
        assert json.loads((output/'prediction_complete.json').read_text())['occurrences']==4
        paths=sorted(output.glob('*.npz'));assert len(paths)==4
        first=None
        for path in paths:
            receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['input_keys_read']==['q','r','cov','base'] and not receipt['query_gt_used']
            assert receipt['method']['retained_poses']==1 and not receipt['method']['abstention']
            with np.load(path,allow_pickle=False) as packet:
                for key in ('constellation','base_control','clipped_base_control','bag_mode_control','aligned_prior_control'):
                    assert packet[key].shape==(131072,)
                if first is None:first=packet['constellation'].copy()
                else:np.testing.assert_array_equal(first,packet['constellation'])
        again=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        assert again.returncode!=0 and 'FileExistsError' in again.stderr
        report=dict(synthetic_occurrences=4,feature_shape=[4096,1024],workers=2,
                    repeated_occurrences_preserved=True,query_gt_sentinel_unread=True,
                    non_abstaining_feature_to_mask_path_checked=True,masks_1024=True,
                    overwrite_rejected=True,real_episodes=0,no_gpu=True,no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
