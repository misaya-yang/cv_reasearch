"""Two-worker synthetic entry check, repeated occurrences and unread GT sentinel."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='huber_runner_check_') as folder:
        folder=Path(folder)
        source=folder/'input.npz'
        y=np.array([[.9,.1,.1,.1,.1,.1,.1]])
        np.savez(source,y=y,a=np.ones_like(y),edge_i=np.zeros(6,dtype=int),
                 edge_j=np.arange(1,7),edge_w=np.full(6,.05),
                 query_gt=np.array([{'forbidden':True}],dtype=object))
        output=folder/'predictions'
        cmd=[sys.executable,str(ROOT/'scripts/run_huber_graph.py'),'--inputs',
             *([str(source)]*4),'--out',str(output),'--workers','2']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        if result.returncode:
            raise RuntimeError(result.stderr)
        completed=json.loads((output/'prediction_complete.json').read_text())
        assert completed['occurrences']==4 and not completed['query_gt_used']
        paths=sorted(output.glob('*.npz'));assert len(paths)==4
        first=None
        for path in paths:
            receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['input_keys_read']==['y','a','edge_i','edge_j','edge_w']
            assert receipt['method']['gap']>=0
            with np.load(path,allow_pickle=False) as packet:
                for key in ('huber','boxed_quadratic_control','original_quadratic_control','pregraph_control'):
                    assert packet[key].shape==(131072,)
                if first is None:first=packet['huber'].copy()
                else:np.testing.assert_array_equal(first,packet['huber'])
        again=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        assert again.returncode!=0 and 'FileExistsError' in again.stderr
        report=dict(synthetic_occurrences=4,workers=2,duplicate_occurrences_preserved=True,
                    forbidden_gt_key_unread=True,all_masks_1024=True,
                    existing_output_overwrite_rejected=True,real_episodes=0,no_gpu=True,no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
