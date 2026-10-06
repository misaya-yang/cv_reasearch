"""Two-worker synthetic entry; repeated inputs and forbidden GT key remain isolated."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='color_bottleneck_check_') as folder:
        folder=Path(folder);source=folder/'input.npz';output=folder/'predictions'
        rgb=np.zeros((128,128,3),dtype=np.uint8);rgb[:,64:]=255
        base=np.full((64,64),.45);base[:,32:]=.55
        base[0,0]=.95;base[0,33]=.05
        np.savez(source,rgb=rgb,base=base,query_gt=np.array([{'forbidden':True}],dtype=object))
        cmd=[sys.executable,str(ROOT/'scripts/run_color_bottleneck.py'),'--inputs',
             *([str(source)]*4),'--out',str(output),'--workers','2']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        if result.returncode:raise RuntimeError(result.stderr)
        assert json.loads((output/'prediction_complete.json').read_text())['occurrences']==4
        paths=sorted(output.glob('*.npz'));assert len(paths)==4
        first=None
        for path in paths:
            receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['input_keys_read']==['rgb','base'] and not receipt['query_gt_used']
            with np.load(path,allow_pickle=False) as packet:
                for key in ('color_bottleneck','original_base_control','same_resize_base_control',
                            'same_resize_clipped_control','color_only_control'):
                    assert packet[key].shape==(131072,)
                if first is None:first=packet['color_bottleneck'].copy()
                else:np.testing.assert_array_equal(first,packet['color_bottleneck'])
        again=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        assert again.returncode!=0 and 'FileExistsError' in again.stderr
        report=dict(synthetic_occurrences=4,workers=2,repeats_preserved=True,
                    query_gt_sentinel_unread=True,masks_1024=True,overwrite_rejected=True,
                    real_episodes=0,no_gpu=True,no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
