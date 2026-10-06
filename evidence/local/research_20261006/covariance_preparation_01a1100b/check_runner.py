"""Two-worker full-dimensional covariance inference; query GT remains unread."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='covariance_runner_check_') as directory:
        root=Path(directory)
        q=np.zeros((4096,1024),dtype=np.float32);r=np.zeros_like(q);q[:,3]=1;r[:,3]=1
        cov=np.zeros((64,64));cov[4:12,4:12]=1
        for row in range(4,12):
            for col in range(4,12):
                i=row*64+col;r[i]=0;r[i,0 if col<8 else 1]=1
        base=np.full((64,64),.1);base[24:32,8:16]=.49;base[24:32,8:12]=.8;base[24:32,40:48]=.87
        blend=np.sqrt(1-.9**2)
        for row in range(24,32):
            for col in range(8,16):
                i=row*64+col;q[i]=0;q[i,0 if col<12 else 1]=.8;q[i,2]=.6
            for col in range(40,48):
                i=row*64+col;q[i]=0;q[i,0]=.9 if col<44 else blend;q[i,1]=blend if col<44 else .9
        sentinel=np.array([{'query_GT_must_remain_unread':True}],dtype=object)
        np.savez_compressed(root/'features.npz',q=q,r=r,query_gt=sentinel)
        np.savez_compressed(root/'packet.npz',cov=cov,score=base,truth=sentinel,native=sentinel)
        row=dict(c=1,fold=0,support='support',query='query',feature_export='features.npz',packet_export='packet.npz')
        manifest=root/'rows.json';manifest.write_text(json.dumps([row,row]))
        output=root/'run'
        cmd=[sys.executable,str(ROOT/'scripts/run_cpu_feature_candidates.py'),'infer','--backend','prepared',
             '--prepared-methods','reference_covariance','--primary-method','reference_covariance',
             '--manifest',str(manifest),'--root',str(root),'--out',str(output),
             '--workers','2','--threads','1','--cpu-budget','2','--memory-gb','2']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        if result.returncode:raise RuntimeError(result.stderr)
        seal=json.loads((output/'sealed.json').read_text());assert seal['n']==2 and not seal['query_gt_opened']
        for path in (output/'receipts').glob('*.json'):
            receipt=json.loads(path.read_text());info=receipt['methods']['reference_covariance']
            assert not info['abstention'] and not receipt['query_gt_opened']
            assert info['reference_templates']==1
            for move in info['search']['moves']:assert move['after']>=move['before']-1e-10
        with np.load(output/'predictions/000000.npz',allow_pickle=False) as packet:
            for key in ('reference_covariance','covariance_trace.control','covariance_bilinear.control','mean_nearest.control'):
                assert packet[key].shape==(131072,)
        report=dict(synthetic_occurrences=2,workers=2,active_covariance_search=True,
                    no_original_image_geometry_needed=True,query_GT_sentinels_unread=True,
                    masks_1024=True,predictions_sealed=True,real_episodes=0,no_gpu=True,no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
