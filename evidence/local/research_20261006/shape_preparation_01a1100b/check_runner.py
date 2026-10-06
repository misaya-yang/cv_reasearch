"""Two-worker shape entry with physical image geometry and unread GT sentinel."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='shape_runner_check_') as directory:
        root=Path(directory)
        q=np.zeros((4096,1024),dtype=np.float32);r=np.zeros_like(q);q[:,2]=1;r[:,2]=1
        cov=np.zeros((64,64));cov[4:12,4:12]=1
        base=np.full((64,64),.1);base[24:32,24:32]=.7;base[24:32,31]=.49
        base[54:56,:]=.9
        q.reshape(64,64,1024)[24:32,24:32]=0;q.reshape(64,64,1024)[24:32,24:32,0]=1
        q.reshape(64,64,1024)[54:56]=0;q.reshape(64,64,1024)[54:56,:,1]=1
        r.reshape(64,64,1024)[4:12,4:12]=0;r.reshape(64,64,1024)[4:12,4:12,0]=1
        sentinel=np.array([{'forbidden_GT':True}],dtype=object)
        np.savez_compressed(root/'features.npz',q=q,r=r,query_gt=sentinel)
        # Source score differs from the desired fixture base, so this check calls
        # the backend with a supplied internally constructed case, not a fake cache parity.
        np.savez_compressed(root/'packet.npz',cov=cov,score=base,truth=sentinel,native=sentinel)
        row=dict(c=1,fold=0,support='support',query='query',feature_export='features.npz',
                 packet_export='packet.npz',support_image_hw=[64,64],query_image_hw=[64,64])
        manifest=root/'rows.json';manifest.write_text(json.dumps([row,row]))
        output=root/'run'
        cmd=[sys.executable,str(ROOT/'scripts/run_cpu_feature_candidates.py'),'infer','--backend','prepared',
             '--prepared-methods','reference_shape','--primary-method','reference_shape',
             '--manifest',str(manifest),'--root',str(root),'--out',str(output),
             '--workers','2','--threads','1','--cpu-budget','2','--memory-gb','2']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
        if result.returncode:raise RuntimeError(result.stderr)
        seal=json.loads((output/'sealed.json').read_text());assert seal['n']==2 and not seal['query_gt_opened']
        for path in (output/'receipts').glob('*.json'):
            receipt=json.loads(path.read_text());info=receipt['methods']['reference_shape']
            assert not receipt['query_gt_opened'] and not info['abstention']
            assert info['reference_image_hw']==[64,64] and info['query_image_hw']==[64,64]
            for move in info['search']['moves']:assert move['after']>=move['before']-1e-10
        with np.load(output/'predictions/000000.npz',allow_pickle=False) as packet:
            for key in ('reference_shape','shape_generic_square.control','shape_bilinear.control','mean_nearest.control'):
                assert packet[key].shape==(131072,)
        # Geometry is a required binding, not inferred from the canonical square.
        missing=dict(row);missing.pop('support_image_hw')
        manifest.write_text(json.dumps([missing]))
        invalid=list(cmd);invalid[invalid.index(str(output))]=str(root/'missing_geometry')
        rejected=subprocess.run(invalid,capture_output=True,text=True,timeout=10)
        assert rejected.returncode!=0 and 'original positive integer H/W' in rejected.stderr
        assert not (root/'missing_geometry').exists()
        report=dict(synthetic_occurrences=2,workers=2,active_shape_search_checked=True,
                    query_GT_sentinels_unread=True,predictions_sealed=True,masks_1024=True,
                    missing_original_geometry_rejected_before_output=True,
                    real_episodes=0,no_gpu=True,no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))


if __name__=='__main__':main()
