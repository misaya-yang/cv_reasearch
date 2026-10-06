"""Shared infer/score protocol checks on synthetic caches; no real benchmark results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np


def run(arguments,success=True):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_cpu_feature_candidates.py'),*arguments],
                          capture_output=True,text=True,timeout=50)
    if success and result.returncode:raise RuntimeError(result.stderr)
    if not success and not result.returncode:raise AssertionError('Expected rejected invocation')
    return result


def main():
    with tempfile.TemporaryDirectory(prefix='prepared_bundle_check_') as directory:
        root=Path(directory)
        q=np.zeros((4096,1024),dtype=np.float32);r=np.zeros_like(q);q[:,4]=1;r[:,4]=1
        cov=np.zeros((64,64));score=np.linspace(0,1,4096,dtype=np.float32).reshape(64,64)
        points=np.array([10+10j,20+10j,10+20j,20+20j])
        for k,point in enumerate(points):
            i=int(point.imag)*64+int(point.real);query=point+28+28j;j=int(query.imag)*64+int(query.real)
            r[i]=0;r[i,k]=1;cov[int(point.imag),int(point.real)]=1
            q[j]=0;q[j,k]=1
        sentinel=np.array([{'query_GT_must_not_be_opened':True}],dtype=object)
        np.savez_compressed(root/'features.npz',q=q,r=r,query_gt=sentinel)
        rgb=np.zeros((128,128,3),dtype=np.uint8);rgb[:,64:]=255
        np.savez_compressed(root/'rgb.npz',rgb=rgb,query_gt=sentinel)
        truth=np.zeros((1024,1024),dtype=bool);truth[600:800,600:800]=True
        native=np.zeros_like(truth);native[650:750,650:750]=True
        np.savez_compressed(root/'packet.npz',cov=cov,score=score,truth=np.packbits(truth),native=np.packbits(native))
        np.savez_compressed(root/'poison.npz',cov=cov,score=score,truth=sentinel,native=np.packbits(native))
        rows=[]
        for occurrence in range(4):
            group=occurrence//2
            rows.append(dict(c=group+1,fold=0,support=f'support_{group}',query=f'query_{group}',
                             support_photo_id=f'support-{group}',query_photo_id=f'query-{group}',
                             key='shared_source',feature_export='features.npz',packet_export='packet.npz',
                             query_rgb_export='rgb.npz',support_image_hw=[64,64],query_image_hw=[64,64]))
        manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
        output=root/'bundle'
        infer=['infer','--manifest',str(manifest),'--root',str(root),'--out',str(output),
               '--backend','prepared','--primary-method','adjacency','--workers','2','--threads','1',
               '--cpu-budget','2','--memory-gb','2','--exposure','synthetic protocol test; not a segmentation result']
        run(infer)
        seal=json.loads((output/'sealed.json').read_text())
        assert seal['n']==4 and not seal['query_gt_opened']
        inference=json.loads((output/'inference_manifest.json').read_text())
        assert all(not any(key in row for key in ('c','fold','support','query','truth')) for row in inference)
        for path in sorted((output/'receipts').glob('*.json')):
            receipt=json.loads(path.read_text())
            assert set(receipt['methods'])=={'adjacency','huber','color_bottleneck','constellation'}
            assert receipt['independent_candidate_methods']==4 and not receipt['query_gt_opened']
        # Metadata tampering must fail before scoring; then restore exact bytes.
        config_path=output/'config.json';before=config_path.read_bytes()
        config=json.loads(before);config['exposure']='tampered'
        config_path.write_text(json.dumps(config))
        rejected=run(['score','--out',str(output)],success=False)
        assert 'Sealed metadata changed' in rejected.stderr
        config_path.write_bytes(before)
        run(['score','--out',str(output)])
        report=json.loads((output/'score/report.json').read_text())
        assert report['n']==4 and report['classes']==2 and report['photo_groups']==2
        assert report['predictions_sealed_before_scoring']
        assert report['primary_methods']==['adjacency','huber','color_bottleneck','constellation']
        # Newly prepared shape method coexists with all original four arms.
        all_output=root/'all_five'
        all_infer=list(infer)
        all_infer[all_infer.index(str(output))]=str(all_output)
        insert=all_infer.index('--primary-method')
        all_infer[insert:insert]=['--prepared-methods','adjacency','huber','color_bottleneck',
                                 'constellation','reference_shape']
        run(all_infer)
        run(['score','--out',str(all_output)])
        five_report=json.loads((all_output/'score/report.json').read_text())
        assert len(five_report['primary_methods'])==5 and 'reference_shape' in five_report['scores']
        six_output=root/'all_six'
        six_infer=list(all_infer)
        six_infer[six_infer.index(str(all_output))]=str(six_output)
        six_infer.insert(six_infer.index('--primary-method'),'reference_covariance')
        run(six_infer)
        run(['score','--out',str(six_output)])
        six_report=json.loads((six_output/'score/report.json').read_text())
        assert len(six_report['primary_methods'])==6 and 'reference_covariance' in six_report['scores']
        seven_output=root/'all_seven'
        seven_infer=list(six_infer)
        seven_infer[seven_infer.index(str(six_output))]=str(seven_output)
        seven_infer.insert(seven_infer.index('--primary-method'),'query_recurrence')
        run(seven_infer)
        run(['score','--out',str(seven_output)])
        seven_report=json.loads((seven_output/'score/report.json').read_text())
        assert len(seven_report['primary_methods'])==7 and 'query_recurrence' in seven_report['scores']
        eight_output=root/'all_eight'
        eight_infer=list(seven_infer)
        eight_infer[eight_infer.index(str(seven_output))]=str(eight_output)
        eight_infer.insert(eight_infer.index('--primary-method'),'reference_prior_shift')
        run(eight_infer)
        run(['score','--out',str(eight_output)])
        eight_report=json.loads((eight_output/'score/report.json').read_text())
        assert len(eight_report['primary_methods'])==8 and 'reference_prior_shift' in eight_report['scores']
        nine_output=root/'all_nine'
        nine_infer=list(eight_infer)
        nine_infer[nine_infer.index(str(eight_output))]=str(nine_output)
        nine_infer.insert(nine_infer.index('--primary-method'),'reference_quadratic')
        run(nine_infer)
        run(['score','--out',str(nine_output)])
        nine_report=json.loads((nine_output/'score/report.json').read_text())
        assert len(nine_report['primary_methods'])==9 and 'reference_quadratic' in nine_report['scores']
        # Same source reused twice is preserved as separate occurrences.
        assert len(list((output/'predictions').glob('*.npz')))==4
        # Poisoned query truth still permits inference; scoring fails when it finally opens truth.
        poison_row=dict(rows[0],packet_export='poison.npz')
        poison_manifest=root/'poison_manifest.json';poison_manifest.write_text(json.dumps([poison_row]))
        poison_output=root/'poison_bundle'
        poison_infer=list(infer)
        poison_infer[poison_infer.index(str(manifest))]=str(poison_manifest)
        poison_infer[poison_infer.index(str(output))]=str(poison_output)
        run(poison_infer)
        poison_score=run(['score','--out',str(poison_output)],success=False)
        assert 'Object arrays cannot be loaded' in poison_score.stderr
        # The default occupancy route remains callable with its existing schema.
        one=np.zeros_like(q);one[:,0]=1
        np.savez_compressed(root/'one_mode.npz',q=one,r=one)
        old_row=dict(rows[0],feature_export='one_mode.npz')
        old_manifest=root/'old_manifest.json';old_manifest.write_text(json.dumps([old_row]))
        old_output=root/'old_route'
        run(['infer','--manifest',str(old_manifest),'--root',str(root),'--out',str(old_output),
             '--workers','1','--threads','1','--cpu-budget','1','--memory-gb','2'])
        assert json.loads((old_output/'config.json').read_text())['primary']=='regional.l1'
        assert json.loads((old_output/'sealed.json').read_text())['n']==1
        # Shared original quadratic producer matches the existing MEAN on this fixture.
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from ics.methods.mean_graph import predict
        original,_=predict(q,r,cov,score)
        with np.load(output/'fields/000000.npz',allow_pickle=False) as data:
            difference=float(np.max(np.abs(data['mean.control']-original)))
        assert difference<1e-6,difference
        result=dict(synthetic_occurrences=4,methods=report['primary_methods'],independent_methods=4,
                    shared_backend_not_a_method=True,repeated_sources_preserved=True,
                    photo_groups=2,query_identifiers_removed_from_worker_manifest=True,
                    feature_and_RGB_GT_sentinels_unread=True,packet_GT_deferred_until_score=True,
                    metadata_tampering_rejected=True,default_occupancy_backend_checked=True,
                    mean_control_maximum_field_difference=difference,
                    five_method_shared_infer_and_score_checked=True,
                    six_method_shared_infer_and_score_checked=True,
                    seven_method_shared_infer_and_score_checked=True,
                    eight_method_shared_infer_and_score_checked=True,
                    nine_method_shared_infer_and_score_checked=True,
                    all_predictions_sealed_before_scoring=True,real_episodes=0,no_gpu=True,no_server=True)
        Path(__file__).with_name('bundle_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))


if __name__=='__main__':main()
