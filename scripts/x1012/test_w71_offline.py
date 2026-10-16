"""Engineering checks: exact ties, persisted deadlines, recovery and offline IO."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import numpy as np
import w71_runtime as rt
import w71_mechanism as mech
import w71_offline as queue


class OfflineTests(unittest.TestCase):
    def test_exact_tied_thresholds(self):
        rng=np.random.default_rng(71)
        for _ in range(30):
            values=rng.integers(0,7,size=(8,9)).astype(np.float32)
            gt=rng.random((8,9))<.4; valid=rng.random((8,9))>.1
            got=mech.exact_curve(values,gt,valid)
            candidates=[(values>=t)&valid for t in np.unique(values[valid])]
            wanted=max(100*(p&gt).sum()/max((p|(gt&valid)).sum(),1) for p in candidates)
            self.assertAlmostEqual(got['best_threshold_iou'],wanted)
            for recall,v in got['at_recall'].items():
                self.assertGreaterEqual(v['recall'],float(recall))
                expected=min((int((p&~gt).sum()) for p in candidates if (p&gt).sum()>=np.ceil(float(recall)*(gt&valid).sum())))
                self.assertEqual(v['FP'],expected)

    def test_atomic_record_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); prediction=root/'prediction'; prediction.write_bytes(b'committed')
            record=root/'record.json'; rt.write(record,dict(prediction_path=str(prediction),prediction_sha256=rt.sha(prediction)))
            self.assertIsNotNone(rt.valid_record(record))
            prediction.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError): rt.valid_record(record)

    def test_deadline_does_not_reset(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(rt,'OUT',Path(temp)):
            rt.write(rt.OUT/'session.json',dict(started_epoch=time.time()-48*3600-5,deadline_epoch=time.time()-5))
            with self.assertRaises(rt.StopRun): rt.checkpoint()
            self.assertLess(rt.deadline(),time.time())

    def test_network_is_denied_in_fresh_process(self):
        code='import w71_runtime as r,socket; r.offline(); socket.getaddrinfo("example.com",443)'
        run=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0); self.assertIn('W71_OFFLINE_DNS_DENIED',run.stderr)

    def test_retry_and_continue_independent_stages(self):
        real_sleep=time.sleep
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(rt,'OUT',Path(temp)):
            root=Path(temp)
            rt.write(root/'session.json',dict(started_epoch=time.time(),deadline_epoch=time.time()+60))
            cfg=dict(root=temp,retries_per_stage=1,stage_timeout_seconds=10)
            def command(stage,config):
                path=root/(stage.replace(':','_')+'.calls')
                code=f'from pathlib import Path; import sys; p=Path({str(path)!r}); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); '
                code+=f'sys.exit(2 if {stage!r}=="mechanism-predict" or ({stage!r}=="official-predict:isaid" and n==1) else 0)'
                return [sys.executable,'-B','-c',code]
            with mock.patch.object(queue,'verify_package',return_value=cfg), mock.patch.object(queue,'stage_command',side_effect=command), \
                 mock.patch.object(queue,'write_report'), mock.patch.object(rt,'event'), \
                 mock.patch.object(queue.time,'sleep',side_effect=lambda n:real_sleep(min(n,.005))):
                queue.supervise()
            self.assertEqual(rt.read(root/'result.json')['status'],'COMPLETE_WITH_FAILURES')
            self.assertEqual((root/'mechanism-predict.calls').read_text(),'2')
            self.assertFalse((root/'mechanism-score.calls').exists())
            self.assertEqual((root/'official-predict_isaid.calls').read_text(),'2')
            self.assertEqual(rt.read(root/'stages/official-score_lvis.json')['status'],'complete')

    def test_process_group_cleanup(self):
        process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
        queue.terminate_group(process)
        self.assertIsNotNone(process.poll())

    def test_memory_worker_budget(self):
        self.assertEqual(rt.worker_budget(4,14),4)
        self.assertEqual(rt.worker_budget(4,10),2)
        self.assertEqual(rt.worker_budget(4,8),1)
        self.assertEqual(rt.worker_budget(2,30),2)
        self.assertEqual(rt.worker_budget(4,30,current_job_gib=18),2)
        self.assertEqual(rt.worker_budget(4,30,current_job_gib=20),1)

    def test_resource_wait_is_bounded(self):
        import psutil
        with mock.patch.object(psutil,'cpu_percent',return_value=100), \
             mock.patch.object(psutil,'virtual_memory',return_value=type('Memory',(),{'available':20*2**30})()), \
             mock.patch.object(rt,'checkpoint'),mock.patch.object(rt,'heartbeat'),mock.patch.object(rt.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'Resource budget'):
                rt.capacity(maximum_wait=0)

    def test_factor_effect_and_null_uncertainty(self):
        rows=[]
        for cohort in ('S','AVSeg','FIVES_train'):
            for n in range(3):
                arms={}
                for layer in mech.LAYERS:
                    for arm in mech.ARMS:
                        i=50+5*int(arm[0]); p=i+40
                        arms[f'L{layer}_{arm}']=dict(I=i,U=100,TP=i,FP=40,FN=60-i,GT=60,predicted=p,
                            best_threshold_iou=float(i),at_recall={str(t):dict(FP_per_GT=2/3) for t in mech.RECALLS})
                rows.append(dict(cohort=cohort,qimg=f'{cohort}_q{n}',rimg=f'{cohort}_r{n}',apd={'19':True,'23':True},arms=arms))
        summary=mech.summarize(rows)
        for group in summary['cohorts'].values():
            effects=group['single_factor_effects']['23']
            self.assertEqual(effects['soft_label']['current_minus_control_pp'],5.)
            np.testing.assert_allclose(effects['soft_label']['family_ci_photo'],[5,5])
            self.assertEqual(effects['always_debias']['current_minus_control_pp'],0.)


if __name__=='__main__': unittest.main()
