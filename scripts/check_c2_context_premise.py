#!/usr/bin/env python3
"""C2 prerequisite failures perform no work; verify paired macro and six gates."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import run_c2_context_premise as runner
import score_c2_context_premise as scorer
from score_c2_context_premise import summarize
from ics.official_data import array_hash


def main():
    choice='C2concat/raw'
    report=dict(passed=True,choice=choice,datasets={name:dict(passed=True,n=5,means=dict(real=.995,wrong=.94))
        for name in ('coco','lvis','pascal_part','paco_part','suim')})
    assert runner.self_qualified(report,choice)
    for change in ('count','iou','wrong','choice'):
        bad=copy.deepcopy(report)
        if change=='count':bad['datasets']['lvis']['n']=4
        if change=='iou':bad['datasets']['lvis']['means']['real']=.97
        if change=='wrong':bad['datasets']['lvis']['means']['wrong']=.99
        if change=='choice':bad['choice']='C2concat/deb'
        assert not runner.self_qualified(bad,choice)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);decision=root/'decision.json';self_run=root/'self';self_run.mkdir()
        args=['context','--decision',str(decision),'--self-run',str(self_run),'--parent-run','/missing/parent',
              '--basis','/missing/basis','--raw-cache-profile','/missing/profile.json','--out',str(root/'run')]
        for stage in ('representation','self'):
            decision.write_text(json.dumps(dict(candidates={'C2':dict(passed=stage=='self',representation=choice)})))
            (self_run/'report.json').write_text(json.dumps(dict(passed=False)))
            with patch.object(sys,'argv',args),patch.object(runner,'load_inputs',side_effect=AssertionError('input opened')), \
                 patch.object(runner,'cache_host',side_effect=AssertionError('host built')),contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            assert not (root/'run').exists()
        label=np.zeros((7,11),np.uint8);label[:,:4]=1
        path=root/'label.png';Image.fromarray(label*255).save(path)
        row=dict(query_mask_path=str(path),query_mask_hash=array_hash(label))
        # Independent nearest indexing followed by explicit16x16 block means.
        raster=label[(np.arange(1024)*7//1024)[:,None],(np.arange(1024)*11//1024)[None,:]]
        truth=(raster.reshape(64,16,64,16).mean((1,3))>=.5).ravel()
        fields=dict(probability=truth.astype(float),direct_probability=np.full(4096,.5),wrong_probability=1-truth.astype(float))
        assert scorer.episode_auc(row,np.ones((1024,1024),np.uint8),fields)==dict(minimum=1.,direct=.5,wrong=0.)
        circle=np.zeros((1024,1024),np.uint8);circle[:,:256]=1
        assert scorer.episode_auc(row,circle,fields)=={}
        # An unsealed result cannot open even one query annotation.
        unsealed=root/'unsealed';unsealed.mkdir()
        (unsealed/'sealed.json').write_text(json.dumps(dict(state='INCOMPLETE')))
        (unsealed/'config.json').write_text('{}')
        with patch.object(sys,'argv',['score','--run',str(unsealed),'--out',str(root/'score')]), \
             patch.object(scorer.Image,'open',side_effect=AssertionError('query GT opened before seal')):
            try:scorer.main();raise AssertionError('Unsealed run scored')
            except ValueError:pass
    records=[]
    for name in runner.DATASETS:
        for half in (0,1):
            for cls,repeats,direct,gain in ((0,20,.5,.04),(1,1,.7,.02)):
                for _ in range(repeats):
                    records.append(dict(dataset=name,half=half,fold=0,class_id=cls,
                                        auc=dict(minimum=direct+gain,direct=direct,wrong=.45)))
    table,passed=summarize(records)
    assert passed and all(abs(row['delta_direct']-.03)<1e-10 for row in table)
    # A high episode-weighted average cannot compensate a failing data/half.
    failed=copy.deepcopy(records)
    for row in failed:
        if row['dataset']=='paco_part' and row['half']==1:
            row['auc']['minimum']=row['auc']['direct']+.005
    assert not summarize(failed)[1]
    missing=copy.deepcopy(records)
    for row in missing:
        if row['dataset']=='coco' and row['half']==0:row['auc']={}
    table,passed=summarize(missing)
    assert not passed and table[0]['delta_direct'] is None
    print('PASS: C2 prerequisite gates, no labels before seal, raw-to-CLI geometry, single-label N/A, paired class macro and six AUC gates')


if __name__=='__main__':main()
