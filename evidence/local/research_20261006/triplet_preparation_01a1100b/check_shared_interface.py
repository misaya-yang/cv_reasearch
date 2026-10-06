"""Small shared-hook checks: synthetic inputs, no server/GT/full shared suite."""
from argparse import Namespace
import ast
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
from ics.methods import prepared_cpu_bundle as bundle
from ics.methods import reference_triplet_relations as triplet
from ics.methods import huber_graph
from ics import experiment

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared_runner', ROOT/'scripts/run_cpu_feature_candidates.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
torch.set_num_threads(1)
original_methods = ('adjacency','huber','color_bottleneck','constellation','reference_shape','reference_covariance','query_recurrence','reference_prior_shift','reference_quadratic')
assert bundle.METHODS == original_methods and 'reference_triplet_relations' in bundle.AVAILABLE_METHODS
cfg = triplet.Config(**json.loads(json.dumps(asdict(triplet.Config()))))
triplet._validate_config(cfg)
assert cfg == triplet.Config()
q = np.eye(6, dtype=np.float32); r = np.roll(q, 1, axis=0)
cov = np.array([[.1,.3,.5],[.4,.6,.9]])
score = np.array([[.13,.71,.55],[.33,.23,.95]], dtype=np.float32)
base = np.full((2,3), .123, dtype=np.float32)
seen = {}
original_predict = triplet.predict

def observe(q_, r_, cov_, score_, cfg_, **kwargs):
    assert np.array_equal(score_, score) and not np.array_equal(score_, base)
    assert kwargs['include_pair_control'] is True
    seen['original_packet_score'] = True
    return original_predict(q_, r_, cov_, score_, cfg_, **kwargs)

with tempfile.TemporaryDirectory(prefix='triplet_shared_interface_') as tmp:
    directory = Path(tmp)
    run = directory/'episode'; run.mkdir()
    for name in ('predictions','fields','receipts'):(run/name).mkdir()
    configuration = dict(root=str(directory), prepared_methods=['reference_triplet_relations'],
                         method_configs={'reference_triplet_relations':json.loads(json.dumps(asdict(cfg)))})
    row = dict(occurrence_id='000000', key='synthetic_only')
    with patch.object(experiment,'load_inputs',return_value=((q,r,cov,score),dict(synthetic=True))), \
         patch.object(huber_graph,'make_mean_inputs',return_value=((),dict(graph_storage_dtype='float64'))), \
         patch.object(bundle,'mean_base',return_value=(base,dict(synthetic_base=True))), \
         patch.object(triplet,'predict',side_effect=observe):
        receipt = bundle.one_episode(row,run,configuration)
    expected = {'mean.control','clipped_mean.control','reference_triplet_relations','triplet_zero.control',
                'triplet_absolute.control','triplet_no_third.control','triplet_pro_m2.control'}
    with np.load(run/'predictions/000000.npz',allow_pickle=False) as predictions:
        assert set(predictions.files)==expected
        assert all(predictions[key].shape==(131072,) for key in predictions.files)
    with np.load(run/'fields/000000.npz',allow_pickle=False) as fields:
        normalized=(score-score.min())/max(float(score.max()-score.min()),1e-6)
        assert np.array_equal(fields['triplet_zero.control'],normalized.astype(float))
        assert np.array_equal(fields['triplet_no_third.control'],fields['triplet_zero.control'])
    json.dumps(receipt,allow_nan=False)
    assert not receipt['query_gt_opened'] and receipt['independent_candidate_methods']==1
    # Generate the actual inference configuration, stopping before worker startup.
    np.savez(directory/'features.npz',q=q,r=r)
    np.savez(directory/'packet.npz',cov=cov,score=score)
    manifest=directory/'manifest.json'
    manifest.write_text(json.dumps([dict(c=0,fold=0,support='synthetic_ref',query='synthetic_query',
                                  feature_export='features.npz',packet_export='packet.npz')]))
    args=Namespace(backend='prepared',prepared_methods=['reference_triplet_relations'],primary_method='reference_triplet_relations',
                   base='mean',strengths=[1.],primary_strength=1.,workers=1,threads=1,cpu_budget=1,memory_gb=1,
                   manifest=manifest,limit=None,root=directory,out=directory/'config_only',base_key='mean.control',
                   exposure='synthetic interface check only')
    class ConfigurationReady(Exception):pass
    with patch.object(runner.mp,'get_context',side_effect=ConfigurationReady):
        try:runner.infer(args)
        except ConfigurationReady:pass
        else:raise AssertionError('Unexpected worker launch')
    stored=json.loads((args.out/'config.json').read_text())
    assert stored['method_configs']['reference_triplet_relations']==json.loads(json.dumps(asdict(cfg)))
    assert stored['primary']=='reference_triplet_relations' and stored['independent_methods']==1
    hashes=stored['code_sha256']
    assert 'src/ics/methods/reference_triplet_relations.py' in hashes
    assert 'src/ics/methods/pro_reference_relations.py' in hashes
    for path in ('src/ics/methods/reference_triplet_relations.py','src/ics/methods/pro_reference_relations.py'):
        assert hashes[path]==experiment.sha(ROOT/path)
# Inspect only the actual CLI declarations; no second shared inference run.
syntax=ast.parse((ROOT/'scripts/run_cpu_feature_candidates.py').read_text())
options={}
for node in ast.walk(syntax):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument' and node.args and isinstance(node.args[0],ast.Constant):
        key=node.args[0].value
        if key in ('--prepared-methods','--primary-method'):
            options[key]={item.arg:ast.literal_eval(item.value) for item in node.keywords if item.arg in ('default','choices')}
assert options['--prepared-methods']['default']==['adjacency','huber','color_bottleneck','constellation']
assert all('reference_triplet_relations' in options[key]['choices'] for key in options)
report=dict(original_methods9_preserved=True,default4_preserved=True,triplet_available=True,
            original_packet_score_used=seen['original_packet_score'],all_four_controls_preserved=True,
            pair_control_enabled=True,packed1024_outputs=True,receipt_JSON_serializable=True,
            JSON_configuration_roundtrip=True,config_generation_checked_without_workers=True,
            triplet_and_Pro_dependency_hashes_preserved=True,query_GT_opened=False,
            CLI_choices_checked=True,shared_full_suite_rerun=False,
            code_sha256={path:experiment.sha(ROOT/path) for path in ('src/ics/methods/reference_triplet_relations.py',
                        'src/ics/methods/prepared_cpu_bundle.py','scripts/run_cpu_feature_candidates.py')})
(OUT/'shared_interface_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
