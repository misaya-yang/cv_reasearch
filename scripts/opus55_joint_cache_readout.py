"""Paired raw/APD readouts of already sealed real separate/joint O24.

No encoder is constructed, no cache is written, and query labels are denied
until the standard instrument seals all fields. The existing FoRIS position
basis and one shared gate from the separate arm are fixed controls, not fitted
to this experiment. This does not isolate attention from joint-canvas geometry.
"""
from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'evidence/local/false_alarm_floor_20261010'),
               str(REPO / 'scripts'), str(REPO / 'src')]
import inmask_evidence as base
from joint_encoding import deny_query_labels
from raw_feature_cache import RawFeatureCache, tensor_hash


def main():
    parser = base.arguments(['fields'])
    parser.add_argument('--joint-source', required=True,
                        help='sealed joint run under inmask, or absolute path')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source, out = base.opened(args)
    prior = Path(args.joint_source)
    if not prior.is_absolute():
        prior = base.ROOT / 'inmask' / prior
    prior = prior.resolve()
    seal = base.read(prior / 'sealed.json')
    success = base.read(prior / 'SUCCESS.json')
    original = base.read(prior / 'config.json')
    recipe = base.read(prior / 'joint_plan.json')
    assert success['state'] == 'REAL_FIELDS_SEALED'
    assert success['sealed_sha256'] == base.sha(prior / 'sealed.json')
    assert seal['config_sha256'] == base.sha(prior / 'config.json')
    assert original['manifest_sha256'] == base.sha(source.folder / 'manifest.json')
    assert original['where'] == str(source.folder)
    assert original['foris'] == str(source.foris)
    assert seal['n'] == success['n'] == len(original['ids'])
    assert [source.describe(r)['id'] for r in source.rows[:seal['n']]] == original['ids']
    assert args.limit is None or args.limit == seal['n']
    args.limit = seal['n']
    grid, side = original['grid'], recipe['arguments']['side']
    assert grid * 16 == side
    records = [base.read(prior / 'records' / f'{i:06d}.json') for i in range(seal['n'])]
    assert [r['id'] for r in records] == original['ids']
    for i, r in enumerate(records):
        assert r['sha256'] == seal['fields'][r['id']]
        assert base.sha(prior / 'fields' / f'{i:06d}.npz') == r['sha256']
        assert r['query_GT_reads'] == r['query_label_attempts'] == 0
    assets = Path(args.assets).resolve()
    basis_path = assets / 'native_assets/positional_basis.pt'
    basis = torch.load(basis_path, map_location='cpu', weights_only=True)['basis'].float()
    projection = torch.eye(1024) - basis @ basis.T
    out.mkdir(parents=True, exist_ok=True)
    freeze = dict(state='FROZEN_BEFORE_READOUT_OR_QUERY_GT', n=seal['n'], grid=grid,
        side=side, joint_source=str(prior), encoder_calls=0, cache_writes=0,
        source_seal_sha256=base.sha(prior / 'sealed.json'),
        source_recipe_sha256=base.sha(prior / 'joint_plan.json'),
        source_config_sha256=base.sha(prior / 'config.json'),
        record_sha256={str(i): base.sha(prior / 'records' / f'{i:06d}.json') for i in range(seal['n'])},
        basis_sha256=base.sha(basis_path), script_sha256=base.sha(__file__),
        instrument_sha256=base.sha(base.__file__),
        gate='Both arms use the same separate-arm lawful reference-FG/query-mean cosine < .8 gate',
        projection='Fixed existing FoRIS positional basis; identical projection for both arms',
        fields='Original raw fields copied exactly; fixed evidence fields recomputed after common APD',
        limitations='Joint canvas also changes geometry; APD does not identify pure cross-attention causality')
    plan_path = out / 'readout_plan.json'
    if plan_path.exists():
        assert base.read(plan_path) == freeze
    else:
        base.write(plan_path, freeze)
    caches, attempts = {}, []

    def no_write(*unused, **kwargs):
        raise RuntimeError('This readout must not write any raw cache')
    RawFeatureCache._write = no_write

    def raw(request):
        entry = Path(request['entry_path'])
        assert base.sha(entry) == request['entry_sha256']
        input_path, profile_path = Path(request['input_path']), Path(request['profile_path'])
        assert base.sha(input_path) == request['input_file_sha256']
        array = np.load(input_path, allow_pickle=False)
        assert tensor_hash(array) == request['input_tensor_hash']
        if str(profile_path) not in caches:
            cache = RawFeatureCache(profile_path.parent.parent, base.read(profile_path))
            assert cache.folder == profile_path.parent
            assert cache.profile_id == request['profile_id']
            assert cache.profile['weights_sha256'] == recipe['source_sha256'].get('weights_sha256',
                base.read(recipe['profile_path'])['weights_sha256'])
            caches[str(profile_path)] = cache
        cache = caches[str(profile_path)]
        assert cache.folder / cache.key(array) / 'entry.json' == entry
        value = cache.read(array, ('O/24',))['O/24']
        assert tensor_hash(value) == request['raw_tensor_sha256']
        gh, gw = array.shape[1] // 16, array.shape[2] // 16
        unit = F.normalize(torch.from_numpy(value.reshape(gh, gw, 1024)), dim=2)
        assert tensor_hash(unit.numpy()) == request['unit_tensor_sha256']
        if request['operation'] == 'joint.canvas':
            assert any(p.get('actual_full_canvas') for p in base.read(entry)['provenance'])
        return unit

    def compute(current, rows, described, i):
        record = records[i]
        assert base.sha(prior / 'records' / f'{i:06d}.json') == freeze['record_sha256'][str(i)]
        requests = {r['operation']: r for r in record['raw_requests']}
        assert set(requests) == {'separate.reference', 'separate.query', 'joint.canvas'}
        rs, qs = raw(requests['separate.reference']), raw(requests['separate.query'])
        joint = raw(requests['joint.canvas'])
        assert tuple(rs.shape) == tuple(qs.shape) == (grid, grid, 1024)
        assert tuple(joint.shape) == (grid, 2 * grid, 1024)
        rs, qs = rs.reshape(-1, 1024), qs.reshape(-1, 1024)
        rj, qj = joint[:, :grid].reshape(-1, 1024), joint[:, grid:].reshape(-1, 1024)
        _, mask, _ = current.images(rows[i])
        coverage = base.reference_coverage(mask, grid=grid, side=side)
        candidate = base.candidate_tokens(current.prediction(rows[i]), grid)
        inside = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                               (grid, grid), mode='nearest')[0, 0].reshape(-1) > .5
        mean_q = qs.mean(0)
        semantic = float(F.normalize(rs[inside].mean(0), dim=0) @ mean_q /
                         (mean_q.norm() + 1e-6)) if inside.any() else None
        apply = semantic is None or semantic < .8
        source_file = prior / 'fields' / f'{i:06d}.npz'
        with np.load(source_file, allow_pickle=False) as original_fields:
            fields = {'raw.' + k: original_fields[k].copy() for k in original_fields.files}
        notes = {}
        for arm, R, Q in [('separate', rs, qs), ('joint', rj, qj)]:
            if i < 3:
                replay, _ = base.evidence(R, Q, coverage, candidate)
                for key, value in replay.items():
                    if key.startswith('aux.'):
                        continue
                    assert np.array_equal(value.astype(np.float32), fields[f'raw.{arm}.{key}']), \
                        f'Raw evidence replay differs: {i}/{arm}/{key}'
            Rp, Qp = (F.normalize(R @ projection.T, dim=1), F.normalize(Q @ projection.T, dim=1)) if apply else (R, Q)
            found, note = base.evidence(Rp, Qp, coverage, candidate)
            fields.update({f'apd.{arm}.{k}': v for k, v in found.items() if not k.startswith('aux.')})
            notes[arm] = dict(note, reference_cosine_to_raw=float((Rp * R).sum(1).mean()),
                             query_cosine_to_raw=float((Qp * Q).sum(1).mean()))
        return fields, dict(side=side, shared_apd_gate=apply, separate_semantic=semantic,
            arms=notes, confusable=notes['separate'].get('confusable'), encoder_calls=0,
            raw_cache_writes=0, query_GT_reads=0, query_label_attempts=len(attempts),
            source_record_sha256=freeze['record_sha256'][str(i)],
            readout_plan_sha256=base.sha(plan_path))

    with deny_query_labels(source.rows, assets, attempts):
        base.make_fields(args, compute=compute, grid=grid)
    assert not attempts
    print(json.dumps(dict(state='READOUT_FIELDS_READY', n=seal['n'], encoder_calls=0,
                          raw_cache_writes=0, query_label_attempts=0)), flush=True)


if __name__ == '__main__':
    main()
