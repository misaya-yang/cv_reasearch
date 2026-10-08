#!/usr/bin/env python3
"""Gated fixed C1 + same-form last-O control, with complete inherited B0 readout.

--identity-only runs a fixed implementation probe with c=0; it cannot admit a
candidate. Actual C1 runs require the frozen representation prerequisite.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from ics import fine_readout
from ics.foris import run_foris
from ics.methods import mean_control, rcg
from ics.official_data import file_hash, load_inputs
from ics.representations import observe_branches, reference_margin, source_reference_roles, unit_tokens
from frozen_dino_plan import locked_run, prediction_file, read_rows, write
from frozen_unary_candidates import c1_unary, solve_parent_system
from run_insid3_baseline import build_model
from run_m4_baselines import render


def qualified_choice(decision):
    item = decision.get('candidates', {}).get('C1', {})
    if item.get('passed') is not True:
        return None
    choice = item['representation']
    branch, layer, mode = choice.split('/')
    if branch not in ('O', 'Q', 'K', 'V') or not 1 <= int(layer) <= 24 or mode not in ('raw', 'deb'):
        raise ValueError('Invalid frozen C1 representation')
    return choice


@contextmanager
def parent_system():
    """Read-only tap of the actual locked RCG matrix, RHS and initialization."""
    saved, captured = rcg.cg, {}
    def tap(matrix, rhs, **kwargs):
        captured.update(matrix=matrix.copy(), rhs=rhs.copy(), unary=kwargs['x0'].copy())
        return saved(matrix, rhs, **kwargs)
    rcg.cg = tap
    try:
        yield captured
    finally:
        rcg.cg = saved


def infer_episode(host, row, assets, device, choice, basis, identity_only):
    branch, layer, mode = choice.split('/')
    bank = {}
    original_extract = host._extract_features
    def extract(imgs):
        if imgs.shape[0]*imgs.shape[1] == 2:
            with observe_branches(host.encoder, layers=(int(layer),), branches=branch) as tapped:
                output = original_extract(imgs)
            bank.update(tapped)
            return output
        return original_extract(imgs)
    host._extract_features = extract
    try:
        image, reference_mask, query = load_inputs(row, assets)
        sampling_rng = np.random.get_state()
        native, got, mask, target = run_foris(host, image, reference_mask, query)
        processed = F.normalize(got['deb'][0].float(), dim=1)
        q, r = (processed[i].flatten(1).T.half().contiguous() for i in (1, 0))
        coverage = F.interpolate(mask[None, None].float(), (64, 64), mode='area')[0, 0].numpy()
        score = got['score'].float().numpy()
        with parent_system() as system:
            coarse, solver = rcg.predict(q, r, coverage, score)
        source_s = rcg.minmax(score).ravel()
        a = rcg.CONFIG['confidence_floor']+np.abs(2*source_s-1)
        a = (a/a.mean()).astype(np.float64)
        if not np.array_equal(system['rhs'], a*system['unary']):
            raise ValueError('Captured parent system differs from locked confidence/unary')
        mean, _ = mean_control.predict(q, r, coverage, score)
        debiased = bool((F.normalize(got['raw'][0].float(), dim=1)-got['deb'][0]).abs().max() > 1e-4)
        fine_features = fine_readout.shifted_features(host, target, debiased)
        parent_fine = fine_readout.field(fine_features, q, coarse)
        parent_mask = fine_readout.mask(parent_fine)
        foreground, background = source_reference_roles(got['reference_features'][0, 0], got['reference_fg'][0])
        chosen = unit_tokens(bank[f'{branch}/{layer}'], basis if mode == 'deb' else None)
        margin, valid = reference_margin(chosen, foreground, background)
        control_margin, control_valid = reference_margin(
            unit_tokens(got['raw'][0].flatten(2).transpose(1, 2)), foreground, background)
        fields, details, masks = {}, {}, dict(**{'foris.crf':native.numpy(), 'rcg':rcg.mask_from_field(coarse),
                                               'rcg.fine':parent_mask, 'mean':rcg.mask_from_field(mean)})
        for arm, d, eligible in [('c1', margin, valid), ('c1.last_o_control', control_margin, control_valid)]:
            unary, probability, confidence = c1_unary(system['unary'].reshape(64, 64), d.numpy(), parent_mask,
                                                     reference_valid=eligible and not identity_only)
            updated = solve_parent_system(system['matrix'], a, unary)
            fine = fine_readout.field(fine_features, q, updated)
            masks[arm] = fine_readout.mask(fine)
            fields[arm+'/coarse'], fields[arm+'/fine'] = updated, fine
            fields[arm+'/p'], fields[arm+'/c'] = probability, confidence
            if not np.any(confidence) and (not np.array_equal(updated, coarse) or not np.array_equal(fine, parent_fine)
                                  or not np.array_equal(masks[arm], parent_mask)):
                raise ValueError('C1 c=0 changed the complete inherited parent field/mask')
            details[arm] = dict(reference_valid=eligible, nonzero_confidence=int(np.count_nonzero(confidence)),
                                c0_complete_identity=not np.any(confidence))
        if not all(np.array_equal(x, y) for x, y in zip(sampling_rng, np.random.get_state())):
            raise ValueError('Inference altered frozen sampling RNG')
        fields.update(parent_coarse=coarse, parent_fine=parent_fine, source_score=score)
        shape = (query.height, query.width)
        payload = {frame+'/'+arm:np.packbits(value if frame == 'cli' else render(value, shape))
                   for arm,value in masks.items() for frame in ('original', 'cli')}
        if identity_only:
            for arm in ('c1', 'c1.last_o_control'):
                for frame in ('original', 'cli'):
                    if not np.array_equal(payload[frame+'/'+arm], payload[frame+'/rcg.fine']):
                        raise ValueError('C1 c0 differs in a final metric frame')
        return payload, shape, fields, dict(solver=solver, arms=details)
    finally:
        host._extract_features = original_extract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=REPO.parent/'cv_data')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--decision', type=Path)
    p.add_argument('--basis', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    p.add_argument('--split', choices=['smoke', 'dev', 'val'], required=True)
    p.add_argument('--identity-only', action='store_true')
    a = p.parse_args()
    a.assets, a.out = a.assets.resolve(), a.out.resolve()
    if a.identity_only:
        if a.split != 'smoke':
            p.error('Identity probe is implementation-only, not a development candidate')
        choice = 'O/24/raw'
    else:
        if a.decision is None:
            p.error('Actual C1 requires its frozen representation decision')
        choice = qualified_choice(json.loads(a.decision.read_text()))
        if choice is None:
            print('C1 prerequisite failed; no model/reference work', flush=True);return
    rows = read_rows(a.manifest)
    prepared = json.loads((a.manifest.parent/'prepared.json').read_text())
    if not a.identity_only and (prepared.get('state') != 'FRESH_MANIFEST_FROZEN' or prepared.get('split_role') != a.split):
        raise ValueError('Candidate requires the matching fresh development/validation split')
    config = dict(candidate_id='C1' if not a.identity_only else 'C1 c0 implementation probe',
                  split_role=a.split, arms=['foris.crf', 'rcg', 'rcg.fine', 'mean', 'c1', 'c1.last_o_control'],
                  prepared_protocol=prepared, choice=choice, identity_only=a.identity_only,
                  manifest_sha256=file_hash(a.manifest), device=a.device,
                  code_sha256=file_hash(Path(__file__)), operator_sha256=file_hash(REPO/'scripts/frozen_unary_candidates.py'),
                  decision_sha256=file_hash(a.decision) if a.decision else None,
                  weights_sha256=file_hash(a.assets/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors'),
                  native_basis_sha256=file_hash(a.assets/'native_assets/positional_basis.pt'),
                  query_mask_in_inference=False, frozen_math='C1 linear p/c; actual parent H/A/CG; unchanged fine/0.5')
    config['branch'] = subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip()
    config['commit'] = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    source_files = [Path(__file__), REPO/'scripts/frozen_unary_candidates.py', REPO/'scripts/run_insid3_baseline.py',
                    REPO/'scripts/frozen_dino_plan.py', REPO/'scripts/run_m4_baselines.py',
                    *sorted((REPO/'src/ics').rglob('*.py')),
                    *sorted((a.assets/'third_party/foris_official').rglob('*.py')),
                    *sorted((a.assets/'third_party/crf_source/src/CRF').glob('*.py'))]
    native_source=a.assets/'third_party/crf_source/src/PermutohedralFiltering/source/cpu'
    source_files += sorted(native_source.rglob('*.cpp'))+sorted(native_source.rglob('*.h'))
    config['source_sha256'] = {str(f):file_hash(f) for f in source_files}
    basis = None
    if choice.endswith('/deb'):
        if a.basis is None:
            p.error('Chosen representation requires its own black basis')
        key = '/'.join(choice.split('/')[:2]);complete=json.loads((a.basis/'complete.json').read_text());entry=complete['bases'][key]
        path=a.basis/entry['path']
        if file_hash(path) != entry['sha256']:
            raise ValueError('Chosen own branch basis changed')
        config['own_basis_sha256']=entry['sha256'];basis=torch.load(path,map_location='cpu',weights_only=True)['basis']
    torch.set_num_threads(2)
    with locked_run(a.out):
        if any(f.name != 'run.lock' for f in a.out.iterdir()):
            raise FileExistsError('Use a fresh candidate output; unfinished attempts require inspection')
        write(a.out/'config.json', config);write(a.out/'manifest.json', rows)
        for file in source_files:
            target=a.out/'source'/('repo' if file.is_relative_to(REPO) else 'assets')/file.relative_to(REPO if file.is_relative_to(REPO) else a.assets)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target)
            if file_hash(target) != config['source_sha256'][str(file)]:
                raise ValueError('Implementation changed during candidate snapshot')
        for directory in ('predictions', 'fields'):
            (a.out/directory).mkdir()
        host = build_model(a.assets, a.device, baseline='foris')
        with torch.inference_mode(), (a.out/'inference.jsonl').open('w', buffering=1) as log:
            for row in rows:
                started=time.monotonic();payload,shape,fields,details=infer_episode(host,row,a.assets,a.device,choice,basis,a.identity_only)
                file=prediction_file(row);np.savez_compressed(a.out/'predictions'/file,original_hw=np.array(shape),**payload)
                np.savez_compressed(a.out/'fields'/file,**fields)
                record=dict(episode_id=row['episode_id'],prediction_file=file,
                            prediction_sha256=file_hash(a.out/'predictions'/file),fields_sha256=file_hash(a.out/'fields'/file),
                            inference_seconds=time.monotonic()-started,**details)
                log.write(json.dumps(record)+'\n');print(json.dumps(dict(episode_id=row['episode_id'],seconds=record['inference_seconds'],identity=a.identity_only)),flush=True)
        write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=len(rows),
              manifest_sha256=file_hash(a.out/'manifest.json'),config_sha256=file_hash(a.out/'config.json'),
              inference_index_sha256=file_hash(a.out/'inference.jsonl'),query_labels_opened=False))


if __name__ == '__main__':
    main()
