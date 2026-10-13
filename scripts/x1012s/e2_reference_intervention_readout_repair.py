"""Resume E2 with only its K1 mask-check render on B1's original MPS device.

The six intervention masks retain v1's CPU readout. Neither existing predictions
nor the v1/config/pilot seal nor the first repair binding is changed. This fixes
a control check where an identical FP32 field straddles 0.5 after CPU versus MPS
bilinear interpolation by one pixel; the experiment itself is unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import e2_reference_intervention_repair as partition

base = partition.base
V1_COMPUTE = partition.ORIGINAL_COMPUTE
RUNTIME = base.OUT / 'readout_control_repair_runtime.json'


def compute(binding, sources):
    readout = base.pc.readout
    calls = 0

    def matching_control_readout(field):
        nonlocal calls
        calls += 1
        # Frozen v1's first readout is exclusively K1 reproduction; its next
        # six readouts are the actual intervention masks and stay unchanged.
        if calls == 1:
            field = torch.as_tensor(field, dtype=torch.float32, device=base.lab.DEV)
        return readout(field)

    base.pc.readout = matching_control_readout
    try:
        payload, receipt = V1_COMPUTE(binding, sources)
    finally:
        base.pc.readout = readout
    assert calls == 7, 'Frozen v1 readout call contract changed'
    receipt['diagnostics'].update(K1_control_render_device='mps', intervention_render_device='cpu')
    payload['receipt_json'] = np.frombuffer(json.dumps(receipt, allow_nan=False).encode(), dtype=np.uint8)
    return payload, receipt


def bind_runtime():
    partition.bind_runtime()
    path = str(Path(__file__).resolve())
    if not RUNTIME.exists():
        base.pc.write(RUNTIME, {
            'repair': 'Use B1 MPS rendering only for K1 mask reproduction assertion; v1 six candidate CPU readouts unchanged.',
            'case': 'k_171c709aa474793b811d', 'K1_field_max_abs': 0.0,
            'identical_saved_field_CPU_mask_xor': 1, 'identical_saved_field_MPS_mask_xor': 0,
            'source': path, 'sha256': base.pc.sha(path),
            'original_runtime_sha256': base.pc.sha(partition.REPAIR),
            'original_config_sha256': base.pc.sha(base.OUT / 'config.json'),
            'pilot_seal_sha256': base.pc.sha(base.OUT / 'pilot/sealed.json'),
            'candidate_predictions_changed': 0, 'query_GT_reads': 0})
    runtime = base.pc.read(RUNTIME)
    assert base.pc.sha(runtime['source']) == runtime['sha256']
    assert base.pc.sha(partition.REPAIR) == runtime['original_runtime_sha256']
    assert base.pc.sha(base.OUT / 'config.json') == runtime['original_config_sha256']
    assert base.pc.sha(base.OUT / 'pilot/sealed.json') == runtime['pilot_seal_sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    bind_runtime()
    if args.mode == 'produce':
        partition.ORIGINAL_COMPUTE = compute
        base.compute = partition.compute
        base.produce('full')
        path = base.OUT / 'full/sealed.json'
        seal = base.pc.read(path)
        seal.update(runtime_repair_sha256=base.pc.sha(partition.REPAIR),
                    readout_control_repair_sha256=base.pc.sha(RUNTIME),
                    K64_recovered_n=sum(r['diagnostics'].get('K64_recovered', False) for r in seal['records']))
        base.pc.write(path, seal)
    else:
        seal = base.pc.read(base.OUT / 'full/sealed.json')
        assert seal['runtime_repair_sha256'] == base.pc.sha(partition.REPAIR)
        assert seal['readout_control_repair_sha256'] == base.pc.sha(RUNTIME)
        base.score('full', args.workers)
