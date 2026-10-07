#!/usr/bin/env python3
"""Score sealed C outputs with flattened binary-count inputs; no inference.

The original frozen C scorer passed64x64 predictions against4096 truth.
This adapter repairs only that shape mismatch and preserves the recorded
predictor, fields, source bindings, metric, draws, and candidate contract.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    config_path = args.run / 'config.json'
    config = json.loads(config_path.read_text())
    source = args.run / 'source/scripts/bench_pro_repairs.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != config['script_sha256']:
        raise ValueError('Original frozen C source changed')
    spec = importlib.util.spec_from_file_location('_frozen_C_score_shape_adapter', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config['config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    module.verify_config(config)
    bench = module.load_frozen_a(config['a_run'], config['a_source_hashes'])
    native_counts = bench.counts
    bench.counts = lambda prediction, truth: native_counts(prediction.reshape(-1), truth.reshape(-1))
    module.score_phase(config, config['ordered_rows'])
    module.verify_config(config)
    receipt = dict(state='C_SCORED_STOP', repair='flatten prediction/truth before unchanged binary counts',
        inference_rerun=False, new_encoder_forwards=0, original_C_source_sha256=digest,
        original_config_sha256=config['config_sha256'],
        scoring_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inference_seal_sha256=module.sha(args.run / 'inference_seal.json'),
        report_sha256=module.sha(args.run / 'report.json'))
    module.write(args.run / 'score_shape_repair.json', receipt)
    module.write(args.run / 'progress.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
