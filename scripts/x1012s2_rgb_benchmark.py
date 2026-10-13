"""Paired live-encoder RGB benchmark for one externally frozen candidate."""
from pathlib import Path
import argparse
import ast
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time

REPO = Path('/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch')
ASSETS = REPO.parent / 'cv_data'
BASE = ASSETS / 'a/x1012s2_runtime'
RUNTIME = BASE / 'mfu_boundary'
WEIGHTS = ASSETS / 'demo4_cache/models/dinov3-vitl16-timm'
DOMAINS = ('coco', 'paco_part', 'fundus')


def load_module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def research_processes():
    lines = subprocess.check_output(['ps', '-axo', 'pid,ppid,etime,%cpu,command'], text=True).splitlines()
    result = []
    for line in lines[1:]:
        columns = line.split(None, 4)
        if len(columns) != 5 or int(columns[0]) == os.getpid():
            continue
        command = columns[4]
        executable = Path(command.split(None, 1)[0]).name.lower()
        if executable.startswith('python') and ('/cv_data/a/x1012s2' in command or 'scripts/x1012s2' in command):
            result.append(line.strip())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('MFU', 'gated_MFU'), required=True)
    parser.add_argument('--boundary', choices=('none', 'full_crf', 'add_only', 'delete_only', 'area16'), required=True)
    parser.add_argument('--selection-artifact', required=True, help='Existing final program/selection JSON, bound by SHA256')
    parser.add_argument('--output', default=str(BASE / 'final_rgb'))
    args = parser.parse_args()
    selection = Path(args.selection_artifact).resolve()
    assert selection.is_file()
    task = load_module('final_rgb_runtime', RUNTIME / 'frozen/x1012s2_runtime.py')
    config = task.check()
    decision = task.read(selection)
    assert decision['runtime']['candidate'] == args.candidate and decision['runtime']['boundary'] == args.boundary
    assert decision['runtime']['config_sha256'] == task.sha(RUNTIME / 'config.json')
    others = research_processes()
    assert not others, 'Wait for the other research Python jobs to finish: ' + str(others)
    out = Path(args.output).resolve()
    assert not out.exists(), 'Preserve any earlier benchmark attempt'
    out.mkdir(parents=True)
    (out / 'frozen').mkdir()
    shutil.copyfile(__file__, out / 'frozen/final_rgb_benchmark.py')
    shutil.copyfile(selection, out / 'frozen/selection.json')
    cases_by_domain = {c['binding']['dataset']: c for c in task.read(RUNTIME / 'manifest.json')}
    cases = [cases_by_domain[d] for d in DOMAINS]
    task.write(out / 'manifest.json', cases)
    runtime = task.Runtime(config['expressions'], config['linear_rules'])
    import torch
    import timm
    assert torch.backends.mps.is_available()
    original = REPO / 'src/ics/data.py'
    original_text = original.read_text()
    node = next(n for n in ast.parse(original_text).body if isinstance(n, ast.ClassDef) and n.name == 'TimmDINOv3')
    encoder_source = out / 'frozen/original_encoder.py'
    encoder_source.write_text('import json\nimport torch\n' + ast.get_source_segment(original_text, node) + '\n')
    started = time.perf_counter()
    encoder_type = load_module('final_original_encoder', encoder_source).TimmDINOv3
    encoder = encoder_type(str(WEIGHTS)).to('mps').eval().requires_grad_(False)
    torch.mps.synchronize()
    setup_seconds = time.perf_counter() - started
    measurement = {'state': 'FINAL_PROGRAM_RGB_BENCHMARK_FROZEN_BEFORE_TIMING',
                   'candidate': args.candidate, 'boundary': args.boundary,
                   'selection_artifact': str(selection), 'selection_sha256': task.sha(selection),
                   'runtime_config_sha256': task.sha(RUNTIME / 'config.json'),
                   'runtime_source_sha256': task.sha(RUNTIME / 'frozen/x1012s2_runtime.py'),
                   'benchmark_source_sha256': task.sha(out / 'frozen/final_rgb_benchmark.py'),
                   'encoder_source_sha256': task.sha(encoder_source),
                   'weights_sha256': task.sha(WEIGHTS / 'model.safetensors'),
                   'model_config_sha256': task.sha(WEIGHTS / 'config.json'),
                   'manifest_sha256': task.sha(out / 'manifest.json'),
                   'domains': DOMAINS, 'selection': 'First original runtime case in COCO, PACO, and Fundus; no GT or quality selection.',
                   'rounds_per_case_per_flow': 2, 'flows': ['R_plus_Q', 'cached_R_Q_only'],
                   'paired_modes': ['B_only', 'candidate'], 'order': 'Alternates by case, flow, and round.',
                   'torch': torch.__version__, 'timm': timm.__version__, 'platform': platform.platform(),
                   'cpu_brand': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                   'memory_bytes': int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                   'CPU_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                   'encoder_device': 'mps', 'head_device': 'cpu', 'dtype': 'float32', 'image_size': 1024,
                   'other_research_python_processes_before': others,
                   'fresh_encoder_for_every_timed_pipeline': True,
                   'query_GT_reads': 0, 'quality_scoring': False, 'fits': 0,
                   'feature_cache_writes': 0, 'natural_mask_writes': 0,
                   'mask_hash_scheme': 'SHA256(JSON([shape,numpy_dtype.str]) followed by contiguous bool mask bytes)',
                   'timer': 'Actual serial RGB/reference-mask loading, preprocessing, synchronized fresh encoder, CPU transfer, actual head and selected final mask; no component-mean addition.',
                   'limits': ['Three selected inputs and two paired rounds are descriptive local latency, not dataset-wide throughput.',
                              'The head API reloads/reprocesses RGB/reference mask; this equal overhead is included for both modes.',
                              'Cached-R retains only one freshly encoded reference tensor in RAM per case.',
                              'No comparison to archived masks is made; fresh encoder outputs are preserved through the real readouts and recorded by mask SHA.',
                              'One-time setup, warmups, mask hashing, report writing and output serialization are excluded.']}
    task.write(out / 'config.json', measurement)
    encode_count = 0

    def encode(inputs):
        nonlocal encode_count
        torch.mps.synchronize()
        tick = time.perf_counter()
        device_inputs = inputs.to('mps')
        torch.mps.synchronize()
        loaded = time.perf_counter()
        maps = encoder.get_intermediate_layers(device_inputs, n=1, reshape=True)[0]
        torch.mps.synchronize()
        inferred = time.perf_counter()
        raw = maps.cpu().permute(0, 2, 3, 1).reshape(inputs.shape[0], 4096, 1024).contiguous()
        torch.mps.synchronize()
        done = time.perf_counter()
        encode_count += 1
        assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
        return raw, {'actual_encoder_calls': 1, 'encoded_images': int(inputs.shape[0]),
                     'host_to_device_seconds': loaded - tick,
                     'forward_synchronized_seconds': inferred - loaded,
                     'CPU_transfer_and_layout_seconds': done - inferred,
                     'encoder_to_CPU_seconds': done - tick}

    pairs, warmups = [], []
    with torch.inference_mode():
        # Fixed-shape encoder and both actual heads warm up once; never count as trials.
        binding = cases[0]['binding']
        with task.inference_files(binding):
            reference, mask, query = runtime.load_inputs(binding['row'], ASSETS)
            image_pair = torch.stack((runtime.host._transform(reference), runtime.host._transform(query)))
            raw, detail = encode(image_pair)
            warmups.append({'phase': 'pair_encoder', **detail})
            features = {'reference': raw[0], 'query': raw[1]}
            runtime.run(binding, providers=('B',), provided_raw=features)
            runtime.predict(binding, candidate=args.candidate, boundary=args.boundary, provided_raw=features)
            _, detail = encode(image_pair[1:])
            warmups.append({'phase': 'query_encoder', **detail})
        for case_number, case in enumerate(cases):
            binding = case['binding']
            with task.inference_files(binding):
                reference, mask, query = runtime.load_inputs(binding['row'], ASSETS)
                reference_raw, detail = encode(runtime.host._transform(reference)[None])
                cached_reference = reference_raw[0].clone()
                warmups.append({'phase': 'cached_reference_setup', 'dataset': binding['dataset'], **detail})
            for flow_number, flow in enumerate(('R_plus_Q', 'cached_R_Q_only')):
                for repetition in range(2):
                    order = (['B_only', 'candidate'] if (case_number + flow_number + repetition) % 2 == 0
                             else ['candidate', 'B_only'])
                    pair = {'index': task.case_key(binding), 'dataset': binding['dataset'],
                            'flow': flow, 'round': repetition + 1, 'order': order}
                    for mode in order:
                        torch.mps.synchronize()
                        tick = time.perf_counter()
                        with task.inference_files(binding) as opened:
                            reference, mask, query = runtime.load_inputs(binding['row'], ASSETS)
                            inputs = (torch.stack((runtime.host._transform(reference), runtime.host._transform(query)))
                                      if flow == 'R_plus_Q' else runtime.host._transform(query)[None])
                            preprocessed = time.perf_counter()
                            raw, encoder_detail = encode(inputs)
                            provided = ({'reference': raw[0], 'query': raw[1]} if flow == 'R_plus_Q'
                                        else {'reference': cached_reference, 'query': raw[0]})
                            if mode == 'B_only':
                                masks, _, _, head = runtime.run(binding, providers=('B',), provided_raw=provided)
                                prediction = masks['B']
                            else:
                                prediction, head = runtime.predict(binding, candidate=args.candidate,
                                                                   boundary=args.boundary, provided_raw=provided)
                        torch.mps.synchronize()
                        elapsed = time.perf_counter() - tick
                        assert head['provided_in_memory_O24_inputs'] == 2 and head['raw_unique_payloads'] == 0
                        assert not any(Path(path).suffix == '.npz' for path in opened)
                        assert head['query_GT_reads'] == head['fits'] == head['archived_score_mask_reads_in_inference'] == 0
                        pair[mode] = {'total_RGB_to_mask_seconds': elapsed,
                                      'initial_RGB_load_preprocess_seconds': preprocessed - tick,
                                      'actual_encoder_calls_total': encoder_detail['actual_encoder_calls'],
                                      'encoder': encoder_detail, 'head': head,
                                      'mask_sha256': task.array_hash(prediction), 'mask_shape': list(prediction.shape),
                                      'actual_input_files': sorted(set(opened))}
                        del prediction, raw, provided
                    pair['candidate_minus_B_seconds'] = pair['candidate']['total_RGB_to_mask_seconds'] - pair['B_only']['total_RGB_to_mask_seconds']
                    pair['candidate_to_B_ratio'] = pair['candidate']['total_RGB_to_mask_seconds'] / pair['B_only']['total_RGB_to_mask_seconds']
                    pairs.append(pair)
                    task.write(out / 'progress.json', {'completed_pairs': len(pairs), 'n_pairs': 12, 'pairs': pairs})
                    print({'completed_pairs': len(pairs), 'n_pairs': 12, 'dataset': binding['dataset'],
                           'flow': flow, 'round': repetition + 1, 'order': order,
                           'B_seconds': pair['B_only']['total_RGB_to_mask_seconds'],
                           'candidate_seconds': pair['candidate']['total_RGB_to_mask_seconds'],
                           'ratio': pair['candidate_to_B_ratio']}, flush=True)
    summary, head_summary = {}, {}
    for flow in ('R_plus_Q', 'cached_R_Q_only'):
        rows = [r for r in pairs if r['flow'] == flow]
        summary[flow] = {mode: task.statistics([r[mode]['total_RGB_to_mask_seconds'] for r in rows])
                         for mode in ('B_only', 'candidate')}
        summary[flow]['paired_delta_seconds'] = task.statistics([r['candidate_minus_B_seconds'] for r in rows])
        summary[flow]['paired_ratio'] = task.statistics([r['candidate_to_B_ratio'] for r in rows])
        summary[flow]['ratio_of_mean_seconds'] = summary[flow]['candidate']['mean'] / summary[flow]['B_only']['mean']
        summary[flow]['by_domain'] = {
            domain: {mode: task.statistics([r[mode]['total_RGB_to_mask_seconds'] for r in rows if r['dataset'] == domain])
                     for mode in ('B_only', 'candidate')}
            for domain in DOMAINS}
        head_summary[flow] = {
            mode: task.statistics([r[mode]['head']['complete_cached_seconds'] for r in rows])
            for mode in ('B_only', 'candidate')}
        head_summary[flow]['by_domain'] = {
            domain: {mode: task.statistics([r[mode]['head']['complete_cached_seconds'] for r in rows if r['dataset'] == domain])
                     for mode in ('B_only', 'candidate')}
            for domain in DOMAINS}
    assert task.sha(selection) == measurement['selection_sha256'], 'Final program artifact changed during timing'
    mask_repeats = [{'dataset': domain, 'flow': flow,
                     'unique_SHA_count': {mode: len({r[mode]['mask_sha256'] for r in pairs
                                                   if r['dataset'] == domain and r['flow'] == flow})
                                          for mode in ('B_only', 'candidate')}}
                    for domain in DOMAINS for flow in ('R_plus_Q', 'cached_R_Q_only')]
    report = {'state': 'FINAL_PROGRAM_PAIRED_LIVE_ENCODER_RGB_TIMING_COMPLETE',
              'candidate': args.candidate, 'boundary': args.boundary,
              'config_sha256': task.sha(out / 'config.json'), 'selection_sha256': task.sha(selection),
              'n_cases': 3, 'rounds_per_case_per_flow': 2, 'n_paired_trials': len(pairs),
              'n_timed_pipelines': len(pairs) * 2, 'pairs': pairs, 'summary': summary,
              'fresh_in_memory_O24_head_summary': head_summary,
              'head_segment_scope': 'Measured within each actual RGB chain using fresh O24 in RAM. Includes input metadata verification, RGB/reference-mask reload/preprocessing, semantic providers and final CRF when requested. Excludes DINO, output transfer, raw NPZ I/O, setup, and mask hashing.',
              'within_case_flow_mask_repeat_SHA_counts': mask_repeats,
              'actual_encoder_calls_total': encode_count, 'actual_encoder_calls_timed': len(pairs) * 2,
              'warmups_and_reference_setup': warmups, 'model_setup_seconds': setup_seconds,
              'runtime_setup_seconds': runtime.setup_seconds,
              'other_research_python_processes_after': research_processes(),
              'query_GT_reads': 0, 'quality_scoring': False, 'fits': 0, 'feature_cache_writes': 0,
              'natural_mask_writes': 0, 'limits': measurement['limits']}
    assert encode_count == 29 and len(pairs) == 12
    task.write(out / 'report.json', report)
    lines = ['# Final-program paired RGB timing', '', f"Candidate: {args.candidate}, boundary: {args.boundary}.",
             'Three fixed COCO/PACO/Fundus inputs, two paired rounds per input and flow. Every measured pipeline executes DINOv3 afresh with explicit MPS synchronization.', '',
             '|Flow|B mean seconds|Candidate mean seconds|Ratio of means|Paired delta mean seconds|', '|---|---:|---:|---:|---:|']
    for flow, values in summary.items():
        lines.append(f"|{flow}|{values['B_only']['mean']:.6f}|{values['candidate']['mean']:.6f}|{values['ratio_of_mean_seconds']:.6f}|{values['paired_delta_seconds']['mean']:.6f}|")
    lines += ['', '|Flow|Input domain|B mean seconds|Candidate mean seconds|', '|---|---|---:|---:|']
    for flow, values in summary.items():
        for domain, per_domain in values['by_domain'].items():
            lines.append(f"|{flow}|{domain}|{per_domain['B_only']['mean']:.6f}|{per_domain['candidate']['mean']:.6f}|")
    lines += ['', '|Fresh O24 in RAM head segment|B mean seconds|Candidate mean seconds|', '|---|---:|---:|']
    for flow, values in head_summary.items():
        lines.append(f"|{flow}|{values['B_only']['mean']:.6f}|{values['candidate']['mean']:.6f}|")
    lines += ['', report['head_segment_scope']]
    lines += ['', *measurement['limits']]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print({'state': report['state'], 'summary': summary, 'report_sha256': task.sha(out / 'report.json')}, flush=True)


if __name__ == '__main__':
    main()
