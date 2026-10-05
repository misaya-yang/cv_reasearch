#!/usr/bin/env python3
"""Complete released INSID3 logic with the common local DINOv3 adapter.

Preserve paired BF16 extraction, native FP32 positional-basis construction and
the released clustering/aggregation. Report bilinear and640-CRF separately.
This is an implementation comparison, not identity with published hub numerics.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def infer(a):
    import numpy as np
    import torch
    from PIL import Image
    from ics.data import TimmDINOv3
    from ics.experiment import load_rows, sha, packet
    if a.out.exists():
        raise FileExistsError('Fresh output required')
    rows = load_rows(a.family / 'manifest.json')
    parent = json.loads((a.family / 'sealed.json').read_text())
    if parent['state'] != 'ALL_PREDICTIONS_SEALED' or sha(a.family / 'manifest.json') != parent['manifest_sha256']:
        raise ValueError('Invalid complete comparison source')
    if len(rows) != 241:
        raise ValueError('Full DEV241 required')
    if a.smoke:
        rows = [rows[0], next(r for r in rows if r['fold'] != rows[0]['fold'])]
    torch.set_num_threads(1)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.3)
    sys.path.insert(0, str(a.insid3))
    from models.insid3 import INSID3
    from utils.refinement import upsample_mask
    source_paths = [Path(__file__), REPO / 'src/ics/data.py', REPO / 'src/ics/experiment.py']
    source_paths += sorted((a.insid3 / 'models').glob('*.py'))
    source_paths += sorted((a.insid3 / 'utils').glob('*.py'))
    protocol = dict(n=len(rows), seed=0, resolution=1024, shots=1,
        query_gt_in_inference=False, dataset='COCO-20i DEV; previously inspected',
        implementation='released INSID3 logic; common timm DINOv3-L adapter and weights; hub numerical parity unverified',
        config=dict(svd_components=500, tau=.6, merge_threshold=.2, crf_size=640,
                    resize_to_orig_size=False, paired_encoder_batch=2, extraction_autocast='bfloat16',
                    basis_autocast=False, cpu_threads=1, gpu_memory_fraction=.3),
        source_code={str(p.resolve()): sha(p) for p in source_paths},
        weights_sha256=sha(a.weights / 'model.safetensors'),
        encoder_config_sha256=sha(a.weights / 'config.json'),
        family_seal_sha256=sha(a.family / 'sealed.json'), input_receipts={})
    a.out.mkdir(); (a.out / 'predictions').mkdir()
    write(a.out / 'manifest.json', rows); write(a.out / 'protocol.json', protocol)
    begin = time.monotonic()
    encoder = TimmDINOv3(str(a.weights)).to('cuda').eval().requires_grad_(False)
    host = INSID3(encoder=encoder, image_size=1024, svd_components=500, tau=.6,
                 merge_threshold=.2, mask_refiner='crf', crf_size=640,
                 resize_to_orig_size=False, device='cuda').eval().requires_grad_(False)
    torch.cuda.synchronize()
    setup_seconds = time.monotonic() - begin
    original_finalize = host._finalize_mask
    captured = {}
    def finalize(mask, target):
        captured['bilinear'] = upsample_mask(mask, 1024, 1024).clone()
        return original_finalize(mask, target)
    host._finalize_mask = finalize  # Read-only tap; original CRF result returned unchanged.
    hashes = {}; audit = {}
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            key = row['key']; started = time.monotonic()
            source = a.family / 'predictions' / (key + '.npz')
            if sha(source) != parent['predictions'][key]:
                raise ValueError('Changed complete comparison ' + key)
            with np.load(source, allow_pickle=False) as z:
                comparisons = {name: z[name].copy() for name in ('native', 'origin', 'rcg', 'astra.control', 'mean.control', 'insid3.control')}
            support = a.data / row['support']; query = a.data / row['query']
            reference_mask = a.annotations / Path(row['support']).with_suffix('.png')
            # The query annotation is never opened in this phase.
            with Image.open(reference_mask) as im:
                gold = torch.from_numpy((np.asarray(im) == row['c'] + 1).copy())
            protocol['input_receipts'][key] = dict(support_sha256=sha(support),
                query_sha256=sha(query), reference_annotation_sha256=sha(reference_mask),
                packet_sha256=sha(packet(a.root, row)), source_prediction_sha256=sha(source))
            try:
                with Image.open(support) as im: host.set_reference(im.convert('RGB'), gold)
                with Image.open(query) as im: host.set_target(im.convert('RGB'))
                crf = host.segment()
            finally:
                host.reset_state()
            if tuple(crf.shape) != (1024, 1024) or 'bilinear' not in captured:
                raise ValueError('Incomplete released output')
            output = dict(comparisons,
                **{'insid3.release_bilinear.control': np.packbits(captured.pop('bilinear').cpu().numpy()),
                   'insid3.release_crf.control': np.packbits(crf.cpu().numpy())})
            destination = a.out / 'predictions' / (key + '.npz')
            np.savez_compressed(destination, **output); hashes[key] = sha(destination)
            audit[key] = dict(seconds=time.monotonic()-started, encoder_forward_batch=2)
            print(json.dumps(dict(n=number, total=len(rows), seconds=round(time.monotonic()-begin, 2),
                                  episode_seconds=round(audit[key]['seconds'], 2))), flush=True)
    host._finalize_mask = original_finalize
    write(a.out / 'protocol.json', protocol); write(a.out / 'runtime.json', audit)
    write(a.out / 'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=len(rows),
        manifest_sha256=sha(a.out / 'manifest.json'), protocol_sha256=sha(a.out / 'protocol.json'),
        runtime_sha256=sha(a.out / 'runtime.json'), predictions=hashes,
        query_gt_in_inference=False, seconds=time.monotonic()-begin, setup_seconds=setup_seconds,
        cuda_peak_bytes=torch.cuda.max_memory_allocated()))


def score(a):
    import numpy as np
    from ics.experiment import sha, packet, unpack, summarize
    seal = json.loads((a.out / 'sealed.json').read_text())
    if seal['state'] != 'ALL_PREDICTIONS_SEALED' or (a.out / 'report.json').exists():
        raise ValueError('Incomplete or already scored')
    for field, name in [('manifest_sha256', 'manifest.json'), ('protocol_sha256', 'protocol.json'), ('runtime_sha256', 'runtime.json')]:
        if sha(a.out / name) != seal[field]: raise ValueError('Changed scoring input')
    protocol = json.loads((a.out / 'protocol.json').read_text())
    for name, digest in protocol['source_code'].items():
        if sha(name) != digest: raise ValueError('Changed implementation')
    rows = json.loads((a.out / 'manifest.json').read_text())
    arrays = {}; corrections = {}; details = []
    for row in rows:
        key = row['key']; pp = packet(a.root, row)
        if sha(pp) != protocol['input_receipts'][key]['packet_sha256']: raise ValueError('Changed GT packet')
        with np.load(pp, allow_pickle=False) as z: truth = unpack(z['truth'])
        pred = a.out / 'predictions' / (key + '.npz')
        if sha(pred) != seal['predictions'][key]: raise ValueError('Changed prediction')
        with np.load(pred, allow_pickle=False) as z: masks = {name: unpack(z[name]) for name in z.files}
        origin = masks['origin']; record = dict(row, iu={})
        for name, mask in masks.items():
            iu = [int((mask & truth).sum()), int((mask | truth).sum())]
            arrays.setdefault(name, []).append(iu); record['iu'][name] = iu
            addition, deletion = mask & ~origin, origin & ~mask
            corrections.setdefault(name, []).append(dict(key=key, c=row['c'], fold=row['fold'],
                batch=row.get('batch', 'DEV'), add_TP=int((addition & truth).sum()),
                add_FP=int((addition & ~truth).sum()), delete_TP=int((deletion & truth).sum()),
                delete_FP=int((deletion & ~truth).sum())))
        details.append(record)
    report, _ = summarize(rows, {k: np.asarray(v) for k,v in arrays.items()}, corrections)
    report['corrections_vs_raw_DINO'] = report.pop('corrections_vs_native')
    report['corrections_vs_raw_DINO_by_class'] = report.pop('corrections_by_class')
    report['corrections_vs_raw_DINO_by_batch'] = report.pop('corrections_by_batch')
    report.update(implementation=protocol['implementation'], prediction_seal_sha256=sha(a.out/'sealed.json'),
                  producer_runtime=json.loads((a.out/'sealed.json').read_text()))
    write(a.out/'report.json', report)
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in details))
    print(json.dumps(dict(n=len(rows), scores=report['scores'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['infer', 'score']); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/demo9_extent'))
    p.add_argument('--family', type=Path); p.add_argument('--smoke', action='store_true')
    p.add_argument('--insid3', type=Path, default=Path('/root/autodl-tmp/demo4/INSID3'))
    p.add_argument('--weights', type=Path, default=Path('/root/demo4_cache/models/dinov3-vitl16-timm'))
    p.add_argument('--data', type=Path, default=Path('/root/demo4_cache/data/COCO2014'))
    p.add_argument('--annotations', type=Path, default=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations'))
    a = p.parse_args()
    if a.phase == 'infer' and a.family is None: p.error('--family required for inference')
    {'infer': infer, 'score': score}[a.phase](a)


if __name__ == '__main__': main()
