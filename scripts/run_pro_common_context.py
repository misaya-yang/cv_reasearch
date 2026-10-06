#!/usr/bin/env python3
"""Standalone M1 preparation/native audit/inference, without downloads or GT.

Real mode binds RGB, frozen local timm weights, and a provenance-bound host score
packet. It is a complete continuation/readout of that host, not a claim that the
FoRIS RGB producer is implemented here. Query GT is never read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def self_check(destination):
    import dataclasses
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.methods.pro_common_context import Config, audit_native, capture_native, graph_readout, guides, memory_hash, replay, render

    torch.set_num_threads(1)
    torch.manual_seed(71)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(16, 48)
            self.q_norm = torch.nn.LayerNorm(4)
            self.k_norm = torch.nn.LayerNorm(4)
            self.proj = torch.nn.Linear(16, 16)
            self.fused_attn = True

        def forward(self, x, rope):
            batch, tokens, width = x.shape
            q, k, v = self.qkv(x).reshape(batch, tokens, 3, 4, 4).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)
            def rotate(value):
                first, last = value.chunk(2, -1)
                return torch.cat((-last, first), dim=-1)
            q, k = q*rope[0]+rotate(q)*rope[1], k*rope[0]+rotate(k)*rope[1]
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
            return self.proj(out.transpose(1, 2).reshape(batch, tokens, width))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = torch.nn.LayerNorm(16), torch.nn.LayerNorm(16)
            self.attn = Attention()
            self.ls1 = torch.nn.Parameter(torch.linspace(.2, .7, 16))
            self.ls2 = torch.nn.Parameter(torch.linspace(.3, .8, 16))
            self.mlp = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16))

        def forward(self, x, rope=None):
            x = x+self.ls1*self.attn(self.norm1(x), rope)
            return x+self.ls2*self.mlp(self.norm2(x))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = torch.nn.Conv2d(3, 16, 2, 2)
            self.prefix = torch.nn.Parameter(torch.randn(1, 2, 16))
            self.blocks = torch.nn.ModuleList([Block() for _ in range(5)])
            self.norm = torch.nn.LayerNorm(16)
            self.num_prefix_tokens = 2

        def forward_features(self, images):
            x = self.patch(images).flatten(2).transpose(1, 2)
            x = torch.cat((self.prefix.expand(len(x), -1, -1), x), dim=1)
            angles = torch.arange(x.shape[1], device=x.device)[None, None, :, None]*torch.tensor([.03, .07, .03, .07], device=x.device)
            rope = (angles.cos(), angles.sin())
            for block in self.blocks:
                x = block(x, rope=rope)
            return self.norm(x)

    model = Toy().eval().requires_grad_(False)
    cfg = Config(start_block=2, suffix_blocks=3, chunk_size=7)
    images = torch.randn(2, 3, 12, 12)
    cache = capture_native(model, images, cfg)
    audit = audit_native(model, cache, cfg)
    before = memory_hash(cache)
    raw = cache['raw_h20'][1, cache['prefix']:]
    chunked = replay(model, cache, raw, cfg)
    full = replay(model, cache, raw, dataclasses.replace(cfg, chunk_size=36))
    chunk_difference = float((chunked-full).abs().max())
    if not torch.allclose(chunked, full, atol=1e-6, rtol=1e-6):
        raise AssertionError('Chunking changed formal replay')
    coverage = np.zeros((6, 6), dtype=np.float32)
    coverage[1:4, 1:4] = 1
    gs, info = guides(model, cache, coverage, cfg)
    score = np.linspace(0, 1, 36, dtype=np.float32).reshape(6, 6)
    fields, graph_info = graph_readout(cache['final'][1, cache['prefix']:], score, gs, cfg)
    masks = {arm: render(field, (17, 23), working_size=32) for arm, field in fields.items()}
    if set(fields) != {'ctx', 'plain', 'mean-unit', 'uniform'}:
        raise AssertionError('Missing complete control')
    if not all(work.shape == (32, 32) and original.shape == (17, 23) for work, original in masks.values()):
        raise AssertionError('Renderer shapes wrong')
    _, all_fg = guides(model, cache, np.ones((6, 6), dtype=np.float32), cfg)
    empty, empty_info = guides(model, cache, np.zeros((6, 6), dtype=np.float32), cfg)
    if all_fg['background_probe_ids'] or empty is not None or not empty_info['empty_reference']:
        raise AssertionError('Reference edge cases changed')
    after = memory_hash(cache)
    if before != after:
        raise AssertionError('Native memory was changed')
    # A deliberately corrupted native KV cache must fail the native self audit.
    with torch.inference_mode():
        cache['layers'][cfg.start_block]['v'].add_(.2)
    rejected = False
    try:
        audit_native(model, cache, cfg)
    except RuntimeError:
        rejected = True
    if not rejected:
        raise AssertionError('Corrupted memory was not detected by native parity')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(state='TOY_CONTRACT_CHECK_PASSED', real_DINO_parity_verified=False,
                  real_segmentation_episodes=0, config=dataclasses.asdict(cfg),
                  native_self=audit, chunk_maxabs=chunk_difference, memory_unchanged=before == after,
                  corrupted_KV_rejected=True, complete_controls=list(fields), graph=graph_info,
                  missing_actual_local_assets=['timm package/Eva source', 'DINO weights', 'native host episode assets'],
                  actual_CPU_time_is_toy_only=True)
    (destination/'toy_check.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(destination/'toy_fields.npz', **fields)
    print(json.dumps(report, indent=2))


def actual(args):
    import numpy as np
    import torch
    from PIL import Image
    from ics.data import TimmDINOv3
    from ics.methods.pro_common_context import Config, audit_native, capture_native, predict
    if not all((args.reference, args.reference_mask, args.query, args.weights, args.host_packet, args.host_receipt, args.out)):
        raise ValueError('Real mode requires reference/mask/query/weights/host-packet/host-receipt/out')
    if args.device != 'cpu':
        raise ValueError('This standalone entry currently supports authorized CPU execution only')
    out = Path(args.out)
    if out.exists():
        raise ValueError('Choose a fresh output directory')
    identity = json.loads(Path(args.host_receipt).read_text())
    bound = dict(reference_sha256=sha(args.reference), reference_mask_sha256=sha(args.reference_mask),
                 query_sha256=sha(args.query), weights_sha256=sha(Path(args.weights)/'model.safetensors'),
                 config_sha256=sha(Path(args.weights)/'config.json'), host_packet_sha256=sha(args.host_packet))
    for key, value in bound.items():
        if identity.get(key) != value:
            raise ValueError(f'Host provenance mismatch or missing binding: {key}')
    if identity.get('score_stage') != 'complete_FoRIS_before_binarize_CRF' or identity.get('query_gt_used') is not False:
        raise ValueError('Bind full host continuous score before binarize/CRF and explicit GT-free producer')
    if not identity.get('producer_recipe') or not identity.get('producer_source_sha256'):
        raise ValueError('Bind actual FoRIS producer recipe and source hash')
    if identity.get('image_transform') != 'PIL_RGB_bilinear1024_ImageNet_FP32':
        raise ValueError('Host transform must bind the same PIL RGB/1024/ImageNet FP32 producer')
    projection = bool(identity.get('apply_native_projection'))
    if 'apply_native_projection' not in identity:
        raise ValueError('Native gate decision must be explicitly bound once')
    basis = None
    if projection:
        if not args.basis:
            raise ValueError('Projection gate requires existing normalized-black native basis')
        from ics.native_basis import load_native_basis
        basis, basis_receipt = load_native_basis(args.basis)
        if identity.get('basis_sha256') != basis_receipt['sha256']:
            raise ValueError('Native basis hash mismatch')
    with np.load(args.host_packet, allow_pickle=False) as packet:
        score = packet['score'].copy()  # no GT/native mask packet keys read
    if score.shape != (64, 64) or not np.isfinite(score).all():
        raise ValueError('Require finite pre-binarize full FoRIS score on 64x64')
    def image_tensor(path):
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            hw = (rgb.height, rgb.width)
            value = np.asarray(rgb.resize((1024, 1024), Image.Resampling.BILINEAR)).copy()
        tensor = torch.from_numpy(value).permute(2, 0, 1).float()/255
        return (tensor-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None], hw
    reference, _ = image_tensor(args.reference)
    query, hw = image_tensor(args.query)
    with Image.open(args.reference_mask) as im:
        raw = np.asarray(im.convert('L'))
        if not np.isin(raw, [0, 1, 255]).all():
            raise ValueError('Reference mask must be explicit binary mask, not class-coded annotation')
        mask = np.asarray(im.convert('L').resize((1024, 1024), Image.Resampling.NEAREST)).copy() > 0
    cov = F_area(mask)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = TimmDINOv3(args.weights).to('cpu').float().eval().requires_grad_(False).m
    if (len(model.blocks) != 24 or int(model.num_prefix_tokens) != 5
            or int(getattr(model, 'num_features', -1)) != 1024
            or int(model.blocks[0].attn.num_heads) != 16):
        raise ValueError('Expected bound timm Eva DINOv3-L: 24 blocks, CLS+4 registers')
    images = torch.stack((reference, query))
    cfg = Config(chunk_size=args.chunk_size)
    started = time.perf_counter()
    if args.audit_only:
        cache = capture_native(model, images, cfg)
        result = dict(native_audit=audit_native(model, cache, cfg), source=cache['source'],
                      capture_seconds=cache['capture_seconds'], scope='real_weights_native_self_audit_only')
    else:
        prediction = predict(model, images, cov, score, hw, cfg, basis=basis)
        result = prediction['info']
    out.mkdir(parents=True)
    if not args.audit_only:
        np.savez_compressed(out/'fields.npz', **prediction['fields'])
        for arm, (working, original) in prediction['masks'].items():
            Image.fromarray((working.astype(np.uint8)*255)).save(out/f'{arm}_1024.png')
            Image.fromarray((original.astype(np.uint8)*255)).save(out/f'{arm}_original.png')
    import resource
    receipt = dict(state='NATIVE_AUDIT_ONLY' if args.audit_only else 'INFERENCE_SEALED_NO_GT_SCORE',
                   assets=bound, host_identity=identity, result=result,
                   complete_entry_seconds=time.perf_counter()-started,
                   peak_rss_platform_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   quality='unscored', historical_baseline_parity='not_asserted',
                   deployment_cost_includes_host_producer=False,
                   source_sha256={str(Path(__file__).resolve()):sha(__file__),
                                  str(Path(__file__).resolve().parents[1]/'src/ics/methods/pro_common_context.py'):sha(Path(__file__).resolve().parents[1]/'src/ics/methods/pro_common_context.py')})
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


def F_area(mask):
    import numpy as np
    import torch
    import torch.nn.functional as F
    return F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], (64, 64), mode='area')[0, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    for name in ('reference', 'reference-mask', 'query', 'weights', 'host-packet', 'host-receipt', 'basis', 'out'):
        parser.add_argument('--'+name)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=64)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 32:
        parser.error('Threads must fit the authorized CPU budget')
    if args.self_check:
        self_check(args.out or 'evidence/local/research_20261006/pro_context_preparation_01a1100b')
    else:
        actual(args)


if __name__ == '__main__':
    main()
