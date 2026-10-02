"""Shared-GPU head-only attribution; never complete-decoder latency evidence."""
import json
import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/compute_structure'))
from pretrained_decoder_round import load_decoder, load_inputs
from baselines import PositionCache
from prototype import (DensePhaseCache, dense_read, dense_write, plain_attention,
                       phase_outputs, shape_plan)
from benchmark import measure, profile_forward
from takeover_compact_head_infer import head, load_student
from takeover_head_fallback import grid
from takeover_compact_phase_fallback import head as phase_fallback_head, phase_patches

OUT = ROOT / 'results/takeover_20261001_v1/local_student_head_v1/head_profile_p32'


def gpu_state():
    return {
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,memory.free,power.draw,clocks.sm', '--format=csv,noheader'], text=True).strip(),
        'compute_apps': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip(),
    }


def main(phase_fallback=False):
    OUT.mkdir(exist_ok=False)
    report = {'status': 'WAITING_VRAM', 'pid': os.getpid(), 'records': [],
              'protocol': 'Shared-GPU head-only attribution, FP32/TF32off. First32 distinct real prompts from the existing128 input, same strong dense cached transformer states, frozen student. MB32, all4 masks/4IQ; transformer executes once untimed. Compare original dense phase head and real compact0/5/10% head, actual Inductor fullgraph/dynamicFalse, no fallback. Four rotating block orders, reps10. Contention and smaller batch prevent publication latency or P128 complete-decoder speed claims. Separate compiled traces and eager labeled stage trace only locate implementation bottlenecks; eager trace is not compiled cost. No new quality tuning or masks persisted.'}
    if phase_fallback:
        report['protocol'] += ' INTERVENTION: add phase5/phase10 local original function computed with cached phase matrices instead of cuDNN deconvolution, same trained weights/margins/budgets. Six balanced rotating rounds. Untimed candidate-output differences and head-only FP64 patch check are separate, not whole-model equivalence or quality claims.'

    def save():
        tmp = OUT / 'report.tmp'
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(OUT / 'report.json')

    save()
    telemetry = None
    telemetry_log = None
    try:
        while int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) < 4500:
            time.sleep(10)
        telemetry_log = (OUT / 'gpu_telemetry.csv').open('x')
        telemetry = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,clocks.sm', '--format=csv', '--loop-ms=1000'], stdout=telemetry_log, stderr=subprocess.STDOUT)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device('cuda:0')
        source = OUT.parent.parent
        model, _, _ = load_decoder(source / 'real_original/mask_decoder_state.pt', device)
        model.requires_grad_(False)
        image, pe, dense, sparse, _ = load_inputs(source / 'real_p128_inputs/encoded_inputs.npz', device)
        sp = sparse['perf'][:32]
        assert len(torch.unique(sp, dim=0)) == 32
        plans = shape_plan(model, 4096, sp.shape[1]+5, 'dense_assoc', 'auto', 'auto')
        cache = DensePhaseCache.build(model, image, dense, PositionCache.build(model, pe, plans))
        student, center, basis = load_student(OUT.parent / 'deployment.pt', device)
        student.requires_grad_(False)
        report['status'] = 'EXTRACT_REAL_FINAL_STATES'
        save()
        with torch.inference_mode():
            learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), 0)
            tokens = torch.cat((learned[None].expand(len(sp), -1, -1), sp), 1)
            q, x = tokens, cache.first.base.expand(len(sp), -1, -1)
            for i, block in enumerate(model.transformer.layers):
                if block.skip_first_layer_pe:
                    q = plain_attention(block.self_attn, q, q, q, 'explicit')
                else:
                    q = q + plain_attention(block.self_attn, q+tokens, q+tokens, q, 'explicit')
                q = block.norm1(q)
                q = block.norm2(q + dense_read(block.cross_attn_token_to_image, q+tokens, x, cache.first, i, plans[2*i], 'explicit'))
                q = block.norm3(q + block.mlp(q))
                x = block.norm4(x + dense_write(block.cross_attn_image_to_token, q+tokens, q, x, cache.first, i, plans[2*i+1], 'explicit'))
            q = model.transformer.norm_final_attn(q + dense_read(model.transformer.final_attn_token_to_image, q+tokens, x, cache.first, len(model.transformer.layers), plans[-1], 'explicit'))
            hyper = torch.stack([mlp(q[:, i+1]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)
            torch.cuda.synchronize()

            def original():
                return phase_outputs(model, cache, x @ cache.conv_matrix + cache.conv_bias, q)

            def current_hyper():
                return torch.stack([mlp(q[:, i+1]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)

            def compact0():
                return head(model, student, x, current_hyper(), center, basis, 0), model.iou_prediction_head(q[:, 0])

            def compact5():
                return head(model, student, x, current_hyper(), center, basis, .05), model.iou_prediction_head(q[:, 0])

            def compact10():
                return head(model, student, x, current_hyper(), center, basis, .1), model.iou_prediction_head(q[:, 0])

            def phase5():
                return phase_fallback_head(model, cache, student, x, current_hyper(), center, basis, .05), model.iou_prediction_head(q[:, 0])

            def phase10():
                return phase_fallback_head(model, cache, student, x, current_hyper(), center, basis, .1), model.iou_prediction_head(q[:, 0])

            compiled = {}
            report['status'] = 'COMPILING_HEADS'
            save()
            arms = [('dense_phase_head', original), ('compact0', compact0), ('compact5', compact5), ('compact10', compact10)]
            if phase_fallback:
                arms += [('phase5', phase5), ('phase10', phase10)]
            for name, fn in arms:
                report['current_arm'] = name
                save()
                start = time.monotonic()
                compiled[name] = torch.compile(fn, backend='inductor', fullgraph=True, dynamic=False)
                value = compiled[name]()
                torch.cuda.synchronize()
                assert value[0].shape == (32, 4, 256, 256) and value[1].shape == (32, 4)
                del value
                report.setdefault('compile_seconds', {})[name] = time.monotonic()-start
                save()
            if phase_fallback:
                for budget in (5, 10):
                    a, b = compiled['compact'+str(budget)](), compiled['phase'+str(budget)]()
                    report.setdefault('phase_layout_diagnostics', {})[str(budget)] = {
                        'max_logit_difference': float((a[0]-b[0]).abs().max()),
                        'lowres_pixel_flip_fraction': float(((a[0]>0)!=(b[0]>0)).float().mean()),
                        'max_IQ_difference': float((a[1]-b[1]).abs().max()),
                    }
                    del a, b
                # Only original head and real FP32 final states promoted to FP64.
                # This proves patch layout, not full transformer/encoder precision.
                from types import SimpleNamespace
                patch_model = SimpleNamespace(output_upscaling=copy.deepcopy(model.output_upscaling).double())
                patch_cache = SimpleNamespace(**{key: getattr(cache, key).double() for key in ('conv_matrix', 'conv_bias', 'second_matrix', 'second_bias')})
                states = x[:, :64].reshape(-1, 256).double()
                expected = patch_model.output_upscaling(states.reshape(-1, 256, 1, 1)).flatten(2)
                actual = phase_patches(patch_model, patch_cache, states)
                error = float((expected-actual).abs().max())
                report['head_only_FP64_patch_check'] = {'parents': len(states), 'max_error': error, 'tolerance': 1e-10, 'passed': error<=1e-10}
                save()
                assert error <= 1e-10, report['head_only_FP64_patch_check']
                del expected, actual, states, patch_model, patch_cache
            names = tuple(compiled)
            report['status'] = 'SHARED_HEAD_ONLY_DIAGNOSTIC'
            save()
            for r in range(len(names)):
                for name in names[r:]+names[:r]:
                    before = gpu_state()
                    for _ in range(3):
                        value = compiled[name]()
                        del value
                    timing = measure(compiled[name], device, 10)
                    report['records'].append({'round': r+1, 'arm': name, 'timing': timing, 'before': before, 'after': gpu_state()})
                    save()
                    print(json.dumps({'round': r+1, 'arm': name, 'median_ms': timing['median_ms']}), flush=True)
            report['status'] = 'PROFILING_KERNELS_AND_STAGES'
            save()
            for name in ('dense_phase_head', 'compact10') + (('phase10',) if phase_fallback else ()):
                report.setdefault('profiles', {})[name] = profile_forward(compiled[name], device, OUT / (name+'.trace.json'))
                save()

            def labeled_compact10():
                with torch.profiler.record_function('original_hypernetworks'):
                    hyper = current_hyper()
                with torch.profiler.record_function('coarse_coefficients_and_logit_contraction'):
                    coeff = student(x)
                    weights = torch.einsum('pmc,ucr->pmur', hyper, basis.reshape(16, 32, 64)).reshape(32, 64, 64)
                    bias = torch.einsum('pmc,uc->pmu', hyper, center.reshape(16, 32)).reshape(32, 64)
                    ph = (torch.bmm(coeff, weights.transpose(1, 2)) + bias[:, None]).reshape(32, 4096, 4, 16)
                with torch.profiler.record_function('normalized_margin_and_topk'):
                    margin = (ph.abs()/hyper.norm(dim=-1)[:, None, :, None].clamp_min(1e-6)).amin((-2, -1))
                    indices = margin.topk(math.ceil(4096*.1), dim=-1, largest=False).indices
                    batch = torch.arange(32, device=device)[:, None]
                with torch.profiler.record_function('gather_and_original_local_deconvolution'):
                    patches = model.output_upscaling(x[batch, indices].reshape(-1, 256, 1, 1)).flatten(2)
                    h = hyper[:, None].expand(-1, indices.shape[1], -1, -1).reshape(-1, 4, 32)
                    exact = torch.bmm(h, patches).reshape(32, indices.shape[1], 4, 16)
                with torch.profiler.record_function('scatter_and_output_layout'):
                    ph[batch, indices] = exact
                    masks = grid(ph)
                return masks, model.iou_prediction_head(q[:, 0])

            a = labeled_compact10()
            b = compact10()
            report['stage_decomposition_max_logit_difference'] = float((a[0]-b[0]).abs().max())
            assert report['stage_decomposition_max_logit_difference'] == 0
            del a, b
            report['profiles']['eager_labeled_compact10'] = profile_forward(labeled_compact10, device, OUT/'eager_labeled_compact10.trace.json')
        report['status'] = 'COMPLETED_SHARED_HEAD_ATTRIBUTION_NOT_DECODER_SPEED'
    except BaseException as exc:
        report.update(status='ERROR', error=repr(exc))
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=5)
        if telemetry_log is not None:
            telemetry_log.close()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--phase-fallback', action='store_true')
    args = parser.parse_args()
    if args.output_dir is not None:
        OUT = args.output_dir
    main(args.phase_fallback)
