#!/usr/bin/env python3
"""Sealed complete FoRIS/CRF inference with one shared frozen DINOv3 instance.

Fixed reference-value candidate, matched-norm uniform-value control, foreground
ablation and existing key-only control. Never reads query truth. No downloads.
Use --self-check for tiny local synthetic contracts without constructing DINO.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_intervention import file_sha, paired_input_audit


def self_check():
    import torch
    import torch.nn.functional as F
    from ics.methods.reference_value import (ReferenceValueConfig, bounded_message,
                                             reference_messages, reference_value)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    cfg = ReferenceValueConfig()
    ref, qry, val = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
    cov = torch.tensor([1., .75, .25, 0.])
    msg = reference_messages(ref, qry, val, cov)
    similarity = F.normalize(qry, dim=-1) @ F.normalize(ref, dim=-1).T / cfg.temperature
    wf = (similarity + torch.where(cov > 0, cov.log(), -torch.inf)).softmax(-1)
    wb = (similarity + torch.where(cov < 1, (1-cov).log(), -torch.inf)).softmax(-1)
    assert torch.allclose(msg["foreground"], wf @ val)
    assert torch.allclose(msg["background"], wb @ val)
    order = torch.tensor([3, 1, 0, 2])
    permuted = reference_messages(ref[order], qry, val[order], cov[order])
    assert torch.allclose(msg["foreground"], permuted["foreground"], atol=1e-6)
    assert torch.allclose(msg["background"], permuted["background"], atol=1e-6)
    native = torch.randn_like(qry)
    deltas = {a: bounded_message(msg, native, a) for a in
              ("control", "reference_value", "foreground_only")}
    for delta in deltas.values():
        assert torch.allclose(delta.norm(dim=-1), .1 * native.norm(dim=-1), atol=1e-6)
    assert not torch.allclose(deltas["reference_value"], deltas["control"])
    unit_control = F.normalize(deltas["control"], dim=-1)
    assert torch.allclose(unit_control, unit_control[:1].expand_as(unit_control), atol=1e-6)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fused_attn = True
            self.qkv = torch.nn.Linear(8, 24)
            self.norm = torch.nn.LayerNorm(8)
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            q, k, v = self.qkv(x).reshape(2, 6, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
            y = F.scaled_dot_product_attention(q, k, v)
            return self.proj(self.norm(y.transpose(1, 2).reshape(2, 6, 8)))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens = 2
            self.blocks = torch.nn.ModuleList([Block() for _ in range(6)])

    vit = Encoder().eval().requires_grad_(False)
    attn, x = vit.blocks[0].attn, torch.randn(2, 6, 8)
    mask = torch.tensor([[1., 1.], [0., 0.]])
    original_sdpa, original_forward = F.scaled_dot_product_attention, attn.forward.__func__
    with torch.inference_mode():
        expected = attn(x)
        for arm, config in (("identity", cfg), ("reference_value", replace(cfg, residual_ratio=0.))):
            with reference_value(vit, mask, arm=arm, config=config) as receipt:
                actual = attn(x)
            assert torch.equal(actual, expected)
            assert receipt["sdpa_calls"] == receipt["attention_calls"] == 1
        capture = {}
        with reference_value(vit, mask, capture=capture) as receipt:
            actual = attn(x)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1, :2], expected[1, :2])
        assert not torch.equal(actual[1, 2:], expected[1, 2:])
        assert torch.allclose(capture["injected_delta"].norm(dim=-1),
                              .1 * capture["native_sdpa"].norm(dim=-1), atol=1e-6)
        try:
            with reference_value(vit, torch.zeros(2, 2)):
                attn(x)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty foreground was accepted")
    assert F.scaled_dot_product_attention is original_sdpa
    assert "forward" not in attn.__dict__ and attn.forward.__func__ is original_forward
    assert all(not p.requires_grad for p in vit.parameters())
    return dict(state="SYNTHETIC_CONTRACTS_PASSED", actual_timm_or_real_image_test=False,
                checks=["conditional value means equal direct weighted sums", "reference permutation",
                        "per-token norm match", "conditional versus uniform differs", "identity exact",
                        "zero residual exact", "reference and prefixes exact", "query changes",
                        "empty-role fail closed", "exception restoration", "frozen parameters"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--foris-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return
    if not args.manifest or not args.out:
        parser.error("--manifest and --out are required for real inference")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    import numpy as np
    import torch
    from PIL import Image
    from ics.data import TimmDINOv3
    from ics.experiment import load_rows
    from ics.foris import build_host, run_foris
    from ics.methods.intervention import CONFIG as KEY_CONFIG, attention_intervention
    from ics.methods.reference_value import CONFIG, ReferenceValueConfig, reference_value

    man = json.loads(Path(args.manifest).read_text())
    rows = load_rows(args.manifest)[:args.limit]
    root = Path(args.out)
    if root.exists():
        parser.error("Output exists; choose a fresh directory")
    root.mkdir(parents=True)
    for name in ("packets", "predictions"):
        (root / name).mkdir()
    (root / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    sources = [Path(__file__), Path(__file__).parents[1] / "src/ics/methods/reference_value.py",
               Path(__file__).with_name("run_intervention.py"),
               Path(__file__).parents[1] / "src/ics/methods/intervention.py",
               Path(__file__).parents[1] / "src/ics/foris.py"]
    protocol = dict(state="INFERENCE_ONLY", config=CONFIG, key_only_config=KEY_CONFIG,
                    manifest=str(Path(args.manifest).resolve()), manifest_sha256=file_sha(args.manifest),
                    source_sha256={str(p.resolve()): file_sha(p) for p in sources},
                    dataset="COCO-20i", split="existing exposed DEV", seed=0,
                    episodes=len(rows), model_resolution=1024, finalization="complete public FoRIS and CRF",
                    encoder_instances=1, frozen_weights=True, added_images=0, added_models=0,
                    query_gt_usage="never read; separate sealed-run scoring after inference exits",
                    reference_input="one image with its complete supplied foreground annotation",
                    native_basis=man.get("projection_basis"), development_only=True,
                    controls=["native", "reference_value.control", "reference_value.key_only.control"],
                    ablation="reference_value.foreground_only", per_arm_paired_forwards=1,
                    ordinary_pair_forwards_per_episode=5, first_episode_total_pair_forwards=7,
                    additional_audit_pair_forwards=2, added_method_encoder_forwards=0,
                    value_control_budget="equal per-token L2 norm; identical retrieval computed then discarded",
                    key_only_budget="existing C key-only RMS-matched routing; not value-norm-matched",
                    data_root=str(Path(man["data_root"]).resolve()),
                    annotation_root=str(Path(man["annotation_root"]).resolve()))
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    host = build_host(man, args.device, args.foris_root)
    encoders = [m for m in host.modules() if isinstance(m, TimmDINOv3)]
    if len(encoders) != 1:
        raise RuntimeError("Requires exactly one shared encoder instance")
    vit = encoders[0].m
    started, hashes, audit_rows = time.monotonic(), {}, {}
    with torch.inference_mode(), (root / "episodes.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            with Image.open(Path(man["data_root"]) / row["support"]) as im:
                support = im.convert("RGB")
            with Image.open(Path(man["data_root"]) / row["query"]) as im:
                query = im.convert("RGB")
            with Image.open(Path(man["annotation_root"]) / Path(row["support"]).with_suffix(".png")) as im:
                support_mask = torch.from_numpy((np.asarray(im) == row["c"] + 1).copy())
            masks, scores, pre_masks, diag, receipts, timings, references = {}, {}, {}, {}, {}, {}, {}
            arms = ["native"] + (["plain_audit", "zero_audit"] if index == 0 else [])
            arms += ["key_only", "control", "reference_value", "foreground_only"]
            baseline = None
            for arm in arms:
                capture = {}
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                before = time.monotonic()
                with paired_input_audit(vit, host) as pair:
                    if arm == "plain_audit":
                        pred, observed, _, _ = run_foris(host, support, support_mask, query)
                        receipt = {}
                    elif arm == "key_only":
                        with attention_intervention(vit, lambda: host._ref_masks[0], arm="control") as receipt:
                            pred, observed, _, _ = run_foris(host, support, support_mask, query)
                    else:
                        cfg = ReferenceValueConfig()
                        value_arm = "identity" if arm == "native" else arm
                        if arm == "zero_audit":
                            value_arm, cfg = "reference_value", replace(cfg, residual_ratio=0.)
                        with reference_value(vit, lambda: host._ref_masks[0], arm=value_arm,
                                             config=cfg, capture=capture) as receipt:
                            pred, observed, _, _ = run_foris(host, support, support_mask, query)
                    if pair["calls"] != 1 or (receipt and receipt["attention_calls"] != 1):
                        raise RuntimeError("Each arm must execute exactly one actual paired encoder forward")
                if args.device == "cuda":
                    torch.cuda.synchronize()
                timings[arm] = dict(seconds=time.monotonic()-before,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
                masks[arm] = pred.cpu().numpy().astype(bool)
                scores[arm] = observed["score"].float().cpu().numpy()
                pre_masks[arm] = observed["pre"].bool().cpu().numpy()
                if masks[arm].shape != (1024, 1024) or not np.isfinite(scores[arm]).all():
                    raise RuntimeError("Expected finite complete 1024 FoRIS/CRF outputs")
                receipts[arm] = dict(attention=dict(receipt), paired_input=dict(pair))
                if index == 0:
                    raw = observed["raw"]
                    if raw.ndim == 5 and tuple(raw.shape[:2]) == (1, 2):
                        ref = raw[0, 0]
                    elif raw.ndim == 4 and raw.shape[0] == 2:
                        ref = raw[0]
                    else:
                        raise RuntimeError(f"Unknown raw reference layout: {tuple(raw.shape)}")
                    references[arm] = ref.float().cpu().numpy().copy()
                if arm == "native":
                    baseline = capture
                elif capture:
                    if not torch.equal(capture["native_sdpa"], baseline["native_sdpa"]):
                        raise RuntimeError("Pre-injection native SDPA changed across arms")
                    block_delta = capture["attention_output"] - baseline["attention_output"]
                    diag[arm + "_projected_attention_delta_norm"] = block_delta.norm(dim=-1).cpu().numpy()
                    diag[arm + "_projected_attention_native_norm"] = baseline["attention_output"].norm(dim=-1).cpu().numpy()
                    for key in ("density_margin", "entropy_fg", "entropy_bg", "contrast_uniform_cosine"):
                        diag[arm + "_" + key] = capture[key].cpu().numpy()
                    diag[arm + "_injected_delta_norm"] = capture["injected_delta"].norm(dim=-1).cpu().numpy()
                    diag[arm + "_native_sdpa_norm"] = capture["native_sdpa"].norm(dim=-1).cpu().numpy()
                    del capture
                del observed
            if index == 0:
                checks = {}
                for arm in ("plain_audit", "zero_audit"):
                    checks[arm] = dict(mask_exact=bool(np.array_equal(masks[arm], masks["native"])),
                        score_close=bool(np.allclose(scores[arm], scores["native"], atol=1e-5, rtol=1e-5)),
                        score_max_abs=float(np.max(np.abs(scores[arm]-scores["native"]))),
                        reference_close=bool(np.allclose(references[arm], references["native"], atol=1e-5, rtol=1e-5)))
                for arm in ("key_only", "control", "reference_value", "foreground_only"):
                    checks[arm + ".reference"] = dict(
                        exact=bool(np.array_equal(references[arm], references["native"])),
                        close=bool(np.allclose(references[arm], references["native"], atol=1e-5, rtol=1e-5)),
                        max_abs=float(np.max(np.abs(references[arm]-references["native"]))))
                (root / "first_episode_audit.json").write_text(json.dumps(checks, indent=2)+"\n")
                if (any(not all(checks[a][k] for k in ("mask_exact", "score_close", "reference_close"))
                        for a in ("plain_audit", "zero_audit"))
                        or any(not checks[a+".reference"]["close"] for a in
                               ("key_only", "control", "reference_value", "foreground_only"))):
                    (root / "audit_failure.json").write_text(json.dumps(dict(
                        state="NATIVE_OR_REFERENCE_PARITY_FAILURE", inference_sealed=False, details=checks), indent=2)+"\n")
                    raise RuntimeError("Native/zero/reference audit failed; predictions are unsealed")
            names = dict(native="native", key_only="reference_value.key_only.control",
                         control="reference_value.control", reference_value="reference_value",
                         foreground_only="reference_value.foreground_only")
            name = row["key"] + ".npz"
            np.savez_compressed(root / "predictions" / name,
                                **{names[a]: np.packbits(masks[a]) for a in names})
            packet = {a: np.packbits(m) for a, m in masks.items()}
            packet.update({a+"_score": s.astype(np.float32) for a, s in scores.items()})
            packet.update({a+"_pre": np.packbits(m) for a, m in pre_masks.items()})
            packet.update({a+"_pre_shape": np.asarray(m.shape, dtype=np.int64) for a, m in pre_masks.items()})
            packet.update({k: v.astype(np.float32) for k, v in diag.items()})
            np.savez_compressed(root / "packets" / name, **packet)
            hashes[row["key"]] = file_sha(root / "predictions" / name)
            audit_rows[row["key"]] = dict(timings=timings, receipts=receipts)
            stream.write(json.dumps(dict(row, timings=timings, receipts=receipts))+"\n")
            stream.flush()
            print(json.dumps(dict(completed=index+1, total=len(rows), timings=timings)), flush=True)
            del baseline, references, masks, scores, pre_masks, packet
    (root / "audits.json").write_text(json.dumps(audit_rows, indent=2)+"\n")
    (root / "sealed.json").write_text(json.dumps(dict(state="ALL_PREDICTIONS_SEALED",
        manifest_sha256=file_sha(root/"manifest.json"), protocol_sha256=file_sha(root/"protocol.json"),
        predictions=hashes, query_labels_opened=False), indent=2)+"\n")
    (root / "completion.json").write_text(json.dumps(dict(state="COMPLETED", episodes=len(rows),
        elapsed_seconds=time.monotonic()-started), indent=2)+"\n")


if __name__ == "__main__":
    main()
