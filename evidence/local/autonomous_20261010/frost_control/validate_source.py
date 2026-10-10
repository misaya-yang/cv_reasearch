"""Reviewed official FROST math fixtures and reference-only cache prerequisites.

No backbone is constructed, no query image/mask is loaded, and no real query
prediction is run. This is evidence code, not an approximate FROST method.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ASSETS = REPO.parent / "cv_data"
SOURCE = ASSETS / "third_party/frost_official"
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from ics.official_data import array_hash, decoded_rgb
from raw_feature_cache import canonical_hash, file_hash, tensor_hash


def reviewed_source():
    receipt = json.loads((HERE / "source_receipt.json").read_text())
    for row in receipt["source_files"]:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Pinned official source changed: " + row["relative"])
    # Do not execute __init__ or encoder/build_encoder. Import only the three
    # inspected math/preprocessing modules; no torch.hub path is loaded.
    package = ModuleType("_reviewed_frost_math_20261010")
    package.__path__ = [str(SOURCE / "frost")]
    sys.modules[package.__name__] = package
    density = importlib.import_module(package.__name__ + ".density")
    data = importlib.import_module(package.__name__ + ".data")
    model = importlib.import_module(package.__name__ + ".model")
    return density, data, model


def fixtures(density, data, model):
    rng = torch.Generator().manual_seed(20261010)
    unit = lambda n, d: F.normalize(torch.randn(n, d, generator=rng, dtype=torch.float64), dim=1)
    fg, bg, test = unit(5, 8), unit(7, 8), unit(9, 8)
    sigma = .20  # One independent algebra fixture, not a dataset search.
    actual = density.kde_log_posterior(test, fg, bg, sigma)
    expected = torch.log(torch.exp((test @ fg.T - 1) / sigma).mean(1)) - torch.log(torch.exp((test @ bg.T - 1) / sigma).mean(1))
    kde_error = float((actual - expected).abs().max())
    ff = torch.exp((fg @ fg.T - 1) / sigma); ff.fill_diagonal_(0)
    bb = torch.exp((bg @ bg.T - 1) / sigma); bb.fill_diagonal_(0)
    fb = torch.exp((fg @ bg.T - 1) / sigma)
    fg_loo = torch.log(ff.sum(1) / (len(fg) - 1)) - torch.log(fb.mean(1))
    bg_loo = torch.log(fb.T.mean(1)) - torch.log(bb.sum(1) / (len(bg) - 1))
    margin_expected = float(fg_loo.mean() - bg_loo.mean())
    margin_actual = density.loo_margin_score(fg, bg, sigma)
    margins = [density.loo_margin_score(fg, bg, s) for s in model._SIGMA_GRID]
    selected, selection_margin = density.estimate_bandwidth(fg, bg)
    checks = dict(kde_independent_exp_mean_parity=kde_error < 1e-12,
                  loo_independent_exp_sum_parity=abs(margin_actual - margin_expected) < 1e-12,
                  original_grid_argmax=selected == model._SIGMA_GRID[int(np.argmax(margins))],
                  original_grid_margin=selection_margin == max(margins),
                  singleton_loo_is_zero=density.loo_margin_score(fg[:1], bg, sigma) == 0)
    # The implemented selection margin is balanced and invariant to a constant
    # log-prior shift. The paper's pooled signed margin is not invariant when
    # anchor counts differ. Neither makes zero-threshold error a tuned objective.
    offset = 2.
    balanced = float(fg_loo.mean() - bg_loo.mean())
    balanced_offset = float((fg_loo + offset).mean() - (bg_loo + offset).mean())
    pooled = float((fg_loo.sum() - bg_loo.sum()) / (len(fg) + len(bg)))
    pooled_offset = float(((fg_loo + offset).sum() - (bg_loo + offset).sum()) / (len(fg) + len(bg)))
    checks["balanced_margin_offset_invariant"] = abs(balanced - balanced_offset) < 1e-12
    checks["paper_pooled_margin_offset_response"] = abs((pooled_offset - pooled) - offset * (len(fg) - len(bg)) / (len(fg) + len(bg))) < 1e-12

    # Exact equivariant duplicate views do not change the class densities at a
    # query. Patch-LOO nevertheless sees the other copy as a perfect self-like
    # match, so its mean class separation increases. True encoded flips need
    # not be exact duplicates; this is an informative limiting case only.
    duplicated_fg, duplicated_bg = fg.repeat(2, 1), bg.repeat(2, 1)
    duplicate_score = density.kde_log_posterior(test, duplicated_fg, duplicated_bg, sigma)
    duplicate_margin = density.loo_margin_score(duplicated_fg, duplicated_bg, sigma)
    expected_duplicate_fg_own = (2 * ff.sum(1) + 1) / (2 * len(fg) - 1)
    expected_duplicate_bg_own = (2 * bb.sum(1) + 1) / (2 * len(bg) - 1)
    expected_duplicate_margin = float((torch.log(expected_duplicate_fg_own) - torch.log(fb.mean(1))).mean()
                                      - (torch.log(fb.T.mean(1)) - torch.log(expected_duplicate_bg_own)).mean())
    checks["class_cloud_duplication_preserves_density_ratio"] = float((duplicate_score - actual).abs().max()) < 1e-12
    checks["duplicated_view_patch_loo_exact_alias_formula"] = abs(duplicate_margin - expected_duplicate_margin) < 1e-12
    checks["duplicated_view_patch_loo_margin_increases"] = duplicate_margin > margin_actual

    c, h, w, views = 6, 2, 3, 2
    refs = torch.randn(1, views, c, h, w, generator=rng)
    tgt = torch.randn(1, c, h, w, generator=rng)
    masks = torch.zeros(views, 1, 4, 6, dtype=torch.bool); masks[:, :, :2] = True
    white_ref, white_tgt = model.FROST._shrinkage_whiten(SimpleNamespace(), refs, tgt, masks, views, h, w)
    xf, xb = [], []
    for s in range(views):
        label = data.downsample_mask(masks[s:s + 1], h, w)
        xf.append(refs[0, s, :, label]); xb.append(refs[0, s, :, ~label])
    xf, xb = torch.cat(xf, 1), torch.cat(xb, 1)
    af, ab = xf - xf.mean(1, keepdim=True), xb - xb.mean(1, keepdim=True)
    scatter = (af @ af.T + ab @ ab.T) / (xf.shape[1] + xb.shape[1] - 2)
    lam = .95
    covariance = (1 - lam) * scatter + lam * scatter.diagonal().mean() * torch.eye(c)
    eig, vec = torch.linalg.eigh(covariance)
    transform = (vec * eig.clamp_min(1e-6).rsqrt()[None]) @ vec.T
    expected_ref = (refs.permute(0, 1, 3, 4, 2) @ transform).permute(0, 1, 4, 2, 3)
    expected_tgt = (tgt.permute(0, 2, 3, 1) @ transform).permute(0, 3, 1, 2)
    checks["whiten_exact_reference_parity"] = torch.equal(white_ref, expected_ref)
    checks["whiten_exact_target_parity"] = torch.equal(white_tgt, expected_tgt)
    checks["whiten_no_post_norm"] = not torch.allclose(white_tgt.norm(dim=1), torch.ones(1, h, w))

    # A token can pass the source top-three reference FG vote even if neither
    # foreground reference places it in its top-three query neighbours.
    support_mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool); support_mask[:, :, :2] = True
    matrix = torch.tensor([[40., 4., 10., 8.], [30., 3., 9., 7.],
                           [20., 2., 15., 13.], [10., 1., 14., 12.]])
    sims = [matrix.reshape(1, 2, 2, 2, 2)]
    target = torch.stack([torch.ones(2, 2), torch.zeros(2, 2)])[None]
    candidate = model.FROST._locate_candidates(SimpleNamespace(), sims, support_mask, target, torch.tensor([[1.], [0.]]), 2, 2)
    expected_candidate = torch.tensor([[True, True], [False, False]])
    checks["source_backward_is_query_to_reference_vote"] = torch.equal(candidate, expected_candidate)
    checks["source_gate_passes_nonreciprocal_token"] = bool(candidate.flatten()[1]) and not bool((matrix[:2].topk(3, dim=1).indices == 1).any())
    # With two augmented support views the source ceil(S/2) rule is a union.
    sims_other = [sims[0], matrix[[2, 3, 0, 1]].reshape(1, 2, 2, 2, 2)]
    candidate_two = model.FROST._locate_candidates(SimpleNamespace(), sims_other, support_mask.repeat(2, 1, 1, 1), target, torch.tensor([[1.], [0.]]), 2, 2)
    checks["two_view_vote_threshold_is_union"] = bool(candidate_two.all())

    feature = unit(9, 6); colors = torch.rand(9, 3, generator=rng, dtype=torch.float64)
    yy, xx = torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij")
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
    score = torch.randn(9, generator=rng, dtype=torch.float64)
    fp, bp = unit(1, 6)[0], unit(1, 6)[0]
    smoothed = density.bilateral_propagate(score, feature, colors, coords, fp, bp)
    f, rgb, xy, m = feature.numpy(), colors.numpy(), coords.numpy(), score.numpy()
    weight = np.exp((f @ f.T) / .20) * np.exp(-((rgb[:, None] - rgb[None]) ** 2).sum(-1) / .05)
    weight *= (((xy[:, None] - xy[None]) ** 2).sum(-1) ** .5 <= 16)
    normalized = m / (m.std(ddof=1) + 1e-6)
    weight *= np.exp(-(normalized[:, None] - normalized[None]) ** 2)
    np.fill_diagonal(weight, 0)
    walk = weight / np.maximum(weight.sum(1, keepdims=True), 1e-6)
    expected_smooth = m.copy()
    for _ in range(10):
        expected_smooth = .30 * m + .70 * (walk @ expected_smooth)
    bilateral_error = float(np.abs(smoothed.numpy() - expected_smooth).max())
    checks["bilateral_independent_five_kernel_parity"] = bilateral_error < 1e-12
    tiny = torch.zeros(1, 1, 16, 16, dtype=torch.bool); tiny[:, :, 9, 9] = True
    down = data.downsample_mask(tiny, 2, 2)
    checks["tiny_mask_center_fallback"] = torch.equal(down, torch.tensor([[False, False], [False, True]]))
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(checks=checks, fixture_seed=20261010, real_query_inputs=0,
                kde_max_abs_error=kde_error, loo_margin_abs_error=abs(margin_actual - margin_expected),
                bilateral_max_abs_error=bilateral_error, source_selected_sigma=selected,
                source_sigma_grid=list(model._SIGMA_GRID), source_loo_margins=margins,
                duplicate_view_limit=dict(query_kde_max_abs_error=float((duplicate_score - actual).abs().max()),
                                          source_loo_before=margin_actual, source_loo_after=duplicate_margin,
                                          alias_formula_abs_error=abs(duplicate_margin - expected_duplicate_margin),
                                          scope="Synthetic exact-duplicate limit, not evidence actual encoded hflips are identical."),
                shift=dict(offset=offset, balanced_before=balanced, balanced_after=balanced_offset,
                           paper_pooled_before=pooled, paper_pooled_after=pooled_offset))


def prerequisites(data):
    source_manifest = ASSETS / "a/cache_mechanism_study_20261010/reference_identity/manifest.json"
    cfg_path = ASSETS / "a/cache_mechanism_study_20261010/reference_identity/config.json"
    rows = json.loads(source_manifest.read_text())
    cfg = json.loads(cfg_path.read_text()); profile_path = Path(cfg["raw_profile"])
    profile = json.loads(profile_path.read_text())
    if canonical_hash(profile) != profile_path.parent.name or file_hash(profile_path) != cfg["raw_profile_sha256"]:
        raise ValueError("Existing cache profile identity differs")
    transform = data.build_transform(1024)
    audit = []; stats = defaultdict(lambda: defaultdict(list))
    for row in rows:
        image = decoded_rgb(ASSETS / row["reference_path"], row["reference_crop"])
        if array_hash(np.asarray(image)) != row["reference_rgb_hash"]:
            raise ValueError("Reference image binding differs")
        model_input = transform(image).numpy()
        key_for = lambda x: canonical_hash(dict(profile=profile_path.parent.name, input_tensor_hash=tensor_hash(x)))
        if key_for(model_input) != row["raw_reference"]["key"]:
            raise ValueError("Official FROST and existing input transformations differ")
        flipped = np.ascontiguousarray(model_input[:, :, ::-1])
        flip_key = key_for(flipped); metadata = profile_path.parent / flip_key / "entry.json"
        found = metadata.is_file()
        if found:
            meta = json.loads(metadata.read_text())
            found = "O/24" in meta["features"] and meta["input_tensor_hash"] == tensor_hash(flipped)
        with Image.open(row["reference_mask_path"]) as image_mask:
            mask_array = (np.asarray(image_mask.convert("L")) > 0).astype(np.uint8)
        if array_hash(mask_array) != row["reference_mask_hash"]:
            raise ValueError("Reference mask binding differs")
        canvas = data.load_mask(torch.from_numpy(mask_array.copy()), 1024, "cpu").unsqueeze(1)
        c = F.adaptive_avg_pool2d(canvas.float(), (64, 64))[0, 0]
        hard = data.downsample_mask(canvas, 64, 64)
        pixel_prior = float(canvas.float().mean())
        stat = dict(pixel_prior=pixel_prior, foreground_area_tokens=float(c.sum()),
                    binary_fg_tokens=int(hard.sum()), mixed_tokens=int(((c > 0) & (c < 1)).sum()),
                    hard_fg_area_mass=float(c[hard].sum()), hard_bg_lost_fg_area_mass=float(c[~hard].sum()),
                    source_dilation_radius=int(round(4 * pixel_prior)))
        for name, value in stat.items():
            stats[row["dataset"]][name].append(value)
        audit.append(dict(episode_id=row["episode_id"], dataset=row["dataset"],
                          reference_input_key=row["raw_reference"]["key"], horizontal_flip_input_key=flip_key,
                          horizontal_flip_O24_cached=found, reference_mask=stat))
    basis_path = ASSETS / "native_assets/positional_basis.pt"
    document = torch.load(basis_path, map_location="cpu", weights_only=True)
    basis = document["basis"]
    prefix = basis[:, :250].contiguous()
    eig = torch.linalg.eigvalsh(prefix.double().T @ prefix.double())
    black = torch.zeros(3, 1024, 1024)
    black = ((black - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]).numpy()
    black_key = canonical_hash(dict(profile=profile_path.parent.name, input_tensor_hash=tensor_hash(black)))
    basis_report = dict(path=str(basis_path), sha256=file_hash(basis_path),
                        metadata={k: v for k, v in document.items() if k != "basis"},
                        prefix250_tensor_sha256=tensor_hash(prefix.numpy()),
                        prefix250_gram_spectral_defect=float((eig - 1).abs().max()),
                        source_code_input="ImageNet-normalized black torch.zeros; FROST .float(), native artifact claims FP32",
                        black_input_tensor_sha256=tensor_hash(black), black_input_cache_key=black_key,
                        black_input_O24_entry_exists=(profile_path.parent / black_key / "entry.json").is_file(),
                        producer_input_tensor_hash_in_payload="input_tensor_sha256" in document,
                        producer_checkpoint_hash_in_payload="weights_sha256" in document,
                        exact_FROST_hub_vs_native_timm_prefix_parity_verified=False,
                        implication="The black-input semantics agree; numerical U250 equivalence is not certified by metadata alone.")
    aggregate = {ds: {k: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                             min=float(np.min(v)), max=float(np.max(v))) for k, v in stat.items()} for ds, stat in stats.items()}
    return dict(state="EXACT_FULL_FROST_CACHE_WRAPPER_NOT_CREATED", references_checked=len(rows),
                unique_reference_inputs=len({r["reference_input_key"] for r in audit}),
                flip_entries_found=sum(r["horizontal_flip_O24_cached"] for r in audit),
                unique_flip_entries_found=len({r["horizontal_flip_input_key"] for r in audit if r["horizontal_flip_O24_cached"]}),
                profile_path=str(profile_path), profile_sha256=file_hash(profile_path),
                source_manifest=str(source_manifest), source_manifest_sha256=file_hash(source_manifest),
                transform_input_keys_all_match=True, rows=audit, reference_mask_statistics=aggregate,
                basis=basis_report, query_images_read=0, query_GT_read=0, raw_feature_payloads_read=0,
                new_encoder_calls=0, real_query_predictions=0,
                reason="A full official head requires actual separately encoded flipped references; no token-grid flip substitute or reduced-view wrapper was created.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    density, data, model = reviewed_source()
    result = dict(time_utc=datetime.now(timezone.utc).isoformat(),
                  source_commit="b9ece69d7495a698c298e7cc3d16efacd4497a43",
                  validation_script_sha256=file_hash(__file__), math=fixtures(density, data, model),
                  prerequisites=prerequisites(data))
    (HERE / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(checks=len(result["math"]["checks"]), all_math_checks_passed=True,
                         reference_inputs=result["prerequisites"]["references_checked"],
                         flipped_O24_entries_found=result["prerequisites"]["flip_entries_found"],
                         real_query_predictions=0)))
