#!/usr/bin/env python3
"""Bounded positive/negative witnesses and native-four complete CPU outputs.

Synthetic labels are explicitly constructed; native query labels are never read.
The four exposed RGB128 packs only check activity/cost, not population quality.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from ics.cpu100.common import Episode, Result, load_episode, prototype_margin, render, sha, unit
from ics.cpu100.cross_image_matching import (
    METHODS, CONTROLS, _roles, _compress, gram_objective, gram_gradient,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/local/cpu100_20261006/reports/cross_image_matching"


def result_margin(fn, ep):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = fn(ep)
    out = value.margin if isinstance(value, Result) else value
    if not np.isfinite(out).all():
        raise FloatingPointError("Non-finite actual signed margin")
    return np.asarray(out).ravel()


def confusion(margin, truth):
    prediction = margin > 0
    return dict(tp=int(np.sum(prediction & truth)), fp=int(np.sum(prediction & ~truth)),
                fn=int(np.sum(~prediction & truth)), tn=int(np.sum(~prediction & ~truth)))


def fixtures():
    vectors = lambda a: np.column_stack((np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))))
    r = vectors(np.array([0., 45.]))
    q = vectors(np.r_[np.linspace(-5, 5, 50), -20, 20])
    csls = Episode(q=q, r=r, wf=np.array([1., 0.]), wvalid=np.ones(2),
                   q_hw=(4, 13), r_hw=(1, 2), q_valid=np.ones(52))
    truth = np.zeros(52, bool); truth[-2] = True
    countertruth = truth.copy(); countertruth[-1] = True
    corrected = result_margin(METHODS["cross_image_csls_hubness"], csls)
    control = result_margin(CONTROLS["cross_image_cosine_dictionary_control"], csls)
    csls_record = dict(
        positive=confusion(corrected, truth), control=confusion(control, truth),
        specified_target_margin=float(corrected[-2]), specified_false_margin=float(corrected[-1]),
        control_specified_false_margin=float(control[-1]),
        counterexample=confusion(corrected, countertruth),
        counterexample_control=confusion(control, countertruth),
        reflection="Hub density changes an identity decision but does not prove it is background; identical legal inputs permit an alternative true-target label. Most support tokens remain FP in this fixture.",
    )
    assert corrected[-2] > 0 and corrected[-1] < 0 < control[-1]

    n, d, a = 32, 34, .2
    bg = np.zeros((n, d)); bg[:, 0] = -a
    bg[np.arange(n), np.arange(n) + 1] = np.sqrt(1 - a*a)
    qbg = bg.copy(); qbg[:, 0] = a
    fg = np.eye(d)[0]
    r = np.vstack((np.tile(fg, (50, 1)), bg))
    q = np.vstack((qbg, fg))
    anchor = Episode(q=q, r=r, wf=np.r_[np.ones(50), np.zeros(32)], wvalid=np.ones(82),
                     q_hw=(3, 11), r_hw=(2, 41), q_valid=np.ones(33))
    truth = np.zeros(33, bool); truth[-1] = True
    shifted = result_margin(METHODS["cross_image_background_anchor_shift"], anchor)
    plain = result_margin(prototype_margin, anchor)
    whole = result_margin(CONTROLS["cross_image_whole_mean_shift_control"], anchor)
    exceptional_target = np.zeros(d); exceptional_target[0] = .4
    exceptional_target[1:33] = np.sqrt(.84 / 32)
    qnegative = q.copy(); qnegative[-1] = exceptional_target
    negative = Episode(q=qnegative, r=r, wf=anchor.wf, wvalid=anchor.wvalid,
                       q_hw=anchor.q_hw, r_hw=anchor.r_hw, q_valid=anchor.q_valid)
    negative_shift = result_margin(METHODS["cross_image_background_anchor_shift"], negative)
    negative_plain = result_margin(prototype_margin, negative)
    anchor_record = dict(
        positive=confusion(shifted, truth), prototype_control=confusion(plain, truth),
        whole_mean_control=confusion(whole, truth),
        positive_info=METHODS["cross_image_background_anchor_shift"](anchor).info,
        negative=confusion(negative_shift, truth), negative_control=confusion(negative_plain, truth),
        negative_target_margin=float(negative_shift[-1]), negative_control_margin=float(negative_plain[-1]),
        reflection="The matched BG residual avoids global foreground-composition bias here. Perfect mutual anchors and unanimous residuals still do not imply a style shared by the target: an exceptional target is wrongly deleted.",
    )
    assert np.all(shifted[:-1] < 0) and shifted[-1] > 0
    assert np.all(plain[:-1] > 0) and np.all(whole[:-1] > 0)
    assert negative_shift[-1] < 0 < negative_plain[-1]

    rng = np.random.default_rng(41)
    r = unit(rng.normal(size=(8, 6))); q = -r
    wf = np.r_[np.ones(4), np.zeros(4)]
    gram = Episode(q=q, r=r, wf=wf, wvalid=np.ones(8),
                   q_hw=(2, 4), r_hw=(2, 4), q_valid=np.ones(8))
    truth = wf > 0
    relational = result_margin(METHODS["cross_image_free_column_gram_matching"], gram)
    appearance = result_margin(CONTROLS["cross_image_appearance_assignment_control"], gram)
    profile = result_margin(CONTROLS["cross_image_sorted_gram_profile_control"], gram)
    fg, bg, _, _ = _roles(gram, maximum=32)
    reference = np.concatenate((fg, bg))
    query, _, _ = _compress(gram.q, gram.q_valid, 64)
    cost, dq, dr = 1-query@reference.T, 1-query@query.T, 1-reference@reference.T
    true_assignment = np.eye(len(reference))[(-query@reference.T).argmax(1)]
    gram_record = dict(
        relation=confusion(relational, truth), appearance_control=confusion(appearance, truth),
        profile_control=confusion(profile, truth),
        exact_true_correspondence_objective=gram_objective(true_assignment, cost, dq, dr),
        optimized_info=METHODS["cross_image_free_column_gram_matching"](gram).info,
        reflection="Fixed fused objective fails its intended rotation witness: relation and appearance both classify 1/8 correctly, while cheap sorted profiles classify 8/8. The optimized objective is below the true relation mapping: the cross-cosine term favors semantic mistakes, so lower objective is not success. Do not tune variants into a winner; this fixed version has no positive quality witness.",
        fixed_version_decision="closed_quality_claim; implementation retained as negative result",
    )
    assert np.count_nonzero((relational > 0) == truth) == 1
    assert np.count_nonzero((appearance > 0) == truth) == 1
    assert np.count_nonzero((profile > 0) == truth) == 8

    r2 = unit(rng.normal(size=(7, 5))); q2 = unit(rng.normal(size=(6, 5)))
    p = rng.dirichlet(np.ones(7), 6)
    c, dqq, drr = 1-q2@r2.T, 1-q2@q2.T, 1-r2@r2.T
    direction = rng.normal(size=p.shape); direction -= direction.mean(1, keepdims=True); direction *= .01
    eps = 1e-5
    analytic = float(np.sum(gram_gradient(p, c, dqq, drr) * direction))
    numeric = (gram_objective(p+eps*direction, c, dqq, drr)
               - gram_objective(p-eps*direction, c, dqq, drr)) / (2*eps)
    assert abs(analytic-numeric) < 1e-8

    empty = Episode(q=csls.q, r=csls.r, wf=np.zeros(2), wvalid=csls.wvalid,
                    q_hw=csls.q_hw, r_hw=csls.r_hw, q_valid=csls.q_valid)
    fallback = {key: bool(np.all(result_margin(fn, empty) < 0)) for key, fn in METHODS.items()}
    assert all(fallback.values())
    return dict(csls_hubness=csls_record, background_anchor_shift=anchor_record,
                free_column_gram=gram_record,
                gram_directional_gradient_error=abs(analytic-numeric),
                gradient_audit_scope="row-simplex tangent directions with zero row sums; not an ambient derivative of the substituted objective",
                mirror_update_scaling="row_softmax(logP - 0.25 * number_query_modes * gradient); gradient uses row-mean objective",
                empty_reference_rejects_all=fallback,
                evidence_scope="synthetic_witness and algebraic audit only; no population mIoU")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(schema="CPU100_CROSS_IMAGE_MATCHING_V1", synthetic=fixtures(), native_four=[])
    manifest = ROOT / "evidence/local/research_20261006/direct_dino_input_01a1100b/inputs4/manifest.json"
    inputs = json.loads(manifest.read_text())
    arms = {**METHODS, **CONTROLS, "cross_image_prototype_control": lambda ep: Result(prototype_margin(ep))}
    for row in inputs["rows"]:
        ep = load_episode(row)
        folder = OUT / "native_four" / ep.source_id
        folder.mkdir(parents=True, exist_ok=True)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            baseline = render(ep, Result(prototype_margin(ep)))
        for key, fn in arms.items():
            start = time.perf_counter()
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = fn(ep)
            elapsed = time.perf_counter() - start
            rendered = render(ep, result)
            path = folder / f"{key}.npz"
            np.savez_compressed(path, signed_margin_native=result.margin,
                                signed_margin_work64=rendered["margin"],
                                work_mask=rendered["work"], original_mask=rendered["original"])
            report["native_four"].append(dict(
                episode=ep.source_id, method=key, inference_seconds=elapsed,
                original_fg_pixels=int(rendered["original"].sum()),
                changed_original_pixels_vs_prototype=int(np.count_nonzero(
                    rendered["original"] != baseline["original"])),
                complete_output=str(path), sha256=sha(path), info=result.info,
            ))
    report.update(query_GT_used=False, empirical_quality="unmeasured; no query GT available",
                  native_scope="four exposed native DINO RGB128 final-LN/unit, 8x8x1024; not RGB1024/full cohort",
                  code_sha256=sha(ROOT / "src/ics/cpu100/cross_image_matching.py"))
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(methods=3, native_complete_outputs=len(report["native_four"]),
                          gradient_error=report["synthetic"]["gram_directional_gradient_error"],
                          report=str(OUT / "report.json"))))


if __name__ == "__main__":
    main()
