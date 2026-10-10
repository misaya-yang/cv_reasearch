"""Frozen saved-table contextual-unary and conditional-pair information audit.

No raw features, encoder, graph solver, new segmentation or parameter fitting.
Use prepare to freeze this source/recipe/input identities before run opens GT.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = REPO.parent / "cv_data/a"
COPULA = DATA / "reference_boundary_copula200_20261010"
SCENE = DATA / "autonomous_scene_reconstruction600_20261010"
EPS = 1e-6
PAIR_MODELS = ("factorized_zero", "full_true", "full_shuffle", "delta_true", "delta_shuffle")
ROLE_NAMES = ("00", "01", "10", "11")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def components(table):
    a, b, c, d = table[:, 0, 0], table[:, 0, 1], table[:, 1, 0], table[:, 1, 1]
    return (a+b+c+d)/4, (c+d-a-b)/4, (b+d-a-c)/4, a+d-b-c


def context_mean(table, edges, nodes=4096):
    _, left, right, delta = components(table)
    values = np.zeros(nodes, np.float64)
    degree = np.zeros(nodes, np.int64)
    np.add.at(values, edges[:, 0], left)
    np.add.at(values, edges[:, 1], right)
    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)
    assert (degree > 0).all()
    return values/degree, delta, degree


def pair_probabilities(probability, edges, tables):
    p = np.clip(np.asarray(probability, np.float64), EPS, 1-EPS)
    role = np.stack((1-p, p), axis=1)
    base = np.log(role[edges[:, 0], :, None]) + np.log(role[edges[:, 1], None, :])
    output = {}
    for name, table in tables.items():
        logits = base+table
        logits -= logits.max((1, 2), keepdims=True)
        e = np.exp(logits)
        output[name] = e/e.sum((1, 2), keepdims=True)
    return output


def rank(score, positive, negative):
    score = np.asarray(score).reshape(-1)
    positive, negative = np.asarray(positive), np.asarray(negative)
    total_p, total_n = float(positive.sum()), float(negative.sum())
    if total_p <= 0 or total_n <= 0:
        return dict(auc=None, ap=None, positive_mass=total_p, negative_mass=total_n)
    order = np.argsort(score, kind="stable")
    values = score[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    p, n = np.add.reduceat(positive[order], starts), np.add.reduceat(negative[order], starts)
    occupied = p+n > 0
    p, n = p[occupied], n[occupied]
    numerator = float(np.sum(p*(np.cumsum(n)-n+.5*n)))
    tp, fp = np.cumsum(p[::-1]), np.cumsum(n[::-1])
    return dict(auc=numerator/(total_p*total_n),
                ap=float(np.sum((p[::-1]/total_p)*tp/(tp+fp))),
                positive_mass=total_p, negative_mass=total_n,
                auc_numerator=numerator, auc_denominator=total_p*total_n)


def pair_scores(prob, truth):
    probability, target = prob.reshape(-1, 4), truth.reshape(-1, 4)
    n = len(probability)
    if not n:
        return None
    state_mass = target.sum(0)
    nll = -np.log(np.maximum(probability, np.finfo(np.float64).tiny))
    # Expected categorical Brier over the soft target: E_s ||P-one_hot(s)||^2.
    brier = (probability*probability).sum(1)[:, None]-2*probability+1
    nll_sum, brier_sum = (target*nll).sum(0), (target*brier).sum(0)
    balanced = None
    if (state_mass > 0).all():
        balanced = dict(nll=float(np.mean(nll_sum/state_mass)),
                        brier=float(np.mean(brier_sum/state_mass)))
    return dict(n=n, raw_nll=float(nll_sum.sum()/n), raw_brier=float(brier_sum.sum()/n),
                soft_target_squared_error=float(np.mean(((probability-target)**2).sum(1))),
                four_state_balanced=balanced, state_mass=state_mass.tolist(),
                state_nll_sum=nll_sum.tolist(), state_brier_sum=brier_sum.tolist())


def fixtures():
    table = np.array([[[1., 2.], [3., 4.]], [[4., -1.], [-2., 3.]]])
    g, hi, hj, delta = components(table)
    sigma = np.array([-1., 1.])
    rebuilt = g[:, None, None]+hi[:, None, None]*sigma[None, :, None]+hj[:, None, None]*sigma[None, None, :]+delta[:, None, None]*sigma[None, :, None]*sigma[None, None, :]/4
    assert np.allclose(table, rebuilt, rtol=0, atol=1e-14)
    edges = np.array([[0, 1], [1, 2]])
    context, _, degree = context_mean(table, edges, 3)
    assert np.array_equal(degree, [1, 2, 1])
    assert np.allclose(context, [hi[0], (hj[0]+hi[1])/2, hj[1]])
    probabilities = pair_probabilities([.2, .7, .6], edges, {"zero": np.zeros_like(table)})["zero"]
    assert np.allclose(probabilities[0], [[.24, .56], [.06, .14]], rtol=0, atol=1e-15)
    symmetric = np.array([[[0., 1.], [0., 1.]], [[0., -2.], [0., -2.]]])
    _, _, _, d = components(symmetric)
    assert np.array_equal(d, [0, 0])
    updated = pair_probabilities([.2, .7, .6], edges, {"unary": symmetric})["unary"]
    assert np.allclose(updated.sum(2), [[.8, .2], [.3, .7]], atol=1e-14)
    expected_right = np.array([.7*np.e/(.3+.7*np.e), .6*np.exp(-2)/(.4+.6*np.exp(-2))])
    assert np.allclose(updated.sum(1)[:, 1], expected_right, atol=1e-14)
    perfect = np.array([[[1., 0.], [0., 0.]]])
    s = pair_scores(perfect, perfect)
    assert s["raw_nll"] == s["raw_brier"] == 0 and s["four_state_balanced"] is None
    uniform = np.full((4, 2, 2), .25)
    target = np.eye(4).reshape(4, 2, 2)
    s = pair_scores(uniform, target)
    assert np.isclose(s["raw_nll"], np.log(4)) and s["raw_brier"] == .75
    assert s["four_state_balanced"]["brier"] == .75
    assert rank([1, 0], [1, 0], [0, 1])["auc"] == 1
    assert rank([.5, .5], [1, 0], [0, 1])["auc"] == .5
    return {"exact_Hadamard_decomposition": True, "incident_degree_mean": True,
            "factorized_proxy_identity": True, "delta_zero_context_updates_marginal": True,
            "proper_NLL_Brier": True, "exact_rank_ties": True, "real_GT_reads": 0}


def prepare():
    assert not (OUT / "protocol.json").exists(), "Do not refreeze an analysis after query scoring"
    checks = fixtures()
    copula_cfg, copula_seal = read(COPULA / "config.json"), read(COPULA / "sealed.json")
    scene_cfg, scene_seal = read(SCENE / "config.json"), read(SCENE / "sealed.json")
    assert copula_seal["n"] == 200 and scene_seal["n"] == 600
    assert sha(COPULA / "config.json") == copula_seal["config_sha256"]
    assert sha(SCENE / "config.json") == scene_seal["config_sha256"]
    assert sha(SCENE / "sealed.json") == copula_cfg["source_seal_sha256"]
    inputs = {}
    for root, cfg, seal in ((COPULA, copula_cfg, copula_seal), (SCENE, scene_cfg, scene_seal)):
        for file in ("config.json", "sealed.json", "manifest.json", "tasks.json"):
            inputs[str(root / file)] = sha(root / file)
        for relative, digest in cfg["source_sha256"].items():
            p = root / "frozen" / relative
            assert sha(p) == digest
            inputs[str(p)] = digest
    manifest, tasks = read(COPULA / "manifest.json"), read(COPULA / "tasks.json")
    scene_manifest, scene_tasks = read(SCENE / "manifest.json")[:200], read(SCENE / "tasks.json")[:200]
    assert manifest == scene_manifest and tasks == scene_tasks
    assert len(manifest) == len(tasks) == len(copula_seal["receipts"]) == 200
    for row, task, a, b in zip(manifest, tasks, copula_seal["receipts"], scene_seal["receipts"][:200]):
        assert row["episode_id"] == task["episode_id"] == a["episode_id"] == b["episode_id"]
        assert a["apd_applied"] == b["apd_applied"]
        for root, rec in ((COPULA, a), (SCENE, b)):
            p = root / "fields" / rec["filename"]
            assert sha(p) == rec["fields_sha256"]
            inputs[str(p)] = rec["fields_sha256"]
        baseline = task["baselines"]["foris.crf"]
        assert sha(baseline["path"]) == baseline["sha256"]
        inputs[baseline["path"]] = baseline["sha256"]
    source_digest = sha(__file__)
    (OUT / "frozen").mkdir(exist_ok=True)
    shutil.copyfile(__file__, OUT / "frozen/analysis.py")
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(), n=200,
        source_sha256=source_digest, input_SHA256=inputs, no_GT_fixtures=checks,
        inputs="Saved copula2x2 tables/edges and exact first200 frozen scene600 scalar fields; no raw features",
        contextual_decomposition="h_i=(L10+L11-L00-L01)/4; h_j=(L01+L11-L00-L10)/4; exact L=g+h_i*sigma_i+h_j*sigma_j+delta*sigma_i*sigma_j/4",
        node_pooling="Mean endpoint h over actual incident4-neighborhood edges; boundary degree2/3, interior4; diagnostic only, not graph-product energy",
        node_ROI="whole/FoRIS_FG/FoRIS_BG are canonical1024 pixel masks; exact FG/BG pixel mass within each64 cell; all fields native64 score, exact-score ties",
        node_scores=["context_true", "context_shuffle", "context_zero", "support_ridge", "source_kernel", "scene"],
        pair_base_probability="fixed p=clip((frozen support_ridge+1)/2,1e-6,1-1e-6); lawful but uncalibrated; BG=1-p",
        pair_models=list(PAIR_MODELS),
        pair_likelihood="P(y,z) proportional p_i(y)*p_j(z)*exp(L_yz); pure_delta uses delta*(2y-1)*(2z-1)/4; factorized L exactly0",
        pair_truth="c=exact1024 GT patch coverage; T_yz=role_c_i(y)*role_c_j(z); a soft pixel-mass product diagnostic, not observed actual adjacent-pixel joint truth",
        pair_ROI="all; FoRIS_FG_incident=b_i OR b_j; FoRIS_border=b_i XOR b_j, with b=CLI1024 FoRIS patch foreground coverage strictly>.5",
        proper_scores="raw mean -sum_s T_s logP_s and expected categorical Brier sum_s T_s*(sum_t P_t^2-2P_s+1); soft-target squared L2 also recorded",
        balanced_scores="Mean over all4 states of state-weighted proper loss; per-episode4state result NA if any state mass0; pooled4state metric separately reported",
        aggregation="Macro per-episode means with valid n, paired deltas on identical episodes/ROI; edge-pooled raw and globally4state-balanced means separate",
        no_parameters_fit=True, query_GT_read_before_freeze=False, new_segmentation=False,
        exposure="Same previously exposed Deep100/PACO100; freeze is before this auditor's new GT access, not independent preregistration")
    write(OUT / "protocol.json", protocol)
    write(OUT / "FROZEN.json", dict(source_sha256=source_digest, protocol_sha256=sha(OUT / "protocol.json"),
                                   query_GT_reads=0, created_at=protocol["created_at"]))
    print(json.dumps(dict(state="ANALYSIS_FROZEN", n=200, source_sha256=source_digest,
                         protocol_sha256=sha(OUT / "protocol.json"), query_GT_reads=0)), flush=True)


def canonical_truth(row):
    with Image.open(row["query_mask_path"]) as image:
        value = (np.asarray(image.convert("L")) > 0).astype(np.uint8)
    h = hashlib.sha256(json.dumps([list(value.shape), value.dtype.str]).encode())
    h.update(np.ascontiguousarray(value).tobytes())
    assert h.hexdigest() == row["query_mask_hash"]
    ys = np.arange(1024, dtype=np.int64)*value.shape[0]//1024
    xs = np.arange(1024, dtype=np.int64)*value.shape[1]//1024
    return value[ys[:, None], xs[None, :]].astype(bool)


def mass64(value):
    return value.reshape(64, 16, 64, 16).sum((1, 3), dtype=np.float64).reshape(-1)


def numeric(values):
    if not values:
        return None
    x = np.asarray(values, np.float64)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)))


def aggregate(episodes):
    out = {}
    for ds in ("deepglobe_road", "paco_part"):
        group = [r for r in episodes if r["dataset"] == ds]
        node, pair, delta = {}, {}, {}
        for roi in ("whole", "FoRIS_FG", "FoRIS_BG"):
            node[roi] = {}
            for name in group[0]["node_rank"][roi]:
                valid = [r["node_rank"][roi][name] for r in group if r["node_rank"][roi][name]["auc"] is not None]
                node[roi][name] = dict(auc=numeric([r["auc"] for r in valid]), ap=numeric([r["ap"] for r in valid]),
                    within_episode_pair_weighted_auc=(sum(r["auc_numerator"] for r in valid)/sum(r["auc_denominator"] for r in valid)) if valid else None)
        for roi in ("all", "FoRIS_FG_incident", "FoRIS_border"):
            pair[roi], delta[roi] = {}, {}
            for name in PAIR_MODELS:
                valid = [r["pair_scores"][roi][name] for r in group if r["pair_scores"][roi][name] is not None]
                balanced = [r["four_state_balanced"] for r in valid if r["four_state_balanced"] is not None]
                state_mass = np.sum([r["state_mass"] for r in valid], axis=0) if valid else np.zeros(4)
                nll_sum = np.sum([r["state_nll_sum"] for r in valid], axis=0) if valid else np.zeros(4)
                brier_sum = np.sum([r["state_brier_sum"] for r in valid], axis=0) if valid else np.zeros(4)
                total_n = sum(r["n"] for r in valid)
                pair[roi][name] = dict(raw_nll=numeric([r["raw_nll"] for r in valid]),
                    raw_brier=numeric([r["raw_brier"] for r in valid]),
                    four_state_balanced_nll=numeric([r["nll"] for r in balanced]),
                    four_state_balanced_brier=numeric([r["brier"] for r in balanced]),
                    edge_count=total_n, state_mass=state_mass.tolist(),
                    pooled_raw_nll=float(nll_sum.sum()/total_n) if total_n else None,
                    pooled_raw_brier=float(brier_sum.sum()/total_n) if total_n else None,
                    pooled_four_state_balanced_nll=float(np.mean(nll_sum/state_mass)) if (state_mass > 0).all() else None,
                    pooled_four_state_balanced_brier=float(np.mean(brier_sum/state_mass)) if (state_mass > 0).all() else None)
            for a, b in (("full_true", "factorized_zero"), ("full_true", "full_shuffle"),
                         ("delta_true", "factorized_zero"), ("delta_true", "delta_shuffle")):
                valid = [(r["pair_scores"][roi][a], r["pair_scores"][roi][b]) for r in group
                         if r["pair_scores"][roi][a] is not None and r["pair_scores"][roi][b] is not None]
                balanced = [(x["four_state_balanced"], y["four_state_balanced"]) for x, y in valid
                            if x["four_state_balanced"] is not None and y["four_state_balanced"] is not None]
                delta[roi][a+" - "+b] = dict(raw_nll=numeric([x["raw_nll"]-y["raw_nll"] for x, y in valid]),
                    raw_brier=numeric([x["raw_brier"]-y["raw_brier"] for x, y in valid]),
                    four_state_balanced_nll=numeric([x["nll"]-y["nll"] for x, y in balanced]),
                    four_state_balanced_brier=numeric([x["brier"]-y["brier"] for x, y in balanced]))
        out[ds] = dict(n=len(group), node_rank=node, pair_scores=pair, paired_loss_deltas=delta)
    return out


def run():
    started = time.monotonic()
    assert not (OUT / "results.json").exists(), "Do not replace scored analysis"
    frozen, protocol = read(OUT / "FROZEN.json"), read(OUT / "protocol.json")
    assert sha(__file__) == sha(OUT / "frozen/analysis.py") == frozen["source_sha256"]
    assert sha(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, digest in protocol["input_SHA256"].items():
        assert sha(path) == digest, path
    manifest, tasks = read(COPULA / "manifest.json"), read(COPULA / "tasks.json")
    copula_receipts, scene_receipts = read(COPULA / "sealed.json")["receipts"], read(SCENE / "sealed.json")["receipts"][:200]
    episodes, checks, bank_summary = [], Counter(), defaultdict(list)
    sigma = np.array([-1., 1.])
    interaction_sign = sigma[:, None]*sigma[None, :]
    for row, task, a, b in zip(manifest, tasks, copula_receipts, scene_receipts):
        with np.load(COPULA / "fields" / a["filename"], allow_pickle=False) as z:
            edges = z["edge_index"].copy()
            real, shuffled = z["log_copula_true"].copy(), z["log_copula_fixed_shuffle"].copy()
            context, delta, degree = context_mean(real, edges)
            null_context, null_delta, null_degree = context_mean(shuffled, edges)
            assert np.array_equal(degree, null_degree) and np.array_equal(delta, z["delta_true"])
            assert np.array_equal(null_delta, z["delta_fixed_shuffle"]) and np.all(z["delta_factorized"] == 0)
        for table in (real, shuffled):
            g, hi, hj, d = components(table)
            rebuilt = g[:, None, None]+hi[:, None, None]*sigma[None, :, None]+hj[:, None, None]*sigma[None, None, :]+d[:, None, None]*interaction_sign/4
            assert np.max(np.abs(rebuilt-table)) < 1e-12
            checks["exact_decomposition"] += 1
        with np.load(SCENE / "fields" / b["filename"], allow_pickle=False) as z:
            scores = {k: z[k].reshape(-1).copy() for k in ("scene", "source_kernel", "support_ridge")}
        scores.update(context_true=context, context_shuffle=null_context, context_zero=np.zeros(4096))
        tables = dict(factorized_zero=np.zeros_like(real), full_true=real, full_shuffle=shuffled,
                      delta_true=delta[:, None, None]*interaction_sign/4,
                      delta_shuffle=null_delta[:, None, None]*interaction_sign/4)
        proxy = np.clip((scores["support_ridge"]+1)/2, EPS, 1-EPS)
        probabilities = pair_probabilities(proxy, edges, tables)
        assert all(np.isfinite(v).all() and np.max(np.abs(v.sum((1, 2))-1)) < 1e-12 for v in probabilities.values())
        checks["normalized_pair_probability_tables"] += len(probabilities)
        states = a["diagnostics"]["construction"]["states"]
        for state, diag in states.items():
            true = np.asarray(diag["true_pair_ids"], dtype=np.int64).reshape(-1, 2)
            null = np.asarray(diag["fixed_shuffle_pair_ids"], dtype=np.int64).reshape(-1, 2)
            assert len(true) == len(null) in (0, 64)
            assert not (true[:, 0] == true[:, 1]).any()
            for endpoint in (0, 1):
                assert np.array_equal(np.bincount(true[:, endpoint], minlength=4096), np.bincount(null[:, endpoint], minlength=4096))
                checks["endpoint_marginal_exact"] += 1
            bank_summary[row["dataset"]+"/"+state].append({k: diag[k] for k in
                ("occurrence_count", "effective_physical_true_pairs", "pair_multiset_changed_fraction", "shuffle_self_pair_count", "missing_state_independence_fallback")})
        # Every statistic/source/formula above was frozen before this query GT opens.
        gt = canonical_truth(row)
        checks["query_GT_array_identity"] += 1
        baseline = task["baselines"]["foris.crf"]
        with np.load(baseline["path"], allow_pickle=False) as z:
            pixels = np.unpackbits(z[baseline["keys"]["cli1024"]], count=1024**2).reshape(1024, 1024).astype(bool)
        rois = dict(whole=np.ones_like(gt), FoRIS_FG=pixels, FoRIS_BG=~pixels)
        node_rank = {}
        for roi, keep in rois.items():
            positive, negative = mass64(gt & keep), mass64(~gt & keep)
            node_rank[roi] = {k: rank(s, positive, negative) for k, s in scores.items()}
        c = mass64(gt)/256
        i, j = edges.T
        endpoint_role = np.stack((1-c, c), axis=1)
        target = endpoint_role[i, :, None]*endpoint_role[j, None, :]
        assert np.max(np.abs(target.sum((1, 2))-1)) < 1e-14
        coarse_b = mass64(pixels)/256 > .5
        edge_rois = dict(all=np.ones(len(i), bool), FoRIS_FG_incident=coarse_b[i] | coarse_b[j], FoRIS_border=coarse_b[i] != coarse_b[j])
        pair = {roi: {name: pair_scores(value[keep], target[keep]) for name, value in probabilities.items()}
                for roi, keep in edge_rois.items()}
        episodes.append(dict(episode_id=row["episode_id"], dataset=row["dataset"], node_rank=node_rank, pair_scores=pair,
            context_abs_mean=dict(true=float(np.abs(context).mean()), shuffle=float(np.abs(null_context).mean())),
            proxy_clipped_nodes=int(((scores["support_ridge"]+1)/2 < EPS).sum()+((scores["support_ridge"]+1)/2 > 1-EPS).sum()),
            input_field_SHA=dict(copula=a["fields_sha256"], scene=b["fields_sha256"])))
        if len(episodes) % 50 == 0:
            print(json.dumps(dict(scored=len(episodes), n=200, seconds=time.monotonic()-started)), flush=True)
    result = dict(state="COMPLETE_SAVED_TABLE_ANALYSIS", n=200, results=aggregate(episodes),
        bank_artifacts={k: {metric: numeric([float(v[metric]) for v in values]) for metric in values[0]} for k, values in bank_summary.items()},
        checks=dict(checks), seconds=time.monotonic()-started,
        frozen_analysis=sha(OUT / "FROZEN.json"), protocol_sha256=frozen["protocol_sha256"], source_sha256=frozen["source_sha256"],
        query_GT_reads=200, raw_feature_reads=0, encoder_calls=0, graph_solves=0, new_segmentation_masks=0,
        limits=["The probability proxy is lawful but not calibrated, so lower loss can repair its confidence rather than prove novel dependence.",
                "Context energy-unaries depend on both endpoint features; zero label coupling does not imply feature independence.",
                "Incident means are diagnostics, not sums in a graph-product energy or independent evidence from overlapping edges.",
                "Soft endpoint coverage product is not actual spatial adjacent-pixel joint truth or original-frame mIoU.",
                "Full true vs shuffle preserves selected empirical per-state endpoint marginals, not a universal source-independent unary model.",
                "KDE marginals differ across role states, and all fields/proxies use the same annotated reference; their evidence is statistically dependent.",
                "Finite pair budgets, duplicate occurrences and shuffle self-pairs can affect the contrast; artifacts are separately reported.",
                "Four-state-balanced macro scores exclude episodes missing any state mass; pooled balanced means and n are separate.",
                "Previously exposed Deep100/PACO100; no parameter tuning, calibrated confidence, new segmentation or generalization claim."])
    (OUT / "episodes.jsonl").write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in episodes))
    write(OUT / "results.json", result)
    print(json.dumps(dict(state=result["state"], n=200, seconds=result["seconds"], checks=result["checks"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.action]()
