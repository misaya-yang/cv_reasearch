#!/usr/bin/env python3
"""CPU readout for the frozen query-only SAM3 second-pass candidates.

Reads first-pass anchor/candidate masks, second-pass masks, and query labels only
after the second-pass JSONL is complete. Chooses one second-pass retention rule on
DEV, then reports that frozen rule on CONFIRM and the already-exposed rest shard 0.
All IoUs are original-resolution class mIoU; intervals use connected support/query
photo-group paired bootstrap from sam3_stitch.paired.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import sam3_stitch as S


RULES = ("absolute_0.3", "absolute_0.5", "relative_0.6", "relative_0.7", "relative_0.8", "fallback_top1")
SELECTABLE = tuple("anchor_plus_" + r for r in RULES)


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def key(r):
    return tuple(map(int, r["key"] if "key" in r else (r["fold"], r["e"], r["c"])))


def choose(scores, areas, rule):
    valid = [i for i, area in enumerate(areas) if int(area) > 0]
    if rule.startswith("absolute_"):
        t = float(rule.split("_")[1])
        return [i for i in valid if float(scores[i]) > t]
    if not valid:
        return []
    top = max(valid, key=lambda i: float(scores[i]))
    if rule == "fallback_top1":
        keep = [i for i in valid if float(scores[i]) > 0.5]
        return keep or [top]
    frac = float(rule.split("_")[1])
    return [i for i in valid if float(scores[i]) >= frac * float(scores[top])]


def unpack(path, count, height, width):
    with np.load(path, allow_pickle=False) as z:
        flat = np.unpackbits(z["proposal_query"], axis=1)[:, :height * width]
    if flat.shape[0] != count:
        raise ValueError(f"candidate count mismatch: {path}")
    return flat.reshape(count, height, width).astype(bool)


def original_iu(canvas_mask, truth, rect):
    full = np.zeros((S.CANVAS, S.CANVAS), np.uint8)
    x, y, w, h = rect
    full[y:y+h, x:x+w] = canvas_mask.astype(np.uint8)
    pred = S.to_original(full, truth.shape[::-1])
    return [int((pred & truth).sum()), int((pred | truth).sum())]


def load_split(root, split, manifest_path, second_rel):
    run_name = "rest" if split == "rest0" else split
    run = root / "results/sam3_keep_v1" / run_name
    pred_path = run / "episodes_shard0.jsonl"
    first = rows(pred_path)
    second = rows(run / second_rel / "second.jsonl")
    first_by = {key(r): r for r in first}
    second_by = {key(r): r for r in second}
    if len(first_by) != len(first) or len(second_by) != len(second):
        raise ValueError(f"duplicate episode keys in {split}")
    if first_by.keys() != second_by.keys():
        raise ValueError(f"first/second pass episode set differs in {split}: {len(first_by)} vs {len(second_by)}")
    manifest = json.loads(Path(manifest_path).read_text())
    rect = S.rectangles()[1]
    out = []
    for base in first:
        k = key(base); sp = second_by[k]
        if not sp.get("candidate_file"):
            sh, sw = base["proposal_shape"][1:]
            smasks = np.zeros((0, sh, sw), bool)
            sscores, sareas = [], []
        else:
            n, sh, sw = map(int, sp["proposal_shape"])
            smasks = unpack(run / second_rel / sp["candidate_file"], n, sh, sw)
            sscores, sareas = sp["score"], sp["area"]
            if len(sscores) != n or len(sareas) != n:
                raise ValueError(f"second-pass score/mask count differs in {split} {k}")
        n, h, w = map(int, base["proposal_shape"])
        fmasks = unpack(run / base["candidate_file"], n, h, w)
        meta = base["proposals"]
        # The second-pass record freezes the first-pass top nonempty proposal index.
        anchor_idx = sp.get("anchor")
        anchor = fmasks[int(anchor_idx)] if anchor_idx is not None else np.zeros((h, w), bool)
        truth = np.asarray(Image.open(Path(manifest["annotation_root"]) / Path(base["query"]).with_suffix(".png"))) == int(base["c"]) + 1
        if truth.shape != tuple(base["query_shape"]):
            raise ValueError(f"query label shape mismatch in {split} {k}")
        canvas_arms = {}
        canvas_arms["first_relative_0.7"] = fmasks[S_relative_pick("relative_0.7", meta)].any(0) if S_relative_pick("relative_0.7", meta) else np.zeros((h, w), bool)
        base_fallback = S_relative_pick("fallback_top1", meta)
        canvas_arms["first_fallback_top1"] = fmasks[base_fallback].any(0) if base_fallback else np.zeros((h, w), bool)
        canvas_arms["anchor_only"] = anchor
        for rule in RULES:
            idx = choose(sscores, sareas, rule) if smasks.shape[0] else []
            second_union = smasks[idx].any(0) if idx else np.zeros((h, w), bool)
            canvas_arms["second_only_" + rule] = second_union
            canvas_arms["anchor_plus_" + rule] = anchor | second_union
        rec = {"fold": int(base["fold"]), "e": int(base["e"]), "c": int(base["c"]),
               "support": base["support"], "query": base["query"], "split": split, "anchor_index": anchor_idx,
               "original_iu": {name: original_iu(mask, truth, rect) for name, mask in canvas_arms.items()}}
        out.append(rec)
    return out


def S_relative_pick(rule, proposals):
    """Label-free reproduction of sam3_relative_decision.pick using score and area only."""
    q = [i for i, p in enumerate(proposals) if int(p[1]) > 0]
    if rule.startswith("absolute_"):
        t = float(rule.split("_")[1])
        return [i for i in q if float(proposals[i][0]) > t]
    if not q:
        return []
    top = max(q, key=lambda i: float(proposals[i][0]))
    if rule == "fallback_top1":
        keep = [i for i in q if float(proposals[i][0]) > .5]
        return keep or [top]
    frac = float(rule.split("_")[1])
    return [i for i in q if float(proposals[i][0]) >= frac * float(proposals[top][0])]


def metric(recs, arm, baseline="first_relative_0.7"):
    get = lambda r: r["original_iu"][arm]
    base = lambda r: r["original_iu"][baseline]
    return S.paired(recs, get, base)


def report(a):
    root = Path(a.root)
    configs = {
        "dev": (a.dev_manifest, "second"),
        "confirm": (a.confirm_manifest, "second"),
        "rest0": (a.rest_manifest, "second"),
    }
    prior = json.loads(Path(a.frozen_from).read_text()) if a.frozen_from else None
    out = prior or {"state": "RUNNING", "method": "query-only full-resolution SAM3 second pass; union first-pass top-score anchor with selected second-pass top-20 masks",
                    "labels_exposure": {"dev": "development, previously exposed", "confirm": "historically exposed by prior SAM3 readouts", "rest0": "previously scored for relative-threshold study; exploratory only"},
                    "selectable_rules": list(SELECTABLE), "splits": {}}
    dev_rows = None
    if "dev" in a.splits:
        dev = load_split(root, "dev", *configs["dev"])
        dev_rows = {name: metric(dev, name) for name in SELECTABLE}
        frozen = max(SELECTABLE, key=lambda name: dev_rows[name]["miou"])
        out["selection_rule_frozen_on_dev"] = frozen
    else:
        frozen = (prior or {}).get("selection_rule_frozen_on_dev")
        if not frozen:
            raise ValueError("CONFIRM/rest0 scoring requires a rule frozen on DEV or --frozen-from")
    for split in ("dev", "confirm", "rest0"):
        if split not in a.splits:
            continue
        if split == "confirm" and "dev" not in a.splits and not prior:
            raise ValueError("CONFIRM cannot be opened before a DEV-frozen rule")
        recs = dev if split == "dev" else load_split(root, split, *configs[split])
        base = "first_relative_0.7"
        names = ["first_relative_0.7", "first_fallback_top1", "anchor_only", "second_only_relative_0.7", frozen]
        if split == "dev":
            names.extend(SELECTABLE)
        rows_out = {}
        for name in dict.fromkeys(names):
            if name not in {k for r in recs for k in r["original_iu"]}:
                raise ValueError(f"missing arm {name} on {split}")
            rows_out[name] = dev_rows[name] if split == "dev" and dev_rows is not None and name in dev_rows else metric(recs, name, base)
        # Comparisons against the second strong baseline are paired on the same records.
        second_control = [metric(recs, frozen, "first_fallback_top1"), metric(recs, frozen, "anchor_only")]
        out["splits"][split] = {"episodes": len(recs), "groups": metric(recs, frozen)["groups"], "rows_vs_first_relative_0.7": rows_out,
                                 "frozen_rule_vs_fallback": second_control[0], "frozen_rule_vs_anchor_only": second_control[1],
                                 "per_fold_gain_vs_relative_0.7": {str(f): S.paired([r for r in recs if r["fold"] == f],
                                     lambda r: r["original_iu"][frozen], lambda r: r["original_iu"]["first_relative_0.7"], draws=1)["gain"]
                                     for f in sorted({r["fold"] for r in recs})}}
    out["state"] = "COMPLETED" if all(k in out["splits"] for k in ("dev", "confirm", "rest0")) else "PARTIAL_READOUTS_COMPLETE"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    summary = {"state": out["state"], "frozen_rule": frozen, "scored_splits": sorted(a.splits)}
    for split in a.splits:
        summary[split + "_gain_vs_relative_0.7"] = out["splits"][split]["rows_vs_first_relative_0.7"][frozen]["gain"]
        summary[split + "_ci95"] = out["splits"][split]["rows_vs_first_relative_0.7"][frozen]["ci95"]
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/root/autodl-tmp/demo9_transductive_ics")
    p.add_argument("--dev-manifest", default="results/extent_head_t1_isolated_v1/dev_episodes.json")
    p.add_argument("--confirm-manifest", default="results/extent_head_t1_isolated_v1/confirm_episodes.json")
    p.add_argument("--rest-manifest", default="results/sam3_keep_v1/rest/rest_episodes.json")
    p.add_argument("--out", default="results/sam3_relative_v1/second_pass_analysis.json")
    p.add_argument("--splits", nargs="+", choices=("dev", "confirm", "rest0"), default=("dev", "confirm", "rest0"))
    p.add_argument("--frozen-from", help="prior report containing the DEV-selected rule; use to read later splits without reopening earlier labels")
    a = p.parse_args()
    for attr in ("dev_manifest", "confirm_manifest", "rest_manifest", "out", "frozen_from"):
        if not getattr(a, attr):
            continue
        value = Path(getattr(a, attr))
        if not value.is_absolute():
            setattr(a, attr, str(Path(a.root) / value))
    report(a)


if __name__ == "__main__":
    main()
