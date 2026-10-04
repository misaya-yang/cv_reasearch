#!/usr/bin/env python3
"""Pair frozen SAM3 proposal-rule outputs with FoRIS on the same CONFIRM episodes.

CPU-only: joins saved per-episode I/U records, validates their episode/image identities,
and bootstraps connected support/query photo groups. It opens no annotation files.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    out = {}
    for row in rows:
        key = (row["fold"], row["e"], row["c"])
        if key in out:
            raise ValueError("duplicate episode key: " + repr(key))
        out[key] = row
    return out


def main():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import sam3_stitch as S

    sam3_path = ROOT / "results/sam3_relative_v1/confirm.json.episodes.jsonl"
    uid_path = ROOT / "results/sam3_relative_v1/sam3_confirm_uid_iu.jsonl"
    foris_path = ROOT / "results/decision_v1/infer_1_confirm/episodes.jsonl"
    audit_path = ROOT / "results/fixed_host_union_v1/confirm_episodes.jsonl"
    sam3, uid, foris, audit = (read_jsonl(p) for p in (sam3_path, uid_path, foris_path, audit_path))
    if not (len(sam3) == len(uid) == len(foris) == len(audit) == 600 and
            sam3.keys() == uid.keys() == foris.keys() == audit.keys()):
        raise ValueError("The four frozen records do not cover the same 600 episodes")

    records = []
    top20_native_mismatches = 0
    for key in sorted(sam3):
        s, u, f, a = sam3[key], uid[key], foris[key], audit[key]
        if any(s.get(k) != f.get(k) or f.get(k) != a.get(k) or u.get(k) != f.get(k)
               for k in ("fold", "e", "c")):
            raise ValueError("episode identity mismatch: " + repr(key))
        if (u.get("support") != f.get("support") or u.get("query") != f.get("query") or
                f.get("support") != a.get("support") or f.get("query") != a.get("query")):
            raise ValueError("support/query identity mismatch: " + repr(key))
        foris_iu = f["original_iu"]["native"]
        if list(foris_iu) != list(a["original_iu"]["foris"]):
            raise ValueError("FoRIS native I/U does not match the frozen host audit: " + repr(key))
        top20_native_mismatches += (list(s["rule_iu"]["absolute_0.5"]["original"]) !=
                                    list(u["full_visual_iu"]))
        records.append(dict(fold=key[0], e=key[1], c=key[2], support=u["support"], query=u["query"],
                            sam3=s["rule_iu"], sam3_full_visual=u["full_visual_iu"], foris=foris_iu))

    result = dict(state="COMPLETED", dataset="COCO-20i", split="CONFIRM", seed=0, episodes=len(records),
                  paired_key_join=True, sam3_image_uid_matches_foris=600,
                  foris_native_matches_host_audit=True,
                  top20_absolute_0p5_differs_from_full_visual_episodes=top20_native_mismatches,
                  result_scope="SAM3 top-20 proposal rules; full native visual union differs in the reported cases",
                  annotation_files_opened_during_pairing=False,
                  bootstrap="2000 draws over connected support/query photo groups, seed 0",
                  comparisons={})
    for rule in ("relative_0.7", "fallback_top1", "absolute_0.3"):
        sam3_get = lambda r, rule=rule: r["sam3"][rule]["original"]
        foris_get = lambda r: r["foris"]
        paired = S.paired(records, sam3_get, foris_get)
        reverse = S.paired(records, foris_get, sam3_get)
        full_get = lambda r: r["sam3_full_visual"]
        vs_full = S.paired(records, sam3_get, full_get)
        per_fold = [S.paired([r for r in records if r["fold"] == fold], sam3_get, foris_get, draws=1)["gain"]
                    for fold in sorted({r["fold"] for r in records})]
        per_fold_full = [S.paired([r for r in records if r["fold"] == fold], sam3_get, full_get, draws=1)["gain"]
                         for fold in sorted({r["fold"] for r in records})]
        case_delta = [sam3_get(r)[0] / max(sam3_get(r)[1], 1) - r["foris"][0] / max(r["foris"][1], 1)
                      for r in records]
        result["comparisons"][rule] = dict(sam3_miou=paired["miou"], foris_miou=reverse["miou"],
                                             gain=paired["gain"], ci95=paired["ci95"], groups=paired["groups"],
                                             largest_group=paired["largest_group"], per_fold=per_fold,
                                             episodes_up=sum(v > 0 for v in case_delta),
                                             episodes_down=sum(v < 0 for v in case_delta),
                                             full_visual_miou=S.paired(records, full_get, sam3_get)["miou"],
                                             gain_over_full_visual=vs_full["gain"],
                                             ci95_over_full_visual=vs_full["ci95"],
                                             per_fold_over_full_visual=per_fold_full)

    out = ROOT / "results/sam3_relative_v1/foris_paired_confirm.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    for rule, row in result["comparisons"].items():
        lo, hi = row["ci95"]
        print("%s SAM3 %.2f vs FoRIS %.2f: %+0.2f [%+0.2f, %+0.2f]; vs full SAM3 %.2f: %+0.2f [%+0.2f, %+0.2f]; folds %s; groups %d" %
              (rule, row["sam3_miou"], row["foris_miou"], row["gain"], lo, hi,
               row["full_visual_miou"], row["gain_over_full_visual"], *row["ci95_over_full_visual"],
               " ".join("%+.2f" % v for v in row["per_fold"]), row["groups"]))
    print("saved", out)


if __name__ == "__main__":
    main()
