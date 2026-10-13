"""Score and independently reconstruct the sealed remaining-domain outputs."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "src")]
import composition_core_interval_audit as core
from sol_verify_consensus900 import bilinear, unpack
from sol_score_bg_admission900 import connected_intervals, edits, stats
from edit_balance import summarize_edits
from ics.metrics import summarize
from ics.official_data import array_hash

OUT = REPO.parent / "cv_data/a/consensus_remaining120_v1_20261011"
ARMS = ("anchor.global", "shared_edits_complete", "same_edit_budget_source_order")


def select(eligible, score, k, descending):
    indices = np.flatnonzero(eligible.ravel())
    values = score.ravel()[indices]
    order = np.lexsort((indices, -values if descending else values))
    result = np.zeros(eligible.size, bool)
    assert 0 <= k <= len(indices)
    result[indices[order[:k]]] = True
    return result.reshape(eligible.shape)


def official(rows, arm, expected):
    totals = defaultdict(lambda: np.zeros(2, np.int64))
    for row in rows:
        totals[str(row["fold"]), str(row["class_id"])] += row["iu"][arm]
    folds = []
    for fold, classes in expected.items():
        values = []
        for cls in classes:
            i, u = totals[str(fold), str(cls)]
            values.append(100 * int(i) / int(u) if u else 0.)
        folds.append(sum(values) / len(values))
    return sum(folds) / len(folds)


def main():
    core.torch.set_num_threads(2)
    tick = time.monotonic()
    seal, cfg = core.read(OUT / "sealed.json"), core.read(OUT / "config.json")
    assert seal["state"] == "ALL120_REMAINING_PREDICTIONS_SEALED" and seal["n"] == 120
    assert seal["query_gt_access_attempts"] == []
    assert core.sha(OUT / "config.json") == seal["config_sha256"]
    assert core.sha(OUT / "manifest.json") == seal["manifest_sha256"]
    assert not (OUT / "report.json").exists(), "Do not repeat completed scoring"
    expected = core.read(OUT / "expected_classes.json")
    sources = core.read(OUT / "manifest.json")
    rows, checks, mismatches, ridge_mismatches = [], 0, 0, 0
    for source, rec in zip(sources, seal["records"]):
        meta = source["row"]
        eid, shape = meta["episode_id"], tuple(meta["query_size_hw"])
        assert eid == rec["episode_id"]
        assert meta["loader_class_id"] in expected[meta["dataset"]][str(meta["fold"])]
        packet = OUT / "predictions" / rec["file"]
        field = OUT / "fields" / rec["file"]
        assert core.sha(packet) == rec["prediction_sha256"]
        assert core.sha(field) == rec["field_sha256"]
        for k in ("baseline", "mean", "source_score"):
            assert core.sha(source[k]) == source[k + "_sha256"]
            checks += 1
        with np.load(source["baseline"], allow_pickle=False) as z:
            b = unpack(z["original/" + source["baseline_key"]], shape)
        with np.load(source["mean"], allow_pickle=False) as z:
            m = unpack(z[source["mean_key"]], shape)
        with np.load(source["source_score"], allow_pickle=False) as z:
            s = z[source["score_key"]].reshape(64,64).astype(float)
        with np.load(packet, allow_pickle=False) as z:
            masks = {k:unpack(z[k],shape) for k in ARMS}
        r, c, control = [masks[k] for k in ARMS]
        # Independent Boolean reconstruction, without producer majority helper.
        assert np.array_equal(c, np.where(r != m, b, r))
        span = float(np.ptp(s))
        norm = (s-s.min()) / max(span,1e-6)
        confidence = np.abs(2*bilinear(norm,shape)-1) if span else np.zeros(shape)
        g = (2*b.astype(float)-1)*confidence
        with np.load(field,allow_pickle=False) as z:
            h = z["evidence_global"]
        decision = g + .5*bilinear(h,shape)
        replay = (decision>0)|((decision==0)&b)
        different = int(np.count_nonzero(replay!=r))
        ridge_mismatches += different
        assert different == 0, (eid,"independent ridge readout",different)
        mutable = confidence<=.5
        add, delete = c&~b, b&~c
        assert int(add.sum()) == rec["added_pixels"] and int(delete.sum()) == rec["deleted_pixels"]
        independent_control=(b|select((~b)&(r|m)&mutable,bilinear(s,shape),int(add.sum()),True)) & ~select(b&((~r)|(~m))&mutable,bilinear(s,shape),int(delete.sum()),False)
        different=int(np.count_nonzero(independent_control!=control))
        mismatches+=different
        assert different==0,(eid,"source control",different)
        checks+=5
        with Image.open(meta["query_mask_path"]) as im:
            raw=(np.asarray(im.convert("L"))>0).astype(np.uint8)
        assert array_hash(raw)==meta["query_mask_hash"]
        truth=core.resize(raw,shape,"nearest")>.5
        yi=np.minimum(np.arange(shape[0])*raw.shape[0]//shape[0],raw.shape[0]-1)
        xi=np.minimum(np.arange(shape[1])*raw.shape[1]//shape[1],raw.shape[1]-1)
        gt=raw[np.ix_(yi,xi)].astype(bool)
        assert np.array_equal(gt,truth)
        masks.update({"foris.crf":b,"mean":m})
        metrics={a:stats(mask,gt) for a,mask in masks.items()}
        for a,mask in masks.items():
            table=np.bincount((gt.astype(np.uint8)+2*mask.astype(np.uint8)).ravel(),minlength=4)
            assert [int(table[3]),int(table[1:].sum()),int(table[2]),int(table[1])]==[metrics[a][k] for k in ("I","U","FP","FN")]
            checks+=1
        accounting={}
        for a in ARMS:
            accounting[a]={base:edits(masks[a],masks[base],gt) for base in masks}
            for base,e in accounting[a].items():
                at,af,dt,df=e
                assert metrics[a]["I"]==metrics[base]["I"]+at-dt
                assert metrics[a]["U"]==metrics[base]["U"]+af-df
                checks+=1
        rows.append(dict(episode_id=eid,dataset=meta["dataset"],fold=meta["fold"],class_id=meta["loader_class_id"],
                         query_photo_id=meta["query_photo_id"] if meta.get("query_photo_id") is not None else "rgb/"+meta["query_rgb_hash"],
                         reference_photo_id=meta["reference_photo_id"] if meta.get("reference_photo_id") is not None else "rgb/"+meta["reference_rgb_hash"],
                         metrics=metrics,iu={a:[v["I"],v["U"]] for a,v in metrics.items()},edits=accounting,
                         inference_record=rec))
    core.write(OUT/"scored_episodes.json",rows)
    domains={}
    for ds in ("lvis","pascal_part","suim"):
        sub=[r for r in rows if r["dataset"]==ds]; src=[s for s in sources if s["row"]["dataset"]==ds]
        bases=("foris.crf","mean","anchor.global","same_edit_budget_source_order")
        summary=summarize(sub,baselines=bases,expected_classes=expected[ds],repetitions=10000,unit="query_photo")
        for a,value in summary["miou"].items():
            assert abs(value-official(sub,a,expected[ds]))<1e-10
            checks+=1
        observed={str(f):sorted({r["class_id"] for r in sub if str(r["fold"])==str(f)}) for f in expected[ds]}
        domains[ds]=dict(summary=summary,official_slots=sum(len(x) for x in expected[ds].values()),
                        observed_slots=sum(len(x) for x in observed.values()),
                        observed_class_only_secondary={a:official(sub,a,observed) for a in summary["miou"]},
                        connected_RQ_photo_intervals=connected_intervals(sub,src,expected[ds],bases),
                        edit_balance={b:summarize_edits(sub,b,"shared_edits_complete",expected_classes=expected[ds]) for b in bases})
    result=dict(state="COMPLETE_SCORED_AND_INDEPENDENTLY_RECONSTRUCTED120",datasets=domains,checks=checks,
                seal_sha256=core.sha(OUT/"sealed.json"),scored_records_sha256=core.sha(OUT/"scored_episodes.json"),
                expected_classes_sha256=core.sha(OUT/"expected_classes.json"),scorer_sha256=core.sha(__file__),
                seconds=time.monotonic()-tick,ridge_readout_mismatch_pixels=ridge_mismatches,
                source_control_mismatch_pixels=mismatches,new_encoder_calls=0,new_predictions=0,
                exposure=cfg["exposure"],metric="Fixed exposed120 original-frame official class/fold pooling including all absent zeros; observed-class-only score is separately labelled.")
    core.write(OUT/"report.json",result)
    print({"state":result["state"],"checks":checks,"seconds":result["seconds"],"scores":{ds:v["summary"]["miou"] for ds,v in domains.items()}},flush=True)


if __name__=="__main__":main()
