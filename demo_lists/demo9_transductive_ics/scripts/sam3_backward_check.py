#!/usr/bin/env python3
"""Backward check of SAM3's query proposals: does a proposal, used as the exemplar, find the reference object?

  GPU   python scripts/sam3_backward_check.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT \
            --out results/sam3_relative_v1/backward_dev.jsonl [--limit 10]
  CPU   python scripts/sam3_backward_check.py --report --run RUN/dev --out results/sam3_relative_v1/backward_dev.jsonl
  CPU   python scripts/sam3_backward_check.py --fixture /tmp/backward_fixture

Why. After the per-episode level, the largest error left is a kept proposal that is not the target (DEV241, approximate
reading: 57 of 241 episodes). SAM3's forward score separates on-target from off-target proposals with AUC 0.716 only.
"Same concept" is symmetric: if a query region is the same concept as the reference object, then prompting with that
region must return the reference object, and the reference mask is known. So the check has a known answer and needs no
query label.

What it does. RUN is a `sam3_stitch.py` stage whose proposal bitmaps were kept. For each episode the canvas is stitched
again, and each of the K most confident query proposals is used, alone, as the positive exemplar box (the same call as
the forward pass, only the box differs). Read inside the reference rectangle, per proposal:
  union_iou   IoU of the kept union (score > 0.5) with the reference mask
  best_iou    the best IoU of any of the 20 most confident masks with the reference mask, and that mask's score
  top_iou     the IoU of the most confident mask that touches the reference rectangle, and its score
The GPU stage reads no query annotation. --report (CPU) joins the rows with the scored records and prints how well each
signal separates on-target from off-target proposals (AUC), next to the forward score on the same proposals; it writes
no selection rule: rules are chosen afterwards on DEV241 from this table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
FIELDS = ("index", "forward_score", "union_iou", "best_iou", "best_score", "top_iou", "top_score", "presence", "kept")


def records(run, name, shards):
    paths = sorted(Path(run).glob(name + "_shard*.jsonl"))
    if shards is not None:
        paths = [p for p in paths if int(p.stem.split("shard")[1]) in shards]
    return [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]


def check_episode(run, S, canvas, raw, meta, ref_mask, k):
    """Rows of FIELDS for the k most confident query proposals. raw: [n, h, w] bool bitmaps in the query rectangle."""
    import numpy as np
    import torch
    s, t = S.rectangles()
    order = [i for i in sorted(range(len(meta)), key=lambda i: -meta[i][0]) if meta[i][1] > 0][:k]
    rows = []
    for i in order:
        ys, xs = np.nonzero(raw[i])
        if not len(xs):
            continue
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        box = [(t[0] + (x0 + x1) / 2) / S.CANVAS, (t[1] + (y0 + y1) / 2) / S.CANVAS, (x1 - x0) / S.CANVAS, (y1 - y0) / S.CANVAS]
        union, prob, back, kept, presence = run(canvas, box, None)
        ts = torch.from_numpy(S.on_canvas(ref_mask, s)).to(back.device)
        rs = back[:, s[1]:s[1] + s[3], s[0]:s[0] + s[2]]
        area, inter = rs.flatten(1).sum(1), (rs & ts).flatten(1).sum(1)
        iou = inter / (area + ts.sum() - inter).clamp(min=1)
        u = union[s[1]:s[1] + s[3], s[0]:s[0] + s[2]]
        best = int(iou.argmax())
        touching = [j for j in range(len(prob)) if int(area[j]) > 0]  # prob is sorted, most confident first
        top = touching[0] if touching else None
        rows.append([i, float(meta[i][0]), float((u & ts).sum() / (u | ts).sum().clamp(min=1)), float(iou[best]), float(prob[best]),
                     float(iou[top]) if top is not None else 0.0, float(prob[top]) if top is not None else 0.0, float(presence), int(kept)])
    return rows


def work(a):
    import numpy as np
    import torch
    from PIL import Image
    import sam3_stitch as S
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    recs = records(a.run, "predictions", a.shards)[:a.limit]
    if not recs:
        raise SystemExit("no prediction records under %s" % a.run)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {tuple(json.loads(line)["key"]) for line in out.read_text().splitlines() if line} if out.exists() else set()
    report = out.with_suffix(".report.json")
    report.write_text(json.dumps(dict(state="RUNNING", expected=len(recs), done=len(done))))
    sys.path.insert(0, str(a.sam3))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model = build_sam3_image_model(bpe_path=str(Path(a.sam3) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), device="cuda",
                                   checkpoint_path=str(a.checkpoint), load_from_HF=False,
                                   eval_mode=True, enable_inst_interactivity=False, compile=False).float().eval()
    S.configure_fp32_mlp_backend(model)
    run = S.Runner(Sam3Processor(model, device="cuda", confidence_threshold=.5))
    start, passes = time.monotonic(), 0
    with open(out, "a") as stream, torch.inference_mode():
        for n, r in enumerate(recs):
            key = (r["fold"], r["e"], r["c"])
            if key in done:
                continue
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            ref_mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            canvas, _ = S.stitch(ref, query, r["exemplar_box"])
            count, h, w = r["proposal_shape"]
            with np.load(Path(a.run) / r["candidate_file"], allow_pickle=False) as z:
                raw = np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(count, h, w).astype(bool)
            rows = check_episode(run, S, canvas, raw, r["proposal_metadata"], ref_mask, a.k)
            passes += len(rows)
            stream.write(json.dumps(dict(key=key, rows=rows)) + "\n")
            stream.flush()
            if n % 20 == 0:
                print("%d/%d episodes, %d backward passes, %.2f s per pass" % (n + 1, len(recs), passes, (time.monotonic() - start) / max(passes, 1)), flush=True)
    elapsed = time.monotonic() - start
    report.write_text(json.dumps(dict(state="COMPLETED", episodes=len(recs), backward_passes_this_run=passes, elapsed_s=elapsed,
                                      seconds_per_pass=elapsed / max(passes, 1), k=a.k, fields=FIELDS, query_annotation_opened=False), indent=1))
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), passes=passes, elapsed_s=round(elapsed, 1))))


def auc(pos, neg):
    """Probability that a positive scores above a negative (ties count half)."""
    import numpy as np
    if not len(pos) or not len(neg):
        return None
    v = np.concatenate([pos, neg])
    order = v.argsort(kind="stable")
    rank = np.empty(len(v))
    rank[order] = np.arange(1, len(v) + 1)
    for value in np.unique(v):  # average ranks of ties
        m = v == value
        rank[m] = rank[m].mean()
    return float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report(a):
    import numpy as np
    scored = {(r["fold"], r["e"], r["c"]): r for r in records(a.run, "episodes", a.shards)}
    rows = [json.loads(line) for line in Path(a.out).read_text().splitlines() if line]
    table = {"all checked proposals": [], "proposals kept by relative_0.7": []}
    episodes = 0
    for item in rows:
        r = scored.get(tuple(item["key"]))
        if r is None:
            continue
        episodes += 1
        q = [p[0] for p in r["proposals"] if p[1] > 0]
        for row in item["rows"]:
            p = r["proposals"][row[0]]
            entry = (p[2] > .5 * p[1], row)
            table["all checked proposals"].append(entry)
            if p[0] >= .7 * max(q):
                table["proposals kept by relative_0.7"].append(entry)
    out = dict(state="COMPLETED", episodes=episodes, scope="AUC of each signal for on-target against off-target proposals; no rule chosen here")
    for name, entries in table.items():
        on, off = [e[1] for e in entries if e[0]], [e[1] for e in entries if not e[0]]
        out[name] = dict(on_target=len(on), off_target=len(off),
                         auc={f: auc(np.array([x[j] for x in on]), np.array([x[j] for x in off])) for j, f in enumerate(FIELDS) if f not in ("index", "kept")})
        print("%s: %d on target, %d off target" % (name, len(on), len(off)))
        for f, v in out[name]["auc"].items():
            print("  AUC %-14s %s" % (f, "%.3f" % v if v is not None else "-"))
    Path(a.out).with_suffix(".auc.json").write_text(json.dumps(out, indent=1))


def fixture(a):
    """The exact loop on the CPU with a model that answers: exemplars from the left half of the query find the reference."""
    import numpy as np
    import torch
    import sam3_stitch as S
    from PIL import Image
    s, t = S.rectangles()
    ref_mask = np.zeros((90, 120), bool); ref_mask[20:60, 30:80] = True
    ts = torch.from_numpy(S.on_canvas(ref_mask, s))
    calls = []

    def run(canvas, box, text=None):
        calls.append(box)
        good = box[0] * S.CANVAS < t[0] + t[2] / 2
        raw = torch.zeros((20, S.CANVAS, S.CANVAS), dtype=torch.bool)
        if good:
            raw[0, s[1]:s[1] + s[3], s[0]:s[0] + s[2]] = ts
        else:
            raw[0, s[1]:s[1] + 5, s[0]:s[0] + 5] = True
        return raw[0].clone(), [0.9 if good else 0.6] + [0.01] * 19, raw, 1, 0.99
    raw = np.zeros((3, t[3], t[2]), bool)
    raw[0, 10:60, 20:200] = True; raw[1, 10:60, t[2] - 220:t[2] - 20] = True  # proposal 0 on the left, 1 on the right, 2 empty
    meta = [[0.4, int(raw[0].sum()), 0, 0], [0.8, int(raw[1].sum()), 0, 0], [0.9, 0, 0, 0]]
    rows = check_episode(run, S, Image.new("RGB", (S.CANVAS, S.CANVAS)), raw, meta, ref_mask, 5)
    root = Path(a.fixture); (root / "run").mkdir(parents=True, exist_ok=True)
    (root / "back.jsonl").write_text(json.dumps(dict(key=[0, 1, 0], rows=rows)) + "\n")
    props = [[0.4, meta[0][1], meta[0][1], 0, 0], [0.8, meta[1][1], 0, 0, 0], [0.9, 0, 0, 0, 0]]  # proposal 0 is on the target
    (root / "run/episodes_shard0.jsonl").write_text(json.dumps(dict(fold=0, e=1, c=0, proposals=props)) + "\n")
    a.run, a.out, a.shards = str(root / "run"), str(root / "back.jsonl"), None
    report(a)
    got = json.loads((root / "back.auc.json").read_text())["all checked proposals"]
    checks = [("two query proposals checked, the most confident first", [r[0] for r in rows] == [1, 0]),
              ("the exemplar box lies inside the query rectangle", all(t[1] / S.CANVAS <= b[1] <= (t[1] + t[3]) / S.CANVAS for b in calls)),
              ("the left proposal finds the reference", rows[1][3] > .99 and rows[1][2] > .99 and rows[1][5] > .99),
              ("the right proposal does not", rows[0][3] < .05 and rows[0][5] < .05),
              ("the report separates them where the forward score does not", got["auc"]["best_iou"] == 1.0 and got["auc"]["forward_score"] == 0.0)]
    bad = [n for n, ok in checks if not ok]
    print("%d/%d checks pass%s" % (len(checks) - len(bad), len(checks), "" if not bad else "; FAILED: " + "; ".join(bad)))
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="output folder of a sam3_stitch stage with its candidates kept")
    p.add_argument("--manifest")
    p.add_argument("--sam3", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out", help="the jsonl of backward rows")
    p.add_argument("--shards", type=lambda v: [int(x) for x in v.split(",")], help="shard indices of the run to read (default: all)")
    p.add_argument("--limit", type=int)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--report", action="store_true")
    p.add_argument("--fixture")
    a = p.parse_args()
    fixture(a) if a.fixture else report(a) if a.report else work(a)


if __name__ == "__main__":
    main()
