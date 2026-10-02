#!/usr/bin/env python3
"""CPU, seconds: where the unrealised gain sits. Reads existing result files only.

  python scripts/analyze_gap.py            (from the direction directory)

1. Four folds, old cache (results/episodes_eval_f*.json): per class, headroom (true - 1shot) = realised (ours - 1shot) + gap.
2. Fold 0, exact protocol (results/probe_decoder_f0_400.json): the gap by the query's one-shot IoU and by the quality of the
   pool's pseudo masks; (results/probe_seed_f0_200.json) patch precision / recall of the query mask per class.
"""
import collections, json
import numpy as np

C = ("person bicycle car motorcycle airplane bus train truck boat trafficlight firehydrant stopsign parkingmeter bench bird cat dog horse "
     "sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sportsball kite baseballbat "
     "baseballglove skateboard surfboard tennisracket bottle wineglass cup fork knife spoon bowl banana apple sandwich orange broccoli "
     "carrot hotdog pizza donut cake chair couch pottedplant bed diningtable toilet tv laptop mouse remote keyboard cellphone microwave "
     "oven toaster sink refrigerator book clock vase scissors teddybear hairdrier toothbrush").split()

rows = []
for f in range(4):
    pc = json.load(open(f"results/episodes_eval_f{f}.json"))["per_class"]
    rows += [(int(c), pc["1shot"][c], pc["agree round4"][c], pc["true"][c]) for c in pc["1shot"]]
rows = np.array(rows); c = rows[:, 0].astype(int); a, o, t = rows[:, 1], rows[:, 2], rows[:, 3]; gain, head, gap = o - a, t - a, t - o
print(f"80 classes, old cache: 1shot {a.mean():.1f}  ours {o.mean():.1f}  true {t.mean():.1f}   headroom {head.mean():.1f} = realised {gain.mean():.1f} + gap {gap.mean():.1f}")
print(f"classes where ours < 1shot: {int((gain < 0).sum())}, costing {-gain[gain < 0].sum() / 80:.2f} mIoU;  top 10 classes hold {np.sort(gap)[::-1][:10].sum() / gap.sum():.0%} of the gap, top 20 hold {np.sort(gap)[::-1][:20].sum() / gap.sum():.0%}")
print("largest gaps (1shot / ours / true):", ", ".join(f"{C[c[i]]} {a[i]:.0f}/{o[i]:.0f}/{t[i]:.0f}" for i in np.argsort(-gap)[:12]))
print("largest harms (1shot / ours):", ", ".join(f"{C[c[i]]} {a[i]:.0f}/{o[i]:.0f}" for i in np.argsort(gain)[:8]))

R = json.load(open("results/probe_decoder_f0_400.json"))["records"]; iou = lambda r, k: r["iu"][k][0] / max(r["iu"][k][1], 1)
b1, bo, bt = (np.array([iou(r, k) for r in R]) for k in ("1shot", "ours", "true")); g = bt - bo
print("\nfold 0, exact protocol, by the query's one-shot IoU:   n | 1shot / ours / true | share of the gap")
for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
    m = (b1 >= lo) & (b1 < hi); print(f"  [{lo:.1f},{hi:.1f}) n={m.sum():3d} | {b1[m].mean():.2f} / {bo[m].mean():.2f} / {bt[m].mean():.2f} | {g[m].sum() / g.sum():.2f}")
ids = np.array([r["id"] for r in R if "id" in r]); gg = np.array([iou(r, "true") - iou(r, "ours") for r in R if "id" in r])
for name, col in (("precision", 0), ("recall", 1)):
    q = np.quantile(ids[:, col], [1 / 3, 2 / 3]); parts = [ids[:, col] < q[0], (ids[:, col] >= q[0]) & (ids[:, col] < q[1]), ids[:, col] >= q[1]]
    print(f"  mean per-episode gap by tercile of the pool's pseudo-mask {name} (low, mid, high): " + ", ".join(f"{gg[m].mean():+.3f}" for m in parts))

S = collections.defaultdict(list)
for r in json.load(open("results/probe_seed_f0_200.json"))["records"]: S[r["c"]].append(r)
print("\nfold 0, first 200 episodes, patch precision/recall of the query mask:  1shot | pseudo pool | true pool")
for cc, v in sorted(S.items()):
    pr = lambda k: (sum(r["pr"][k][0] for r in v) / max(sum(r["pr"][k][1] for r in v), 1), sum(r["pr"][k][0] for r in v) / max(sum(r["pr"][k][2] for r in v), 1))
    x, y, z = pr("1shot"), pr("p1:insid3"), pr("true:insid3"); print(f"  {C[cc]:13s} n={len(v):2d}  {x[0]:.2f}/{x[1]:.2f} | {y[0]:.2f}/{y[1]:.2f} | {z[0]:.2f}/{z[1]:.2f}")
