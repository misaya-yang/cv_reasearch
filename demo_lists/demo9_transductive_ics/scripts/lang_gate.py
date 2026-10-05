#!/usr/bin/env python3
"""Does the language channel carry what FoRIS lacks? Read on a CPU from the bank (lang_bank.py) and FoRIS's saved
packets; nothing is encoded here. The decision rule is written below before any number exists.

  python scripts/lang_gate.py --bank B/dev --packets P/packets --manifest M/dev_episodes.json --out R/gate_dev.json
  python scripts/lang_gate.py --fixture positive --bank /tmp/fx --packets P/packets --manifest M/dev_episodes.json \
      --out /tmp/fx.json --limit 60      # code check: synthetic tokens built from the labels must PASS; `negative` must FAIL

Level: FoRIS's own cut on the model-size image, before its CRF (the packet's `pre`); class mIoU; paired interval over
photograph groups. Arms: 3 token views x 8 ways to carry the reference's concept x 4 weights, all label-free, and the
true class name as the ceiling of the channel. The reported label-free gain is nested: for each fold the arm is chosen
on the other three folds, so choosing among 96 arms cannot inflate it.

Verdict (fixed in advance):
  PASS         nested label-free gain >= +2.0 and its interval above 0  -> the arm chosen on all of DEV is frozen
  NAMING       the true name gives >= +3.0 (interval above 0) but label-free does not pass -> the words are the weak link
  FAIL         the true name gives < +3.0 -> this text head does not carry the category on FoRIS's errors
"""
import argparse
import json
import time
from pathlib import Path

from lang_common import (LAMBDAS, MODES, NAMINGS, ORACLES, binarise, class_miou, contested_auc, evidence, fuse, iu, naming, paired,
                         photo_groups, to_grid)


def make_fixture(kind, rows, packets, out, dim=64, words=40):
    """A bank whose tokens are built from the labels (positive) or from nothing (negative)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    e = F.normalize(torch.randn(words, dim, generator=gen), dim=1)
    np.savez(out / "text.npz", names=np.array(["w%d" % i for i in range(words)]), e_patch=e.numpy(),
             e_full=F.normalize(torch.cat([e, e], 1), dim=1).numpy(), coco_patch=e[torch.arange(80) % words].numpy(),
             coco_full=F.normalize(torch.cat([e, e], 1), dim=1)[torch.arange(80) % words].numpy(), scale=30.0)
    for r in rows:
        z = np.load(packets / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])))
        truth = torch.from_numpy(np.unpackbits(z["truth"])[:1024 * 1024].reshape(1024, 1024).astype(np.float32))
        tq, tr, k = F.adaptive_avg_pool2d(truth[None, None], 64)[0, 0] >= .5, torch.from_numpy(z["cov"]) >= .5, r["c"] % words

        def tokens(target):
            other = torch.randint(0, words - 1, (8, 8), generator=gen)
            other = (other + (other >= k)).repeat_interleave(8, 0).repeat_interleave(8, 1)  # blocks of other words
            word = torch.where(target, torch.full_like(other, k), other) if kind == "positive" else other
            return e[word.flatten()] + 0.08 * torch.randn(4096, dim, generator=gen)
        q, rr = tokens(tq), tokens(tr)
        grid = lambda t: F.normalize(F.avg_pool2d(t.view(64, 64, dim).permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0), dim=-1)
        ys, xs = torch.nonzero(tr, as_tuple=True)
        if len(ys) == 0:
            ys, xs = torch.tensor([0, 63]), torch.tensor([0, 63])
        box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        c_tok = rr.view(64, 64, dim)[box[0]:box[1], box[2]:box[3]]
        c_cov = torch.from_numpy(z["cov"])[box[0]:box[1], box[2]:box[3]]
        half = lambda t: t.half().numpy()
        np.savez(out / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])), q_w=half(q), r_w=half(rr), r_cov_w=z["cov"].astype(np.float16),
                 q_s=half(grid(q)), r_s=half(grid(rr)), q_b=half(grid(q)), r_b=half(grid(rr)), r_cov_s=half(F.avg_pool2d(torch.from_numpy(z["cov"])[None, None], 2)[0, 0]),
                 c_cls=half((c_cov[..., None] * c_tok).sum((0, 1)) / c_cov.sum().clamp_min(1e-6)), c_tok=half(c_tok), c_cov=half(c_cov))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True); p.add_argument("--packets", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True); p.add_argument("--limit", type=int); p.add_argument("--device", default="cpu")
    p.add_argument("--fixture", choices=("positive", "negative")); p.add_argument("--draws", type=int, default=2000)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    dev, bank, packets = a.device, Path(a.bank), Path(a.packets)
    rows = json.loads(Path(a.manifest).read_text())["episodes"][:a.limit]
    if a.fixture:
        make_fixture(a.fixture, rows, packets, bank)
    text = np.load(bank / "text.npz")
    names = list(text["names"])
    e_patch, e_full, coco_patch, coco_full = (torch.from_numpy(text[k]).float().to(dev) for k in ("e_patch", "e_full", "coco_patch", "coco_full"))
    view = dict(W="w", S="s", B="b")
    scale = float(text["scale"])
    arms = [(m, k, lam) for m in MODES for k in NAMINGS for lam in LAMBDAS]
    oracle = [(m, k, lam) for m in MODES for k in ORACLES for lam in LAMBDAS]
    alone = [(m, k, None) for m in MODES for k in NAMINGS + ORACLES]  # the channel without FoRIS: its own quality
    I_U = {arm: [] for arm in arms + oracle + alone}
    base, auc, named, mismatch, began = [], {}, [], 0, time.monotonic()
    tensor = lambda v: torch.from_numpy(np.asarray(v, np.float32)).to(dev)
    with torch.inference_mode():
        for n, r in enumerate(rows):
            key = "%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
            z, b = np.load(packets / key), np.load(bank / key)
            bits = lambda k: torch.from_numpy(np.unpackbits(z[k])[:1024 * 1024].reshape(1024, 1024).astype(bool)).to(dev)
            truth, pre, score = bits("truth"), bits("pre"), tensor(z["score"])  # the query label, for the reading only
            replica = binarise(score, (1024, 1024))
            mismatch += int((replica != pre).sum())
            base.append(iu(replica, truth))
            target = F.adaptive_avg_pool2d(truth[None, None].float(), 64)[0, 0] >= .5
            native = F.adaptive_avg_pool2d(replica[None, None].float(), 64)[0, 0] >= .5
            auc.setdefault("foris", []).append(contested_auc(score, target, native))
            crop = dict(c_cls=tensor(b["c_cls"]), c_tok=tensor(b["c_tok"]).flatten(0, 1), c_cov=tensor(b["c_cov"]).flatten())
            for m in MODES:
                q = tensor(b["q_" + view[m]])
                hw = (int(round(len(q) ** .5)),) * 2 if m == "W" else q.shape[:2]
                q = F.normalize(q.reshape(-1, q.shape[-1]), dim=-1)
                rt = tensor(b["r_" + view[m]])
                ref = dict(tok=F.normalize(rt.reshape(-1, rt.shape[-1]), dim=-1), cov=tensor(b["r_cov_w"] if m == "W" else b["r_cov_s"]).flatten(), **crop)
                p_q = (scale * q @ e_patch.T).softmax(1)
                for k in NAMINGS + ORACLES:
                    ev = to_grid(evidence(k, q, ref, e_patch, e_full, scale, true_row=coco_patch[r["c"] % len(coco_patch)], p_q=p_q), hw)
                    auc.setdefault(m + ":" + k, []).append(contested_auc(ev, target, native))
                    I_U[(m, k, None)].append(iu(binarise(ev, (1024, 1024)), truth))
                    for lam in LAMBDAS:
                        fused = fuse(score, ev, lam)
                        I_U[(m, k, lam)].append(iu(binarise(fused, (1024, 1024)), truth))
                        if lam == 1.0:
                            auc.setdefault(m + ":" + k + "+foris", []).append(contested_auc(fused, target, native))
                if m == "W":  # where the true name stands among the words, by the two posteriors
                    rec = dict(fold=r["fold"], e=r["e"], c=r["c"])
                    for k in ("crop_bma", "pool_bma"):
                        w = naming(k, ref, e_patch, e_full, scale)
                        rec[k] = [str(names[i]) for i in w.topk(3).indices.tolist()]
                    g = torch.cat([ref["c_cls"], (ref["c_cov"][:, None] * ref["c_tok"]).sum(0) / ref["c_cov"].sum().clamp_min(1e-6)])
                    both = F.normalize(g, dim=0) @ torch.cat([e_full, coco_full[r["c"] % len(coco_full)][None]]).T
                    rec["true_rank"] = int((both[:-1] > both[-1]).sum()) + 1
                    named.append(rec)
            if n % 20 == 0:
                print("%d/%d, %.1f s" % (n + 1, len(rows), time.monotonic() - began), flush=True)
    cls, fold = np.array([r["c"] for r in rows]), np.array([r["fold"] for r in rows])
    groups, base = photo_groups(rows), np.array(base, float)
    I_U = {k: np.array(v, float) for k, v in I_U.items()}
    label = lambda arm: "%s:%s:%s" % arm

    def nested(pool):
        """Per fold, the arm that is best on the other folds; the stitched result is free of the choice."""
        got, picks = np.zeros_like(base), {}
        for f in np.unique(fold):
            rest = fold != f
            best = max(pool, key=lambda arm: class_miou(I_U[arm][rest], cls[rest])) if rest.any() else pool[0]
            got[fold == f], picks[int(f)] = I_U[best][fold == f], label(best)
        return dict(picks=picks, **paired(got, base, cls, groups, a.draws))
    free, top = nested(arms), nested(oracle)
    frozen = max(arms, key=lambda arm: class_miou(I_U[arm], cls))
    above = lambda res, bar: res["gain"] >= bar and res["ci95"][0] > 0
    verdict = "PASS" if above(free, 2.0) else "NAMING" if above(top, 3.0) else "FAIL"
    mean = lambda v: None if not [x for x in v if x is not None] else float(np.mean([x for x in v if x is not None]))
    table = {label(arm): round(class_miou(v, cls) - class_miou(base, cls), 2) for arm, v in I_U.items()}
    ranks = np.array([x["true_rank"] for x in named])
    result = dict(
        state="COMPLETED", verdict=verdict, episodes=len(rows), level="FoRIS cut at model size, before the CRF", fixture=a.fixture,
        foris=class_miou(base, cls), replica_mismatch_pixels=mismatch, label_free_nested=free, true_name_nested=top,
        frozen=dict(mode=frozen[0], naming=frozen[1], lam=frozen[2], in_sample=paired(I_U[frozen], base, cls, groups, a.draws)),
        gain_by_arm=table, contested_auc={k: mean(v) for k, v in auc.items()},
        true_name_rank=dict(first=float((ranks == 1).mean()), top5=float((ranks <= 5).mean()), median=float(np.median(ranks))),
        vocabulary=len(names), scale=scale, seconds=round(time.monotonic() - began, 1))
    Path(a.out).write_text(json.dumps(result, indent=1))
    Path(a.out).with_suffix(".names.jsonl").write_text("".join(json.dumps(x) + "\n" for x in named))
    print(json.dumps({k: result[k] for k in ("state", "verdict", "episodes", "foris", "replica_mismatch_pixels", "label_free_nested",
                                             "true_name_nested", "frozen", "true_name_rank")}, indent=1))
    best = sorted(table.items(), key=lambda kv: -kv[1])
    print("best arms:", best[:8]); print("contested AUC:", {k: None if v is None else round(v, 3) for k, v in result["contested_auc"].items()})


if __name__ == "__main__":
    main()
