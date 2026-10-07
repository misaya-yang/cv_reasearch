#!/usr/bin/env python3
"""Finite complete-output ablation plus a fixed signed-CSLS-guide candidate.

Reuses the historical FoRIS host and the exact CPU/FP16 RCG arithmetic from
run_rcg2_stream.py. Query annotations are opened only after all masks exist.
The candidate changes only the ranking guide from FG CSLS to FG minus BG CSLS.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.stats import rankdata


def rank(x):
    return ((rankdata(x.ravel(), method="average") - .5) / x.size).astype(np.float32)


def graph_fields(q, r, cov, score):
    """All inputs are inference-available; preserve rounded legacy token norms."""
    sim = q @ r.T
    dq = sim.topk(10, dim=1).values.mean(1)
    dr = sim.topk(10, dim=0).values.mean(0)
    fi = np.flatnonzero(cov.ravel() >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov.ravel() == cov.max())
    bi = np.flatnonzero(cov.ravel() <= .1)
    gf = ((2 * sim[:, fi] - dr[fi][None, :]).max(1).values - dq).numpy()
    gb = ((2 * sim[:, bi] - dr[bi][None, :]).max(1).values - dq).numpy() if len(bi) else None
    s = np.asarray(score, np.float32)
    s = ((s - s.min()) / max(float(s.max() - s.min()), 1e-6)).ravel()
    y = (s + .5 * (rank(gf) - rank(s))).astype(np.float64)
    ys = (s + .5 * (rank(gf - gb) - rank(s))).astype(np.float64) if gb is not None else y.copy()
    mu = F.normalize(r[fi].mean(0), dim=0)
    ym = (s + .25 * (rank((q @ mu).numpy()) - rank(s))).astype(np.float64)
    sim = q @ q.T
    sim.fill_diagonal_(-2)
    values, idx = sim.topk(20, dim=1)
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), idx.numpy().ravel())), shape=(4096, 4096))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    w = w / max(float(np.asarray(w.sum(1)).mean()), 1e-8)
    a = .1 + np.abs(2 * s - 1)
    a = (a / a.mean()).astype(np.float64)
    h = sparse.diags(a) + 16 * (sparse.diags(np.asarray(w.sum(1)).ravel()) - w)
    maps = {"pre": s, "rerank": y}
    solve_s = {}
    for name, unary in (("smooth", s.astype(np.float64)), ("rcg", y), ("signed", ys), ("mean", ym)):
        start = time.monotonic()
        field, status = cg(h, a * unary, x0=unary, rtol=1e-7, atol=1e-9, maxiter=600)
        if status:
            raise RuntimeError(f"CG failed: {name} {status}")
        maps[name] = field
        solve_s[name] = time.monotonic() - start
    return {k: v.astype(np.float32).reshape(64, 64) for k, v in maps.items()}, w, {
        "pure_fg": len(fi), "pure_bg": len(bi), "no_bg_fallback": gb is None, "solve_seconds": solve_s}


def image_id(im):
    return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def count(mask, truth):
    return [int((mask & truth).sum()), int((mask | truth).sum())]


def edits(base, new, truth):
    add, delete = new & ~base, base & ~new
    return [int((add & truth).sum()), int((add & ~truth).sum()),
            int((delete & truth).sum()), int((delete & ~truth).sum())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--packs", nargs="+", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    src = Path(__file__).read_bytes()
    (a.out / "runner.py").write_bytes(src)
    (a.out / "contract.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(src).hexdigest(), "packs": a.packs, "limit_per_pack": a.limit,
        "candidate": "rank(FG_CSLS - BG_CSLS), alpha=.5, pureFG>=.9, pureBG<=.1; no BG -> RCG guide",
        "fixed": "mutual20 graph, lambda16, confidence .1+abs(2s-1), CPU FP16-rounded features",
        "controls": ["native CRF", "pre", "rerank", "smooth", "rcg", "mean_a.25"],
        "render": "continuous64->bilinear1024->>.5->binary1024 bilinear original->>.5",
        "query_gt_in_inference": False, "early_stop_for_quality": False,
        "exposure": "same previously evaluated cross-dataset packs; development re-use",
    }, indent=2) + "\n")
    host_root = Path("/root/autodl-tmp/demo9_extent")
    sys.path[:0] = [str(host_root), str(host_root / "scripts"), str(a.root / "scripts")]
    from extent_experiment import build_host, run_foris
    from run_rcg2 import solve as legacy_solve
    host = None
    start = time.monotonic()
    all_n = 0
    with torch.inference_mode():
        for pack in a.packs:
            old = a.root / "outputs" / ("claude_rcg2_" + pack)
            rows = json.loads((old / "rows.json").read_text())[:a.limit]
            if pack.startswith("group"):
                # Only paths/configuration are used; rows remain the exact saved group.
                man = json.loads((a.root / "outputs/claude_official/batch0.json").read_text())
            else:
                man = json.loads((a.root / "outputs/claude_packs" / pack / "episodes.json").read_text())
            sealed = np.load(old / "counts.npz", allow_pickle=False)
            if host is None:
                host = build_host(SimpleNamespace(fixture=None, foris_root=None,
                                  demo4_root="/root/autodl-tmp/demo4"), man, "cuda")
                torch.set_num_threads(4)
            out = a.out / pack
            out.mkdir()
            (out / "manifest.json").write_text(json.dumps(rows) + "\n")
            allfields = []
            with (out / "episodes.jsonl").open("w") as stream:
                for n, row in enumerate(rows):
                    begin = time.monotonic()
                    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
                    sp = Image.open(data / row["support"]).convert("RGB")
                    qp = Image.open(data / row["query"]).convert("RGB")
                    gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
                    native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
                    torch.cuda.synchronize()
                    host_s = time.monotonic() - begin
                    deb = got["deb"][0]
                    q, r = (F.normalize(deb[i].flatten(1).T.float(), dim=1).half().float().cpu() for i in (-1, 0))
                    score = got["score"].float().cpu().numpy()
                    cov = F.interpolate(ref_mask[None, None].float(), score.shape, mode="area")[0, 0].cpu().numpy()
                    post_start = time.monotonic()
                    fields, w, info = graph_fields(q, r, cov, score)
                    post_s = time.monotonic() - post_start
                    if n < 5:
                        replica = legacy_solve(q, r, cov, score, 16., "cpu")
                        assert np.array_equal(replica, fields["rcg"]), np.max(np.abs(replica-fields["rcg"]))
                    maps = {k: torch.from_numpy(v)[None, None] for k, v in fields.items()}
                    masks = {k: F.interpolate(v, (1024, 1024), mode="bilinear", align_corners=False)[0, 0] > .5 for k, v in maps.items()}
                    masks["native"] = native.cpu()
                    pred_o = {k: F.interpolate(v[None, None].float(), (qp.height, qp.width), mode="bilinear", align_corners=False)[0, 0] > .5 for k, v in masks.items()}
                    # Evaluation only: every prediction above is already fixed.
                    truth = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy())
                    assert tuple(truth.shape) == (qp.height, qp.width)
                    iu = {k: count(v, truth) for k, v in pred_o.items()}
                    tc = F.interpolate(truth[None, None].float(), (64, 64), mode="area")[0, 0].numpy().ravel()
                    wr, wc = w.nonzero()
                    leakage = float(w.data[(tc[wr] > .5) != (tc[wc] > .5)].sum() / max(w.data.sum(), 1e-9))
                    rec = dict(row, pack=pack, photo_support=image_id(sp), photo_query=image_id(qp),
                        iu=iu, edits_vs_native={k: edits(pred_o["native"], v, truth) for k, v in pred_o.items()},
                        signed_edits_vs_rcg=edits(pred_o["rcg"], pred_o["signed"], truth),
                        graph_cross_gt_weight=leakage, truth_area=float(truth.float().mean()), info=info,
                        host_seconds=host_s, shared_post_seconds=post_s,
                        old_iu={k: sealed["orig:" + k][n].tolist() for k in ("native", "rcg")},
                        sealed_fp16_field_maxdiff=float(np.max(np.abs(fields["rcg"]-sealed["fields"][n]))),
                        seconds=time.monotonic()-begin)
                    stream.write(json.dumps(rec) + "\n")
                    stream.flush()
                    allfields.append(np.stack([fields[k] for k in ("pre", "rerank", "smooth", "rcg", "signed", "mean")]))
                    all_n += 1
                    if n < 5 or (n + 1) % 25 == 0:
                        print(json.dumps(dict(pack=pack, n=n+1, total=len(rows), all_n=all_n,
                            elapsed_s=time.monotonic()-start, episode_s=rec["seconds"],
                            native_iu_match=iu["native"] == rec["old_iu"]["native"],
                            rcg_iu_match=iu["rcg"] == rec["old_iu"]["rcg"],
                            field_diff=rec["sealed_fp16_field_maxdiff"])), flush=True)
            np.savez_compressed(out / "fields.npz", names=np.array(["pre", "rerank", "smooth", "rcg", "signed", "mean"]), fields=np.stack(allfields))
            sealed.close()
    (a.out / "receipt.json").write_text(json.dumps({"state": "COMPLETE", "n": all_n,
        "wall_seconds": time.monotonic()-start, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}, indent=2)+"\n")


if __name__ == "__main__":
    main()
