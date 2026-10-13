"""W06: do last-block internals of the same forward pass carry extent (grouping) evidence
that the output tokens lack?

Motivation (W04): complete FoRIS and the independent candidate rank equally well and fail on
the same episodes; what is left is where the target ends inside the query, and every extent
rule tested so far read that from output tokens (O24).  Here the same forward pass is read
differently: the last block's keys / queries / values and its native attention.

Data: the 119 last-block inputs (X23) cached on 2026-10-10 for 60 dev pairs (COCO, PACO, SUIM).
Each image with an object mask is one sample (reference image + reference mask, query image +
query GT).  No encoder call.  The test is estimator-independent and seeded from the mask itself
(diagnostic): 8 tokens inside the mask, field = affinity of every token to the seeds.

Pre-stated: P1 the best last-block source beats O24 raw cosine by >= 5 points of GT-optimal-level
IoU in each domain; P2 its natural (min-max > .5) to optimal-level gap is at least one third
smaller.  If both fail, same-forward internals are closed as an extent source.

    python scripts/x1012/w06_grouping_source.py OUT_DIR
"""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(6)
import w02_scale_eval as w
from timm.layers import apply_rot_embed_cat

A = lab.ASSETS / "a/task_attention_relation60_20261010"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
par = np.load(A / "last_qk_parameters.npz")
rope = torch.from_numpy(np.load(A / "rope64.npy")).float()
Wn, Bn, eps = torch.from_numpy(par["norm1_weight"]), torch.from_numpy(par["norm1_bias"]), float(par["norm1_eps"])
Wqkv = torch.from_numpy(par["qkv_weight"])
assert par["qkv_bias"].size == 0 and par["q_norm_weight"].size == 0
H, P5, K_SEED = 16, 5, 8


def internals(x23):
    xn = F.layer_norm(x23, (1024,), Wn, Bn, eps)
    q, k, v = (xn @ Wqkv.T).reshape(-1, 3, H, 64).permute(1, 2, 0, 3)          # each (H, 4101, 64)
    flat = lambda t: t[:, P5:].permute(1, 0, 2).reshape(4096, -1)
    rot = lambda t: torch.cat([t[:, :P5], apply_rot_embed_cat(t[:, P5:], rope)], 1)
    return xn[P5:], flat(q), flat(k), flat(v), (q, k), (rot(q), rot(k))


def attention_fields(qk, seeds):
    """Mean over heads of native attention: seeds -> token (fwd) and token -> seed set (bwd), log domain."""
    q, k = qk
    fwd, bwd = torch.zeros(4096), torch.zeros(4096)
    for h in range(H):
        a = torch.softmax((q[h] * 0.125) @ k[h].T, dim=-1)[P5:, P5:]              # (4096, 4096), patch part
        fwd += a[seeds].mean(0)
        bwd += a[:, seeds].sum(1)
    return torch.log(fwd / H + 1e-9), torch.log(bwd / H + 1e-9)


def kmeans_labels(x, K, iters=12):
    c = [int(x.norm(dim=1).argmax())]
    d = 1 - x @ x[c[0]]
    for _ in range(K - 1):                                                      # farthest-point init
        c.append(int(d.argmax())); d = torch.minimum(d, 1 - x @ x[c[-1]])
    C = x[c]
    for _ in range(iters):
        lab_ = (x @ C.T).argmax(1)
        C = lab.unit(torch.stack([x[lab_ == j].mean(0) if (lab_ == j).any() else C[j] for j in range(K)]))
    return (x @ C.T).argmax(1)


def union_oracle(labels, m):
    L = F.interpolate(labels.reshape(1, 1, 64, 64).float(), size=(1024, 1024), mode="nearest")[0, 0].long()
    K = int(labels.max()) + 1
    tot = torch.bincount(L.flatten(), minlength=K).float()
    hit = torch.bincount(L.flatten(), weights=m.flatten().float(), minlength=K)
    order = torch.argsort(hit / tot.clamp_min(1), descending=True)
    I, P = torch.cumsum(hit[order], 0), torch.cumsum(tot[order], 0)
    return float((I / (P + m.sum() - I)).max() * 100)


rows = {(b["dataset"], b["row"]["episode_id"]): b for b in w.rows_all(0)}
pairs = [p for p in json.loads((A / "pairs.json").read_text()) if (p["dataset"], p["row"]["episode_id"]) in rows]
print("pairs matched", len(pairs), flush=True)
recs = []
for pi, p in enumerate(pairs):
    b = rows[(p["dataset"], p["row"]["episode_id"])]
    for role, mpath in (("reference", b["row"]["reference_mask_path"]), ("query", b["row"]["query_mask_path"])):
        m = lab.mask1024(mpath)
        y = lab.coverage64(m).flatten()
        if float(m.float().mean()) > 0.6 or int((y > 0.5).sum()) < 4:
            continue
        pure = torch.where(y >= 0.9)[0]
        if len(pure) < K_SEED:
            pure = torch.sort(torch.topk(y, K_SEED).indices).values
        seeds = pure[torch.linspace(0, len(pure) - 1, K_SEED).long()]
        o24 = lab.unit(lab.feats(b, role))
        xn, fq, fk, fv, pre, nat = internals(torch.from_numpy(np.load(A / "x23" / f"{p['raw_binding'][role]['key']}.npy")))
        feats = {"O24 raw": o24, "O24 debiased": lab.unit(lab.debias(o24)), "X23": lab.unit(xn), "key": lab.unit(fk), "query": lab.unit(fq), "value": lab.unit(fv)}
        fields = {k: (x @ x[seeds].T).mean(1) for k, x in feats.items()}
        f_, b_ = attention_fields(nat, seeds)
        fields.update({"attn seed->tok": f_, "attn tok->seed": b_, "attn both": f_ + b_})
        f_, b_ = attention_fields(pre, seeds)
        fields.update({"attn both (no RoPE)": f_ + b_})
        r = {"d": p["dataset"], "role": role, "area": float(m.float().mean())}
        for k, s in fields.items():
            u = lab.up(s.reshape(64, 64))
            io, uo, t = lab.best_threshold_iou(u, m)
            r[k] = {"auc": float(lab.auroc(u, m, stride=5)), "orc": 100 * float(io) / max(float(uo), 1),
                    "nat": 100 * (lambda iu: iu[0] / max(iu[1], 1))([float(v) for v in lab.iu(lab.minmax(u) > 0.5, m)]),
                    "t": float((t - float(u.min())) / max(float(u.max() - u.min()), 1e-9))}
        for k in ("O24 raw", "O24 debiased", "key", "value", "X23"):
            for K in (8, 16):
                r[k][f"u{K}"] = union_oracle(kmeans_labels(feats[k], K), m)
        recs.append(r)
    if pi % 10 == 0:
        print(pi, len(pairs), len(recs), flush=True)
(OUT / "records.json").write_text(json.dumps(recs))
SRC = [k for k in recs[0] if k not in ("d", "role", "area")]
for scope in ("coco", "paco_part", "suim", "ALL"):
    L = [r for r in recs if scope == "ALL" or r["d"] == scope]
    if not L:
        continue
    print(f"\n=== {scope}: n={len(L)} (image, mask) samples; seeded completion from 8 in-mask tokens (diagnostic)")
    print(f"{'affinity source':22s}{'AUROC':>7s}{'opt-level':>10s}{'natural':>8s}{'gap':>6s}{'sd t*':>7s}{'corr(t*,log area)':>18s}{'union K=8':>10s}{'K=16':>6s}")
    for k in SRC:
        g = lambda f: np.mean([r[k][f] for r in L])
        t = np.array([r[k]["t"] for r in L]); a = np.log([r["area"] for r in L])
        un = f"{g('u8'):10.1f}{g('u16'):6.1f}" if "u8" in L[0][k] else ""
        print(f"{k:22s}{g('auc'):7.3f}{g('orc'):10.1f}{g('nat'):8.1f}{g('orc') - g('nat'):6.1f}{t.std():7.2f}{np.corrcoef(t, a)[0, 1]:18.2f}{un}")
