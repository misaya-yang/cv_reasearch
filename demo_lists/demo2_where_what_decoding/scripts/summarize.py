"""Markdown tables from result JSONs (never copy numbers by hand).

  python scripts/summarize.py results/ade20k [results/coco ...] > RESULTS_TABLES.md
Prints every table the files present allow. With `*.perimage.pt` files next to the JSONs (and torch installed), also prints paired
bootstrap intervals for the headline comparisons.
"""
import glob, json, os, sys

ORDER = ["segformer-b0", "segformer-b2", "m2f-swin-tiny", "m2f-swin-small", "segformer-b5", "upernet-convnext-large", "m2f-swin-base-in21k", "m2f-swin-large",
         "eomt-large", "eomt-dinov3-large"]
NAMES = {"segformer-b0": "SegFormer-B0", "segformer-b2": "SegFormer-B2", "segformer-b5": "SegFormer-B5", "m2f-swin-tiny": "Mask2Former Swin-T",
         "m2f-swin-small": "Mask2Former Swin-S", "m2f-swin-base-in21k": "Mask2Former Swin-B (IN-21k)", "m2f-swin-large": "Mask2Former Swin-L",
         "upernet-convnext-large": "UperNet ConvNeXt-L", "eomt-large": "EoMT-L (DINOv2)", "eomt-dinov3-large": "EoMT-L (DINOv3)"}
tag_of = lambda n: n.split("_", 1)[1].replace("-ade", "").replace("-coco", "")
key = lambda t: ORDER.index(t) if t in ORDER else 99
nm = lambda t: NAMES.get(t, t)
g = lambda d, k: d[k]["mIoU"] if k in d else None
f = lambda v, base=None: "—" if v is None else (f"{v:.2f}" if base is None else f"{v:.2f}（{v - base:+.2f}）")


def table(title, header, rows, note=None):
    if not rows: return
    print(f"\n### {title}\n")
    if note: print(note + "\n")
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for r in rows: print("| " + " | ".join(r) + " |")


def best(d, prefix):
    """Best mIoU over the swept exponents of one family of variants (an upper bound: the exponent is chosen on the evaluation set)."""
    c = [v["mIoU"] for k, v in d.items() if isinstance(v, dict) and "mIoU" in v and k.startswith(prefix)]
    return max(c) if c else None


def pick(R, need, prefer=("a0_", "h1_", "final_", "ens_", "h2_", "h4_", "h2cf_")):
    """One result per segmenter among the files that contain every key in `need`."""
    out = {}
    for pre in prefer:
        for n, d in R.items():
            if n.startswith(pre) and all(k in d for k in need): out.setdefault(tag_of(n), d)
    return dict(sorted(out.items(), key=lambda kv: key(kv[0])))


for root in sys.argv[1:]:
    R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{root}/*.json"))}
    if not R: continue
    any_d = next(iter(R.values())); print(f"\n## {any_d['dataset']}（验证集 {any_d['images']} 张）")

    rows = []
    for tag, d in pick(R, ["oracle/region_majority"]).items():
        a = g(d, "argmax"); dg = d["diagnostics"]; rg = next((v for k, v in d.items() if k.endswith("/regions")), {})
        rec_gt = next((g(d, k) for k in ("ext/gt_regions", "dino/gt_regions") if k in d), None)
        rows.append([nm(tag), f(a), f"{100 * dg['error_pixels_predicting_absent_class']:.0f}%", f"{dg['spurious_classes_per_image']:.1f}", f(g(d, "oracle/presence")),
                     f(g(d, "oracle/region_majority")), f(g(d, "oracle/gt_regions+seg_what")), f(rec_gt), f"{100 * rg['seg_acc']:.1f}%" if rg else "—"])
    table("表 1　诊断：在哪 / 是什么", ["分割器", "argmax", "错误像素中预测了图里没有的类", "每图幻觉类数", "上限：只在真实存在的类里选", "上限：预测区域 + 真值命名",
                                 "真值区域 + 分割器自己命名", "真值区域 + 冻结 DINOv2-L 识别器命名", "预测区域的命名准确率"], rows,
          "“区域”指一张图里被标成同一类的全部像素。命名准确率按区域个数计，标准答案是区域内占多数的真值类。")

    rows = []
    for tag, d in pick(R, ["ext/fuse^2"]).items():
        a = g(d, "argmax")
        rows.append([nm(tag), f(a), f(g(d, "ext/fuse^1"), a), f(g(d, "ext/fuse^2"), a), f(g(d, "ext/fuse^4"), a), f(g(d, "ext/relabel^2"), a), f(g(d, "ext/rec_only"), a),
                     f(g(d, "ext/query^2+fuse^2"), a) if "ext/query^2+fuse^2" in d else "—"])
    table("表 2　区域识别器（冻结 DINOv2-L + MLP，在训练集真值区域上训练）", ["分割器", "argmax", "融合 γ=1", "融合 γ=2", "融合 γ=4", "整区域重标 γ=2", "只用识别器命名",
                                                         "查询级 + 类区域级，γ=2"], rows, "γ=2 是事先固定的设置，其余列只为显示敏感度。")

    rows = []
    for tag, d in pick(R, ["lin/alone", "ext/fuse^2"]).items():
        a = g(d, "argmax")
        rows.append([nm(tag), f(a), f(g(d, "lin/alone")), f(g(d, "lin/mean"), a), f(best(d, "lin/ens^"), a), f(best(d, "lin/region_vote/fuse"), a), f(g(d, "ext/fuse^2"), a),
                     f(g(d, "ext/query^2+fuse^2"), a) if "ext/query^2+fuse^2" in d else "—", f(g(d, "ext/fuse^2*lin/region_vote^1"), a)])
    table("表 3　H1 对照：同一个冻结 DINOv2-L 的三种用法", ["分割器", "argmax", "逐 patch 头单独", "像素级算术平均", "像素级乘积（取最优 γ）", "逐 patch 头在区域内平均后融合（取最优 γ）",
                                           "区域识别器（γ=2 固定）", "区域识别器，查询级 + 类区域级", "区域识别器 + 逐 patch 头区域平均"], rows,
          "逐 patch 头：`[patch token, CLS]` 上的两层 MLP，训练 8 轮，与区域识别器容量相同、看到的信息相同。对照取对它最有利的指数。")

    rows = []
    for n in sorted((n for n in R if n.startswith("ens_")), key=lambda n: key(tag_of(n))):
        d = R[n]; a = g(d, "argmax")
        for name in sorted({k.split("/")[0] for k in d if "/region_vote/" in k and not k.startswith("ext/") and not k.startswith("lin/")}):
            label = {"v:flip": "同一分割器：水平翻转", "v:all": "同一分割器：全部视角", "b5": "SegFormer-B5", "m2fl": "Mask2Former Swin-L", "eomt": "EoMT-L",
                     "upernet": "UperNet ConvNeXt-L"}.get(name, "同一分割器：缩放 " + name[3:] if name.startswith("v:x") else name)
            rows.append([nm(tag_of(n)), label, f(a), f(g(d, f"{name}/alone")), f(g(d, f"{name}/mean"), a), f(best(d, f"{name}/ens^"), a), f(best(d, f"{name}/region_vote/"), a),
                         f(g(d, "ext/fuse^2"), a), f(g(d, f"ext/fuse^2*{name}/region_vote^1"), a)])
    table("表 4　其他“第二意见”：像素级与区域级", ["分割器", "第二意见", "argmax", "第二意见单独", "像素级算术平均", "像素级乘积（最优 γ）", "区域内平均后融合（最优）",
                                   "区域识别器 γ=2（参照）", "区域识别器 + 第二意见区域平均"], rows)

    rows = []
    for n in sorted((n for n in R if n.startswith("h2_")), key=lambda n: key(tag_of(n))):
        d = R[n]; a = g(d, "argmax")
        rows.append([nm(tag_of(n)), f(a)] + [f(g(d, f"{r}/fuse^2"), a) for r in ("own_gt", "own_gtpred", "dino_gt", "dino_gtpred")] +
                    [f"{100 * d[r + '/regions']['rec_acc']:.1f}%" for r in ("own_gt", "own_gtpred", "dino_gt", "dino_gtpred")] + [f"{100 * d['ext/regions']['seg_acc']:.1f}%"])
    table("表 5　H2：识别器用分割器自己的骨干特征（训练集 2 万张，融合 γ=2）",
          ["分割器", "argmax", "自身特征，真值区域训练", "自身特征，真值 + 预测区域训练", "DINOv2-L，真值区域训练", "DINOv2-L，真值 + 预测区域训练",
           "命名准确率：自身 / 真值", "自身 / 真值 + 预测", "DINOv2-L / 真值", "DINOv2-L / 真值 + 预测", "分割器自己"], rows,
          "自身特征 = 骨干最后一级 + 解码器输出，在区域内池化，再拼上它们的全图平均；不需要额外的骨干前向。")

    rows = []
    for n in sorted((n for n in R if n.startswith("h2cf_")), key=lambda n: key(tag_of(n))):
        d = R[n]; a = g(d, "argmax")
        rows.append([nm(tag_of(n)), f(a)] + [f(g(d, f"{r}/fuse^2"), a) for r in ("own_cf_gt", "own_cf_gtpred", "ext_cf_gt", "ext_cf_gtpred", "ext")])
    table("表 6　训练数据量的影响：只用 1000 张图训练识别器（验证集 2 折交叉拟合，融合 γ=2）",
          ["分割器", "argmax", "自身特征，真值区域", "自身特征，真值 + 预测区域", "DINOv2-L，真值区域", "DINOv2-L，真值 + 预测区域", "DINOv2-L，训练集 2 万张（参照）"], rows)

    rows = []
    for n in sorted((n for n in R if n.startswith("h4_")), key=lambda n: key(tag_of(n))):
        d = R[n]; a = g(d, "argmax")
        rows.append([nm(tag_of(n)), f(a)] + [x for r in ("dino", "sig", "both") for x in (f(g(d, f"{r}/fuse^2"), a), f(g(d, f"{r}/gt_regions")))] +
                    [f(g(d, "both/query^2+fuse^2"), a) if "both/query^2+fuse^2" in d else "—"])
    table("表 7　H4：识别器的编码器（融合 γ=2；“真值区域”列是把真值区域交给识别器命名的 mIoU）",
          ["分割器", "argmax", "DINOv2-L", "真值区域", "SigLIP2-so400m", "真值区域", "两者拼接", "真值区域", "两者拼接，查询级 + 类区域级"], rows)

    rows = [[nm(tag), f(g(d, "argmax"))] + [f(g(d, k), g(d, "argmax")) for k in ("mc/sum_pow^2", "mc/sum_pow^4", "mc/self_sharpen^2", "mc/drop_conf<0.3", "mc/drop_conf<0.5")]
            for tag, d in pick(R, ["mc/drop_conf<0.5"]).items()]
    table("表 8　掩码分类模型的无标签读出规则（对照）", ["分割器", "argmax", "Σ(m·p)²", "Σ(m·p)⁴", "查询类别分布自乘", "丢弃置信度 < 0.3 的查询", "丢弃 < 0.5 的查询"], rows)

    # ---- paired bootstrap over images
    try: import torch
    except ImportError: continue
    def boot(path, names, B=2000):
        d = torch.load(path); C, N, V = d["C"], d["images"], d["variants"]; names = [n for n in names if n in V]
        if len(names) < 2: return []
        def dense(n):
            t = torch.zeros(N, 3, C, dtype=torch.float64); idx = V[n]["idx"].long(); t[idx[:, 0], :, idx[:, 1]] = V[n]["val"].double(); return t
        miou = lambda t: torch.nanmean(t[..., 0, :] / (t[..., 2, :] + t[..., 1, :] - t[..., 0, :]), -1) * 100
        torch.manual_seed(0); w = torch.zeros(B, N, dtype=torch.float64).scatter_add_(1, torch.randint(0, N, (B, N)), torch.ones(B, N, dtype=torch.float64))
        T = {n: dense(n) for n in names}; bs = {n: miou(torch.einsum("bn,nkc->bkc", w, t)) for n, t in T.items()}; out = []
        for n in names[1:]:
            df = bs[n] - bs[names[0]]; lo, hi = (float(x) for x in torch.quantile(df, torch.tensor([0.025, 0.975], dtype=torch.float64)))
            out.append((n, float(miou(T[n].sum(0))) - float(miou(T[names[0]].sum(0))), lo, hi))
        return [(names[0], float(bs[names[0]].std()), None, None)] + out
    LABEL = {"ext/fuse^2": "区域识别器 γ=2", "ext/query^2+fuse^2": "区域识别器，查询级 + 类区域级", "both/fuse^2": "两个编码器的识别器 γ=2",
             "both/query^2+fuse^2": "两个编码器，查询级 + 类区域级", "lin/ens^0.5": "逐 patch 头，像素级乘积 γ=0.5", "lin/ens^1": "逐 patch 头，像素级乘积 γ=1",
             "lin/region_vote/fuse^0.5": "逐 patch 头，区域内平均 γ=0.5", "lin/region_vote/fuse^1": "逐 patch 头，区域内平均 γ=1", "v:flip/alone": "只用水平翻转后的输入",
             "v:flip/mean": "翻转测试时增强（算术平均）", "own_gt/fuse^2": "自身特征识别器（真值区域训练）", "own_gtpred/fuse^2": "自身特征识别器（真值 + 预测区域训练）",
             "dino_gtpred/fuse^2": "DINOv2-L 识别器（真值 + 预测区域训练）", "eomt/alone": "换成 EoMT-L", "eomt/ens^1": "与 EoMT-L 像素级乘积", "upernet/ens^1": "与 UperNet 像素级乘积",
             "m2fl/ens^1": "与 Mask2Former Swin-L 像素级乘积", "mc/drop_conf<0.5": "丢弃置信度 < 0.5 的查询"}
    rows = []
    for p in sorted(glob.glob(f"{root}/*.perimage.pt"), key=lambda p: (key(tag_of(os.path.basename(p)[:-12])), p)):
        n = os.path.basename(p)[:-12]; res = boot(p, ["argmax"] + list(LABEL))
        for k, (name, d, lo, hi) in enumerate(res):
            if k == 0: sd = d; continue
            rows.append([nm(tag_of(n)), n.split("_")[0], LABEL[name], f"{d:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{sd:.2f}"])
    table("表 9　配对自助法：相对 argmax 的 mIoU 差及 95% 区间", ["分割器", "来自哪次运行", "变体", "差", "95% 区间", "argmax 自身在重采样下的标准差"], rows,
          "对图像做 2000 次有放回重采样，每次重算数据集级 mIoU。区间不含 0 才算有差别。")
