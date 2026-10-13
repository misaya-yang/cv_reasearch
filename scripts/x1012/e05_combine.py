"""Offline: complete-mask combinations of FoRIS (B), MEAN (G), coverage transfer (T / Treg) on all ten dev sets.

Every rule is per episode and uses only the reference mask and the candidate masks; no dataset identity.
  vote(a,b,c)      pixel-wise majority
  thin gate        rho = share of reference FG area lying in impure tokens (coverage < 0.9); if rho > theta use T
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
ARMS = lab.ASSETS / "a/x1012/arms"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
key = lambda b: b["index"] if isinstance(b["index"], str) else format(b["index"], "06d")
rows = [b for b in (lab.subset(N) if N > 0 else lab.bindings()) + lab.ext_subset(None) if (ARMS / f"{key(b)}.npz").exists()]
DOMS = tuple(dict.fromkeys(b["dataset"] for b in rows))
un = lambda x: torch.from_numpy(np.unpackbits(x)[: 1024 * 1024].reshape(1024, 1024).astype(bool))
maj = lambda a, b, c: (a & b) | (a & c) | (b & c)
M = lab.defaultdict(lab.Meter)
sub = lab.defaultdict(lambda: lab.defaultdict(lab.Meter))
for b in rows:
    z = np.load(ARMS / f"{key(b)}.npz")
    gt = lab.mask1024(b["row"]["query_mask_path"])
    ig = lab.ignore1024(b)
    B, T, Tr = lab.foris_cli(b), un(z["T"]), un(z["Treg"])
    G = lab.arm_cli(b, "G") if "G" in b["old_arms"] else un(z["G"])
    rho = float(z["ref_stats"][2])
    c = {"B  FoRIS complete": B, "G  MEAN": G, "T  coverage transfer": T, "Treg region transfer": Tr,
         "vote(B,G,T)": maj(B, G, T), "vote(B,G,Treg)": maj(B, G, Tr)}
    for a in ("R", "C", "N"):
        if a in b["old_arms"]:
            c[{"R": "R  whole ridge (repo)", "C": "C  vote(B,R,G) (repo)", "N": "N  supported_mean (repo)"}[a]] = lab.arm_cli(b, a)
    for th in (0.5, 0.8):
        thin = rho > th
        c[f"rho>{th}: T else B"] = T if thin else B
        c[f"rho>{th}: T else G"] = T if thin else G
        c[f"rho>{th}: T else vote(B,G,Treg)"] = T if thin else c["vote(B,G,Treg)"]
        c[f"rho>{th}: T else vote(B,G,T)"] = T if thin else c["vote(B,G,T)"]
        if "N" in b["old_arms"]:
            c[f"rho>{th}: T else N (repo)"] = T if thin else c["N  supported_mean (repo)"]
    g = gt if ig is None else gt & ~ig
    for n, p in c.items():
        if ig is not None:
            p = p & ~ig
        r = lab.iu(p, g)
        M[n].add(b, *r)
        if n[:1] in "BGT":
            sub["rho>0.5" if rho > 0.5 else "rho<=0.5"][n].add(b, *r)

cnt = {d: sum(r["dataset"] == d for r in rows) for d in DOMS}
print("class-pooled mIoU; first row absolute, other rows delta vs FoRIS; episodes " + ", ".join(f"{d[:6]}={n}" for d, n in cnt.items()))
print(f"{'method':36s}" + "".join(f"{d[:8]:>9s}" for d in DOMS) + "   #wins")
base = M["B  FoRIS complete"].row(DOMS)
print(f"{'B  FoRIS complete':36s}" + "".join(f"{base[d][0]:9.2f}" for d in DOMS))
for n, m in M.items():
    if n.startswith("B "):
        continue
    r = m.row(DOMS)
    dl = [r[d][0] - base[d][0] if d in r else float('nan') for d in DOMS]
    print(f"{n:36s}" + "".join("        -" if np.isnan(x) else f"{x:+9.2f}" for x in dl) + f"   {sum(x > 0 for x in dl if not np.isnan(x))}/{sum(not np.isnan(x) for x in dl)}")
print("\nby reference thinness (class-pooled within the subset): B | G | T | Treg  (episodes)")
for s in ("rho<=0.5", "rho>0.5"):
    for d in DOMS:
        if d in sub[s]["B  FoRIS complete"].eps and len(sub[s]["B  FoRIS complete"].eps[d]) >= 8:
            print(f"  {s:9s} {d:15s}" + "  ".join(f"{sub[s][n].miou(d):6.2f}" for n in ("B  FoRIS complete", "G  MEAN", "T  coverage transfer", "Treg region transfer")) + f"   ({len(sub[s]['B  FoRIS complete'].eps[d])})")
