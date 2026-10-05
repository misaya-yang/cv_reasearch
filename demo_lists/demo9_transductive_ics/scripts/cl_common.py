"""The query partitioned by its own features (k-means on cosine, no reference). Ceiling when each segment is labelled by the
truth; how many segments make the target; how well label-free segment quantities rank target segments; simple rules."""
import json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/root/autodl-tmp/demo9_lang/scripts" if Path("/root/autodl-tmp").exists() else "/Users/yang/projects/CVPR2027/demo_lists/demo9_transductive_ics/scripts")
from lang_common import binarise, class_miou, iu, normalise, paired, photo_groups
import os
SERVER = Path("/root/autodl-tmp").exists()
if SERVER:  # the rented machine: everything at its real path
    L = Path("/root/autodl-tmp/demo9_lang/cpu"); ROOT = Path("/root")
    FEAT = Path("/root/autodl-tmp/demo9_extent/cache/evidence_v1/feat")
else:
    L = Path.home() / "cvpr2027_local"; ROOT = L / "dino_local/root"; FEAT = L / "feat"
man = json.loads((ROOT / "autodl-tmp/demo9_transductive_ics/results/extent_head_t1_isolated_v1/dev_episodes.json").read_text())
packets = ROOT / "autodl-tmp/demo9_extent/results/extent_v1/run/packets"
DATA = Path(man["data_root"]) if SERVER else L / "dino_local" / man["data_root"].lstrip("/")
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
rows = man["episodes"]
def kmeans(x, k, iters=15, seed=0):
    g = torch.Generator().manual_seed(seed); c = x[torch.randperm(len(x), generator=g)[:1]]
    for _ in range(k - 1):  # farthest-point start
        c = torch.cat([c, x[(x @ c.T).max(1).values.argmin()][None]])
    for _ in range(iters):
        a = (x @ c.T).argmax(1); c = F.normalize(torch.zeros_like(c).index_add_(0, a, x), dim=-1)
    return (x @ c.T).argmax(1)
def up(m): return F.interpolate(m.float().view(1, 1, 64, 64), size=(1024, 1024), mode="bilinear", align_corners=False)[0, 0] > .5
def auc(v, y):
    v, y = np.asarray(v), np.asarray(y, bool)
    if y.all() or not y.any(): return np.nan
    r = np.argsort(np.argsort(v)) + 1.; n1 = y.sum(); return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
