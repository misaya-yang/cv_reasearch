"""Print the mIoU of every variant in result JSONs:  python scripts/show.py results/ade20k/*.json [--all]"""
import json, sys
full = "--all" in sys.argv
for p in [a for a in sys.argv[1:] if not a.startswith("--")]:
    d = json.load(open(p)); print("==", p, d.get("segmenter"), d.get("images"))
    for k, v in d.items():
        if isinstance(v, dict) and "mIoU" in v: print(f"  {k:42s} {v['mIoU']:7.2f}" + (f"  mAcc {v['mAcc']:6.2f}  aAcc {v['aAcc']:6.2f}" if full else ""))
        elif k == "diagnostics" or k.endswith("/regions"): print(f"  {k}: {v}")
