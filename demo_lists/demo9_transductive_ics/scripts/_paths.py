"""Locations. demo9 reuses demo4's encoder wrapper, episode list and COCO loader (demo_lists/demo4_incontext_seg/icx/common.py)
and the INSID3 checkout next to it. On the server: code /root/autodl-tmp/demo9, demo4 /root/autodl-tmp/demo4,
shared env / weights / COCO masks /root/demo4_cache (read-only for this project), feature caches /root/demo9_cache (delete after use)."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO4 = os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4")
CACHE9 = os.environ.get("DEMO9_CACHE", "/root/demo9_cache")
# COCO-20i masks. Default: the official masks (crowd regions excluded). The tables made before 2026-10-02 used masks rebuilt from the
# instance annotations, which include crowd regions: set DEMO9_COCO_ANN=/root/demo4_cache/data/COCO2014/annotations to reproduce them.
COCO_ANN = os.environ.get("DEMO9_COCO_ANN", "/root/autodl-tmp/datasets/ics/COCO2014/annotations")
sys.path.insert(0, HERE)
