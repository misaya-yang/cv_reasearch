#!/usr/bin/env python3
"""Fetch files of a Hugging Face repo through a mirror with parallel, resumable byte ranges.

The default hub client stalls on this server; plain ranged curl does not.
Usage: hf_fetch.py <repo_id> <dest_dir> <file> [<file> ...]      (HF_ENDPOINT defaults to hf-mirror.com)
A file argument may be "a|b" meaning: try a, fall back to b (e.g. model.safetensors|pytorch_model.bin).
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EP = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"); PARTS = int(os.environ.get("PARTS", 12))

def size_of(url):
    out = subprocess.run(["curl", "-sIL", "-m", "30", url], capture_output=True, text=True).stdout
    blocks = [b for b in out.replace("\r", "").split("\n\n") if b.strip()]   # text mode already folds CRLF
    if not blocks or " 200" not in blocks[-1].splitlines()[0]: return None
    for line in blocks[-1].splitlines():
        if line.lower().startswith("content-length:"): return int(line.split(":")[1])
    return None

def part(url, path, lo, hi):
    want = hi - lo + 1
    for _ in range(40):
        have = path.stat().st_size if path.exists() else 0
        if have == want: return True
        if have > want: path.unlink(); have = 0
        with path.open("ab") as f:
            subprocess.run(["curl", "-sL", "--connect-timeout", "15", "-m", "300", "-r", f"{lo + have}-{hi}", url], stdout=f)
    return path.stat().st_size == want

def fetch(repo, dest, spec):
    for name in spec.split("|"):
        url = f"{EP}/{repo}/resolve/main/{name}"; out = dest / name
        n = size_of(url)
        if n is None: continue
        if out.exists() and out.stat().st_size == n: print("have", out, n); return True
        out.parent.mkdir(parents=True, exist_ok=True)
        k = 1 if n < 4_000_000 else PARTS; step = -(-n // k)
        jobs = [(url, dest / f".{name}.part{i}", i * step, min(n, (i + 1) * step) - 1) for i in range(k)]
        with ThreadPoolExecutor(k) as ex: ok = list(ex.map(lambda j: part(*j), jobs))
        if not all(ok): print("FAILED", name); return False
        with out.open("wb") as f:
            for _, p, _, _ in jobs: f.write(p.read_bytes()); p.unlink()
        assert out.stat().st_size == n; print("fetched", out, n); return True
    print("NOT FOUND", spec); return False

if __name__ == "__main__":
    repo, dest = sys.argv[1], Path(sys.argv[2]); dest.mkdir(parents=True, exist_ok=True)
    sys.exit(0 if all([fetch(repo, dest, s) for s in sys.argv[3:]]) else 1)
