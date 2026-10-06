"""Edits of a host mask as counts, for sweeping many proposals and many kinds of evidence at once.

A proposer P edits the host H in two disjoint places: additions P & ~H lie outside H, deletions H & ~P inside it.
The universe proposer (P = None) offers every pixel outside H for addition and every pixel inside H for deletion.
An edit pixel is good (an added true pixel, a deleted false pixel) or bad (the side effect). With I0, U0 the host's
intersection and union with the truth,

    I = I0 + add_good - delete_bad        U = U0 + add_bad - delete_good

so the class mIoU of any composition is exact from per-episode counts, without rendering a mask.

Evidence is one byte per token (64 x 64, BINS levels), constant over the token's 16 x 16 pixels. Additions are
accepted from the highest evidence down, deletions from the lowest up. A rule is one threshold: accept every edit
pixel whose token priority is at or above it. `rank` rules meet a per-episode pixel budget, `absolute` rules use one
threshold for all episodes, placed where the fitting episodes meet the same pooled budget. Budgets are label-free
(a share of the proposer's edit set, or of the host's area for the universe). The same quantisation renders the
mask, so counts and masks agree by construction.
"""
from __future__ import annotations

import numpy as np

BINS = 256
OPS = ("add", "delete")
SHARES = (.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)   # of a proposer's own edit set
BUDGETS = (.02, .05, .1, .2, .3, .5)               # of the host's area, for the universe proposer


def blocks(mask):
    """Pixels of a 1024 x 1024 mask inside each of the 4096 tokens."""
    return np.asarray(mask, dtype=bool).reshape(64, 16, 64, 16).sum((1, 3)).reshape(4096)


def expand(tokens):
    return np.repeat(np.repeat(np.asarray(tokens).reshape(64, 64), 16, 0), 16, 1)


def edit_set(host, proposer, op):
    if proposer is None:
        return ~host if op == "add" else host.copy()
    return proposer & ~host if op == "add" else host & ~proposer


def priority(bins, op):
    """Higher is accepted earlier: high evidence for additions, low evidence for deletions."""
    b = np.asarray(bins).astype(np.int64)
    return b if op == "add" else BINS - 1 - b


def histograms(bins, good, bad, op):
    """[N, BINS, 2] good and bad edit pixels per priority level. bins, good, bad: [N, 4096]."""
    n = bins.shape[0]
    index = (np.arange(n)[:, None] * BINS + priority(bins, op)).ravel()
    return np.stack([np.bincount(index, weights=np.asarray(x, np.float64).ravel(), minlength=n * BINS).reshape(n, BINS)
                     for x in (good, bad)], -1)


def from_top(hist):
    """[N, BINS + 1, 2]: counts accepted at each threshold t (levels >= t); t = BINS accepts nothing."""
    c = np.flip(np.cumsum(np.flip(hist, 1), 1), 1)
    return np.concatenate([c, np.zeros_like(c[:, :1])], 1)


def thresholds(total_top, need):
    """Highest threshold at which the accepted pixels reach the budget. total_top [N, BINS + 1], need [..., N]."""
    need = np.minimum(np.asarray(need, np.float64), total_top[:, 0])
    return (total_top >= need[..., None]).sum(-1) - 1


def pooled_thresholds(total_top, need):
    """One threshold for all episodes: total_top [N, BINS + 1] of the fitting episodes, need [...] pooled budgets."""
    pooled = total_top.sum(0)
    return (pooled >= np.minimum(np.asarray(need, np.float64), pooled[0])[..., None]).sum(-1) - 1


def counts_at(top, thr, common=False):
    """Accepted good and bad pixels. top [N, BINS + 1, 2]; thr [..., N] per episode, or [...] thresholds common to
    all episodes when `common` -> [..., N, 2]."""
    thr = np.asarray(thr)
    if common:
        return np.moveaxis(top[:, thr], 0, -2)
    return top[np.arange(top.shape[0]), thr]


def compose(base, add=None, delete=None):
    """Host [N, 2] (I, U) after accepted additions and deletions, each [..., N, 2] (good, bad)."""
    out = np.asarray(base, dtype=np.float64)
    if add is not None:
        out = out + np.stack([add[..., 0], add[..., 1]], -1)
    if delete is not None:
        out = out - np.stack([delete[..., 1], delete[..., 0]], -1)
    return out


def selectivity(hist):
    """[N] probability that a good edit pixel outranks a bad one inside each episode (0.5 = no selection; nan when
    an episode lacks one kind)."""
    g, b = hist[..., 0], hist[..., 1]
    below = np.cumsum(b, 1) - b
    with np.errstate(invalid="ignore", divide="ignore"):
        return (g * (below + .5 * b)).sum(1) / (g.sum(1) * b.sum(1))


def accepted(edit, bins, op, thr):
    """Edit pixels a rule accepts in one episode: tokens whose priority is at or above the threshold."""
    return edit & expand(priority(bins, op) >= thr)


def self_check():
    """Counts equal rendered masks; perfect evidence takes every good pixel before any bad one."""
    rng = np.random.RandomState(0)
    truth = expand(rng.rand(4096) < .3)
    host, prop = rng.rand(1024, 1024) < .3, rng.rand(1024, 1024) < .3
    perfect = np.where(blocks(truth) > 0, 230, 20).astype(np.uint8)
    noise = rng.randint(0, BINS, 4096).astype(np.uint8)
    base = np.array([[(host & truth).sum(), (host | truth).sum()]], np.float64)
    for op in OPS:
        for proposer, levels in ((prop, SHARES), (None, BUDGETS)):
            e = edit_set(host, proposer, op); good = e & (truth if op == "add" else ~truth)
            g, b = blocks(good)[None], blocks(e & ~good)[None]
            ref = float(e.sum()) if proposer is not None else float(host.sum())
            for ev in (perfect, noise):
                top = from_top(histograms(ev[None], g, b, op)); total = top.sum(-1)
                need = np.array(levels)[:, None] * ref
                for common, thr in ((False, thresholds(total, need)), (True, pooled_thresholds(total, need[:, 0]))):
                    c = counts_at(top, thr, common)
                    for k in range(len(levels)):
                        t = int(np.ravel(thr[k])[0]); acc = accepted(e, ev, op, t)
                        mask = (host | acc) if op == "add" else (host & ~acc)
                        iu = compose(base, add=c[k]) if op == "add" else compose(base, delete=c[k])
                        assert [int(iu[0, 0]), int(iu[0, 1])] == [int((mask & truth).sum()), int((mask | truth).sum())]
                        assert c[k].sum() >= min(need[k, 0], e.sum()) - 1e-9
            s = selectivity(histograms(noise[None], g, b, op))[0]
            assert abs(s - .5) < .05, (op, s)
            assert selectivity(histograms(perfect[None], g, b, op))[0] > .999
    return "edit_counts self-check passed"


if __name__ == "__main__":
    print(self_check())
