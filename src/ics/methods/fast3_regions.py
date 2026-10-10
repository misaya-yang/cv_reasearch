"""Fixed upper/middle/lower native512 views, whole-guide outside coverage."""
import numpy as np
from scipy.ndimage import label
from .shared_query_regions import WINDOWS

INDICES = (1, 4, 7)
BOXES = tuple(WINDOWS[i] for i in INDICES)


def fuse(local_masks, selections, global_guide, boxes=BOXES):
    """Strict majority of evaluated windows; untouched pixels retain FoRIS."""
    guide = np.asarray(global_guide, dtype=bool)
    if guide.shape != (1024, 1024) or len(local_masks) != len(boxes) or len(selections) != len(boxes):
        raise ValueError('Require the same1024 guide and one selection vector per window')
    coverage = np.zeros(guide.shape, dtype=np.uint8)
    votes = np.zeros(guide.shape, dtype=np.uint8)
    for local, accept, box in zip(local_masks, selections, boxes):
        if np.asarray(local).shape != (512, 512):
            raise ValueError('Require the original512 local masks')
        x0, y0, x1, y1 = box
        ids, n = label(local, np.ones((3, 3), dtype=bool))
        if n != len(accept):
            raise ValueError('Component IDs no longer match the saved decisions')
        lookup = np.asarray([False]+list(accept), dtype=bool)
        coverage[y0:y1, x0:x1] += 1
        votes[y0:y1, x0:x1] += lookup[ids]
    result = 2*votes > coverage
    result[coverage == 0] = guide[coverage == 0]
    return result, coverage
