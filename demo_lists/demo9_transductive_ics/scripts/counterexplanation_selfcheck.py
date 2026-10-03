"""Local CPU checks; never marks the REAL public-source GPU path ready."""
import argparse
import json
from pathlib import Path

import numpy as np

from counterexplanation_core import (cache_public_extractor, counterexplanation_cut,
                                    negative_proposals, positive_only_cut, response_density)


def check():
    levels = np.linspace(.2, .9, 57)
    s = np.arange(64, dtype=float).reshape(8, 8)
    records = []
    def record(name, assertion):
        assert assertion, name
        records.append(dict(name=name, passed=True))
    record("same_field_native", counterexplanation_cut(s, [s], levels)[0] == .5)
    record("constant_negative_native", counterexplanation_cut(s, [np.ones_like(s)], levels)[0] == .5)
    record("constant_positive_native", counterexplanation_cut(np.ones_like(s), [s], levels)[0] == .5)
    record("uniform_response_native", counterexplanation_cut(s, [np.ones_like(s)/s.size], levels)[0] == .5)
    naive=positive_only_cut(s, levels)
    uniform=counterexplanation_cut(s, [np.ones_like(s)/s.size], levels, uniform_control=True)
    record("uniform_control_not_fallback", uniform[1] == "selected")
    record("positive_only_uniform_same_operator", naive[0] == uniform[0] and naive[1] == uniform[1])
    p = response_density(s)
    record("density_is_mass_not_posterior", abs(p.sum()-1) < 1e-12)
    d = p-response_density(s[::-1])
    record("original_empty_full_tie", abs(d.sum()) < 1e-12 and np.array([]).sum() == 0)
    # Ten label-only fixtures: geometric proposals remain strictly known BG.
    for seed in range(10):
        fg = np.zeros((16, 16), bool); fg[6:10, 6:10] = True
        priority = np.random.default_rng(seed).random(fg.shape)
        hard, _ = negative_proposals(fg, priority)
        random, _ = negative_proposals(fg, priority, True, seed)
        record("legal_area_connected_fixture_%d" % seed,
               len(hard) == len(random) == 2 and
               all(not (m & fg).any() and m.sum() == fg.sum() for m in hard+random))
    fg = np.ones((8, 8), bool); fg[0, 0] = False
    record("insufficient_BG_refused", not negative_proposals(fg, np.zeros_like(fg, float))[0])
    # A small incorrect hot region can beat a broad true object under mean evidence.
    evidence = np.zeros(100); evidence[:4] = 2; evidence[4:44] = 1
    small = np.arange(100) < 4; large = np.arange(100) < 44
    contrast = lambda m: evidence[m].mean()-evidence[~m].mean()
    record("remaining_small_region_bias_counterexample", contrast(small) > contrast(large))
    pp=np.repeat([.45,.35,.15,.05],4).reshape(4,4)/4
    nn=np.repeat([.49,.49,.01,.01],4).reshape(4,4)/4
    candidates=[np.arange(16)<k for k in (4,8,12)]
    mass=[(pp.ravel()[m]-nn.ravel()[m]).sum() for m in candidates]
    record("unit_mass_negative_can_harm_true_FG", mass[0] > mass[1] and abs(pp.sum()-1)<1e-12 and abs(nn.sum()-1)<1e-12)
    pp_mean=np.repeat([.48,.32,.15,.05],4).reshape(4,4)/4
    record("mean_repair_can_still_delete_TP", counterexplanation_cut(pp_mean,[nn],[.2,.5,.9])[0] == .9)
    import torch
    torch.set_num_threads(1)
    class Fake:
        def _extract_features(self, images):
            return images * 2
    host=Fake(); images=torch.arange(16).view(1,2,2,2,2).float()
    with cache_public_extractor(host) as state:
        a=host._extract_features(images); a.zero_()
        b=host._extract_features(images)
        record("raw_cache_no_mutation", torch.equal(b, images*2))
        record("raw_cache_single_encoding", state["calls"] == state["reuses"] == 1)
        try:
            host._extract_features(images+1)
        except RuntimeError:
            record("RGB_change_refused", True)
        else:
            raise AssertionError("changed image accepted")
    record("cache_restored_and_released", "_extract_features" not in host.__dict__ and state["raw"] is None)
    return dict(state="CPU_PRIMITIVES_PASSED", checks=records,
                real_public_source_native_cache_exact=False,
                GPU_ready=False, scientific_gain_measured=False,
                pending=["actual public-source synthetic end-to-end",
                         "negative masks after source preprocessing preserve known-BG labels",
                         "actual DINO cached/uncached native mask exact gate",
                         "finite runner/guard with shutdown disabled"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    args=parser.parse_args(); result=check(); path=Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)); print(json.dumps(result))
