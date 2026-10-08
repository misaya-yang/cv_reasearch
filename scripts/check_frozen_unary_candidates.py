#!/usr/bin/env python3
"""Independent scalar/linear-system checks for prepared C1/C2 operators."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

from frozen_unary_candidates import (blend_unary, c1_unary, first_context_roles,
                                     minimal_context, reference_walk, solve_parent_system)


def main():
    torch.set_num_threads(2)
    rng = np.random.RandomState(7)
    y = rng.rand(64, 64)
    margin = np.linspace(-2, 2, 4096, dtype=np.float32).reshape(64, 64)
    mask = np.zeros((1024, 1024), bool);mask[:512, :512]=True
    changed, p, c = c1_unary(y, margin, mask)
    assert np.array_equal(changed[32:], y[32:]) and np.array_equal(changed[:, 32:], y[:, 32:])
    assert c.max() <= .5 and p.min() == 0 and p.max() == 1
    absent, _, confidence = c1_unary(y, margin, mask, reference_valid=False)
    assert np.array_equal(absent, y) and not np.any(confidence)
    n = 16
    a = np.linspace(.5, 1.5, n)
    w = sparse.diags(np.ones(n-1), 1)+sparse.diags(np.ones(n-1), -1)
    h = sparse.diags(a)+16*(sparse.diags(np.asarray(w.sum(1)).ravel())-w)
    before = h.toarray().copy()
    u = np.linspace(.1, .9, n)
    replacement = blend_unary(u, np.full(n, .5), np.zeros(n))
    native = solve_parent_system(h, a, u)
    identity = solve_parent_system(h, a, replacement)
    assert np.array_equal(native, identity)
    assert np.array_equal(h.toarray(), before)
    np.testing.assert_allclose(native, np.linalg.solve(before, a*u), atol=2e-7)

    app = F.normalize(torch.from_numpy(rng.randn(n, 8).astype(np.float32)), dim=-1)
    walk = reference_walk(app, (4, 4)).toarray()
    np.testing.assert_allclose(walk.sum(1), 1, atol=2e-7)
    assert not np.any(walk.diagonal())
    coverage = np.zeros(n);coverage[[5, 6]]=1
    roles = first_context_roles(app, [6, 5], [10, 9], coverage, (4, 4))
    assert roles['foreground'] == 5
    # Dense powers independently reproduce the reference pair/anchor choice.
    possibilities = []
    for steps in (2, 4, 8):
        power = np.linalg.matrix_power(walk, steps)
        for bg in (9, 10):
            possibilities.append((float(np.sqrt(power[5, bg]*power[bg, 5])), bg, steps, power))
    best = sorted(possibilities, key=lambda v: (-v[0], v[1], v[2]))[0]
    assert roles['background'] == best[1] and roles['walk_steps'] == best[2]
    shared = np.minimum(best[3][5], best[3][best[1]]);shared[[5, best[1]]]=-1
    assert roles['anchor'] == int(shared.argmax())

    branches = {f'{kind}/{layer}':torch.from_numpy(rng.randn(n, 8).astype(np.float32))
                for layer in (16, 24) for kind in ('Q', 'K')}
    fixed = dict(foreground=0, background=1, anchor=2)
    scored = minimal_context(app, app, branches, branches, fixed, num_heads=2)
    assert scored['query_anchor'] == 2 and scored['wrong_anchor'] == 10
    normalized = {key:F.normalize(value.reshape(n, 2, 4), dim=-1).numpy() for key,value in branches.items()}

    def scalar_relation(i, anchor):
        result = []
        for layer in (16, 24):
            q, k = normalized[f'Q/{layer}'], normalized[f'K/{layer}']
            result.extend([sum(float(q[i,h,d])*float(k[anchor,h,d]) for d in range(4)) for h in range(2)])
            result.extend([sum(float(q[anchor,h,d])*float(k[i,h,d]) for d in range(4)) for h in range(2)])
        return np.asarray(result)
    expected = []
    for j in range(n):
        relation = scalar_relation(j, 2)
        ef = .5*(1-float(app[j]@app[0]))/2+.5*np.mean((relation-scalar_relation(0, 2))**2)/4
        eb = .5*(1-float(app[j]@app[1]))/2+.5*np.mean((relation-scalar_relation(1, 2))**2)/4
        expected.append(.5 if j == 2 else 1/(1+np.exp(-(eb-ef)/.07)))
    np.testing.assert_allclose(scored['probability'].numpy(), expected, atol=3e-7)
    swapped = minimal_context(app, app, branches, branches,
                              dict(foreground=1, background=0, anchor=2), num_heads=2)
    np.testing.assert_allclose(swapped['probability'].numpy(), 1-scored['probability'].numpy(), atol=1e-7)
    print('PASS: C1 no outside unary edits/c0 identity, inherited solve, source kernel/walk roles, scalar bidirectional C2, shared anchor, fixed mismatch and role symmetry')


if __name__ == '__main__':
    main()
