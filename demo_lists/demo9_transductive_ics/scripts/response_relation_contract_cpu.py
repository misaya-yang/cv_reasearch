#!/usr/bin/env python3
"""Exact small witnesses for response-relation inference claims.

This audits a withdrawn proposal, not a segmentation method. No encoder,
features, images, downloads, CUDA or scientific performance claims.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random


def enumerate_energy(probabilities, observations=(), strength=1.0):
    """Toy Bernoulli unary plus stated finite noisy pair observations.

    Each observation is (i, j, observed_state, weight). Correct-state
    likelihood .97; all other states .01. This channel is GIVEN, not
    claimed to be a model of DINO responses. The constant denominator
    in the Bayes likelihood ratio cancels when comparing masks.
    """
    energies = {}
    for labels in itertools.product((0, 1), repeat=len(probabilities)):
        energy = -sum(math.log(p if y else 1-p)
                      for p, y in zip(probabilities, labels))
        for i, j, state, weight in observations:
            likelihood = .97 if (labels[i], labels[j]) == state else .01
            energy -= strength * weight * math.log(likelihood)
        energies[labels] = energy
    return min(energies, key=energies.get), energies


def audit():
    checks = []
    rng = random.Random(2131)
    error = 0.0
    for _ in range(128):
        l00, l01, l10, l11 = [rng.uniform(-8, 8) for _ in range(4)]
        kappa = l11+l00-l10-l01
        for a, b in itertools.product((0, 1), repeat=2):
            truth = (l00, l01, l10, l11)[2*a+b]
            reconstructed = l00+a*(l10-l00)+b*(l01-l00)+a*b*kappa
            error = max(error, abs(truth-reconstructed))
    assert error < 1e-13
    checks.append(dict(name='four_state_exact_unary_interaction_decomposition',
                       tables=128, max_abs_error=error))

    # Separable p0 and p1 give a log ratio with no pair interaction.
    p0 = [.25, .75]
    p1 = [.6, .4]
    q0 = [.8, .2]
    q1 = [.3, .7]
    table = [[math.log(p1[a]*q1[b]/(p0[a]*q0[b])) for b in (0, 1)]
             for a in (0, 1)]
    kappa = table[1][1]+table[0][0]-table[1][0]-table[0][1]
    assert abs(kappa) < 1e-14
    checks.append(dict(name='separable_ratio_is_only_unary', kappa=kappa))

    native, _ = enumerate_energy((.9, .2, .7))
    recovered, _ = enumerate_energy((.9, .2, .7),
        [(0, 1, (1, 1), .5), (0, 2, (1, 0), .5)])
    assert native == (1, 0, 1) and recovered == (1, 1, 0)
    checks.append(dict(name='given_discriminating_channel_can_fix_FN_and_FP',
        native=list(native), recovered=list(recovered),
        inference_scope='synthetic correct channel; no DINO evidence'))

    native, _ = enumerate_energy((.9, .1))
    corrupted, energies = enumerate_energy((.9, .1), [(0, 1, (1, 1), 1.0)])
    assert native == (1, 0) and corrupted == (1, 1)
    advantage = energies[(1, 0)]-energies[(1, 1)]
    assert abs(advantage-(math.log(97)-math.log(9))) < 1e-13
    checks.append(dict(name='same_formula_harms_under_query_response_reversal',
        native=list(native), corrupted=list(corrupted),
        wrong_mask_energy_advantage=advantage))

    # Four perfectly coupled labels, prior P(all1)=.2. Every graph edge
    # reports the SAME binary observation, not six independent observations.
    # Given all1 the observation has probability .7; given all0 it is .3.
    prior_odds = .2/.8
    likelihood_ratio = .7/.3
    correct_odds = prior_odds*likelihood_ratio
    edges = list(itertools.combinations(range(4), 2))
    edge_weight = 1/3
    max_degree_weight = max(sum(edge_weight for e in edges if i in e)
                            for i in range(4))
    false_odds = prior_odds*likelihood_ratio**(len(edges)*edge_weight)
    correct = correct_odds/(1+correct_odds)
    repeated = false_odds/(1+false_odds)
    assert abs(correct-7/19) < 1e-14
    assert abs(repeated-49/85) < 1e-14
    assert max_degree_weight <= 1 and correct < .5 < repeated
    checks.append(dict(name='degree_normalization_does_not_remove_shared_evidence',
        nodes=4, edges=len(edges), maximum_incident_weight=max_degree_weight,
        proper_posterior=correct, repeated_edge_pseudo_posterior=repeated,
        proper_decision=0, pseudo_decision=1))

    # E(y)=sum y_i-.8 sum_{i<j} y_i*y_j. Every proper block of the all0
    # mask raises energy, but the all1 mask lowers it. Monotone local
    # convergence therefore provides no global optimality guarantee.
    energies = {y: sum(y)-.8*sum(a*b for a, b in itertools.combinations(y, 2))
                for y in itertools.product((0, 1), repeat=4)}
    base = (0, 0, 0, 0)
    proper_moves = [e for y, e in energies.items() if 0 < sum(y) < 4]
    global_best = min(energies, key=energies.get)
    assert min(proper_moves) > energies[base]
    assert global_best == (1, 1, 1, 1) and energies[global_best] < energies[base]
    checks.append(dict(name='monotone_ICM_and_small_blocks_can_miss_best_mask',
        local_solution=list(base), global_solution=list(global_best),
        minimum_proper_block_delta=min(proper_moves),
        optimum_gap=energies[base]-energies[global_best]))
    return dict(state='CPU_MATHEMATICAL_WITNESSES_PASSED', checks=checks,
                model_loaded=False, images_opened=False, GPU_used=False,
                real_task_gain_measured=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: preserve prior receipts even on repeated commands.
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(state=result['state'], checks=len(result['checks']),
                          GPU_used=False, real_task_gain_measured=False)))


if __name__ == '__main__':
    main()
