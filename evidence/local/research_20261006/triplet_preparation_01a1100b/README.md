# Reference three-role label interactions: bounded preparation

One independent CPU candidate, **not another name or parameter combination for Pro M2**: its transferred statistic is the three-label log interaction on spatial right-angle hyperedges, rather than pairwise relations. The existing Pro M2 dictionary/top2 assignment is reused as a component. Three-body statistics and entropy inference are established ideas; this preparation claims neither novelty nor real DINO gains. No remote/GPU/encoder/download work occurred.

## Observable inference and fixed recipe

`reference_triplet_relations.predict(q,r,coverage,score)` accepts frozen query/reference features, the complete reference mask coverage, and the raw cached source score. There is no query GT/class/fold argument. The default score normalization matches Pro M2: float32 minmax with denominator floor 1e-6. `score_is_normalized=True` accepts a caller's [0,1] score without changing its values. The role dictionary is learned deterministically on reference then query tokens, with the existing Pro M2 32-role spherical Lloyd and top2 assignments.

At distances 1,2,4, sample right-angle triads `(center,vertical,horizontal)` in all four signed directions; this is twelve fixed geometric types, no feature-neighbor search in the candidate. Reference coverage gives fractional weights for all eight binary label states. Each state's joint distribution over the three roles is estimated separately and mixed, with the same weight .1, with the product of Laplace-smoothed label-conditional role marginals. Missing any label state disables that type; it does not use a query label or guess the absent statistic.

For each query triad, marginalize the role likelihoods using its top2 assignments, yielding eight positive likelihoods. Set

`J = (1/8) sum_state product(2*state-1) * log likelihood_state`.

This is the degree-three Walsh/Hadamard coefficient. By orthogonality, adding any constant, unary or pairwise function of the three label spins to these logs leaves J unchanged. Small coefficients below 1e-10 are zeroed and raw coefficients clipped to [-2,2]. On the canonical hyperedge union, repeated geometric types are averaged. One global downscale caps the largest incident absolute coefficient mass at .5 without amplifying weak raw relations.

The fields are signed, zero, absolute and no-third controls. The last control removes the degree-three coefficient **after query-role marginalization**, retaining all lower log components. Since this inference consumes only degree three, it is algebraically equivalent to zero; it is not an independent method. Removing degree three before a soft-role log-sum-exp does not guarantee its absence afterward: the included concrete counterexample produces 0.04396486 from lower-order input logs.

The standalone runner also computes the original Pro M2 signed pair-only control, using its full spatial1/2/4 + mutual20 graphs and fixed recipe. It shares the same dictionary and normalized source field; all original six-arm work is disclosed in metadata. The small injected-role test uses all three pairs of its injected triads for that pair control, rather than claiming reproduction of a full feature graph experiment.

## Mathematical checks

Let m_i=2p_i-1 and u=s-.5. Minimize

`E(p)=sum_i[p_i log p_i+(1-p_i)log(1-p_i)-u_i*p_i] - .9*sum_triads J*m_i*m_j*m_k`.

The stationary equation is `p=sigmoid(u+2*.9*messages(m))`, with messages_i=sum_incident J*m_j*m_k. Since sigmoid'<=1/4 and each incident triad has two other endpoints, the infinity-norm Lipschitz bound is `2*.9*max_incident_sum_abs_J <= .9`. The entropy Hessian diagonal is at least 4; the interaction's absolute off-diagonal row mass is at most `8*.9*max_incident_sum_abs_J <= 3.6`. Thus the Hessian is bounded below by .4 I everywhere in the open probability cube. The energy has a unique interior minimizer and the fixed-point map is globally contractive.

The returned scalar field is `t=s+2*.9*messages(m)`. It is a threshold score, not the internal p or a calibrated semantic probability. A zero interaction returns s exactly. The solver uses float64, tolerance 1e-7, maximum 250 iterations, raises on nonconvergence and reports probability residual/(1-contraction). Complete masks use existing float32 bilinear expansion to 1024, align_corners=False, threshold >.5. Numerical optimization guarantees do not establish correct semantic labels.

`checks.json` records an independent L-BFGS-B energy minimization: maximum probability difference 3.53e-9; observed minimum Hessian eigenvalue 3.6549 against the analytic .4 lower bound; fixed-point residual 3.45e-9. The full legal 64x64 geometry has 45,652 query triads, takes four fixed-point updates and has residual 1.82e-9. All zero/no-third and missing-state baseline checks pass.

## Synthetic evidence and counterexample

An injected reference has 384 tokens / 128 disjoint triads. It contains all eight label states and all eight position-role bit states, with multiplicities 3 versus 1 according to their parity agreement. All label-conditioned pair marginals agree, while the joint triple relation differs. The positive query triad has raw J=.5465776, normalized to .5. For base `[.499,.8,.8]`, signed output is `[.519119,.801286,.801286]`; the pair-only Pro M2 control retains the exact base. This is an algebraic witness on observable synthetic roles, not a DINO segmentation measurement.

For the opposite role parity, base `[.501,.8,.8]` becomes `[.480881,.801286,.801286]`; absolute gives `[.521137,.801423,.801423]`. If the first token is truly BG, the signed deletion is useful. If it is truly FG, exactly the same observable input wrongly deletes that target. With the synthetic full query designated all-FG, baseline1024 TP is 1,048,576, and signed deletes 195,584 true pixels. The designated GT is used only for this post-inference counterexample. Numerical stability is not semantic confidence.

`synthetic_outputs.npz` stores these fields and complete masks. `legal_inputs.npz` is a separate synthetic feature packet exercising the default feature dictionary and spatial geometry; its random labels do not measure gain.

## Deployment and verification

```sh
python3 scripts/run_reference_triplet_relations.py --inputs /bound/episode_*.npz --out /new/triplet_run --workers 2
python3 scripts/run_reference_triplet_relations.py --verify /new/triplet_run
```

Each input provides only required `q,r,cov,score`: q/r4096x1024 and cov/score64x64. Extra keys are not opened. **This runner consumes raw score, unlike the hull runner's cached MEAN base input.** Worker threads are fixed to one; repeated occurrences are preserved; complete predictions, continuous fields and receipts are written separately. A new output directory is required. The runner seals only after all occurrences finish and code hashes remain unchanged, then verifies contract/prediction/field/receipt hashes and packed1024 structure. Real GT scoring belongs in a separate subsequent operation.

`two_worker_run/` and `verification.json` demonstrate two distinct worker PIDs, repeated inputs with identical complete predictions, 0.937/1.138 local synthetic seconds per occurrence including the full Pro M2 control, ignored object-valued GT poison key, and detection of a corrupted field in a temporary copy. The existing Pro control emits macOS matrix-multiplication floating-flag warnings; all actual fields are finite and solver residual checks pass. Its shared source was preserved. These runtimes are not estimates of real dataset throughput.

Reproduce bounded checks with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 evidence/local/research_20261006/triplet_preparation_01a1100b/check.py
```

No shared prepared entry, previous hull code, PLAN or STATUS was changed.

## Shared prepared integration

Added `reference_triplet_relations` to `AVAILABLE_METHODS` and both prepared CLI choices, while preserving the original nine `METHODS` and default four. The shared branch uses the original packet score, enables the original pair-only Pro M2 control, and preserves zero/absolute/no-third controls and complete scalar metadata. Configuration stores Config/asdict and hashes both the triplet source and its Pro M2 dependency. `Config.__post_init__` restores the distances tuple after JSON serialization; the fixed-recipe check accepts the resulting roundtrip.

`shared_interface_checks.json` and `check_shared_interface.py` record a small synthetic hook test, actual config generation stopped before worker startup, unchanged defaults, all packed1024 outputs, receipt serialization and both dependency hashes. No full shared suite, server operation or GT read was performed. The previous standalone sealed run retains the historical pre-integration source hashes; the new JSON metadata normalization changes no inference values. Shared source changes are now complete and released to the primary agent.
