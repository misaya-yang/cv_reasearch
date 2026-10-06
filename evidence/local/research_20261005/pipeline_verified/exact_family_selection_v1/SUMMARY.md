# Exact existing-mask A/B family selection

PUBLIC4000 retains all sampled draws and the natural repeated episode identity. The estimate is class-summed intersection/union over 80 classes. Paired intervals use 2,000 connected-photograph bootstrap draws from RandomState(0).

| Search | Recipes actually enumerated | Global DEV optimum | Train-three/held-one recipe readout |
|---|---:|---:|---:|
| Original 11 complete families | 11,534,336 | 63.062305 | not separately rerun |
| Strict 12 complete families, including scalar graft | 50,331,648 | 63.064818 | 62.964699 |
| Extended 13 families, including fixed fold-temperature size-cut control | 218,103,808 | 63.238541 | 63.102199 |

Let P be conservative_delete, V be fine.rcg16.control, F be fine.rcg64, G be the fixed scalar graft, and S be the fixed fold-temperature size-cut control. The original 11 best is (P & V) | (~P & F). The strict 12 best is (V & G) | (~V & F). The extended 13 best is (P & V) | (~P & S). Their minimum outer complete-mask producer-count proxy is 3. Composite rows retain several upstream dependencies; this is not runtime or atomic-component count.

The extended best is native +2.306800 [1.886799,2.768258], MEAN +.725569 [.400378,1.094699], fine16 +.529441 [.270524,.830262], but strongest complete S +.089340 [-.086422,.299029]. Extended heldfold is native +2.170458 [1.730749,2.628629] and S -.047002 [-.248330,.149315]. Strict heldfold is native +2.032958 [1.657151,2.387729], MEAN +.451727 [.233044,.659919], fine16 +.255599 [.094337,.399785], but strongest strict scalar graft +.120162 [-.103035,.319822]. Global strict best vs graft is +.220281 [.057350,.366154].

S uses the frozen six thresholds fitted on all fresh 600 GT, and existing own-fold temperatures (.07 on folds0/3, .15 on folds1/2). It is a separately named control, not the supplied uniform-tau15 primary. Those 600 cases are included in PUBLIC4000. Holding a fold out of recipe selection does not remove producer-level label exposure.

All actual Pareto finalist masks were constructed without GT, fully sealed, then counted against GT. Every per-draw I/U matches the membership-histogram prediction exactly. Original six-arm per-draw I/U matches the original scored PUBLIC4000 counts exactly. All single producers are algebraically contained under every allowed origin. No epsilon is used to choose score ties.

Global optima use full 4000 GT for development selection, followed by one fixed deployable recipe. Conditional paired intervals do not repeat or correct recipe selection. These are reused development data; the native gain interval extends below 2, and superiority over S is unresolved.

DEV241 fixes unchanged rawNN origin O and the PUBLIC4000-aligned shared family subset. This is explicitly not an exhaustive search of the entire 187-family DEV library. Strict 13 sources: 12 operators/side and 16,777,216 recipes. Extended 14 sources: 13 operators/side and 67,108,864 recipes. Both global optima are 62.773896 with (~O & P) | (O & V & Astra), outer producer proxy 4. Strict heldfold is 61.284097: native +2.209272 [-.023515,3.778239], MEAN +.603980 [-1.452451,1.788880], fine16 -.216565 [-2.276171,.993915]. Extended heldfold is also 61.284097: every per-draw I/U and sealed NPZ hash is identical to strict, and all four selected recipes omit S.

The DEV-fixed recipe requires actual rawNN O. The complete PUBLIC4000 O archive is absent. The expression depends on O on 34,591,588 pixels across 3,953 draws, so its application is explicitly incomplete without a substitute origin. See dev241_to_public4000_application.json.

Per-run reports, exact optimizations, recipes, seals, counts, scopes and edit-accounting files are retained. Four signed per-class edit terms sum exactly to native gain; these are accounting identities rather than causal explanations.
