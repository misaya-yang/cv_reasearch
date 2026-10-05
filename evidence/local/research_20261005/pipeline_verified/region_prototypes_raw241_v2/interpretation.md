# Decision: no positive feasibility evidence for fixed mean-prototype recovery

Full exposed DEV241, fixed raw last-layer and cached FoRIS prototype formulas, CPU2, 33.77 seconds. This uses privileged GT geometry, not legal object proposals or a complete method. Only18 episodes contain both an eligible entirely missed GT component and an eligible zero-GT-touch fine64 component.

The raw region descriptor obtains episode AUROC0.3944 [0.1680,0.6072]; the cached FoRIS descriptor0.2556 [0.0733,0.4556]; the stored nearest-reference margin averaged over the same regions0.3272 [0.1176,0.5556]. All four fold raw means are below0.5:0.400,0.400,0.375,0.400, with4/5/4/5 eligible episodes. No cue sign is reversed.

Raw exceeds the cached prototype by+0.1389 [0.0278,0.2813], but exceeds stored NN by only+0.0673 [-0.0626,0.2250]. Relative improvement over a failing control does not establish useful positive semantic separation. Raw's interval crosses0.5; its true generalization effect is unresolved, not proven harmful. This tested fixed mean-prototype construction does not justify a semantic-region recovery component. Stop this branch on the present evidence; the result does not exhaust raw DINO features or legal region proposals.

There are170 entirely missed GT regions, of which114 have pure tokens and cover1,305,139 of1,333,939 missed pixels(97.84%). The56 excluded missed regions cover28,800 pixels;29 have area<256. There are111 stray fine64 regions, of which83 have pure tokens and cover1,526,749 of1,527,467 pixels(99.95%). The28 excluded stray regions cover718 pixels and all have area<256. Region-pooled raw AUROC is0.4477, and pixel-mass-weighted pooled raw AUROC0.3434; these are secondary descriptive statistics across episodes.

All241 references contain cov>=0.9 FG tokens: no fallback. Reference FG token count has minimum5 and median171; exact-BG count has minimum53 and median3841. Selected FG mean coverage has minimum0.9656. Poor reference purity is therefore not an observed explanation for this negative diagnostic.

239 cached episodes record positional projection;2 do not. All18 eligible paired episodes applied projection. Raw FP32 versus cached FP16 includes quantization differences, so this is not a pure isolated projection ablation. The fine64 field control's AUROC0 is conditioned on errors of its own rendered mask and is partly tautological; it does not imply that its underlying representation is weak.

Every raw/projected/packet/mask/field source hash and old/new(class,reference,query) identity matched, and all241 six-arm I/U counts reproduce the scored1200 source. Per-case truth-packed and reference-coverage array hashes are in receipt.json. Local verification independently reconstructs AUROC from region-pair comparisons, pooled/pixel-mass AUC, exact2000 RandomState(0) connected-photo bootstrap draws, paired CIs, fold means and region totals; maximum difference0. Source masks/features and GT geometry were checked remotely, not reopened locally.

The first v1 run stopped at an incorrect strict-debiased=True source contract. Its snapshot, log, partial output and failure receipt are retained separately. V2 repairs only source-flag recording and subset reporting; fixed scores, orientation and thresholds are unchanged. No GPU, encoder, image, download, new inference component, grid, deletion, commit or shared document edit was performed.
