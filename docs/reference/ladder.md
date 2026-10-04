# In-context segmentation methods: evidence ladder

Updated 2026-10-04 from six independent first-party paper-and-code checks supplied for this task. This is a map of what each paper actually changes and what its own ablations support; it is not a causal decomposition across papers.

## Reading this ladder

All scores below are published COCO-20i 1-shot class-mIoU unless a row says otherwise. They come from different protocols, backbones, training sets, input information and implementations. A difference between two papers is therefore descriptive, not an estimate of the causal effect of the intervening method. Module deltas are interpretable only within the corresponding paper and ablation setup. In particular, neither the 76.1 reported for SegIC nor the 77.8 reported for UNICL-SAM is a strict held-out-category result: both are in-domain results. Running one checkpoint across four folds does not make its training category-disjoint.

The current project objective is an independent method that beats FoRIS on COCO-20i 1-shot; correcting FoRIS outputs is outside that objective. Current user-reported measurements supersede older repository prose: FoRIS CONFIRM600 is 59.78; the only +3.54 result is a FoRIS correction head trained with base-class labels; today's unlabeled learning attempts lost 7–28 points; transfer was +0.97 on PASCAL-Part with CI crossing zero and +0.08 on PACO-Part with CI crossing zero. Those results do not establish a new independent method. The paper ladder below is prior art and constraints on novelty, not evidence for a proposed method.

All differences are mIoU percentage points. No cross-paper delta is attributed to a single component without a same-setting ablation. Speculation is labeled explicitly.

## Methods × five questions

| Method and reported score | a. What the reference mask becomes and how it acts on the query | b. Knowledge beyond one reference image and training | c. What the ablations establish | d. What is evidenced relative to the preceding method | e. Reported failures and limits |
|---|---|---|---|---|---|
| [INSID3](https://arxiv.org/html/2603.28480v1), 57.6 | Masked reference features form a foreground prototype; the mask also supplies foreground labels for query-to-reference nearest neighbours. The method clusters query features, selects the best matching seed, then aggregates regions using cross-image and within-query similarity. Official code: [models/insid3.py at 0c165a1](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py) (`predict_mask`, `_locate_candidates`, `_seed_and_aggregate`, `_finalize_mask`); [utils/clustering.py](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/clustering.py) (`agglomerative_clustering`). | Frozen DINOv3-L at 1024², inheriting large-scale image pretraining (LVD-1689M and other sources). No SAM, language input or task-trained decoder. Supplement B nevertheless selects τ, α and SVD dimension by three-fold cross-validation on the COCO training split. “No weight updates” therefore does not mean “no use of training data to select settings.” | Table 3: fine clustering without aggregation 42.8 → cross-image aggregation 54.6 → cross- plus within-query aggregation 57.6; aggregation group +14.8, last addition +3.0. Table 5 default debiasing +3.1. Table 8 DINOv2→DINOv3 +12.5, a backbone change. | A mechanism-level reference: strong pretrained visual features plus matching-aware seed localization and region recovery. The interpretation “locate a matching part, then recover the target extent” is a description of the mechanism, not evidence that one image provides all class knowledge. | Supplement F: single concept, complete reference mask required, same-class instances merge, and performance is bounded by the backbone representation. In a constructed no-target evaluation, empty-mask accuracy is 85%. Worst COCO category was not found. |
| [FoRIS](https://arxiv.org/html/2609.03384v1), 60.9 published | The mask defines foreground/background samples. FoRIS forms multiple foreground and hard-background prototypes, a contrastive score, reverse-match voting and a query-region prior, followed by semantic-conflict and region-purity corrections. Official code: [models/foris.py at 1aa02a1](https://github.com/Xi-Mu-Yu/FoRIS/blob/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e/models/foris.py): `_reference_contrastive_prototypes`, `_part2_stage1_feature_gating`, `_part2_stage2_contrastive_score`, `_locate_candidates`, `_build_seed_cluster_prior`, `_semantic_disagreement_penalty`, `_semantic_cluster_reweight_map`. | Frozen DINOv3-L at 1024² with CRF; no newly trained base-class segmenter, SAM or language branch. Uses pretrained visual knowledge, RGB and spatial priors in addition to the reference mask. | Table 2 largest single added module is foreground localization (FL): 43.7→51.9, +8.2. Full combination without foreground consolidation (FC) is 58.5; adding FC gives 60.9, +2.4. Table 3 adaptive versus fixed debiasing is 0 on COCO; candidate densification +0.8. | Table 1 reports 57.6→60.9 (+3.3) against INSID3 with the same DINOv3 and author-claimed matching protocol. But the FoRIS ablation starts at 43.7, not at INSID3. Assigning the +3.3 among FoRIS modules has no direct evidence. | Fine vessels and roads are difficult; the authors attribute this to patch mixtures of foreground/background and dispersed features along thin structures. Fundus 1-shot 19.7 and DeepGlobe-18 13.3; more references help little. Worst COCO category was not found. |
| [FROST](https://arxiv.org/html/2606.31136v1), COCO 48.4 | The mask splits support tokens into foreground/background for within-class whitening and two kernel-density estimates. Query tokens are classified by foreground/background log-density ratio, with candidate gating and bilateral propagation. Official code: [frost/model.py at b9ece69](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/model.py) (`_shrinkage_whiten`, `_locate_candidates`, `predict_mask`); [frost/density.py](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/density.py) (`estimate_bandwidth`, `kde_log_posterior`, `density_ratio_predict`). | Frozen DINOv3-L; no SAM, language or base-class segmentation training. Retains foreground/background density anchors and uses support flips, support-derived statistics and fixed spatial rules; estimating statistics is not neural-network training. | Table 4 averages 17 remote-sensing datasets: full 48.3; removing debiasing and whitening together −4.6; removing debiasing alone −2.1 (largest). This is not a COCO ablation. | This is not a step upward from INSID3. Table 3 gives COCO 1-shot 48.4 versus INSID3 56.5 (−8.1); at 10-shot FROST is 64.3 versus 64.0. Its stronger evidence is remote sensing and more references. | Authors note that one support image may poorly represent the query distribution. Equal priors and multimodal density work better in remote sensing; object-centred COCO remains weaker. One class per run. Worst COCO category was not found. |
| [UINO-FSS](https://arxiv.org/html/2504.15669v4), 64.5 (folds 62.2/66.0/65.5/64.4) | One mask path pools a foreground prototype and creates a sparse visual prompt. A second path subtracts background correlation from multi-layer support–query foreground correlation to form a 4D correlation volume; MHCM turns it into a dense query prior, which is passed with adapted query features to a SAM-initialized decoder. Official code/functions were not found; this description is from paper Sections IV-D/E. | Frozen DINOv2-B, but the method is trained: adapter distillation against SAM-H features on 1% of SA-1B, then prompt generator and SAM-initialized decoder trained with base-class pixel masks. Distillation does not directly use masks as its target, but its teacher inherits SAM segmentation supervision. No class-text input. | Table VII simplified model: without/with distillation 52.8/59.6, +6.8. Table V: SVP +1.4, MHCM +2.7, CE +2.2. Replacing CP4D convolution with MHCM is only 64.0→64.5, +0.5. | No direct controlled comparison to FoRIS. The 3.6-point cross-paper difference cannot be assigned to SAM distillation, base supervision or learned correlation reading. | No separate limitations section, systematic full-model failure cases or classwise IoU found. Figure 5 mainly shows errors after removing CE/MHCM, not worst cases of the complete method. |
| [FSS-SAM3](https://arxiv.org/html/2604.05433v1), paper body 66.1; main table 66.6 | Core design is one instance box plus a shared canvas: concatenate support/query, give an in-box visual exemplar to SAM3, then crop out the query prediction. COCO code reads an instance bbox from instance JSON and does not return the support mask. [data/dataset_tool.py](https://github.com/WongKinYiu/FSS-SAM3/blob/main/data/dataset_tool.py) (`COCOFSSDataset._load_ref`); [model/evaluate.py](https://github.com/WongKinYiu/FSS-SAM3/blob/main/model/evaluate.py) (`create_input`, `get_norm_box`, `run_episode`); [geometry_encoders.py](https://github.com/WongKinYiu/FSS-SAM3/blob/main/sam3/model/geometry_encoders.py) (`SequenceGeometryEncoder._encode_boxes`). | FSS stage frozen, no new training, but SAM3 already learned language–vision concept segmentation from large phrase–mask corpora. Visual-only does not give the target class name; +text does. COCO uses instance boxes; PASCAL also reads instance annotations to select the largest object. The public implementation is not accurately described as consuming only one complete semantic mask. | Table 5 visual 66.6, text 74.9, both 75.8: adding class name +9.2; adding visual input when class name is already known +0.9. Table 4 visual layout range 64.8–66.6; bottom layout support fraction 0.5→0.6 +0.1. | No direct controlled comparison to UINO-FSS. The change from explicit cross-image correlation to invoking SAM3's pretrained within-image concept segmentation is visible in the mechanism; assigning the 1.6–2.1 points to it has no direct evidence. | Table 3 adding automatic negative prompts reduces score, up to −21.9; Figure 3 shows all-background collapse. Pseudo-video propagation and selecting classes by direct text–image embedding also failed, without reported numbers. Worst class was not found. |
| [CG-ICS](https://arxiv.org/pdf/2606.28149v2), 72.3 | An MLLM names concepts for the masked object. SAM3 segments each concept on the reference and scores IoU (RF), then assigns query presence (QM); RF×QM guides concept search. A visual mosaic separately yields a query box; final segmentation uses concept plus box. [tree_search.py](https://github.com/Kakarot1103/CG-ICS/blob/main/model/tree_search.py) (`TreeSearcher.select`, `_score_single_ref_raw`); [main.py](https://github.com/Kakarot1103/CG-ICS/blob/main/main.py) (`evalute`); [sam3.py](https://github.com/Kakarot1103/CG-ICS/blob/main/model/sam3.py) (`Sam3Segmenter.infer`). | Frozen Qwen3-VL-4B-Instruct and SAM3; no CG-ICS parameters trained. It does not take the user's GT class name, but it uses language-model knowledge. Qwen3-VL grounding pretraining explicitly includes COCO, Objects365, OpenImages and RefCOCO-family data. This alone does not prove test-image leakage. | Table 4 is a reference-selection robustness protocol (500 queries/fold, 50 references/query), while main-table 72.3 is Standard ICS (1000 queries/fold). In Table 4: single-concept system 67.6→full 72.1; RF +2.8 on candidates; QM after RF +0.8; search +1.1; visual +0.9. Largest one-switch recovery is adding RF to QM-only: 62.8→70.1, +7.3, a weak baseline recovery. | No direct controlled comparison to FSS-SAM3. Added automatic naming, reference-mask concept validation, query-presence scoring and search. The 72.3 score cannot be decomposed using the different-protocol 72.1 ablation as a causal account of the gap from 66.1/66.6. | Authors acknowledge MLLM hallucinations, overly fine concepts and unreliable multi-image reasoning. Table 3 largest domain perturbation drop is 5.2. No dedicated CG-ICS failure panel or worst class found; failures in Figure 5 are mainly GF-SAM and must not be attributed to CG-ICS. |
| [SegIC](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05519.pdf), 76.1 in-domain | Pools the reference mask into a visual vector that matches query features to make a coarse mask and points. It also encodes class/task text as a meta prompt for a trained decoder. [model/segic.py](https://github.com/MengLcool/SEGIC/blob/master/model/segic.py): `extract_inst_feat`, `maks_decoding` (source spelling), `extract_dift_feature`, `forward`. Default semantic text is “a photo of a [class]”. | DINOv2 frozen, but about 5M decoder-related parameters are trained; training uses COCO semantic/instance masks, ADE20K, LVIS and CLIP language knowledge. The 76.1 setup trained on the evaluation categories. Strict four-fold table 2 is 53.6; adding ADE/LVIS/FSS-1000 is 62.3, with external category overlap not audited. | Table 4 uses DINOv2-B and a smaller sampling budget: no prompt 48.9; geometry only 68.8; visual only 71.0; meta only 73.7; all 74.6. Largest single prompt gap is meta +24.8; all prompts over meta-only +0.9. These are not a module ledger for the 76.1 main model. | No direct controlled comparison to CG-ICS. The 76.1 includes supervised training, class-name prompting and an in-domain setting; it is not an extra 3.8 points from better matching on a held-out class. | Meta-only scores 63.7 on FSS-1000 and 21.6 on DAVIS17, weaker out of domain than all prompts. Worst COCO class or dedicated failure figure was not found; success examples do not establish failures. |
| [UNICL-SAM](https://openaccess.thecvf.com/content/CVPR2025/html/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.html), 77.8 in-domain | The mask selects foreground; a graph, uncertainty estimation/refinement and clustering discover part prototypes. Part prototypes, global pooling, original features and mask form conditions read by 50 learned queries to produce sparse SAM embeddings. [unicl_sam/model/model.py](https://github.com/ImmortalSdm/UNICL-SAM/blob/main/unicl_sam/model/model.py): `get_graph`, `get_uncertainty_estimation`, `get_uncertainty_refinement`, `get_cluster_feats`, `get_support_embeddings`, `generate_img`. | DINOv2-L and SAM-H frozen, but MFA/UGGN/QPG and related modules have about 55M trained parameters. Main paper trains on COCO plus ADE semantic masks. Inference does not appear to use class text. SAM inherits 11M images and over 1B masks. The 77.8 result is in-domain; running a single checkpoint on four folds does not make four held-out training runs. | Small-data Table 2: all modules 71.77→74.64, +2.87. UPM alone +1.22; adding UPM after Part +2.20. Supplement C3: more COCO data 74.6→77.3, then ADE 77.8. Supplement C4: with full COCO and MFA, adding UGGN changes COCO 77.4→77.3 (−0.1). | No direct controlled SegIC comparison; the main table does not include SegIC. The 1.7-point paper difference cannot be attributed to uncertainty or part prototypes; architecture, training, data and prompt source differ. | Supplement D5: replacing support mask with box gives 79.8→69.0; Gaussian noise gives 71.0. Supplement E.3 notes semantic ambiguity introduced by boxes. Worst COCO class not found. |

## Pretraining and supervision scale reported by the sources

- [DINOv3 paper §3.1](https://arxiv.org/html/2508.10104v1) and [official weights table](https://github.com/facebookresearch/dinov3#pretrained-models): LVD-1689M is about 1.689B images; other sources are mixed in, so this is not a deduplicated total for the full training mixture.
- [DINOv2 appendix](https://arxiv.org/html/2304.07193v2): LVD-142M is about 142M images; this is an image count, not a segmentation-label count.
- [SAM paper](https://arxiv.org/abs/2304.02643): SA-1B is about 11M images and more than 1B masks. UINO's exact sample count for 1% of the images was not found.
- [SAM3 paper, Table 23 and Appendices C/E](https://arxiv.org/html/2511.16719v1): PE 5.4B image–text pairs; SA-Co/HQ 5.2M images and 52.3M phrase–mask pairs; EXT 9.3M images and 70.5M pairs; SYN 39.4M images and 1.4B pairs. Automatic and synthetic labels are included; these should not all be called human-annotated fine masks.
- [Qwen3-VL paper §3.2.4](https://arxiv.org/html/2511.21631v2#S3.SS2.SSS4): grounding pretraining sources include COCO and the datasets listed in the CG-ICS row.
- SegIC lists roughly 83K COCO training images, 20K ADE20K and 100K LVIS; UNICL lists 118K COCO training images plus ADE. Images can overlap, and neither paper gives a deduplicated total number of independent masks that can be summed directly.

## Within-paper ablations and source table numbers

The tables below preserve the reported COCO 1-shot columns where applicable. Check marks are represented as 1 and blanks as 0; numeric values are transcribed unchanged. An ablation explains only its own experimental setting.

### 1. INSID3 — [Tables 3, 5, 8](https://arxiv.org/pdf/2603.28480v1)

| Table 3: clustering / aggregation | COCO | PASCAL-Part |
|---|---:|---:|
| No clustering; threshold similarity graph @0.55 | 44.2 | 35.4 |
| Coarse clustering τ=0.5, no aggregation | 50.6 | 31.1 |
| Fine clustering τ=0.6, no aggregation | 42.8 | 36.2 |
| Fine clustering + cross aggregation | 54.6 | 48.5 |
| Fine clustering + self/cross aggregation | 57.6 | 50.5 |

- Table 5 debiasing, rows no / 4-view / 12-view / 1-noise SVD / 5-noise / 10-noise: 54.5 / 55.7 / 56.5 / 57.6 / 57.7 / 57.7.
- Table 8 backbone, rows DINOv3 / DINOv2 / Franca / PE / SD: 57.6 / 45.1 / 39.2 / 48.4 / 33.2.
- Largest same-granularity algorithm component group is aggregation +14.8; self versus cross alone is +3.0. The backbone gap DINOv3–SD is 24.4 and is not an inference-module gain.

### 2. FoRIS — [Tables 2, 3](https://arxiv.org/html/2609.03384v1#S4.SS2)

APD = adaptive position debiasing; FR = foreground refinement; FL = foreground localization; FC = foreground consolidation. Every row includes Base.

| APD | FR | FL | FC | COCO |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 43.7 |
| 1 | 0 | 0 | 0 | 46.9 |
| 0 | 1 | 0 | 0 | 48.3 |
| 0 | 0 | 1 | 0 | 51.9 |
| 1 | 1 | 0 | 0 | 52.2 |
| 1 | 0 | 1 | 0 | 52.9 |
| 0 | 1 | 1 | 0 | 56.2 |
| 1 | 1 | 1 | 0 | 58.5 |
| 0 | 1 | 1 | 1 | 58.0 |
| 1 | 1 | 1 | 1 | 60.9 |

- Table 3a fixed debiasing OD / adaptive AD: 60.9 / 60.9.
- Table 3b without / with densification: 60.1 / 60.9.
- Table 3d no FC / semantic disagreement penalty (SDP) / semantic reweighting (SR) / both: 58.5 / 59.9 / 60.2 / 60.9.
- Table 3c position/RGB ablation has no COCO column; it cannot allocate COCO gains.
- Largest single-module addition is FL +8.2; full combination adds +17.2 over its own 43.7 baseline. It is not +17.2 over INSID3.

### 2a. FROST — remote-sensing ablations only

[Table 4](https://arxiv.org/html/2606.31136v1#S4.SS4), mean 1-shot over 17 remote-sensing datasets:

| Variant | mIoU | Change from full |
|---|---:|---:|
| Full | 48.3 | — |
| Remove bilateral propagation | 48.0 | −0.3 |
| Remove whitening | 47.5 | −0.8 |
| Remove support flip | 47.2 | −1.1 |
| Remove position debiasing | 46.2 | −2.1 |
| Remove feature refinement (debiasing + whitening) | 43.7 | −4.6 |

No COCO-specific FROST ablation was found. Appendix Table 9 backbone results are also remote-sensing results: DINOv3-L 49.7, DINOv3-7B 50.3, DINOv2-L 43.6, DINOv3-ConvNeXt-L 42.7, DINOv3-SAT-7B 35.9, DINOv3-SAT-L 35.4. These do not explain COCO differences.

### 3. UINO-FSS — [v4 Tables V–VIII](https://arxiv.org/pdf/2504.15669v4#page=9)

BA = bottleneck adapter; PCS = prototype cosine similarity; SVP = semantic-aware visual prompt; MHCM = Mamba-hypercorrelation module; CE = foreground-minus-background contrast enhancement.

| Table V configuration | COCO | Increment from preceding row |
|---|---:|---:|
| BA + PCS | 58.2 | — |
| + SVP | 59.6 | +1.4 |
| + MHCM | 62.3 | +2.7 |
| + CE | 64.5 | +2.2 |

| Other table | Configuration | COCO |
|---|---|---:|
| VI | CP4D convolution replacing MHCM | 64.0 |
| VI | MHCM | 64.5 |
| VII, simplified Comp | No distillation | 52.8 |
| VII, simplified Comp | Distillation | 59.6 |
| VIII, simplified Comp | DINOv2-S | 54.3 |
| VIII, simplified Comp | DINOv2-B | 59.6 |
| VIII, simplified Comp | DINOv2-L | 60.7 |
| IV, simplified Comp* | Frozen decoder | 57.2 |
| IV, simplified Comp | Trained decoder | 59.6 |

Largest relevant training comparison is +6.8 for distillation in the simplified model; it cannot be added to 64.5 for the full model. MHCM is +2.7 over the preceding no-MHCM row but only +0.5 over the alternate 4D reader.

### 4. FSS-SAM3 — [Tables 3–7](https://arxiv.org/html/2604.05433v1#S5)

| Table | Row / configuration | COCO 1-shot mIoU | FB-IoU |
|---|---|---:|---:|
| 3, negative prompts | 0 | 66.6 | 82.9 |
| 3 | ≤1 | 54.8 | 77.2 |
| 3 | ≤3 | 46.6 | 73.1 |
| 3 | ≤5 | 44.7 | 72.4 |
| 4, layout | ARP, top padding | 66.0 | 82.3 |
| 4 | FR horizontal, left, 0.5 | 64.8 | 81.8 |
| 4 | FR vertical, bottom, 0.5 | 66.5 | 82.8 |
| 4 | FR vertical, bottom, 0.6 | 66.6 | 82.9 |
| 4 | FR vertical, top, 0.5 | 66.1 | 82.9 |
| 4 | FR vertical, top, 0.6 | 66.0 | 82.7 |
| 5, prompt modality | Visual | 66.6 | 82.9 |
| 5 | Class-name text | 74.9 | 86.3 |
| 5 | Both | 75.8 | 87.1 |
| 6, visual + selected text | GT class name | 75.8 | 87.1 |
| 6 | Fold-level candidate class name | 72.4 | 86.1 |
| 6 | Full-dataset candidate class name | 67.8 | 83.2 |

Largest positive prompt-modality gap is visual→both +9.2, which adds class-name information. Negative prompts reduce by up to −21.9. The automatic negative-example script also changes how the positive prompt is produced, so this is not a strict one-variable test of adding any accurate negative prompt.

Table 7 uses a separate multi-class MSCOCO test, not the standard main-table protocol: Positive Only 53.2; Semantic Distractors 26.0; Background Negatives 22.1; Multiple Negatives ≤10, 9.7. Its largest drop must not be moved onto the standard 66.6 setup.

### 5. CG-ICS — [v2 Table 4](https://arxiv.org/pdf/2606.28149v2#page=9)

This is the reference-selection robustness protocol: 500 queries/fold and 50 references/query. The 72.3 main result uses Standard ICS with 1000 queries/fold.

| Candidates | RF | QM | Search | Visual | mIoU | Std | CV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 67.6 | 15.1 | 22.3% |
| 1 | 0 | 0 | 0 | 0 | 66.5 | 15.3 | 23.0% |
| 1 | 1 | 0 | 0 | 0 | 69.3 | 13.9 | 20.5% |
| 1 | 0 | 1 | 0 | 0 | 62.8 | 18.2 | 29.0% |
| 1 | 1 | 1 | 0 | 0 | 70.1 | 11.3 | 16.1% |
| 1 | 1 | 1 | 1 | 0 | 71.2 | 10.5 | 14.7% |
| 1 | 1 | 1 | 1 | 1 | 72.1 | 9.3 | 12.9% |

The first row already uses Qwen-generated single-concept text with SAM3; it is not a no-foundation-model-knowledge baseline. Generating more candidates alone is −1.1; query presence alone is weaker. RF improves concept selection in these ablations. The interpretation that RF mainly corrects concepts present in the query but outside the intended reference meaning is a mechanism hypothesis, not an error-count result: the paper does not report that error type's frequency. This table does not causally explain the entire difference from FSS-SAM3.

### 6. SegIC — [main Tables 2–6](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05519.pdf#page=13)

The prompt ablation uses DINOv2-B and a smaller sampling budget; its full row is 74.6, not the L-backbone 76.1 main result.

| Table 4 prompt | COCO | FSS-1000 | DAVIS17 |
|---|---:|---:|---:|
| Geometry only | 68.8 | 87.3 | 65.8 |
| Visual only | 71.0 | 85.1 | 67.6 |
| Meta only | 73.7 | 63.7 | 21.6 |
| No prompts | 48.9 | 58.1 | 21.1 |
| All | 74.6 | 87.4 | 68.4 |

| Table 5 training augmentation: Reversion / Negative | COCO |
|---|---:|
| 0 / 0 | 70.0 |
| 0 / 1 | 72.9 |
| 1 / 0 | 73.1 |
| 1 / 1 | 74.6 |

| Table 6 training data | COCO | FSS-1000 | DAVIS17 |
|---|---:|---:|---:|
| COCO semantic | 76.3 | 80.9 | 45.1 |
| + ADE semantic | 75.6 | 82.3 | 40.1 |
| COCO semantic + COCO instance | 75.7 | 83.5 | 70.1 |
| COCO semantic + ADE + COCO instance + LVIS | 74.6 | 87.4 | 68.4 |

Thus this table does not support “more segmentation data always raises COCO”; extra instance data helps video more. Table 3 backbone COCO values in row order: DINOv2-B 74.6, L 75.4, G 75.7; DINOv1-B 53.5; SAM-B 59.8; SD2.1 66.6; OpenCLIP-ConvNeXt-B 65.1; CLIP-ViT-B 15.3; MAE-B 12.0. Each changes model and pretraining, not one isolated component.

Table 2 protocol correction:

| SegIC setup | Fold0 | Fold1 | Fold2 | Fold3 | Mean |
|---|---:|---:|---:|---:|---:|
| Strict four-fold training/test category isolation | 55.8 | 54.7 | 52.4 | 51.4 | 53.6 |
| Also combines ADE20K, LVIS, FSS-1000 | 62.3 | 62.5 | 63.3 | 60.9 | 62.3 |

### 7. UNICL-SAM — [main Tables 2–6](https://openaccess.thecvf.com/content/CVPR2025/papers/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.pdf), [supplement C3/C4](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_CVPR_2025_supplemental.pdf)

Table 2 uses about 34K COCO images and about 10% of masks. Checkmarks were checked against the PDF table image to avoid extraction-column errors. MFA = multi-scale feature adaptation; Part = part prototypes; UPM = uncertainty propagation/modeling.

| MFA | Part | UPM | mIoU | MAE |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 71.77 | .039 |
| 0 | 0 | 1 | 72.99 | .037 |
| 0 | 1 | 0 | 72.41 | .038 |
| 1 | 0 | 1 | 73.28 | .036 |
| 1 | 1 | 0 | 73.78 | .036 |
| 0 | 1 | 1 | 74.61 | .034 |
| 1 | 1 | 1 | 74.64 | .034 |

Other main-paper tables, values are mIoU / MAE:

- Table 3 image enhancement / feature refinement: 00 = 73.78/.036; 10 = 73.82/.036; 01 = 73.99/.035; 11 = 74.64/.034.
- Table 4 GCN / UGGN / guided graph enhancement: 100 = 72.12/.038; 010 = 72.41/.038; 011 = 73.82/.036.
- Table 5 top-ratio masking = 43.73/.117; random masking = 73.44/.037; learnable gate = 74.64/.034.
- Table 6 cluster count 5 = 73.36/.037; 10 = 73.82/.036; 15 = 72.50/.039; 20 = 72.38/.040.

Table 5's +30.91 when replacing top-ratio masking is replacement of a very poor alternative, not the contribution of uncertainty. The largest ordinary component contrast in Table 2 is adding UPM when Part is already present, +2.20; full combination versus its baseline is +2.87.

Supplement C3 training data:

| Training data | COCO | FSS-1000 | LVIS |
|---|---:|---:|---:|
| COCO-s subset | 74.6 | 81.7 | 29.9 |
| Full COCO | 77.3 | 82.4 | 32.0 |
| + ADE20K | 77.8 | 84.0 | 34.1 |

Supplement C4 MFA / UGGN:

| MFA | UGGN | Trainable parameters | COCO | FSS-1000 | LVIS |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 55.4M | 77.3 | 82.4 | 32.0 |
| 1 | 0 | 52.5M | 77.4 | 81.2 | 29.8 |
| 0 | 1 | 9.8M | 77.0 | 81.3 | 30.2 |
| 0 | 0 | 7.8M | 75.4 | 80.7 | 28.9 |

On full COCO, adding UGGN while holding MFA gives COCO −0.1, FSS-1000 +1.2 and LVIS +2.2. Its value cannot be summarized as an unconditional COCO gain.

## What the ladder does and does not say

- INSID3→FoRIS has an author-reported same-protocol overall comparison of +3.3, but FoRIS's own ablation does not start from INSID3; no per-module allocation of that cross-paper gap is supported.
- UINO-FSS adds SAM-feature distillation, base-class mask training and learned 4D correlation reading. Its largest explicit simplified-model training contrast is +6.8 for distillation; that is not an additive explanation of the complete 64.5 score.
- FSS-SAM3 invokes SAM3's pretrained concept segmentation through a single-instance box and shared canvas. Its visual score differs from UINO by 2.1, but no controlled test assigns this difference to those mechanisms. The within-paper class-name input comparison is direct: visual 66.6→both 75.8, +9.2.
- CG-ICS adds automatic concept naming, reference-mask verification, query-presence scores and search. Its own ablation supports those steps under its setup; it does not show that they account for the full difference from FSS-SAM3.
- SegIC's 76.1 changes training and evaluation conditions and explicitly uses class names. Its meta-only 73.7, all-prompts 74.6 and strict-fold 53.6 results prevent treating 76.1 as a single-image matching gain on held-out categories.
- UNICL-SAM adds trained prompt generation, parts/uncertainty and SAM priors. Its own data expansion 74.6→77.3→77.8 is controlled within that paper; the 1.7-point difference from SegIC is not attributed to a component.
- If “one reference image, frozen backbone, no training” means no task-specific training at all, UINO's distillation/base-class training, SegIC's decoder training and UNICL's roughly 55M trained parameters do not satisfy it. This is a constraint conflict, not proof that training-free methods can never reach the same score.
- Frozen backbones can still carry learned SAM or MLLM knowledge. FSS-SAM3 and CG-ICS demonstrate that. If the project additionally requires the same DINOv3 only and no extra models, those external knowledge sources would not meet that condition.
- “Available but unused” is not supported here: foreground/background statistics, region propagation, spatial mosaics and concept verification all appear in these methods. How a new transfer of them would score is unknown. Any mechanism explanation without a matching ablation is speculation.

## Missing evidence, numeric conflicts and paper–code differences

### Not found

- UINO-FSS official repository, functions or run configuration. “Not found in this check” does not mean the authors never released code. The IEEE final text was not obtained; table numbers refer to arXiv v4. Version 1 was FS-DINO and must not be mixed in.
- Apart from FoRIS's author-reported overall comparison to INSID3, no same-setting causal test between adjacent methods was found. FoRIS has no ablation that starts from INSID3 and replaces components one at a time.
- Full COCO classwise IoU and verifiable worst-class rankings for all methods. Worst fold or worst dataset is not worst class.
- Per-fold logs for INSID3, FoRIS and CG-ICS main tables. FROST's public evaluation details are insufficient to establish identical episode sampling.
- Four strict category-held-out UNICL-SAM training results for the reported 77.8. No full per-image pretraining overlap audit for any backbone.

### Numeric and interpretation conflicts

- FSS-SAM3 body says visual 66.1 and +text 75.4; main table says 66.6 and 75.8. Main-table visual folds are 64.2/66.8/67.7/67.5; 66.1 is from a different layout. The body 5-shot 71.2/74.5 also conflicts with table 71.8/75.1.
- SegIC 76.1 and UNICL-SAM 77.8 are not verified strict held-out-category results. SegIC separately reports 53.6 and 62.3.
- UINO Table V's 59.6→62.3 is +2.7, while the text says MHCM +3.1.
- CG-ICS main-table 72.3 and ablation 72.1 use different protocols. Table 4 RF-only has 13.9/69.3 ≈20.1%, versus reported CV 20.5%. The text's “3.3 below best” compares to SANSA 75.6, not UNICL-SAM 77.8.
- FROST Table 3 gives INSID3 56.5, not 57.6; FROST itself is 48.4 for COCO 1-shot. Some ± values in its main table are standard deviations across shot counts, not repeated-run error bars.
- FoRIS Fundus text says +9.8 although 19.7−8.6=11.1. FSS-SAM3's special negative-prompt table also has 0.1-scale disagreements. Table values are retained here; text-derived deltas are not used to reconcile them.

### Paper and current official code do not fully match

These are source differences, not measured effects; no experiment was run here to estimate their impact.

- INSID3: paper describes Gaussian noise for the position subspace; code uses a zero image and centres it. Candidates also have a positive-similarity gate, aggregation multiplies by candidate coverage, and seeds are thresholded. Paper has CRF, while the API default does not; CLI evaluation-size handling differs from the paper's original-size restoration description.
- FoRIS: some paper formulas show unit coefficients; code uses background coefficient 0.55 and candidate/seed coefficients 0.20/0.25, with additional powers, clipping and region weights. These implementation details are not expanded in the paper formulas.
- FROST: appendix says bilateral propagation uses whitened tokens, code uses raw normalized features; whitening epsilon is written as 1e−4, code uses a 1e−6 eigenvalue floor. “No test-time augmentation” needs to be distinguished from support flip.
- FSS-SAM3: COCO uses instance bboxes directly; PASCAL reads instance masks. Default script enables class text; visual-only requires explicitly disabling it. Code builds COCO episodes itself rather than consuming `lists_root`. Negative-prompt script also changes positive-prompt generation, so it is not a one-variable “add one negative box” test.
- Metric implementations differ: FSS-SAM3 aggregates intersection/union by class then computes IoU; current CG-ICS code averages per-episode IoUs by class. Same metric name does not guarantee equivalence.
- CG-ICS: paper describes a full red-mask reference view; current code has `crop=True` and zooms the crop. Paper abstracts joint reasoning; code runs text first, then grounding per box. It is not one fixed joint decode.
- SegIC: paper describes dense correspondence; public code computes query responses from a pooled reference foreground vector without explicitly storing a full cross-image matrix. This may be an equivalent computational simplification; it is not labeled an algorithmic contradiction.
- UNICL-SAM: supplement says similarity uses max; some code branches use mean. Train/inference pseudo-mask thresholds differ. Paper lists four graph augmentations; current code randomly uses three and comments out the subgraph branch. Main paper trains on COCO+ADE semantic masks, while default `train.sh` goes through a branch including COCO/LVIS instance data; loss weights also differ from supplement. The default script is therefore not evidence of exact reproduction of 77.8.

## Source scope

The paper and official-code checks above were provided as completed first-party verification for this task. This document preserves those results and does not claim to have rerun any method, reproduced any published score, or performed a new external source audit.
