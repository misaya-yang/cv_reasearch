# In-context segmentation: what the reported score ladder contains

This note compares seven methods and the FROST side branch from their papers and official implementations. The scores are not a single controlled experiment. Unless a table says otherwise, all COCO numbers are 1-shot mIoU; differences within a paper's stated ablation are in percentage points. A difference between papers is descriptive, not a causal module effect.

The most consequential protocol distinction is training data: a frozen backbone does not mean a training-free method, and evaluating all four folds does not mean the tested fold's classes were excluded during training. SegIC 76.1 and UNICL-SAM 77.8 are in-domain numbers. Neither is a strict held-out-class result.

## Methods × questions a–e

| Method and score | a. What the reference mask becomes; how it affects the query | b. Other knowledge and training | c. What the ablation demonstrates | d. What can be attributed relative to the prior row | e. Failure cases and limitations |
|---|---|---|---|---|---|
| **1. [INSID3](https://arxiv.org/html/2603.28480v1), 57.6** | Masked reference features form a foreground prototype; the mask also supplies foreground labels for query-to-reference nearest-neighbour matching. The query is clustered first, a matching seed is selected, and cross-image similarity plus within-query similarity aggregate regions. Official code: [`models/insid3.py`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/models/insid3.py) (`predict_mask`, `_locate_candidates`, `_seed_and_aggregate`, `_finalize_mask`); [`utils/clustering.py`](https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/utils/clustering.py) (`agglomerative_clustering`). | Frozen DINOv3-L at 1024², inheriting large-scale image pretraining including LVD-1689M. No SAM, language input, or task-trained decoder. Supplement B nevertheless selects τ, α, and SVD dimension with three-fold cross-validation on the COCO training split. “No weight updates” does not mean “no setting selected on training data.” | Table 3: fine clustering without aggregation 42.8 → add cross-image aggregation 54.6 → add within-image similarity 57.6; the aggregation group adds 14.8, with the last addition adding 3.0. Table 5's default de-biasing adds 3.1. Table 8 changes DINOv2 to DINOv3 for +12.5, a backbone change. | INSID3 is a mechanistic reference point, not a method that learns all class knowledge from one image: it uses a strong pretrained representation and adds inference that first finds a matchable part, then recovers its query extent. | Supplement F: single-concept input, full reference mask required, same-class instances can merge, and the method is bounded by its representation. In a constructed no-target test, it correctly returns an empty mask in 85% of cases. No worst COCO class was found. |
| **2. [FoRIS](https://arxiv.org/html/2609.03384v1), 60.9** | The mask defines foreground/background samples. FoRIS forms multiple foreground prototypes and hard-background prototypes, contrast scores, reverse-match votes, and a query-region prior; semantic conflict and region purity then reweight the result. Official code: [`models/foris.py`](https://github.com/Xi-Mu-Yu/FoRIS/blob/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e/models/foris.py): `_reference_contrastive_prototypes`, `_part2_stage1_feature_gating`, `_part2_stage2_contrastive_score`, `_locate_candidates`, `_build_seed_cluster_prior`, `_semantic_disagreement_penalty`, `_semantic_cluster_reweight_map`. | Frozen DINOv3-L, 1024², and CRF; no new base-class segmentation training, SAM, or language branch. In addition to the support mask, it uses pretrained vision features, RGB, and spatial priors. | Table 2: largest single module addition is foreground localization (FL), 43.7 → 51.9 (+8.2); full model without foreground consolidation (FC) is 58.5, with FC 60.9 (+2.4). Table 3: adaptive versus fixed de-biasing is 60.9 versus 60.9 on COCO; candidate densification adds 0.8. | Table 1 reports 57.6 → 60.9 against INSID3 on the same DINOv3 and author-declared same protocol. This is a 3.3-point whole-method comparison, not a component attribution: FoRIS's own ablation starts at 43.7, not at INSID3. Assigning the 3.3 to individual FoRIS modules has **no direct evidence**. | Fine vessels and roads fail visibly; the paper attributes this to mixed foreground/background patches and dispersed features on thin branches. Fundus 1-shot is 19.7 and DeepGlobe-18 is 13.3; extra support helps only modestly. No worst COCO class was found. |
| **2a. [FROST](https://arxiv.org/html/2606.31136v1), COCO 48.4** | The mask divides support tokens into foreground/background for within-class whitening and two kernel-density estimates. Query tokens are classified by foreground-versus-background log-density ratio, with candidate gating and bilateral propagation. Official code: [`frost/model.py`](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/model.py) (`_shrinkage_whiten`, `_locate_candidates`, `predict_mask`); [`frost/density.py`](https://github.com/jhpark-ai/FROST/blob/b9ece69d7495a698c298e7cc3d16efacd4497a43/frost/density.py) (`estimate_bandwidth`, `kde_log_posterior`, `density_ratio_predict`). | Frozen DINOv3-L; no SAM, language, or base-class segmentation training. It keeps foreground and background density anchors and uses support flipping, support-estimated statistics, and fixed spatial rules. Estimating statistics is not neural-network training. | Table 4 is an average over 17 remote-sensing datasets, not a COCO ablation: full model 48.3; removing de-biasing and whitening together costs 4.6; removing de-biasing alone costs 2.1, the largest individual removal. | This is a side branch, not the next upward rung. Table 3 reports COCO 1-shot 48.4, 8.1 below INSID3's 56.5 in FROST's comparison; at 10-shot FROST is 64.3 versus 64.0. Its main gains are supported on remote sensing and with more supports. | The authors note that a single support image may not represent the query distribution. Equal-prior, multimodal density estimates suit remote sensing better than object-centred COCO. The method processes one class at a time. No worst COCO class was found. |
| **3. [UINO-FSS](https://arxiv.org/html/2504.15669v4), 64.5** (folds 62.2/66.0/65.5/64.4) | The mask is pooled into a foreground prototype and sparse visual prompt. A second path subtracts background correlation from multi-layer support–query foreground correlation to form a 4D correlation volume. MHCM turns this into a dense query prior; an adapted query feature and the prior feed a SAM-initialized decoder. Official repository/function was not found in this audit; this description is verified from paper Sections IV-D/E only. | Frozen DINOv2-B, but the full method is trained: adapters are distilled against SAM-H features on 1% of SA-1B images, then the prompt generator and SAM-initialized decoder are trained with base-class pixel masks. The teacher inherits SAM segmentation supervision. No class text is used at inference. | Table VII's simplified model improves 52.8 → 59.6 with distillation (+6.8). Table V: SVP +1.4, MHCM +2.7, contrastive enhancement (CE) +2.2. Table VI's actual MHCM versus CP4D convolution is only 64.0 → 64.5 (+0.5). | Relative to FoRIS there is **no direct evidence** for a numerical attribution. UINO adds SAM-feature distillation, base-class supervision, and learned correlation reading; 64.5 − 60.9 = 3.6 is not assigned to those ingredients by a controlled cross-paper comparison. | No standalone limitations section, full-method systematic failures, or per-class IoU was found. Figure 5 mostly shows errors after removing CE/MHCM; it is not evidence of the complete model's worst cases. |
| **4. [FSS-SAM3](https://arxiv.org/html/2604.05433v1), main table 66.6; body 66.1** | The core is one instance box on a shared canvas: support/query are composed into one image, the in-box visual exemplar prompts SAM3, and the query crop is returned. COCO code reads instance JSON boxes and does not return a support mask. Official code: [`data/dataset_tool.py`](https://github.com/WongKinYiu/FSS-SAM3/blob/main/data/dataset_tool.py) (`COCOFSSDataset._load_ref`); [`model/evaluate.py`](https://github.com/WongKinYiu/FSS-SAM3/blob/main/model/evaluate.py) (`create_input`, `get_norm_box`, `run_episode`); [`geometry_encoders.py`](https://github.com/WongKinYiu/FSS-SAM3/blob/main/sam3/model/geometry_encoders.py) (`SequenceGeometryEncoder._encode_boxes`). | Frozen during FSS, no new training, but SAM3 already learned language–vision concept segmentation from extensive phrase–mask supervision. Visual-only omits the category name; the `+text` variant supplies it. COCO uses instance boxes; PASCAL also reads instance annotation to select the largest object. The public implementation is therefore not accurately described as consuming only one full semantic mask. | Table 5: visual 66.6, text 74.9, both 75.8; adding the class name gives +9.2, and adding visual input when the name is present gives +0.9. Table 4's visual layout range is 64.8–66.6; with fixed bottom placement, support share 0.5 → 0.6 adds 0.1. | Relative to UINO-FSS there is **no direct evidence**. The mechanism changes from explicit cross-image correlation to invoking SAM3's learned concept segmentation on a shared canvas. The 1.6–2.1-point cross-paper gap is not a measured causal effect. | Table 3: adding more automatic negative prompts hurts, with a maximum drop of 21.9; Figure 3 shows collapse toward an all-background prediction. Pseudo-video propagation and direct image–text embedding selection also failed without reported numbers. No worst class was found. |
| **5. [CG-ICS](https://arxiv.org/pdf/2606.28149v2), 72.3** (standard ICS) | The mask marks an object to an MLLM, which names a concept. SAM3 segments that concept on the reference and computes reference IoU (RF), then supplies query presence score (QM). RF × QM guides concept search. A visual mosaic also supplies a query box; final prompting uses concept plus box. Official code: [`model/tree_search.py`](https://github.com/Kakarot1103/CG-ICS/blob/main/model/tree_search.py) (`TreeSearcher.select`, `_score_single_ref_raw`); [`main.py`](https://github.com/Kakarot1103/CG-ICS/blob/main/main.py) (`evalute`); [`model/sam3.py`](https://github.com/Kakarot1103/CG-ICS/blob/main/model/sam3.py) (`Sam3Segmenter.infer`). | Frozen Qwen3-VL-4B-Instruct and SAM3; no CG-ICS parameter training. The user does not provide a ground-truth class name, but the system uses language knowledge. Qwen3-VL pretraining explicitly includes COCO, Objects365, OpenImages, and RefCOCO-family sources; this does not establish leakage of the particular test images. | Table 4 is a reference-selection robustness protocol (500 queries/fold, 50 references/query), not the 72.3 main protocol (1000 queries/fold). In that table, one concept is 67.6 → full 72.1. On candidates: RF +2.8; after RF, QM +0.8; search +1.1; visual +0.9. Largest one-switch gain is adding RF to QM-only, 62.8 → 70.1 (+7.3), which repairs a weak control. | Relative to FSS-SAM3 there is **no direct evidence**. CG-ICS adds automatic concept naming, mask-verified concept selection, query presence scoring, and search. Its internal ablation supports those steps in its own protocol, not a full causal account of the gap to FSS-SAM3. | Authors acknowledge MLLM hallucination, over-specific concepts, and unreliable multi-image reasoning. Table 3's largest domain perturbation drop is 5.2. No CG-ICS failure plate or worst class was found; Figure 5's failures belong mainly to GF-SAM and should not be attributed to CG-ICS. |
| **6. [SegIC](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05519.pdf), 76.1 in-domain** | The mask is pooled into a visual vector to match a coarse query mask and points. A meta prompt separately encodes class/task text for the trained decoder. Official code: [`model/segic.py`](https://github.com/MengLcool/SEGIC/blob/master/model/segic.py): `extract_inst_feat`, `maks_decoding` (spelling in source), `extract_dift_feature`, `forward`. Default semantic text is “a photo of a <class name>”. | DINOv2 is frozen, but about 5M decoder-related parameters are trained. Training includes COCO semantic/instance masks and ADE20k/LVIS; CLIP language knowledge is also used. The 76.1 setup has seen the evaluated COCO classes. The strict four-fold result is 53.6; the result after adding ADE/LVIS/FSS-1000 is 62.3, with class overlap in external data not audited here. | Table 4 prompt ablation uses the smaller DINOv2-B setup and scores 74.6 for all prompts, not the 76.1 L-backbone main result. No prompt 48.9, geometry only 68.8, visual only 71.0, meta only 73.7, all 74.6; all versus meta only is +0.9. This does not give a component attribution for the main score. | Relative to CG-ICS there is **no direct evidence**. The 76.1 result changes training, uses class text, and is in-domain; it cannot be described as a +3.8 improvement to matching based on one reference. | Meta-only is 63.7 on FSS-1000 and 21.6 on DAVIS17, substantially below all prompts across those domains. No worst COCO class or dedicated failure example was found; success figures are not failures. |
| **7. [UNICL-SAM](https://openaccess.thecvf.com/content/CVPR2025/html/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.html), 77.8 in-domain** | The mask selects foreground for graph construction, uncertainty estimation, reweighting, and part-prototype clustering. Global pooling, original features, and the mask form conditions; 50 learned queries read them and generate sparse SAM prompt embeddings. Official code: [`unicl_sam/model/model.py`](https://github.com/ImmortalSdm/UNICL-SAM/blob/main/unicl_sam/model/model.py): `get_graph`, `get_uncertainty_estimation`, `get_uncertainty_refinement`, `get_cluster_feats`, `get_support_embeddings`, `generate_img`. | DINOv2-L and SAM-H are frozen, but MFA, UGGN, QPG and related components train about 55M parameters. Main training uses COCO plus ADE semantic masks. SAM also brings 11M images and over 1B masks of supervision. Main inference does not use class text. 77.8 is in-domain: running one checkpoint on all four folds is not four held-out-class training runs. | Table 2 small-data setup: all modules 71.77 → 74.64 (+2.87); uncertainty module (UPM) alone adds 1.22, and with part prototypes already present it adds 2.20. Supplement C3 expands COCO 74.6 → 77.3 and then adds ADE for 77.8. Supplement C4 shows that with full COCO and MFA, adding UGGN changes COCO 77.4 → 77.3. | Relative to SegIC there is **no direct evidence**: the main table does not include SegIC. The 1.7-point difference is not attributed to uncertainty or part prototypes; training architecture, SAM supervision, data and prompt differ. | Supplement D5: replacing reference mask with a box changes 79.8 → 69.0; Gaussian noise gives 71.0. Supplement E.3 attributes ambiguity to box semantics. No worst COCO class was found. |

## Pretraining sources and scale (context, not independent data counts)

- DINOv3: [paper §3.1](https://arxiv.org/html/2508.10104v1) and [official weights table](https://github.com/facebookresearch/dinov3#pretrained-models) report about 1.689B LVD images for the cited pretraining set. Other image sources are also involved; 1.689B is not a deduplicated total of the full mixture.
- DINOv2: [paper appendix](https://arxiv.org/html/2304.07193v2) reports about 142M LVD images. That is an image count, not a segmentation-label count.
- SAM: [paper](https://arxiv.org/abs/2304.02643) reports SA-1B with about 11M images and over 1B masks. UINO-FSS's exact sample count for its 1% distillation subset was not found.
- SAM3: [paper Table 23 and Appendices C/E](https://arxiv.org/html/2511.16719v1) report PE with 5.4B image–text pairs; SA-Co/HQ with 5.2M images and 52.3M phrase–mask pairs; EXT with 9.3M images and 70.5M pairs; SYN with 39.4M images and 1.4B pairs. These include automatic/synthetic labels and should not all be called hand-annotated masks.
- The [Qwen3-VL paper §3.2.4](https://arxiv.org/html/2511.21631v2#S3.SS2.SSS4) lists COCO and other sources for grounding pretraining. This is not proof of overlap with a specific evaluation image.
- SegIC lists about 83K COCO, 20K ADE20k, and 100K LVIS training images; UNICL-SAM lists 118K COCO images plus ADE data. Images may overlap, so these values cannot be added into independent-mask totals.

## Original ablation numbers and what each table supports

All gains below are within their named table and configuration. No cross-paper differences are added to these ablation effects.

### INSID3 — [Tables 3, 5, 8](https://arxiv.org/pdf/2603.28480v1)

**Table 3, clustering and aggregation**

| Configuration | COCO | PASCAL-Part |
|---|---:|---:|
| No clustering, threshold similarity graph @ 0.55 | 44.2 | 35.4 |
| Coarse clustering τ=0.5, no aggregation | 50.6 | 31.1 |
| Fine clustering τ=0.6, no aggregation | 42.8 | 36.2 |
| Fine clustering + cross aggregation | 54.6 | 48.5 |
| Fine clustering + self/cross aggregation | 57.6 | 50.5 |

Table 5 de-biasing rows, in order “none / 4-view / 12-view / 1-noise SVD / 5-noise / 10-noise”: **54.5 / 55.7 / 56.5 / 57.6 / 57.7 / 57.7**.

Table 8 backbone rows, “DINOv3 / DINOv2 / Franca / PE / SD”: **57.6 / 45.1 / 39.2 / 48.4 / 33.2**. The largest same-granularity algorithm group is aggregation at +14.8; self versus cross alone is +3.0. The DINOv3 versus SD gap is 24.4 points and is a backbone/pretraining comparison, not an inference-module effect.

### FoRIS — [Tables 2, 3](https://arxiv.org/html/2609.03384v1#S4.SS2)

APD = adaptive positional de-biasing; FR = foreground refinement; FL = foreground localization; FC = foreground consolidation. Every row includes Base.

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

Table 3a fixed versus adaptive de-biasing: **60.9 / 60.9**. Table 3b without versus with candidate densification: **60.1 / 60.9**. Table 3d “no FC / semantic disagreement penalty / semantic reweighting / both”: **58.5 / 59.9 / 60.2 / 60.9**. Table 3c's position/RGB ablation has no COCO column and cannot allocate COCO gains. FL's +8.2 is the largest single addition; the full combination adds 17.2 over its 43.7 Base. Neither number is a gain over INSID3.

### FROST — [Table 4](https://arxiv.org/html/2606.31136v1#S4.SS4), 17 remote-sensing datasets, 1-shot mean

| Variant | mIoU | Relative to full |
|---|---:|---:|
| Full | 48.3 | — |
| Remove bilateral propagation | 48.0 | −0.3 |
| Remove whitening | 47.5 | −0.8 |
| Remove support flip | 47.2 | −1.1 |
| Remove positional de-biasing | 46.2 | −2.1 |
| Remove refinement (de-biasing + whitening) | 43.7 | −4.6 |

No COCO-specific ablation is given. Appendix Table 9 is also remote-sensing-only: DINOv3-L 49.7; DINOv3-7B 50.3; DINOv2-L 43.6; DINOv3-ConvNeXt-L 42.7; DINOv3-SAT-7B 35.9; DINOv3-SAT-L 35.4. Do not use these gaps to explain COCO.

### UINO-FSS — [v4 Tables V–VIII](https://arxiv.org/pdf/2504.15669v4#page=9)

BA = bottleneck adapter; PCS = prototype cosine similarity; SVP = semantic-aware visual prompt (sparse prompt); MHCM = Mamba-hypercorrelation module; CE = foreground-minus-background contrastive enhancement.

**Table V**

| Configuration | COCO | Adjacent gain |
|---|---:|---:|
| BA + PCS | 58.2 | — |
| + SVP | 59.6 | +1.4 |
| + MHCM | 62.3 | +2.7 |
| + CE | 64.5 | +2.2 |

**Other original rows**

| Table | Configuration | COCO |
|---|---|---:|
| VI | CP4D convolution replacing MHCM | 64.0 |
| VI | MHCM | 64.5 |
| VII, simplified `Comp` | No distillation | 52.8 |
| VII, simplified `Comp` | Distillation | 59.6 |
| VIII, simplified `Comp` | DINOv2-S | 54.3 |
| VIII, simplified `Comp` | DINOv2-B | 59.6 |
| VIII, simplified `Comp` | DINOv2-L | 60.7 |
| IV, simplified `Comp*` | Frozen decoder | 57.2 |
| IV, simplified `Comp` | Trained decoder | 59.6 |

The largest relevant training contrast is +6.8 for distillation on the simplified model; it is not added to the full model's 64.5. MHCM versus no module is +2.7; MHCM versus the alternative 4D reader is +0.5. Those are different controls.

### FSS-SAM3 — [Tables 3–7](https://arxiv.org/html/2604.05433v1)

**Table 3, negative prompts**

| Negative prompts | COCO mIoU | FB-IoU |
|---|---:|---:|
| 0 | 66.6 | 82.9 |
| ≤1 | 54.8 | 77.2 |
| ≤3 | 46.6 | 73.1 |
| ≤5 | 44.7 | 72.4 |

**Table 4, layout**

| Configuration | COCO mIoU | FB-IoU |
|---|---:|---:|
| ARP, pad at top | 66.0 | 82.3 |
| FR horizontal, left, 0.5 | 64.8 | 81.8 |
| FR vertical, bottom, 0.5 | 66.5 | 82.8 |
| FR vertical, bottom, 0.6 | 66.6 | 82.9 |
| FR vertical, top, 0.5 | 66.1 | 82.9 |
| FR vertical, top, 0.6 | 66.0 | 82.7 |

**Table 5, prompt modality**

| Prompt | COCO mIoU | FB-IoU |
|---|---:|---:|
| Visual | 66.6 | 82.9 |
| Text class name | 74.9 | 86.3 |
| Both | 75.8 | 87.1 |

**Table 6, visual plus selected text**

| Text | COCO mIoU | FB-IoU |
|---|---:|---:|
| Ground-truth class name | 75.8 | 87.1 |
| Fold-level candidate name | 72.4 | 86.1 |
| Whole-dataset candidate name | 67.8 | 83.2 |

The largest positive modality gap is visual → both, +9.2, and adds class-name information. Up to five automatic negative prompts cause −21.9. The negative-prompt script also changes how the positive prompt is obtained, so this is not a clean “add an accurate negative box” single-factor test.

Table 7 is a separate multi-class MSCOCO test, not the standard main table: Positive Only 53.2; Semantic Distractors 26.0; Background Negatives 22.1; Multiple Negatives ≤10 9.7. Its largest decrease does not transfer to the 66.6 setup.

### CG-ICS — [v2 Table 4](https://arxiv.org/pdf/2606.28149v2#page=9)

This is the reference-selection robustness protocol: 500 queries/fold, 50 references/query. The 72.3 main score uses a separate Standard ICS protocol with 1000 queries/fold.

| Candidates | RF | QM | Search | Visual | mIoU | Std | CV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 67.6 | 15.1 | 22.3% |
| 1 | 0 | 0 | 0 | 0 | 66.5 | 15.3 | 23.0% |
| 1 | 1 | 0 | 0 | 0 | 69.3 | 13.9 | 20.5% |
| 1 | 0 | 1 | 0 | 0 | 62.8 | 18.2 | 29.0% |
| 1 | 1 | 1 | 0 | 0 | 70.1 | 11.3 | 16.1% |
| 1 | 1 | 1 | 1 | 0 | 71.2 | 10.5 | 14.7% |
| 1 | 1 | 1 | 1 | 1 | 72.1 | 9.3 | 12.9% |

The first row already asks Qwen to generate one SAM3 concept; it is not a no-foundation-model baseline. Generating more candidates alone loses 1.1. Query presence alone is worse. RF uses the reference mask and improves word selection in these comparisons. **Speculation:** it may fix concepts that exist in the query but are not the object selected by the reference; the paper does not count these error categories. This table does not assign the entire gap to FSS-SAM3.

### SegIC — [正文 Tables 2–6](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05519.pdf#page=13)

Tables 4–6 use DINOv2-B and a smaller sampling budget; their full row is 74.6, not the L-backbone 76.1 main result.

**Table 4, prompt ablation**

| Prompt | COCO | FSS-1000 | DAVIS17 |
|---|---:|---:|---:|
| Geometry only | 68.8 | 87.3 | 65.8 |
| Visual only | 71.0 | 85.1 | 67.6 |
| Meta only | 73.7 | 63.7 | 21.6 |
| None | 48.9 | 58.1 | 21.1 |
| All | 74.6 | 87.4 | 68.4 |

**Table 5, training enhancements (reversion / negative)**

| Reversion | Negative | COCO |
|---:|---:|---:|
| 0 | 0 | 70.0 |
| 0 | 1 | 72.9 |
| 1 | 0 | 73.1 |
| 1 | 1 | 74.6 |

**Table 6, training data**

| Data | COCO | FSS-1000 | DAVIS17 |
|---|---:|---:|---:|
| COCO semantic | 76.3 | 80.9 | 45.1 |
| + ADE semantic | 75.6 | 82.3 | 40.1 |
| COCO semantic + COCO instance | 75.7 | 83.5 | 70.1 |
| COCO semantic + ADE + COCO instance + LVIS | 74.6 | 87.4 | 68.4 |

This table does not support “more segmentation data always improves COCO”; extra instance data helps video segmentation more strongly.

Table 3 backbone COCO column, in row order: DINOv2-B 74.6; DINOv2-L 75.4; DINOv2-G 75.7; DINOv1-B 53.5; SAM-B 59.8; SD2.1 66.6; OpenCLIP-ConvNeXt-B 65.1; CLIP-ViT-B 15.3; MAE-B 12.0; ConvNeXt-B 62.0. These change backbone and pretraining jointly, not one module.

**Table 2 protocol correction**

| SegIC setup | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---:|---:|---:|---:|---:|
| Strict four-fold train/test class isolation | 55.8 | 54.7 | 52.4 | 51.4 | 53.6 |
| Also train with ADE20k, LVIS, FSS-1000 | 62.3 | 62.5 | 63.3 | 60.9 | 62.3 |

### UNICL-SAM — [paper Tables 2–6](https://openaccess.thecvf.com/content/CVPR2025/papers/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.pdf); [supplement C3/C4](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_CVPR_2025_supplemental.pdf)

Table 2 uses about 34K COCO images and 10% of masks. Checkmarks below were rechecked against the PDF figure to avoid PDF text-extraction column shifts. MFA = multi-scale feature adaptation; Part = part prototypes; UPM = uncertainty modelling.

**Table 2**

| MFA | Part | UPM | mIoU | MAE |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 71.77 | .039 |
| 0 | 0 | 1 | 72.99 | .037 |
| 0 | 1 | 0 | 72.41 | .038 |
| 1 | 0 | 1 | 73.28 | .036 |
| 1 | 1 | 0 | 73.78 | .036 |
| 0 | 1 | 1 | 74.61 | .034 |
| 1 | 1 | 1 | 74.64 | .034 |

Other original table values, each pair is **mIoU / MAE**:

- Table 3, image enhancement / feature refinement: 00 = 73.78/.036; 10 = 73.82/.036; 01 = 73.99/.035; 11 = 74.64/.034.
- Table 4, GCN / UGGN / guided image enhancement: 100 = 72.12/.038; 010 = 72.41/.038; 011 = 73.82/.036.
- Table 5, top-ratio masking = 43.73/.117; random masking = 73.44/.037; learnable gate = 74.64/.034.
- Table 6, cluster count 5 = 73.36/.037; 10 = 73.82/.036; 15 = 72.50/.039; 20 = 72.38/.040.

Replacing top-ratio masking adds 30.91, but replaces a visibly poor alternative; it is not the contribution of uncertainty modelling. The normal component ablation's largest single switch is adding UPM when Part is already on (+2.20); the full combination adds +2.87 over baseline.

**Supplement C3, training data**

| Data | COCO | FSS-1000 | LVIS |
|---|---:|---:|---:|
| COCO-s subset | 74.6 | 81.7 | 29.9 |
| Full COCO | 77.3 | 82.4 | 32.0 |
| + ADE20k | 77.8 | 84.0 | 34.1 |

**Supplement C4, MFA / UGGN**

| MFA | UGGN | Training parameters | COCO | FSS-1000 | LVIS |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 55.4M | 77.3 | 82.4 | 32.0 |
| 1 | 0 | 52.5M | 77.4 | 81.2 | 29.8 |
| 0 | 1 | 9.8M | 77.0 | 81.3 | 30.2 |
| 0 | 0 | 7.8M | 75.4 | 80.7 | 28.9 |

On full COCO, with MFA fixed, adding UGGN changes COCO −0.1, FSS-1000 +1.2, and LVIS +2.2. Its value cannot be summarized as “always improves COCO.”

## Summary: what changes from row to row?

- **57.6 → 60.9, INSID3 to FoRIS.** The authors report a same-DINOv3, same-protocol whole-method comparison of +3.3. FoRIS adds foreground/background contrast, multiple prototypes, localization, and region-level correction; its own ablation does not start from INSID3, so it cannot split that +3.3 by module.
- **60.9 → 64.5, FoRIS to UINO-FSS.** UINO adds SAM feature distillation, base-class mask training, and learned 4D correlation reading. Its simplified-model distillation contrast is +6.8; the full-model MHCM advantage over an alternative 4D reader is +0.5. Neither number explains the cross-paper +3.6.
- **64.5 → 66.6, UINO-FSS to FSS-SAM3 main table.** The method invokes SAM3's already trained concept segmentation through a single-instance box and shared canvas. The 2.1-point cross-paper difference has no controlled attribution. Within FSS-SAM3, adding a class name is +9.2.
- **66.6 → 72.3, FSS-SAM3 to CG-ICS.** CG-ICS adds MLLM naming, reference-mask verification, query-presence scoring, and search. Its internal ablation supports components on its own protocols but does not assign the gap to those components.
- **72.3 → 76.1, CG-ICS to SegIC.** SegIC changes training and evaluation assumptions and explicitly uses class names; 76.1 is in-domain. Its prompt ablation is 74.6 with a smaller backbone, and strict four-fold training is 53.6. This is not a like-for-like improvement from one-reference matching.
- **76.1 → 77.8, SegIC to UNICL-SAM.** UNICL-SAM adds trained prompt generation, part/uncertainty handling, and SAM priors. Its own data expansion supports 74.6 → 77.3 → 77.8, but the 1.7-point difference from SegIC has no direct component attribution.

**Constraint reading.** If “one reference image, frozen backbone, no training” means the proposed method itself may not train, then UINO's distillation/base-class training, SegIC's decoder training, and UNICL's roughly 55M trained parameters do not satisfy that constraint. This is a mismatch in constraints, not evidence that training-free methods can never reach their scores. Frozen weights may still carry SAM segmentation knowledge or MLLM language knowledge, as in FSS-SAM3 and CG-ICS. If the constraint also requires one DINOv3 model and no external foundation model, those sources of knowledge are excluded.

**Speculation boundary.** “Nobody uses” foreground/background statistics, region propagation, spatial composition, or concept verification is unsupported: these papers already use them. Their transfer to another method and any resulting score change are unmeasured and remain speculation.

## Missing evidence, score conflicts, and paper–code differences

### Not found in this audit

- UINO-FSS official code repository, functions, and run configuration. “Not found here” does not mean the authors never released code. The IEEE final text was not obtained; table references above use arXiv v4. Do not mix v1's FS-DINO with v4's UINO-FSS.
- Same-setting adjacent-rung causal comparisons other than FoRIS authors' overall INSID3 comparison. FoRIS has no ablation that replaces INSID3 modules one at a time.
- Complete per-class COCO IoU or a verifiable “worst class” ranking for all methods. A worst fold or dataset is not a worst class.
- Per-fold logs for the main INSID3, FoRIS, and CG-ICS tables. FROST's public evaluation description does not establish identical episode construction to the other papers.
- Four strict class-held-out UNICL-SAM training/evaluation runs corresponding to 77.8; complete per-image pretraining-overlap audits for any foundation model.

### Number and interpretation conflicts

- **FSS-SAM3:** body text says visual 66.1 and text 75.4; the main table says 66.6 and 75.8. The main-table visual folds are 64.2/66.8/67.7/67.5; 66.1 also occurs for a different layout. The paper's 5-shot body values 71.2/74.5 disagree with table values 71.8/75.1.
- **SegIC / UNICL-SAM:** 76.1 and 77.8 are not verified strict held-out-fold results. SegIC explicitly also reports 53.6 and the expanded-data 62.3 setup.
- **UINO-FSS:** Table V gives 59.6 → 62.3 as +2.7, while the prose says MHCM adds +3.1.
- **CG-ICS:** main-table 72.3 and ablation 72.1 use different protocols. Table 4's RF-only row has 13.9/69.3 ≈ 20.1%, while the table reports CV 20.5%. The prose's “3.3 below best” refers to SANSA at 75.6, not the larger UNICL-SAM 77.8.
- **FROST:** its Table 3 reports the INSID3 comparator as 56.5, not 57.6; FROST's own COCO 1-shot is 48.4. Some reported ± values are standard deviation across shot settings, not repeated-run error bars.
- **FoRIS:** the Fundus table gives 19.7 − GF-SAM 8.6 = 11.1, while prose says 9.8. FSS-SAM3's special negative-prompt table has some 0.1-level text/table mismatches. This note retains table values and does not derive results from inconsistent prose.

### Paper and official implementation do not fully match

No experiment was run to measure the score impact of these differences.

- **INSID3:** paper uses Gaussian noise to estimate the position subspace; code uses a centred all-zero image. Code adds positive-similarity candidate gating, candidate-coverage weighting, and seed thresholding. Paper describes CRF, but the API default does not use it; CLI evaluation size handling also differs from the paper's original-size description.
- **FoRIS:** some equations use unit coefficients, while code uses background coefficient 0.55, candidate/seed coefficients 0.20/0.25, and adds powers, clipping, and region weights. These implementation details should not be omitted when claiming equation-by-equation identity.
- **FROST:** appendix says bilateral propagation uses whitened tokens; code uses raw normalized features. Whitening epsilon is 1e−4 in the paper and eigenvalue floor 1e−6 in code. “No test-time augmentation” needs to be distinguished from support flipping.
- **FSS-SAM3:** COCO reads instance boxes; PASCAL reads instance masks. Default scripts enable category text; visual-only requires disabling it explicitly. The script constructs COCO episodes rather than using the supplied `lists_root`. Negative-prompt code also changes positive-prompt construction, so it is not a single-factor “add one negative box” test.
- **Evaluation protocol:** FSS-SAM3 accumulates intersection/union by class before IoU; current CG-ICS code averages per-episode IoU by class. Same-name mIoU labels do not make the two aggregates equivalent.
- **CG-ICS:** the paper depicts a full red-mask reference view; current `crop=True` code crops and enlarges it. The paper abstracts joint reasoning; code first selects text, then invokes grounding box by box. It is not one fixed decoding pass.
- **SegIC:** the paper expresses dense correspondence; public code pools the reference foreground and computes query response without explicitly materializing the full cross-image matrix. This may be an equivalent computational reduction; it is not established as an algorithmic contradiction.
- **UNICL-SAM:** supplement says similarity uses max, while some code branches use mean; pseudo-mask thresholds differ between training and inference. Paper lists four image augmentations but current code samples three and comments out the subgraph branch. Main training is described as COCO + ADE semantic; default `train.sh` instead follows a branch that includes COCO/LVIS instances and other data, and loss weights also differ from the supplement. The default script is not an exact reproduction recipe for 77.8.
