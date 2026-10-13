# C1 候选冻结

固定候选：VP_thin09；决策基座：K64_region_mean；证据调制：A4_max_J_continuous；FoRIS依赖2步；公共CRF：False。
选择规则：max worst-domain delta; tie max ten-domain equal mean; tie fewer stages; no automatic CRF。开发集十域均≥−0.5：False。
固定参考rho回退：0.9；仅严格超过阈值时直接原T，跳过所有修正/maxJ/graph/CRF。
未达开发门槛时也只确认一次最接近的固定组合，不冒称已达到独立基座目标。确认集不参与选择。
确认清单：2136例；各域例数：{'coco': 360, 'deepglobe_road': 10, 'fundus': 10, 'isaid': 75, 'isic': 30, 'lung': 10, 'lvis': 161, 'paco_part': 1051, 'pascal_part': 369, 'suim': 60}。
confirm原文件SHA256：bba8ea8a253554066cebedb220cda1ecaa2cd3245953b26fe24c2eb7741be1c1；自然输入digest：87f1ea1062e860eb4daf06f3264137f3d3ab3aa5a96690d0d7441692c492b0f4。
允许P0随后只绑定派生FoRIS基线；R/Q/掩码路径与已有O24身份、完整确认队列保持冻结。
推理CPU单进程4线程，只重算同一独立证据与所选0/1/2步。没有DINO模型、编码、窗口或域条件选择。全部自然掩码封存后才评分。
确认评分后禁止换候选、改常数、重新拟合；如需研究新方法，返回explore并另划未暴露确认集。

## 方法与出处

{
  "kind": "b4",
  "source_arm": "VP_thin09",
  "base_arm": "VP",
  "decision_basis": "K64_region_mean",
  "evidence_modulation": "A4_max_J_continuous",
  "lineage_source_arm": "VP_thin09",
  "underlying_source_arm": "VP",
  "active_score_terms": [
    "V",
    "P"
  ],
  "dependencies": [
    "V",
    "P"
  ],
  "FoRIS_stage_count": 2,
  "common_crf": false,
  "dataset_branch": false,
  "ten_domains_ge_minus_05_in_explore": false,
  "explore_delta_pp": {
    "coco": -2.28886948820071,
    "lvis": -1.8148945937241407,
    "pascal_part": -0.26601093073374926,
    "paco_part": 3.776558106522934,
    "suim": 0.2403485278447235,
    "isaid": 2.5337944962345205,
    "isic": -2.480850327393057,
    "lung": -0.42843934070194223,
    "fundus": 15.04071248746762,
    "deepglobe_road": 0.9005585023046159
  },
  "selection_rule": "max worst-domain delta; tie max ten-domain equal mean; tie fewer stages; no automatic CRF",
  "reference_rho_gate": 0.9,
  "reference_rho_fallback": "raw_T",
  "triggered_FoRIS_stage_count": 0,
  "triggered_common_crf": false
}

完整常数、依赖、源码SHA和已评分开发报告出处：/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1012_C1/config.json。
证据快照：/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1012_C1/frozen。
