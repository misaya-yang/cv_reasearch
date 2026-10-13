# confirm1000 通用评分器协议

源码为 `scripts/x1012s2_confirm_score.py`。准备阶段仅运行合成自测与元数据检查。随后 root 已冻结 `G_mfu_crf` 并明确授权评分；4000 个自然 prediction 文件预检通过后，一次完整确认评分及照片配对 CI 已完成。结果见 `CONFIRM1000_RESULT_20261013.md`。

manifest为 `cv_data/a/x1012s2_confirm1000/manifest.json`，SHA `0810bfd1dff670a5a62da8ca8a8bd7dfcac0efcb768c35a574b9f3ed1842c718`，十域各100例。这些是旧照片上的未用R输入/参考mask1024/Q输入任务，不能宣称全新照片留出。

## 统计口径

- 主分数不变：每个观察到的(fold,global_class)池化I/U，再对槽等权；总体十域等权。每折、逐例、大小分层、FN/FP另列，避免混用指标。
- Canonical query photo仅由元数据合并：RGB realpath、原始RGB文件SHA、dataset/official photo_id。不按crop/class拆同一照片，道路reference/query副本归并。当前1000任务合为862张canonical照片，其中1簇跨LVIS/PACO。
- 每次bootstrap对全局canonical photo抽一次共同权重，所有域、参考配对、类别和方法共享该照片权重；再重算每域观察槽分数与总体。1000次，seed2026101302。不能逐域独立抽后忽略跨域复用。
- 固定主方法相对FoRIS、MEAN、U、T、INSID3 bilinear/CRF640均给逐域、macro配对区间和补回TP/新增FP/删除TP/移除FP。
- 最终冻结描述可显式列出消融，但主方法ID不经评分器排序选择；消融只能解释机制，不据确认结果改主公式。

## 冻结后调用

默认或 `inspect` 只读元数据，并禁止打开所有query foreground/ignore文件。`preflight` 同样禁止GT；带 `--verify-predictions` 时仅核验自然mask的SHA、key及packed1024格式。

先由root收到开发结果后明确冻结唯一main，再填 `cv_data/a/x1012s2_confirm1000_scores/final_freeze.template.json` 为一个正式冻结文件。`main` 绑定定义、producer配置SHA、自然seal SHA及唯一mask key；自然库可以含10臂，实际评分仅取声明的main与事先列出的ablation。

```sh
python -B scripts/x1012s2_confirm_score.py preflight --freeze /absolute/path/final_freeze.json --verify-predictions
python -B scripts/x1012s2_confirm_score.py score --freeze /absolute/path/final_freeze.json
```

评分要求主方法及B/M/UTIJ全部1000自然输出封存、manifest/索引一致、全部mask身份通过。首次GT读取前写固定评分协议；中断可恢复相同协议，禁止在开始后换主方法/协议。报告和完整整数写仓库外 `cv_data/a/x1012s2_confirm1000_scores/`。

## 时间口径

FoRIS完整cached-O24时钟、MEAN增量、B+M产物流、UTIJ共享来源流及candidate产物时钟分开，不能把独立来源耗时相加冒充完整主方法。自然库同时生成多臂时，其吞吐不属于单个主方法。

`timing_evidence.catalog.json` 已登记实际MFU/gated_MFU的10域CPU2 cached-O24共享头部报告及SHA；该结果不能套到新增CRF后处理。最终RGB/编码整链证据可稍后通过 `attach-timing --timing-evidence /absolute/path/list.json` 加入；此模式不重读GT、不改主方法和准确率。因此无需为等待RGB计时而延后合法的一次确认评分。

七项合成检查通过。正式冻结文件为 `cv_data/a/x1012s2_composition_search/final_program.json`，SHA `342b57abf45355fb0b65df86dbbc1cd16eaef730e9a4cf6b93eb7baf4007cc00`；其后按唯一 main 完成全部1000任务评分与1000次全局照片bootstrap，未重选主方法。

最终真实 RGB 报告已通过 `attach-timing` 追加；作用域限定为3个输入、每流程2轮、2编码流程共12对24管线，不外推全数据集。追加阶段query GT读取0，main/准确率/CI/四编辑内容SHA保持不变。
