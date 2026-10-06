# 固定简单完整控制：MEAN + RGB128 boundary Potts

已交付一个可直接有限评测的廉价完整边界对照，不声称原创机制，不计入新方法数量。实现为 `src/ics/methods/mean_rgb_potts.py`，独立入口为 `scripts/run_mean_rgb_potts.py`。没有改动 ProM4 活跃实现、PLAN 或服务器文件；本轮未 SSH、未跑真实600例。独立实现完成后，按根代理新授权接入了两个共享入口文件，核验后已释放其独占。

## 完整固定配方

1. 只用绑定的 q/r/cov/source score，按当前 locked MEAN a=.25/lambda=16 原稀疏graph dtype→FP64 CG合同重新计算64×64 MEAN场；不使用历史预测mask作为输入、不新增DINO编码。
2. 使用已绑定 `query_rgb_export` 的 `rgb` 键（128×128×3），不读类别名、fold、目标数量、query GT或额外参考。RGB uint8转[0,1]；已有合法RGB浮点直接用[0,1]。
3. MEAN用FP32 half-pixel双线性插值到128×128。定义 signed margin `ell=2*MEAN128-1`，保留原0.5决策；不裁剪，不重minmax。MEAN0/1对应−1/+1，与ProM4均值归一化margin尺度一致；这只是固定控制映射，没有拟合query GT。
4. 保留整个128×128四邻域，无种子、ROI、候选筛选或边剪枝。RGB边距离 `r_ij=mean_c((RGB_ic-RGB_jc)^2)`，`sigma=max(median_E(r),1e-4)`，`W=exp(-r/sigma)`，平均加权度 `dbar=2*sum(W)/16384`，Potts容量 `.25*W/max(dbar,1e-8)`。
5. 最小化 `sum_i[z_i*softplus(-ell_i)+(1-z_i)*softplus(ell_i)] + sum_E capacity_ij*|z_i-z_j|`。复用 ProM4 `exact_potts_cut` 浮点FP64 Dinic，不整数化。源侧=前景；输出有独立cut/maxflow能量证书。
6. 输出 `field` 是128×128二值全局cut，不是连续概率。复用ProM4二级renderer：binary128→bilinear1024>.5→binary→bilinear原查询尺寸>.5。主评测仍用项目历史1024口径，原尺寸mask额外保存。
7. 两个明确简单控制同行保存：`mean.control`（原MEAN连续场→bilinear1024>.5）与 `mean_rgb_unary.control`（同128插值、二值化、两级renderer，但不加RGB Potts）。后者分离grid/render差异与真实RGB边界正则效应。全空参考按完整empty规则输出空。

`.25`、median scale、`1e-4`和`1e-8`floor来自所给Pro文件的M4 §7.3（约716–735行），未搜索参数。RGB平方欧氏距离替代原M4 frozen-feature余弦距离，图由64改为已绑定RGB128；因此这是明确的新简单对照，**不冒充ProM4方法复现**。指数下溢可使极端边容量数值为0，其数量单独记录；没有按得分或标签删边。

## 本地验证与成本

`check_control.py` / `check.json`：24个六节点图、每图64种二值label穷举，及独立NetworkX min-cut核对；包含全正、全负、零margin和强RGB边界案例。最大独立cut能量差 <9e-16。双worker×1thread的RGB128正/负合成例分别添加36、删除36网格像素；图16,384点/32,512边，cut为0.329/0.216秒，完整附加读出为0.337/0.224秒；每worker峰值RSS约221MB。tiny check证明代码/目标一致，不是分割收益。

`runner_smoke/`：独立runner用两个合法synthetic cached packets全链重算MEAN、RGB cut、保存三臂、原尺寸mask、逐例receipt与seal。packet故意没有任何query GT/native字段，证明infer不用它们。两个worker×1thread每例约0.879/0.903秒，其中MEAN约0.612/0.564秒，cut约0.244/0.310秒。仅为本地合成成本，不预测服务器600例成本。

RGB图构造 O(V+E)、额外内存 O(V+E)，V=16,384/E=32,512；复用浮点Dinic理论最坏 O(V²E)，实际时延由流相位/图像margin决定，不能拿合成例时间作真实时限保证。locked MEAN重算的N²相似度成本单独报告；已有MEAN场可由上层合法共享同producer值，从而只计附加readout。没有新增编码成本。

## 根代理接入

先上传新独立code snapshot，保持source/recipe/RGB producer hash绑定，运行新的out namespace：

```bash
PYTHONPATH=/root/demo4_cache/env /root/miniconda3/bin/python scripts/run_mean_rgb_potts.py infer \
  --manifest /root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/bound600_v2/rows.json \
  --root /root --out <new-independent-output> --workers 2 --threads 1 --memory-gb 4
```

有限smoke可用原 `bound600_v2/smoke4.json`。infer不从评测文件读任何truth键，完整封存后另行：

```bash
PYTHONPATH=/root/demo4_cache/env /root/miniconda3/bin/python scripts/run_mean_rgb_potts.py score \
  --out <new-independent-output>
```

独立runner支持最多两worker、每worker最多两线程，预算默认4 decimal GB。输出兼容现有独立成对scorer的seal格式；`score`复用该评测函数而不改其源文件，并将报告标为simple-control/independent_methods=0。既有color_bottleneck的mask/counts可通过绑定 `evaluation_controls` 同数据成对比较（需要相应hash），不在infer里读取其GT。

当前真实质量增量及是否超过color_bottleneck均未测。根代理只需做同600队列的有限强控制比较，无需重开机制搜索。

## 共享入口注册（当前推荐接入）

新增可选 `mean_rgb_potts`，原9项 `METHODS` 和原默认4项保持；独立increment=0、kind=`strong_simple_control`。共享配置记录本控制Config、module以及复用ProM4 mincut源码hash。只接受已绑定RGB128 packet及原query H/W；当前MEAN由共享producer重算并复用，不重复MEAN计算。

`prepared_cpu_bundle`输出主行 `mean_rgb_potts` 的128×128 binary cut和 `mean_rgb_unary.control`，使用predict直接给出的1024完整mask；既有 `mean.control`继续同行。独立runner的 `mean_rgb_boundary.control` 与共享主行是同一个配方的不同输出名称，不算两个方法。

`shared_smoke1/`、`shared_check.json`：单个无GT synthetic packet跑完整共享infer后，直接predict与保存field逐元素相等(maxerror=0)，主cut/unary完整mask同直接renderer和通用任意2D→1024 renderer的XOR全0。保持128 field，未压到64。Config JSON roundtrip（tuple→list）保持同固定shape/参数并被正确接受；这仅修复序列化接受边界，没有改变图、unary或solver。

初版数学核验的源码hash保留在 `check.json`；最终注册版module hash为 `7869c717102d00e2672850bc797a3d94f0426c586a647518d37de32ce574fb34`，shared parity和当前依赖hash见 `shared_check.json`。没有因这次注册追加真实样本或参数搜索。

```bash
PYTHONPATH=/root/demo4_cache/env /root/miniconda3/bin/python scripts/run_cpu_feature_candidates.py infer \
  --backend prepared --prepared-methods mean_rgb_potts --primary-method mean_rgb_potts \
  --manifest <bound600_v2/rows.json> --root /root --out <new-output> \
  --workers 2 --threads 1 --cpu-budget 2 --memory-gb 4
```

根代理可在同一调用中并列absorption/gaussian等既有选定行，按当时资源自行调度；本次仅提供入口，没有启动服务器作业。
