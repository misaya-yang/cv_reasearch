# 服务器实验计划：确认 FoRIS-free 头 SH

2026-10-14 写定。本计划只做确认，不在服务器上选方法。任何结果出来后，方法、常数和读法都不改。

## 1 要确认什么

**本机结论**（固定种子子样本，每域 ≤ 150 例，详见 `docs/research/x1012_tasks/W19_simple_head.md`、`W23_simple_pipeline.md`）：

| 比较 | S 半（选择用） | T 半（一次性） |
|---|---:|---:|
| SH+ − 完整 FoRIS | +2.80（显著正／负：6／0） | +3.28（6／0） |
| SH+ − MFU | −0.02 | +0.11 |

头部时间（CPU 2 线程）：SH+ 1.28 s，FoRIS 1.56 s；不做 CRF 时 0.29 s。

**服务器要回答的三个问题：**
1. 在双角色都没曝光过的新照片上，SH+ 是否仍然不低于 FoRIS？（E1）
2. 在与 FoRIS 表 1 相同的官方协议上，SH+ 的主表成绩如何？（E2）
3. 在 GPU 端到端条件下，SH 比 FoRIS 快多少？（E3）

E0 是前置门槛：环境和实现一致性不过关，就不进入 E1–E3。

## 2 冻结的方法 SH（不得改动）

参考实现是 `scripts/x1012/w23_simple_pipeline.py::simple_head`（SHA 前缀 5890b666），依赖 `r5_candidate.py`（6be00b51）和 `w19_simple_head.py`（50418a0f）。

**输入：**
- 参考图像和查询图像经 FoRIS 预处理（`build_transform(1024)`）后得到的 DINOv3-L/16 最后一层 token，记为 O24，形状 4096×1024，FP32；
- 参考 mask，最近邻缩放到 1024²；
- 查询 RGB，仅供 CRF 使用。

**步骤：**

1. **去偏与覆盖率。** 用 `positional_basis.pt` 做位置去偏，得到单位特征 r、q。参考覆盖率 y 取每个 16×16 块的平均。

   rho = 1 − Σ(y·[y ≥ .9]) / Σy
2. **核迁移 c。** mu = (r̄ + q̄)/2；sim = unit(q−mu)·unit(r−mu)ᵀ。

   c = Σ_j k_ij·y_j / Σ_j k_ij，其中 k = exp((sim − max)/.05)。

   若 rho > .9，直接输出 minmax(up(c)) > .5（T 回退），不做 CRF。
3. **新奇调制。** 先做 k-means-64 区域（球面、最远点初始化、6 轮迭代）。

   c_mod = c × novelty^s，novelty = minmax(sim_max)。s 从 {0, 1/8, …, 1} 中按 max-J 规则选择：
   - J = P·C/(P + C − P·C)
   - P 是掩码原生 token 上 c 的均值
   - C 是这些 token 上反向迁移 back_f 的和
4. **查询图 L。** 每个 token 取余弦 top-20 近邻（排除自身），d = 1 − cos，权重 exp(−d/d₂₀)，d₂₀ 是第 20 个近邻的距离。用 √(w∘wᵀ) 对称化（只保留互为近邻的边），除以平均度，L = D − W。
5. **点证据线。** U0 = minmax(up(solve(c_mod)))。

   solve 求解 (diag(a) + 16L) f = a·s，其中 a = (.1 + |2s−1|)·sigmoid(log_e − median)。
6. **原型证据线。**

   p = q·μ_fg − .55·q·μ_bg⊥
   - μ_fg 是 y ≥ .5 的参考 token 均值；
   - μ_bg 是与 μ_fg 最相似的 20% 参考背景 token 的均值，再对 μ_fg 正交化。

   M′ = 排序引导求解：y′ = minmax(p) + .25(rank(guide) − rank(minmax p))，同一张图 L，λ=16；guide 是查询与参考纯前景（y ≥ .9）均值的相似度。
7. **输出。** SH+ = 官方 CRF([.5·up(M′) + .5·U0 > .5])，CRF 带宽 10 px、p_core .95，与 FoRIS 的 finalize 相同。

## 3 预注册与封存规则

1. 运行前，把以下各项写入 `cv_data/a/server_<run>/evaluation.json`：
   - 实现代码的 SHA；
   - 参考实现的 SHA；
   - 官方 manifest SHA（前缀 e6d2a515）、protocol SHA（baa10d8d）、留出 manifest SHA（4131586e）；
   - 权重与位置基的 SHA；
   - GPU、CUDA、torch 版本。
2. 全部预测写完并封存（每个预测文件的 SHA 汇总成 `sealed.json`）之后，才读 GT 评分。
3. 新照片留出只评分一次。结果不用于任何选择，方法不因结果改动。
4. 某一域结果不利时，如实报告。原因只在开发数据上诊断，不回到留出或官方集调整。

## 4 实验

### E0 环境与一致性（门槛，约 1 小时）

| 项 | 做法 | 门槛 | 不过关时 |
|---|---|---|---|
| E0.1 编码器 | 服务器 CUDA FP32（关 TF32）编码 20 张图，与本机缓存的 O24 比较 | 逐 token 余弦的最小值 ≥ 0.999 | 查预处理、权重 SHA、TF32 设置 |
| E0.2 SH 实现（精确） | 用本机导出的 O24 特征（40 例，每域 4 例，约 1.3 GB），在服务器 CPU 上运行 SH，与本机的 CRF 前掩码比较 | ≥ 39/40 例逐像素一致；其余例 \|ΔIoU\| < 0.1 | 对照参考实现逐函数排查，常见原因是 FP16 舍入、kmeans 初始化、topk 平局 |
| E0.3 SH 实现（端到端） | 服务器编码，在 W23 T 半子样本中抽 200 例（每域 20 例）运行 SH+ | 与本机 SH+ 的十域均值差 ≤ 0.3，且无一域差超过 1.0 | 分别查编码器（E0.1）和 CRF（E0.4） |
| E0.4 CRF 与 FoRIS 重现 | 官方 Fundus 200、ISIC 600、Lung 600 例上运行完整 FoRIS | 与本机重现（19.13 / 62.73 / 87.71）各差 ≤ 0.5；发表值为 19.7 / 62.9 / 87.6 | 查 CRF 编译和参数（`init_crf(1024)`） |

E0.2 需要的本机导出包，在本机写代码时一并生成：40 例的参考与查询特征、行信息和期望掩码。

### E1 新照片留出（一次性，约 1 小时）

- **数据：** `a/x1013_photo_holdout/manifest.json`，3,211 例、5,034 张图像。参考图和查询图都与开发曝光零交集。此前已用于评分一次 MFU，SH 从未在它上面做过选择。
- **臂：**
  - FoRIS（官方完整）；
  - **SH+（主）**；
  - SH（不做 CRF）；
  - U0、M′ 两条单线（CRF 前，只作描述性消融）。
- **统计：** 与 `docs/research/x1012s_tasks/PHOTO_HOLDOUT_RESULT.md` 相同：
  - 每域把 (fold, class) 槽池化后等权，十域再等权；
  - 2,000 次按 canonical 查询照片的配对 bootstrap，seed 2026101303。
- **主比较：** SH+ − FoRIS。
- **读法（预先写定）：**
  - 所有域点估计 ≥ 0，且十域均值下界 > 0：可以写“十域都不低于 FoRIS”。
  - 有域点估计为负但区间含 0：写成“无一域显著低于 FoRIS，k 域显著更高”。
  - 有域区间上界 < 0：十域主张不成立，如实报告该域。
- **次要（描述性）：**
  - SH+ − MFU，MFU 用本机已封存的留出预测。两者特征来自不同设备，只作参照，不做检验。
  - 服务器 FoRIS − 本机 FoRIS（52.79）：跨设备一致性检查。

### E2 官方协议主表（约 6–10 小时，以 E0 实测为准）

- **数据：** `a/x1013_official/manifest.json`（60,981 例）和 `protocol.json`（各 fold 的期望类别）。

  | 数据集 | 例数 |
  |---|---:|
  | COCO | 4,000 |
  | LVIS | 23,000 |
  | PASCAL-Part | 8,459 |
  | PACO-Part | 10,000 |
  | SUIM | 3,859 |
  | iSAID | 10,063 |
  | ISIC | 600 |
  | Lung | 600 |
  | Fundus | 200 |
  | Road custom200 | 200 |
- **臂：** 与 E1 相同。
- **指标：** 与 FoRIS 官方脚本一致：
  - 每个 fold 内，类 IoU = ΣI / max(ΣU, 1)，在期望类别上平均；
  - 再对 fold 平均；
  - CLI-1024 画面，忽略像素不计。

  评分代码沿用 `scripts/x1012/w12_official.py::score`。
- **置信区间：** 1,000 次照片 bootstrap，seed 2026101303。
- **运行顺序：** 小集先跑，便于早发现问题：Fundus → ISIC → Lung → Road → SUIM → COCO → PASCAL → PACO → iSAID → LVIS。
- **报告：**
  - SH+、SH、FoRIS 重现值与 FoRIS 发表值（COCO 60.9、LVIS 42.8、PASCAL 55.8、PACO 42.3、SUIM 59.1、iSAID 53.6、ISIC 62.9、Lung 87.6、Fundus 19.7）并列；
  - SH+ − FoRIS 重现值及其区间；
  - 其他训练无关方法引用 FoRIS 表 1 的发表值。

### E3 端到端计时（约 30 分钟）

- **样本：** 100 例，每域 10 例，取自官方 manifest，种子固定。分别报告非回退例和全部例。
- **流程：** 从解码后的 RGB 开始，到最终掩码为止。两种编码条件：
  - (a) 参考和查询都新编码；
  - (b) 参考已缓存，只编码查询。
- **方法：** FoRIS 完整；SH+；SH（不做 CRF）。SH 的头部分别在 CPU 和 GPU 上运行，两个结果都报。
- **控制：**
  - 同一进程内运行，方法顺序逐例交替；
  - 丢弃 5 个预热例；
  - 计时前后调用 `torch.cuda.synchronize()`；
  - 报告均值、中位、p95 和 GPU 峰值显存。

## 5 结果如何进入论文

- E1 决定主张 C1 的措辞（读法见上）。E2 是主表。E3 支撑“更简单、更快”。
- U0、M′ 单线的结果构成消融表：点证据与原型证据互补，融合优于任一单线。
- COCO、ISIC 在本机两个半区的点估计都略低于 FoRIS（不显著）。如果 E1/E2 在这两个域出现显著负值，要作为局限写出：区域整合尺度的折中，见 W20、W24。

## 6 移植说明（写服务器代码时用；本计划阶段不写代码）

1. **方法实现。** 逐字复制参考实现用到的函数，不要重写：
   - `r5_candidate.independent/modulate/graph/quality`；
   - `w09_integration_lambda.solve`；
   - `w19_simple_head.mean_solve`；
   - `lab.unit/debias/coverage64/mask1024/kmeans/minmax/up`；
   - `plan_common.readout`。

   必须保留的细节（它们会改变结果）：
   - log_e、sim_max、back_f 先转 FP16 再使用；
   - kmeans 用确定性的最远点初始化；
   - kNN 排除自身；
   - CG 参数为 rtol 1e-7、atol 1e-9，maxiter 300（M′）和 2000（U0）。
2. **编码共享。** 每张图只编码一次。FoRIS 通过回放编码器读取缓存的 O24，做法参照 `scripts/x1012s2_runtime.py::SharedReplay`。FoRIS 由 `ics.foris.build_host(device='cuda', …)` 构建，并用 `reuse_native_basis` 加载同一个位置基。
3. **CRF。** SH 与 FoRIS 共用同一个 `init_crf(1024)` 模块和 `crf_refine`。
4. **进程布局。**
   - 1 个 GPU 进程负责编码，并按图像哈希维护 LRU 缓存，上限 ≤ 96 张（每张 16 MB）。本机曾因上限设为 400 而占用 22 GB；
   - N 个 CPU worker 跑头部；
   - 每例写一条记录，支持断点续跑；
   - 全部完成后再封存。
5. **路径。** manifest 中以 `/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/` 开头的路径，替换为服务器的 cv_data 根。输入读取用 `ics.official_data.load_inputs`，它带 RGB 和 mask 哈希校验。
6. **随机性。** SH 没有随机步骤。FoRIS 用 `torch.manual_seed(0)`，CPU 2 线程或 GPU 都可以，但必须在 evaluation.json 中记录。

## 7 时间预算（估计，以 E0 实测替换）

| 阶段 | 估计 |
|---|---|
| E0 | 1 h |
| E1 | 编码 5k 图 + 3.2k 例头部，约 1 h |
| E2 | 编码 36.8k 图 2–3 h；头部（8 worker）4–7 h |
| E3 | 0.5 h |
| 合计 | 约 1 个工作日（无人值守） |
