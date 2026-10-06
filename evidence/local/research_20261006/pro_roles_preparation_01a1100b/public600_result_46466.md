# M3 固定600例完整缓存评测

**结果为负：**M3 heldout59.563526，低于native61.335314、stored MEAN62.931530和同树zero60.878311；heldout相对all-role也没有正增益证据。可以收束当前固定缓存版本的性能主张，不能由此否定整个结构对应方向。没有启动新M3变体。

## 本次完整运行身份

- 新46466 CPU运行；快照`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/code_318a5564bafd`，方法/参数/四行与此前screen固定一致。原24目录未覆盖。
- 固定原顺序`bound600_v2/rows.json`全部600条，80类、570照片关联组，最大组3例。是既有暴露缓存复用，**不是独立确认**。
- 结果目录`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_roles_public600_v1`；推断全部封存后独立score，二者退出码0。未触发8GiB资源停止。
- **1024工作空间缓存适配版**；没有RGB到新DINO编码，也没有原尺寸第二段renderer。该边界不因例数增加而消失。新编码/GPU/下载均0。
- 使用6workers×1thread=6CPU，taskset154–159。所有自有推断和评分进程已结束，6CPU已释放；未自动新增队列或变体。

## 同例线程核验

先在相同smoke4第一例独立运行1worker/1thread，对原screen1的1worker/4thread比较：四行packed mask、四行token field、未截断heldout/all-role诊断及evaluated标记**全部逐值相等，最大差0**。两次4979次Hungarian一致；单例3.875793秒对2.892896秒。仅此一例是线程数值核验，不用它宣称全队列部署时间。

远端记录：`runs/pro_roles_thread1_parity_v1.json`、`runs/pro_roles_thread1_screen1_v1`。核验通过后直接完整600，不按24结果重选参数。

## 完整成绩与配对比较

| 600例class-summed mIoU | 成绩 |
|---|---:|
| native | 61.335314 |
| M3 heldout | 59.563526 |
| M3 all-role | 59.929347 |
| M3 mean | 58.567621 |
| M3 zero | 60.878311 |
| stored MEAN | 62.931530 |
| RCG | 63.088942 |
| fine16 | 63.464527 |
| fine64 | 63.681184 |

| heldout相对 | 点差pp | 配对照片组95%区间 | 上/下/平例数 |
|---|---:|---|---|
| native | -1.771788 | [-3.103075,-0.670443] | 189/402/9 |
| all-role | -0.365821 | [-1.179057,0.645387] | 119/120/361 |
| mean | +0.995906 | [-0.493012,1.948686] | 157/170/273 |
| zero | -1.314784 | [-2.670559,-0.282979] | 134/189/277 |
| stored MEAN | -3.368004 | [-4.706402,-2.309515] | 155/428/17 |
| RCG | -3.525415 | [-4.851461,-2.411186] | 159/426/15 |
| fine16 | -3.901001 | [-5.313658,-2.807776] | 136/448/16 |
| fine64 | -4.117658 | [-5.626427,-2.243541] | 171/412/17 |

共享评分器2000次RandomState(0)照片关联组配对重采样，原类别/照片身份与基线完整mask沿bound600绑定。没有把episode IoU平均或oracle写成主指标。fine控制保留其历史producer，不称新全局Pro合法版本。

四折heldout−stored MEAN分别约**-3.956694/-3.328140/-2.774700/-3.412480**，均反向；这些是同折已测成绩之差，不跨流水线相加。24例的正点差没有在完整600兑现。

相对native的四类像素改动：add_TP2660917、delete_FP6737408、delete_TP3989232、add_FP4886375。误删真目标多于补回真目标，虽删掉不少FP，完整class mIoU仍下降；不把原始像素数相加宣称mIoU因果贡献。

## 回退与成本

- 活动结构分支**599/600**；`full_foris_native_no_template`回退**1/600=0.1667%**。回退发生occurrence000433（fold2,e133,c30，参考8个粗连通组件但没有≥4非空角色的可用模板），依法只额外读native，未读query truth。该例四行均回到绑定full-native。
- 六worker推断完整守卫窗口 **465.988670秒=7.7665分钟**，四行全含共享加载/Ward/匹配/renderer/封存；评分守卫窗口 **22.323187秒**。二者合计约**8.1385分钟**。
- 单例mean4.584944秒、median4.409701秒、p95 **6.371520秒**、max **13.348076秒**。单例窗口合计2750.966193秒是并发各例时间之和，不冒称墙钟。
- 自有进程组RSS按250ms监控峰 **4940128256 bytes≈4.94GB（4.60GiB）**，低于8GiB；最大worker RSS high-water960327680 bytes。RSS指标来自真实守卫，不用runner的六倍worker高水位估算替代。
- Hungarian总5211719次，最大单例115708次；平均Ward0.601615秒、平均matching0.814505秒。包含all-role控制的匹配调用与共同准备，不将四行研究成本写成单独heldout部署成本。
- **此前约49分钟是24例、1worker×4threads的线性外推，不是600实测。**本次真实600为7.77分钟，明显低于该估算；未实测600的旧并发配置，因此不宣称一个经过同600对照验证的加速倍数。

## 证据与哈希

远端原始产物完整保留：`sealed.json`、`config.json`、两个manifest、`receipts/`、`fields/`、`predictions/`、`score/report.json`、逐例计数、修正账本、bootstrap抽样。精简机器读摘要另存`runs/pro_roles_public600_v1.summary.json`。自有守卫结果为`runs/pro_roles_public600_v1.infer.guard.json`和`.score.guard.json`。

- evaluation_manifest SHA256：`07e21be036f3f5acda24de22fcc196d9f9b65b962e9a07aa7790da13bd024aba`
- sealed SHA256：`19deb8ea8a97b0e6a4ff7d535f23660776d112ddcab3d8ae10fa5a1e86d95fb2`
- config SHA256：`cc15cf1d7dd13189643e42607bf79f2426c58ef74e5f6f7055f8c66467e8f910`
- score/report SHA256：`75fa77e10479eaea6fc9a5a1c58f9b8544782ad2f069f54a32b10f3823a316c0`
- 方法源码SHA256：`cfdd881339202fa52205138e82afb26c4bd30af23924735d39a182f9ff1e1957`
- runner源码SHA256：`c4f8a6ecdd765d81b7e6cc445d98edbc8e0b7a715a06b65e8c2ae50392621c38`

本次没有同合同bug修复、算法变更或共享/root记录修改。仅更新自有成本守卫的worker/thread参数与CPU affinity；没有下载缓存/模型/预测二进制。根代理可直接据上述完整结果进行整体选择；当前M3性能主张应收束，不由负结果自动产生新M3任务。
