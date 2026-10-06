# Pro M2 CPU 缓存变体：部署合同

状态：CPU 数学核心与六个固定臂已实现，小合成检查通过；真实分割未测。未改共用入口、PLAN、STATUS，未访问服务器。主代理负责上传、绑定缓存和运行。

源实现：[pro_reference_relations.py](../../../../src/ics/methods/pro_reference_relations.py:1)。原定义参照用户提供的 Pro 报告 L388–519；renderer L235–250；配置 L1171–1185。身份 `pro_m2_reference_relations_cached_v0`：**不宣称已复现全 RGB / 新 FP32 DINO 编码的 Pro 方法**。

## 调用与输出

```python
from src.ics.methods.pro_reference_relations import predict, render_original

result = predict(q, r, cov64, source_score64)
# q/r: 同生产路径的缓存单位特征或待归一化特征；通常4096x1024。
# cov64: 参考mask面积覆盖率；score: 原始缓存FoRIS pre-CRF连续分数。
# 不把已完成MEAN图求解场默默替代source_score。
fields = {
    'signed': result['field'],
    'zero': result['zero_control'],
    'positive': result['positive_control'],
    'absolute': result['absolute_control'],
    'pair_independent': result['pair_independent_control'],
    'block': result['block_control'],
}
original_mask, work_mask = render_original(fields['signed'], (original_h, original_w))
```

默认 `score` 按当前本地缓存 helper 的 FP32 `minmax=(score-min)/max(span,1e-6)` 转 s；如已绑定归一化 Pro 宿主 s，显式传 `score_is_normalized=True`。不得先生成 MEAN 再冒称同一个宿主。所有六臂共享同 s、字典、图和 renderer；不重新 minmax、不 clamp、不 CRF。输出场求解 FP64，renderer前统一转FP32，bilinear64→1024、严格>.5，再将工作二值mask bilinear到原图、严格>.5，均align_corners=False。为匹配历史1024评分，单独保存 `work_mask`；原图尺寸缺失时不能宣称完整原尺寸Pro终点已完成。

可注入 API：`predict(..., dictionary=centers, ref_groups=..., query_groups=...)`；更小核心 `infer_roles(a_ref,a_query,coverage,normalized_base,ref_groups,query_groups)`。`a_ref/a_query` 是非负、行和1、至多2非零项的角色矩阵；graph dict对应同一类型，值为无向端点对。直接表核验使用 `relation_values(P,a_query,edges)`，连续求解使用 `solve_field(base,J)`。默认图始终是轴向距离1/2/4与精确mutual20NN；小N使用min(20,N−1)，**正式4096路径不降k**。注入图/字典必须在运行收据披露，不能将删图/删角色称为同一默认方法。

## 原样保留的数学合同

- R/Q合并的确定性球面32角色字典，最多20轮；空簇删除、top2温度.07。零向量保留为0；一个实际角色退化top1。Pro没有指定farthest-point首种子，此实现明确取合并序列第0个参考token，所有平手按最小token/角色ID。该未指定实现约定需随recipe保存，不能声称与另一隐含首种子像素等价。
- 四状态表 `P_st=.9*N_st/n_st+.1*p_s*p_t^T`，所有状态同.1；任一状态边质量0则整边类型关系0。两个方向均计入软coverage加权N。`pair_independent` 用每状态**加权端点边缘的外积**替代经验联合表，再做相同混合；不将它错误等同于所有情况下J=0。
- FP64 cross-ratio `Jraw=.25*(log L11+log L00-log L10-log L01)`；无绝对epsilon，残差<1e−10归零、clip±2；重复类型等权平均，包括失效类型的零值。
- signed/zero/positive/absolute/pair-independent/block六臂固定。positive/absolute在聚合raw后变换；各臂独立按最大绝对行和归一。block以8×8参考块坐标奇偶分四组，每次移除涉及该组的边，保持字典固定，重估raw；四次均非零同号才保留并取均值，再归一。不是四份独立数据。
- ℓ=s−.5，λ=.9，同步FP64固定点最多250步，相邻差<1e−6。输出 **t=s+1.8J(2z−1)**；绝不输出sigmoid场。每臂记录残差、最大绝对行和、收缩上界 c≤.9、解误差证书 `||F(z)-z||∞/(1−c)`；非有限/不收敛直接报失败。
- 全空参考六臂输出空；全前景参考关系退化0并保留s；不使用query GT/class/fold/目标数/目标面积。

## 已完成的必要核验

运行：`python3 evidence/local/research_20261006/pro_relations_preparation_01a1100b/check_math.py`。

收据：[math_check.json](math_check.json)，绑定源文件SHA。全部为小合成/独立数学检查，真实episode=0。

- ≤16token的注入核心、软coverage四状态表与独立逐边外积计算、pair-independent表、直接likelihood与cross-ratio对照；最大表差2.22e−16。
- 含低概率项的严格因子分解P得到**精确J=0**，没有epsilon制造伪关系。
- zero臂的1024及41×59原图renderer逐像素相同；故意错误的sigmoid读出与工作mask差1578像素，检查能捕获该错误。
- 8个小严格凸目标与独立BFGS优化器比较，最大z差约4.56e−7，各差异均落入保存的固定点残差证书（最大证书约3.74e−6）内。
- 一个人工给定表例中 signed删除弱错误、zero/positive/absolute保留；signed场包含>1值，确认没有隐藏clamp。**该表例尚未构成从默认字典/自然输入产生的增益证据。**
- 原样block四删除图与独立表重建/组合逐场相等；默认小缓存完整路径实际学习4角色、构建四类边、生成六场；空参考、全前景、零特征单角色均有明确有限输出。
- 25token精确mutual20边与独立全排序一致，包括边界余弦平手和分块批大小变化；等权重复边类型平均核对通过。

默认packet同时覆盖默认字典/图到六场；这只是数值接线检查，未声称它能在匹配简单控制后修复指定真实目标。

## 部署局限与成本

依赖已有numpy/scipy；renderer另需现有CPU torch，不安装资源。正式输入建议仍绑定原4096×1024网格；核心支持不同R/Q矩形网格用于合成核验。`Config`固定Pro参数，只允许改变cosine矩阵分块大小以控制内存，不允许参数搜索。

两图的精确mutual20需要O((Nr²+Nq²)D)乘算；字典O(20(Nr+Nq)32D)。只分块、未删候选、未近似邻居。R图通常不能直接从已有Q图复用，合并32角色也不能复用九项8个FG模式。输出info单列字典/分配、R图、Q图、六臂核心、总CPU数学时长；不含编码、I/O、renderer或GT评分。真实4096路径时间/内存仍未知，不承诺CPU分钟预算。默认六臂同时产生，简单控制不是新增方法数。

缓存q/r通常来自历史FP16量化/位置处理，原始score也需确认producer；原Pro约定新FP32编码与完整宿主，二者不能冒称等价。主代理上传后可先做固定小运行绑定/成本检查，再决定同队列完整评分；所有预测先封存，评分单独读GT。严格主张目前只到“原Pro M2数学对象及六臂在缓存变体中可部署”，自然迁移收益仍未知。
