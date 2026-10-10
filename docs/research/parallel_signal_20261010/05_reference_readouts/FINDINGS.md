# 05阶段证据

2026-10-10，完成，无阻塞。完整结果与建议见本目录 `REPORT.md`；独立配对重聚合与reference coverage统计主收据为对应evidence目录的 `audit.json`。

## 当前排序的独立核验与关键边界

1 CPU、6.31秒，从400份封存字段、200份预测、200份query mask与200份reference mask实际读取校验；独立降序tie-group算法重新算7200个AUC/AP/TPR数值，最大差0。没有拟合、编码或生成新预测。

- Deep100：100个不同reference照片＋100个不同query照片，无跨角色复用。APD ridge local4相对P0，在已有FoRIS前景AUC +.048485（82/100胜，配对照片95%描述区间[.034349,.063114]）；在新增区域+.099951（89/100胜，[.077150,.123349]）。相对更强FoRIS source，新增区AUC仍+.055178（71/100胜），AP+.160018（91/100胜）。
- PACO100只有87个query照片、91个reference照片，二者关联成73个照片连通组，最大组6例；有效ROI为95/96，fold/class共87。按照片连通组配对重采样后，whole相对P0新增区AUC+.089209（67/96胜，[.048702,.131291]）、AP+.056944（64/96胜，[.024668,.092182]）仍为正。
- **PACO whole相对FoRIS source的新增区AUC仅+.004518（48/96胜，照片区间[-.035071,.048606]）**，AP+.037415（51/96胜，[.007357,.072643]）。所以PACO目前更可靠的新证据是相对source的AP改善，不能称AUC也明确胜过完整source。已有FG AUC相对P0+.019316的区间也跨0。
- PACO新增区whole相对P0的within-image FG×BG pair-mass加权AUC差为−.026373，尽管episode均值+.089209；这不是最终官方mIoU，也不是跨图pooled AUC，但说明大面积争议区域未普遍获益。class等权AUC差仍+.089056，重复类别本身没有解释掉宏收益。

## 未拟合reference patch并不等于同分布独立验证

- Deep holdout保留原reference FG质量比例median45.60%，p10 5.24%，最少1.30%；PACO median71.91%，p10 3.78%，最少0.45%。两组都没有完全零FG质量，但已明显改变像素先验与纯度。
- Deep holdout中27/100没有`c>.5` token、59/100没有`c≥.9` token；PACO分别10/100、17/100。硬化coverage会把这37例校准集变成全背景。以全部reference计算，Deep本来只有3例没有majority-FG token，PACO为0，额外缺失主要由采样选择造成。
- holdout FG质量中，与某个拟合patch Chebyshev距离≤1的比例：Deep平均88.77%、中位94.49%；PACO平均85.62%、中位96.51%。加上同图DINO上下文，不能将holdout准确率/最佳阈值称跨图置信度。
- `c>.5`硬化会遗漏Deep全部reference道路FG像素质量的38.95%（按图平均48.33%），PACO为3.31%（按图平均5.97%）。这足以支持保留连续coverage pixel-mass标定语义。

## 已确认的配方身份

主线保存的 `raw/frozen/historical_invariance_support.py` SHA256 为 `63918543831c4c8a1db98033805a895559fcf77df5d07c2662d558bc3f22773a`，与native200正式 `score/report.json` 的该模块SHA完全一致。因此 ridge/Huber 核心可追溯到真实native200执行版本，不能只称未绑定的旧Git代码。

**连续coverage目标不是本轮新改法。** 旧 `_fit_sample` 已用 `y=2*wf/wvalid-1`，权重 `.5*wf/sum(wf)+.5*wb/sum(wb)`，128/role cumulative mass quantiles、去重、lambda=.01，Huber delta=.5/12轮。当前 valid=1 时拟合公式相同；新变化在实际输入、APD、whole→fine128特征插值、真实局部观察与最终读出，而不是把旧binary label改成soft label。

## 合法标定仍需保留的语义

- 该公式是对coverage `c`做role混合权重的回归，不是把一个mixed patch拆成正负两份后做balanced binary regression。后者的等价目标为 `(c/C-(1-c)/B)/(c/C+(1-c)/B)`，通常不等于`2c-1`。不能把score 0或`(score+1)/2`直接解释为balanced role probability。
- 128/role quantile采样按GT合法reference质量挑patch再去重，剩余patch不是独立随机holdout；可能优先耗掉稀少的高coverage前景。将剩余patch用于阈值是合法reference校准，但不能叫独立cross-image验证。正在统计其剩余FG质量与纯度。
- 对binary patch决策评价连续reference覆盖，正确pixel-mass IoU为 `sum(c*pred)/(sum(c)+sum((1-c)*pred))`；用`c>.5`先硬化会改变细道路的目标语义。balanced拟合权重也不应偷偷替代pixel-mass IoU权重。

## 已确认的旧正结果

native200为实际12方法、200复用开发例、74类；不是300卡已执行。正式报告有32个含controls评分，全部从现有逐类I/U复算一致。Huber52.373390、ridge51.662537、average logistic50.346700、constant erosion50.717844、prototype42.580632。Huber相对ridge+0.710854、旧照片连通组bootstrap95%区间[+0.135326,+1.001446]为真实旧开发边际；相对prototype的大头不能全归Huber。DR03/04/08均低于average logistic。尚无本轮同协议完整FoRIS/MEAN可比成绩。

Huber相对ridge在74类中47升、25降、2平；单class53贡献净macro差的61.21%，该类I/U由5161/11955到9324/12371。剔除此类后的描述性差仍+.279501，不能说完全由单例/单类解释，但旧+.711并非均匀收益。当前Deep/PACO相同APD特征下没有重现Huber排序优势：两个新增ROI分别AUC −.000471/+.000688；PACO TPR@5%FPR反为−.015645，照片区间[-.029101,-.003816]。

补查旧processed600原始评分报告后，还确认了应保留但不能抬成强基线胜出的局部正证据：reference quadratic51.607030，比linear+2.473630 [1.123537,3.969725]、homogeneous quadratic+1.954700 [.789729,3.384554]，但比同MEAN−11.324516；reference hull48.915345比affine+.960172 [.195922,1.555419]，但比MEAN−14.016201；query recurrence62.893978比all-seed+.848400 [.372796,1.418694]，但比MEAN−.037568。原报告及差值已实际读取核对，当前缺其逐例I/U，未重算旧CI。它们说明模型/支持选择确能改进受限读出，却不支持原样替代完整宿主。

## 待整合

本线继续核查APD-ridge paired AUC/AP/TPR及类别/照片依赖，不生成候选mask。上述coverage回归语义和非随机holdout是标定解释约束，不是新增候选要求；主线可保留原拟合并仅标定输出，实际是否涨分仍由完整输出决定。
