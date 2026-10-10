# 主代理任务：将已证实的困难区域区分信号变成完整输出

状态：已完成现存字段、原始O24的200例信号对照，以及第一项完整输出检验。完整预测先封存后评分；下面记录实际正结果及未解决部分。

主研究文件：`docs/research/DIFFICULT_REGION_SIGNAL_STUDY_20261010.md`。脚本与收据：`evidence/local/difficult_region_signal_20261010/study.py`，`existing/`与`raw/`均已封存200例并完成评价，原始输入1200次请求全部核对profile、实际transform key、payload与tensor SHA；DINO/新raw写入0。raw阶段105.58秒，其中输入解码/校验54.16秒。全部CPU2线程。

ROI固定为FoRIS预测前景，以及FoRIS=BG而已有P0直接读出=FG的新增区域。按fine128节点分数、canonical1024 ROI内实际FG/BG像素质量计算weighted AUC/AP，不能当原图mIoU。对Deep两个ROI各100有效；PACO各95/96有效。

|困难ROI内宏AUC|Deep已有FG|Deep新增区域|PACO已有FG|PACO新增区域|
|---|---:|---:|---:|---:|
|旧P0|.7408|.7667|.7556|.6749|
|保留候选峰值差|.5782|.6128|.6893|.5963|
|旧整图竞争h|.6316|.6322|.7409|.6577|
|完整FoRIS source response|.6665|.8115|.7455|.7596|
|raw ridge whole|.6901|.7559|.7790|.7439|
|raw ridge local4|.7754|.8355|.7115|.6304|
|source-branch APD ridge whole|.7030|.8015|.7749|.7641|
|source-branch APD ridge local4|.7893|.8667|.7132|.6717|

Deep新增区APD ridge local4的AP .3667（P0 .2389），TPR@5%FPR .5338（P0 .3676）。PACO新增区APD ridge whole的AP .3022（P0 .2453），TPR@5%FPR .3748（P0 .2619）。Huber与ridge近似，无足够新增区排序收益支持增加12轮IRLS。相同特征的双原型比ridge弱。APD按官方whole-query语义分支应用，200中191开启；投影复用原basis，local与whole共用同一分支以隔离投影因素。

峰值差不能直接替代LSE：新增区若按背景峰值更强拒绝，Deep删70.63%干扰同时删50.07%真目标；PACO删78.47%干扰同时删64.41%真目标。都是fine128符号的ROI质量诊断，并非已经实施的原图候选。

继承旧ridge的零阈值也不直接可用：Deep whole37/100例、local22/100例全空；PACO whole15、local17例全空。当前拟合用旧量化子集128/role、连续参考coverage目标y=2c-1、平衡权重、lambda=.01，单位O24；没有用query GT拟合方向。

下一步（尚未执行）：只用合法reference中未入ridge拟合子集的patch选择标量读出，记录空角色与回退；保留未标定直接零阈值作为控制。在相同APD-ridge字段上比较whole/local及明确的融合，query上下文只能作为可检查的视图可靠性证据；不能按数据集名称选最优视图，也不能用query GT选阈值。以完整FoRIS、MEAN、现有九窗、旧whole竞争作同批完整mask对照。当前仅证明排序信号更强，尚无新完整mIoU。

供其他线使用：`raw/episodes.jsonl`包含逐例三个ROI、每个信号的AUC/AP/TPR；`raw/sealed.json`包含APD开关及输入身份。`existing/episodes.jsonl`提供旧P0/峰值/旧h/source score对照。实际field键为`source_apd.ridge.global`和`source_apd.ridge.local4`等，形状128×128。请不要重复运行或修改这些封存产物。

## 已冻结的第一项完整输出检验

入口`evidence/local/difficult_region_signal_20261010/candidate.py`；产物`candidate_reference_v1/`。仍同200，无新编码，仅读取每例已核验reference O24并复用保存query字段。配方、运行代码与数学依赖在生成预测前保存，全部200预测封存后再评价。

- 固定APD-ridge方向，不采用无额外信号收益的Huber。沿用旧subset/lambda，参考拟合子集不变。
- 用未入拟合子集的reference patch及其真实连续coverage，选择该参考上的严格阈值以最大化source IoU；如该补集中一个角色为空，明确记录使用完整reference回退。这是合法reference监督，邻域/同图相关仍存在，不能称独立校准。
- 定义有界参考证据`h=clip((score-t_ref)/(mean_F_ref-mean_B_ref),-1,1)`，两类均值使用同校准reference集；均值差非正时证据为0。它不是校准概率。
- 观察控制为whole、local4、固定等权平均。候选`scene_selected`的local权重alpha，由现有query FoRIS FG/BG分组、原source response绝对置信度构成平衡平方损失，闭式最小化并限制在[0,1]；不使用query GT或数据集名称。此处query上下文可错误，必须实测。
- 完整mask用已有bounded anchor规则`g0+.5*h`，g0符号取同帧完整FoRIS，幅度取原source score minmax响应；平局继承FoRIS。每例验证零证据精确还原FoRIS。
- 保留未标定零阈值global/local、reference阈值独立global/local，以及四个anchored输出。它们只隔离标定、观察和anchor的贡献，不扫权重/阈值菜单。原图同完整FoRIS/MEAN/快九窗/旧whole竞争配对。

标量阈值同分处理、严格阈值参考I/U、已知解析alpha端点控制已检查通过；这只验证实现，不作质量证据。

## 第一项完整输出：实际部分成功，视图选择未解决

|原图同批完整输出|Deep100总I/U|PACO100观察class/fold mIoU|
|---|---:|---:|
|完整FoRIS|29.377542|45.563915|
|MEAN|26.465369|46.687946|
|原快九窗|37.344028|40.956998|
|旧whole角色竞争|29.529416|47.579863|
|新ridge零阈值 whole/local|20.727848 /24.163139|39.090160 /31.581252|
|reference阈值独立 whole/local|28.298623 /31.205260|24.300485 /18.919154|
|bounded anchor + 新whole ridge证据|32.014976|49.227401|
|bounded anchor + 新local ridge证据|34.288729|48.915185|
|bounded anchor + 固定等权证据|33.141307|49.171141|
|bounded anchor + scene-selected证据|32.641402|49.304596|

结论：原始特征中找到的任务方向已经兑现成两组相对FoRIS/MEAN的完整输出增益；PACO也超过旧whole竞争。Deep仍低于已有快九窗37.344，不能称全Strong目标完成。reference单独定阈值在Deep改善零阈值，却严重伤害PACO，说明合法source监督仍不保证绝对分数跨图可用。anchor保留的query场景信息至关重要。

scene selector没有证明必要性：Deep平均local权重.267、55例选全global；PACO平均.100、86例选全global。它在Deep明显低于固定local，在PACO仅比固定global高.0772。未将它设成默认方案，也不凭最高单组分数挑数据集专用规则。两组均无空参考校准角色回退/非正分离度。生成200例18.31秒，评分2.07秒；这是共享查询字段后的CPU/IO增量，不是冷端到端时间。

供融合/身份/数据差异线整合：`candidate_reference_v1/report.json`、`scored_episodes.jsonl`、`sealed.json`（含每例reference threshold、尺度、local权重）、`predictions/`已可读取。若需要对应h，可从`raw/fields`的APD ridge score按receipt中的threshold/separation重建clip值；不要改写本候选。根据信号选择的下一改动必须解释：怎样保住Deep已有局部九窗收益，同时保留PACO的整图身份优势，而非强制global/local平均。
