# 查询基底完整方法600例：已完成，拒绝作为跨数据集方案

2026-10-10，`codex_m4`。本轮真实产生600例完整输出，每例有查询基底、相同流程的参考基底、直接参考 ridge 三臂，各自保存 direct／原 FoRIS CRF、CLI1024／原图。没有新增 DINO 编码；实际原始 FP32 O24 的读取、输入／payload／tensor SHA、源码／参数及掩码均留存。全部预测封存后才读取 query GT。

完整新方法对四组 FoRIS 都失败。它不是十集涨分方法，不继续扫其 K、温度、lambda 或阈值，也不扩到新660例。PACO 的查询基底相对匹配参考基底正差必须保留，但不能将它单独解释成更好的目标身份。

|固定同批，CLI1024 mIoU|n|完整 FoRIS|查询基底＋CRF|参考基底＋CRF|参考 ridge＋CRF|
|---|---:|---:|---:|---:|---:|
|DeepGlobe 自定义探索道路，总 I/U|100|29.377542|15.986349|19.910111|31.545417|
|PACO-Part，87观察 fold/class|100|45.629834|44.100479|36.231942|39.036596|
|COCO，fold0/1 官方40槽位|200|63.170230|41.629344|46.446268|51.097512|
|LVIS，十折200观察类|200|46.679984|28.064457|31.451061|36.476455|

原图口径同向，另存于原表；PACO303缺类口径也另报。所有数据此前已用于开发或诊断，不能称未曝光确认。LVIS200每类一张查询照片，类内照片 bootstrap 退化；PACO100也有大量单照片 strata，不将条件区间解释成跨照片泛化。Deep 不是公开13.3分所用配对，当前 pooled I/U也不替代缺失公开协议。

## 从完整错误和匹配控制得到的机制约束

查询基底相对参考基底在 PACO提高7.868537点，在另外三组降低3.923762／4.816923／3.386604点。因此“查询组织天然能提供更好的任务判别”没有成立。

独立全600诊断发现，查询基底的全图宏AUC在 Deep／PACO／COCO／LVIS为.8772／.8690／.8742／.8884，参考基底为.8972／.9059／.9382／.9395。FoRIS已有前景内查询基底为.6192／.7406／.5965／.6303，参考基底为.6601／.7389／.7088／.7358。这是64网格分数对真实1024像素质量的排序诊断，不是原图分割分数；各 ROI 有效样例另存。新实际新增区中，参考基底对相同 ROI 的排序也普遍更强。

PACO查询基底的完整预测面积／GT面积1.276，参考基底.778；查询基底恢复更多 TP，同时增加更多 FP。它的 source 风险和全图排序均更差，现有正差主要支持读出范围／召回的变化。Deep／COCO／LVIS分别预测3.385／1.541／2.323倍GT面积，误检增加且困难区排序下降。问题已经发生在完整阈值和 CRF之前，不能只归咎于零阈值；原CRF最多修复局部边界，未改变这一判断。

这些结果不否定冻结 DINO信息、参考判别或一切 query transduction；它们否定当前固定128组件、query occupancy基底和同risk重建作为统一方法的充分性。直接参考 ridge保有较强排序，但跨图符号／范围及排他性仍不足。下一项有依据的检验是参考真实邻接关系是否带来**独立于两个 endpoint unary 的角色交互**，先量化真实／保边际打乱／因子化控制，不直接堆图求解器。

## 重现与成本证据

主结果：[report.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/autonomous_scene_reconstruction600_20261010/report.json)、[逐例I/U及编辑](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/autonomous_scene_reconstruction600_20261010/scored_episodes.jsonl)、[源码／参数冻结](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/autonomous_scene_reconstruction600_20261010/config.json)、[600预测封存](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/autonomous_scene_reconstruction600_20261010/sealed.json)。入口 [autonomous_scene_reconstruction.py](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/scripts/autonomous_scene_reconstruction.py)。

独立核验全部600GT身份、1800场、9600掩码几何、2400基线I/U、7200候选I/U及编辑、1200逐例帧账本和3619 SHA；十组汇总与主评分差≤7.1e−15。[核验](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/evidence_validation/scene600_verified.json)、[机制汇总](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/evidence_validation/scene600_mechanism_summary.json)。四类真实首例推断使用 query GT／旧mask读取拦截，编码0；另从冻结代码独立重放LVIS首例，三连续场及12完整mask逐位一致，原CRF准备／未准备backend也一致。[真实重放](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/scene_model_validation/actual_repeat_index400_no_gt.json)。

四CPU进程、每进程2线程，总推断481.13秒；每例 raw读取／核验均.1645秒，三头共.0976秒。三臂CRF均分别1.2452／.8260／.8220秒。吞吐包括三个方法和输出封存，不能当单方法延迟；全部都以已有编码缓存为前提，不是冷输入端到端速度。iSAID／ISIC另有新真实 MPS 编码与完整 FoRIS基线流水，单独记录其时延和协议。

当前自主 goal 仍 active：未达到十集优势、可靠泛化或原创贡献验收。
