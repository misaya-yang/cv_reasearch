# 原LVIS600任务双角色判别：完整配方淘汰

固定历史600的四个完整臂、两frame共4800最终Mask已全部先封存后GT评分和直接验证。主臂原图 **47.812350**，比已有效最强S **52.005138低4.192787pp**，也低于R-only48.975332/Q-only48.061093。预先冻结的S+1pp、CLI>S及完整消融要求未过；本配方结束，不扩6200、不扫λ/阈值/seed/host，不将消融赢家换名作为主方法。原图对FoRIS仅+.032979pp不构成Strong成功。目标仍未完成。

| 方法 | 原图 observed600 | CLI observed600 | 原图 official920 |
|---|---:|---:|---:|
| FoRIS | 47.779371 | 47.619569 | 31.160459 |
| 保留的最强 S | 52.005138 | 51.702409 | 33.916394 |
| 完整 R+Q 判别 | 47.812350 | 47.674206 | 31.181968 |
| 去查询：R-only 完整臂 | 48.975332 | 48.816147 | 31.940434 |
| 去参考：Q-only 完整臂 | 48.061093 | 47.918395 | 31.344191 |
| S 同实际编辑预算 | 47.309988 | 47.174660 | 30.854340 |

600是原独立历史LVIS zoom600与旧6200的精确身份join，不是LVIS1400前600。R/Q图像、Mask、crop、fold、loader class、尺寸和旧S路径SHA沿用原冻结绑定。每fold观察60类；official920每fold92类且未观察槽位按原policy计入，两种池不能混用。这是已暴露开发缓存，不能称新独立验证或五域验证。

完整设计保留S的有效终点；合法R Mask为FG=p、BG=1−p，Q只取CLI四方一致FG S&B&G&R0和BG ~S&~B&~G&~R0 的16×16面积fraction，争议像素不充标签。每图每角色128面积中点采样，同图重复native id合并 y=(nf−nb)/(nf+nb)、a=(nf+nb)/256。角色各半，joint R/Q各半；λ=.01固定。同一Unit O24容量、weighted centered ridge和自由截距，MPS32 Gram/readout、CPU64 small dual/中心/截距/最终frame双线性。单图臂风险总量1，joint总量1；去图证据也改变约束风险分配，不能当保持其它优化条件完全不变的纯单因素标签污染实验。

Dadd=~S&(B|G|R0)，Ddel=S&~(B&G&R0)；strict正logit新增、负logit删除，0留S，1111/0000逐像素保护。无FoRIS confidence/alpha或新零点校准。第四完整臂在同frame、同全D域按原FoRIS margin排序，实际增/删数量分别严格等于joint；row-major打平。旧S最终Mask作为完整处理复用，没有重建九窗/新head/view。

| 完整臂相对 S | 补真 | 补假 | 删真 | 删假 |
|---|---:|---:|---:|---:|
| R+Q | 727,469 | 2,045,191 | 95,353 | 561,940 |
| R-only | 270,052 | 618,369 | 371,934 | 1,050,093 |
| Q-only | 718,956 | 2,007,647 | 108,259 | 591,533 |
| 同预算 | 703,057 | 2,069,603 | 105,806 | 551,487 |

这是原图像素累计，不等于class等权mIoU。逐例对S改善165、下降403、相同32；对R-only改善220、下降349、相同31；对Q-only改善265、下降251、相同84。原图joint对同编辑数量控制+.502363pp，选点有作用，但明显不足以保留S。

原图joint对R-only，实际补749080真/2073933假、删15082真/158958假；对Q-only，补39322真/153043假、删17903真/85906假。这些是同输入、相同宿主/域/读出下的完整对照得失。原参考输入、旧局部观察和encoder保持相同，所以本次坏增量发生在新任务判据/证据融合及其争议域读出，而非新编码或新增视图；没有把融合与strict0读出完全分离的实验，不能锁定唯一首个不可逆步骤。

593个共同可用例，joint合法R角色平均原±1MSE **0.079286**、预测Q角色 **0.052822**；正确补回重复native id合并丢掉的常数Σa(1−y²)，不把混合cell零残差当纯角色。下降403例均值R/Q为 **0.076756/0.049845**，改善165例为 **0.084317/0.059140**：坏例并没有更高拟合损失。可支持的结论是，本配方拟合合法R+自预测Q角色良好仍会在争议区扩大量假前景；不能以角色拟合损失小或reference自一致来证明query任务正确。不能据此断言Q anchors的GT污染量、FG/BG信息天生无用或所有算法存在同一个不可恢复定理。只在这批LVIS重复出现，不冒称五域机制已成立。

R-only600成功；joint和Q-only各593成功、7例Q角色缺失独立回S，没有以Q缺失阻断R-only。实际1786个闭式头，最高相对残差9.56e-16。固定前三例CPU64数值参照因第2例Q缺失只产生joint/Q各2、R-only3共7个比较；max logit误差2.82e-07，native正号差0，未选精度或追加Mask。不能由7个比较推全600逐像素等同FP64。

生产 wall **110.215388s** / CPU 108.626150s / RSS 1.286423GiB，600全MPS，无失败或重启；旧R/Q raw实际1200加载请求/1166unique，container请求18,669,988,053B。原raw不复制，实际物理磁盘IO未知，不能将请求字节当物理读取。raw加载耗78.870016s，三判据10.430069s，双frame读出13.161651s。MPS driver在每例compute返回后采样最大1.026093GiB，live allocated此时0；这不是GPU使用率/真实allocation峰值，未测能耗。评分 wall 22.660097s / CPU 23.748813s / RSS 0.813538GiB。新增packed最终结果7,189,325B、逐例小记录5,153,395B，未新增encoder/CRF/RGB/view/sample/dense archive或全局训练；1786个局部闭式fit属于每例推断，不更新encoder。

生产guard封住600个Q Mask路径，访问尝试0；seal为`92c170fa302305bd8cbdad7022b837883ffd5db8f1101acad7b831c986012846`。评分先4800全包SHA/身份/schema/padding gate再读600QGT，直接核验IU/四编辑/争议域保护、每frame同预算数量与位置。root另 **148263** 项整数/类池/角色/独立回退/实际solver核验通过，0重复GT/raw/mask/fit。后seal case归因只读saved scalars，未重跑生产或增加实验臂。

协议`68cc06cf1f2acafea84e344a4c420c6425c68ee25c694d43f30271b2243b4466` / core`f9321cdb860791b71869286c798400942e37c0c46e73e04d8dd54df55b0bf640`。复现只读原数据/权重/raw，完整输入绑定和score-binding SHA在[冻结协议](../../evidence/local/task_discriminant600_20261011/protocol.frozen.json)。[core](../../evidence/local/task_discriminant600_20261011/core.py)、[producer](../../evidence/local/task_discriminant600_20261011/producer.py)、[postseal scorer](../../evidence/local/task_discriminant600_20261011/score_and_verify.py)、[root verifier](../../evidence/local/task_discriminant600_20261011/root_verify.py)、[逐例IU](../../evidence/local/task_discriminant600_20261011/episodes.jsonl)、[逐例归因](../../evidence/local/task_discriminant600_20261011/case_attribution.json)、[淘汰判定](../../evidence/local/task_discriminant600_20261011/screening_decision.json)、[资产清单](../../evidence/local/task_discriminant600_20261011/completion.json)均保留；predictions/600个最终packed文件由sealed.json逐文件SHA绑定，records/600个小收据由records_manifest.json逐文件SHA绑定。没有大场/新缓存。

当前producer PID61815/session13824与scorer PID62674/session90879均已退出成功；不按历史PID重新启动。主目标继续，旧pure-FG/最近角色/补齐BG/查询无任务图/本双角色ridge配方关闭。下一步须把已有强组合有效处理的条件与参考指定任务的冲突分开解释，形成能否定这些具体坏增量的约束后才写下一完整600方案；不把“再拟合/再传播”当新语义证据。不以本配方失败终止研究，也不按低fitloss、某class/GT或frame选择下一赢家。
