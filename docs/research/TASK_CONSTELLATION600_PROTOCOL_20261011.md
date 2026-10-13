# 同原600：任务Mask布局与语义对应的完整组合

原S gate整体有效，而宽恢复`S|(B&acceptedAny)`已失败。当前固定一条不同的约束：用合法参考Mask中的特征—位置绑定，与已有强S前景锚拟合一个共同的相似变换；把全参考Mask按该变换放到query，只允许在该模板支持内、同时有完整B和原接受局部支持的位置新增。原S逐位保留，原区域margin、选择与窗口heads均不重跑。

这是一条新固定完整组合假设，不是已证明有用的方法；旧R1-G已有Procrustes和位置支持，**不把注册本身称原创**。参考几何不是语义证书；多实例、形变、视角和部分观察可使单变换失效。

legalR softFG按面积中点取512等质量粒子，已有S queryFG取128，legalRBG取128；重复native ID保留成重复槽，一对一匹配指粒子槽，不是不同物理点。R/Q坐标先按实际full编码图像宽高解resize失真，再按合法Mask/S重心和RMS归一。

特征cost为`1-cos(r,q)+max(max_BG cos(r,b),max_BG cos(q,b))`。BG项可能退化为行/列常量，记录其非可分interaction，另有完整去BG对照，不称它必然提供语义否决。矩形Hungarian给初始语义对应，closedform正scale/det(rotation)=+1 similarity pose；随后固定5次`feature+.5*squared_geometry_distance` assignment→pose。每个子步骤相对同一初始完整目标不增，只用机器误差容差。

只匹配R粒子子集，未匹配的R前景也随完整Mask warp；新增点没有参与pose拟合，不能称完整reference/每个新增点已获对应证明。缺role、只有一个坐标、退化方向或非正scale时该pose臂回S。

实际各frame从目标像素中心inverse-warp合法RMask，CPU64 bilinear、域外零、严格>.5。Dadd=`~S_frame & B_frame & old_FP32_render(acceptedAny)`。完整输出=`S | (Dadd & template)`，无删除，不把canonical阈值模板再render原图冒充连续逐帧warp。

同600、六完整臂、CLI与original共7200 Mask，全部seal后才GT：完整联合配准；只有初始语义assignment/pose而无联合迭代；RFG descriptor相对位置固定reverse打乱；去参考BG项；无模板的完整宽恢复；S同实际新增数量、在整个Dadd按原FoRIS字段排序的预算控制。初始化对照仍有Mask几何，不称完全去几何；预算pool不交template，遇到K等于整个pool如实记录退化。

预先门槛沿用Strong600：原图观察600 class/fold mean超过S至少1pp，CLI超过S，完整方法在双frame均超过所有必要完整对照；否则立即淘汰，不扩6200，不扫迭代/pose/尺度/assignment/采样/warp，不拿消融赢家替换主臂。通过600也不代表多域Strong目标完成。

复用600原R/Q whole O24与已封存S/B/窗口/区域收据，流式逐例释放。0新encoder、FoRIS、CRF、编码view、照片、sample、global训练或dense档案；每例临时correspondence/pose优化计实际步数、成功与失败。MPS32特征cost，CPU64 assignment/pose/warp；固定首3只记FP64cost误差，不按误差选precision或候选。

冻结[protocol](../../evidence/local/task_constellation600_20261011/protocol.json)：`f73da7f6b3cd46dd1b2551ceac193d93ebf8d13f1d2f97e94998f307041bf5ef`；[core](../../evidence/local/task_constellation600_20261011/core.py)：`7eaaab93bfd65827b2dff0dd28f72e2bb4570fe0878f2ff6dd5ed07e7edc034f`。[producer](../../evidence/local/task_constellation600_20261011/producer.py)、[scorer](../../evidence/local/task_constellation600_20261011/score_and_verify.py)、[静态审查](../../evidence/local/task_constellation600_20261011/static_review.json)、[纯代数检查](../../evidence/local/task_constellation600_20261011/numeric_sanity.json)。实际运行/成本以activity与production_receipt为准，不从静态检查推断分数。
