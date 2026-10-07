当前决定：关闭曲率单独机制槽，独立方法增量计 0。确有五点真实路径和非端点函数的曲率，但没有证明它提供超越强简单表示的类别判据。这里没有评价查询图质量。

1. 数据：四例持久 native_pair.pt 均包含真实 block20–24 raw states，每图五个 FP32 [4101,1024]；同 patch_ids=5..4100，全部 4096 tokens 的三处转弯有效。
2. 原生时序来自真实 hook output，不由 QKV 倒解、projected_native 或旧 FP16 q 拼凑；保存接口为 pro_message_extrapolation.py:487–490 和 smoke_case_worker.py:145–151。
3. 表示：事后对五个 raw states 施加同一 learned FINAL_LN，再施加该 episode 固定 Pi=I−UUᵀ，最后单位化；这不是 encoder 实际输出的逐层 intermediate LN。
4. epsilon=1e-5 已通过 config architecture 与 SHA-bound Eva factory 行3135–3159核对；use_fc_norm=False，行700–708/817构成 final norm；capture metadata 原本未存 epsilon。
5. 各例 assets_first.json 的 source/config/weights/basis SHA 与当前资产一致，producer 与 binding.json 一致，原 gate feature SHA/debiased 均核对；U 为 FP32 [1024,500]。producer 自身缺 basis SHA，依据逐例 sidecar 补链。
6. 转弯量 C_t=acos(unit((x_(t−1)·x_t)x_t−x_(t−1))·unit(x_(t+1)−(x_(t+1)·x_t)x_t))/π，t=21,22,23；不是仅两个端点的函数，亦不等同 β 消息或 QK response。
7. 合法 Rmask 仅用于 R FG coverage≥.9/BG≤.1；固定 8×8 空间块四折 centroid 判别。Q 仅记录未标注分布，未读 Q GT、预测 mask，未据 Q 类名选择路径。
8. 同 LN 合成可实现状态向量中，同端点 geodesic 与 detour 的 C 不同；相反标签镜像 detour 的 C 完全相同，廉价全状态线性拼接已解正/负例；实际 DINO RGB 正例数=0。
9. 真实 R 四例曲率 AUC=.515/.637/.835/.680；speed=.762/.870/.759/.736；endpoints 与 allstates 均>.996。此为参考图内部空间关联，空间相关性、同图外观可使 AUC 乐观；非 Q 增益、非独立确认。
10. 资源：1 CPU，24.605s，peak RSS 2.008GiB；0 encoder、0 suffix replay，最终 LN/Pi 对 cached projected_native 的 R/Q 八个重建 maxabs 全 0。无后续 variants/队列。

| case | curvature3 | speed4 | endpoints2 | allstates5 |
|---|---:|---:|---:|---:|
| 0 | 0.515150 | 0.761958 | 0.999733 | 0.999745 |
| 1 | 0.636839 | 0.870013 | 0.997603 | 0.996844 |
| 2 | 0.834518 | 0.758869 | 0.999985 | 0.999992 |
| 3 | 0.679818 | 0.736060 | 0.999561 | 0.999421 |

原必要旧表示核对：src/ics/methods/multilayer.py:17–47 为 layer16/24 unit concat 或 final+unit difference；multilayer_controls.py:18–30 为 delta-only。旧 DEV241 编辑证据 evidence/local/RESULTS.md:441–451 显示新增像素副作用，不能把旧 pipeline 的失败当作当前 curvature 的直接质量测量。五状态拼接是本次强简单对照，完整信息仍包含全部方向轨迹；曲率只是它的非线性压缩，并没有新的 encoder 干预。

实现证据：observe_path.py:44–62 核实 eps；95–121 逐例资产/gate/最终表示验证；67–75 固定球面转弯；81–91 空间留出；143–164 仅 R 标签与 Q 无标签统计。toy_path_check.py:19–41 给同端点正例与镜像负例、强简单对照。底层JSON收据和执行源码包含服务器绝对路径，保留本地；本文件是可提交的路径无关结果摘要。

full600 持久 rawH20 不存在，本观察只有以上四例；full600 RGB/Rmask 资产不改变这一点。最有限下一动作是保留可复用数据并关闭本槽，无完整新方法实现。若将来明确重新授权检验 Q 迁移，应先固定具体完整 readout 与简单端点/全状态控制，当前不自动启动。
