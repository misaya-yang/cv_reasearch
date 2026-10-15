# 独立参考任务读出：利用新观察约束旧参考不可见方向

现有三域source-span诊断证明：密集参考监督未补齐主要迁移缺口，query标签oracle可以在原参考quadrature不可见方向改进Q，且安全缩步后4096 R token判断全部相同。这不等于已找到可识别算法。17个PACO缓存focus的平方风险观察判据未稳定超过whole／打乱；排序判据比插值强，但其它区间仍跨零。

现在把该前提变为一个可证伪、完全不读query标签的独立读出。仅用同一R RGB的既有真实focus和当前合法mask。先由whole-R的平衡平方风险与lambda=.01拟合theta0；随后以focus FG/BG角色的期望pairwise平方排序风险（期望margin=2，与±1标签尺度一致）拟合一次闭式增量。增量限制在原reference quadrature augmented矩阵的nullspace，精确保持那些位置的旧logit；仅由全部4096 R token旧分数决定安全步长，保留全部旧二值判断。Query只用既有unit O24，独立signed场沿bilinear1024、>0输出，零FoRIS依赖、零CRF、零跨episode训练、零新增编码。

五臂固定：原whole-R读出；真实focus无约束增量；真实focus nullspace增量；真实focus nullspace＋安全步长；插值focus nullspace＋同安全步长。所有view都使用相同实际参考mask覆盖和物理位置；插值由whole单位描述子按相同focus中心采样，不是真DINO输入。query始终相同。新矩阵/源码、原参考logit身份、原whole-R字段重放、源数值和时延保存；预测全部封存后评分完整1024掩码、前景／背景AUC、两任务身份切换及TP/FP/FN。

固定17个原任务A具真实focus的PACO对，选择只依据旧RGB＋maskA匹配和缓存覆盖。A是此prototype的单参考合法view-policy实验；B共享原A选定focus只作固定观察的反任务诊断，不当作B自身mask生成view的官方结果。若focus缺FG或BG，明确不更新，返回原读出。

这些是已曝光开发照片，候选在oracle观察后提出，不主张独立确认或原创性。若真实更新不胜插值、原读出及完整FoRIS，或空间身份排序不改善，就不扩失败候选；若通过，才冻结机制并在既有其它缓存／合法新增必要观察上检验跨域。最终目标仍是多数据集完整领先，当前未完成。

入口`scripts/task_transfer_null_rank_probe.py`；资产`../cv_data/a/task_transfer_null_rank_probe17_20261010/`。

## 已完成：候选未通过

17对、34任务、五臂170个完整1024 mask均已封存评分，预测无query标签／oracle字段读取。单次缓存矩阵和读出5.388秒，不含原DINO和focus编码。A的17个focus均同时有FG/BG；B只有12个有两角色，另5个明确返回原读出。该B固定原A裁剪，只是观察控制，不能称官方B view-policy。

|臂|原任务A完整IoU|A−原R读出|A−完整FoRIS|A FG/BG AUC|
|---|---:|---:|---:|---:|
|reference only|33.124|0|−9.510|.89503|
|真实focus无约束|37.019|+3.895|−5.615|.90924|
|真实focus投影|29.022|−4.102|−13.612|.88271|
|真实focus投影＋保旧判断|33.190|+.066|−9.444|.89731|
|插值focus投影＋保旧判断|33.162|+.038|−9.472|.89468|

完整FoRIS同17原任务A均值42.634。无约束更新相对原R的区间[−3.111,+11.623]，仍未稳定；安全版本[−.307,+.517]。不扩大这个失败配方。投影后的大步更新确实保持quadrature logit，但不代表对其它R位置或Q正确；安全步长把A更新缩到约.02的典型尺度，原R判决保留不能替代新Q任务识别。这里的投影和安全性是被检验的具体假设，不是算法必须永远保留的要求。

独立核验`scripts/verify_task_transfer_null_rank_probe.py`共867项：全170个mask重新渲染／像素计数、weighted AUC、quadrature logit身份、4096 R token判断安全性和同批FoRIS像素计数均通过。原whole-R字段对前一独立控制最大差2.64e−7，为FP64读出相对旧MPS混合算术的误差；不冒称byte identity。

实验归属Sol；源码／封存／字段／完整TP/FP/FN、A/B分开结果均保留。本项没有证明原创机制或多数据集领先，主goal继续。
