# Pro M4本地实现与部署接口

已实现实际RGB→完整1024/原尺寸掩码入口，六行是paired、mean、class_lda、ref_canvas、ce、ce_self。代码为`src/ics/methods/pro_paired_environment.py`，独立CLI为`scripts/run_pro_paired_environment.py`。保留Pro报告666—775行核心定义；canonical离散像素、矩形半开边界及缩放舍入细节在method.json明确固定，没有查GT选规则。

`check.json`记录非DINO确定性检查：40个背景矩形与穷举一致、32个六节点cut与穷举最优能量差0、薄SVD与直接闭式解约3e−15、同λ的环境/部件散布分解、同canonical正负映射、原Q字节不变及未贴片query像素始终无标签。真实RGB文件CLI四行封存也通过；query GT哨兵路径不存在且未读。之后在复用公开数据的首4例完成真实DINO六臂评分：原尺寸paired=22.142373、class_lda=25.756491、ref_canvas=31.260366，缓存native变体=36.360804；每例17次编码、约363–405秒。该结果只是暴露样本活动/成本检查，非确认；见[完整报告](real_smoke4_46466/report.json)。

API为：

```python
result = predict(reference_rgb, complete_reference_mask, query_rgb,
                 frozen_rgb_encoder, full_native_callback,
                 native_binding={"method": "full_foris_native", "recipe_sha256": "<64位实际sha>"},
                 encoder_binding={"producer": "<实际producer身份>"})
```

编码器接受1024² uint8 RGB，返回raw final-LN `64x64xD`；内置`FrozenTimmRGBEncoder`直接适配当前`ics.data.TimmDINOv3`，强制1024维/FP32/冻结/无autocast、TF32。注入encoder_factory可适配另一份已绑定合法冻结producer，测试encoder不会成为默认。

native callback参数为原R、完整mask、原Q、已经编码的原Q特征（几何回退则None）。返回两个完整二值mask及`info.encoder_forwards`。统计回退必须复用Q，至多多编码1张R；几何回退允许原native的2次编码。API不提供任何默认替代基线。CLI使用提前封存的native mask并核对SHA，因此新增回退编码数0；明确披露没有重新测量native原部署成本。

CLI清单每个出现至少包含：

```json
{
  "id": "occurrence_0000",
  "reference_rgb": "/absolute/reference.jpg",
  "reference_mask": "/absolute/binary_reference.png",
  "query_rgb": "/absolute/query.jpg",
  "native_npz": "/absolute/presealed_native_prediction.npz",
  "native_sha256": "<实际文件sha256>"
}
```

`native_npz`必须有`mask_work` (1024²)和`mask_original` (原Q的H/W)，只读取这两项。参考mask文件必须已经二值化(0/1或0/255)，不接受语义class图后自行选类；query GT/类别/折字段不会进入infer。输出新目录包含四行work/original预测、coarse场、原始文件SHA、逐编码调用与分段时间、回退类别/绑定、进程峰值RSS、cut证书和整体封存SHA。

本地模型绑定JSON需包含`producer`、`checkpoint_sha256`及`config_sha256`；CLI逐项校验本地文件，`TimmDINOv3`仅pretrained=False载入给定safetensors，不联网。native绑定JSON需包含`method="full_foris_native"`及实际`recipe_sha256`，两份JSON均须在运行前生成。

```bash
python scripts/run_pro_paired_environment.py \
  --manifest /absolute/m4_rgb_manifest.json --expected 8 \
  --out /absolute/new_m4_run \
  --model-dir /root/demo4_cache/models/dinov3-vitl16-timm \
  --encoder-binding /absolute/encoder_binding.json \
  --native-binding /absolute/native_binding.json \
  --device cpu --threads 4
```

新服务器CPU32核/60GiB，timm1.0.30可离线匹配Eva318 keys；checkpoint SHA为`45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`。本地实现阶段没有远程编码；随后授权的真实绑定/成本阶段见下文。正常主分支9次编码，含ref-canvas及CE两行共17次；CPU真实成本不能由确定性小编码器推算。

CE与CE自身偏置两行已补齐，复用同一批合法贴片点，无新编码。FP64零初始化L-BFGS-B最多100步，目标梯度∞范数1e−8，完整保存实际梯度、停止信息；未达目标也如实记录。CE方向走主法midpoint/separation及同Potts，CE-self用拟合w、b的raw logit，同Potts/renderer。原文767行要求“自身偏置”但没有写b的损失/惩罚及自身读出的额外尺度，代码采用未惩罚的b和不额外缩放的logit，这是明示的实现约定，供根代理核对，不宣称由原文唯一决定。成功运行会另保存并封存原Q末层特征与两画布干预bank，后续检查这些读出无需重编码。

新端点46466的真实4例已由`bind_smoke4.py`绑定完整参考原标注；不从coverage反推完整mask。native original缺失，因此采用Pro显式binary1024→actual original H/W bilinear重构，标为cache renderer variant。首个180秒成本screen完成7次真实编码，未得到完整预测；根代理随后澄清有进展首例允许12分钟，已另起`first_complete`，保留前次记录。`run_supervised.py`只管理新建自有进程、4线程、8GiB RSS，逐编码进展与停止原因都有记录。`score_smoke4.py`须验证四例全部封存后才打开query GT，报告四类增删、原尺寸/1024 class-summed IoU与完整成本，四例只作smoke而非独立确认。

四例现已全部17forward/六行封存并评分，实际负结果见[real_smoke4.md](real_smoke4.md)：原尺寸paired22.1424、class-LDA25.7565、ref-canvas31.2604、cached native renderer variant36.3608。1线程评分的1024口径复用绑定packet.truth；真实原annotation用于原尺寸口径，nearest-original-GT旧评分作为独立诊断variant保留。root新增并行授权后保留串行case2进度，独立case3并行；guard在case1/2完整输出关闭后封存partial hashes并停自有尾部，避免重复。所有自有encoder进程已结束。
