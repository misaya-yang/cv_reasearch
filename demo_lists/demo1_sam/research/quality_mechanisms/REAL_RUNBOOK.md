# 真实预训练资产诊断入口

这些命令由主代理在有卡实例执行。本子代理只验证本地 CPU 合同；不下载、不 SSH、不使用 GPU。

## 资产与依赖

- 官方 SAM 源码：`assets/source/segment-anything`；启动时核对 decoder/common/transformer 三份源码与包内固定 `dca509fe...` 哈希。
- 官方预训练权重：`assets/checkpoints/sam_vit_b_01ec64.pth`；获取和身份确认由主代理完成，脚本不下载或覆盖。
- 实例已有 COCO：`/root/autodl-pub/COCO2017/annotations_trainval2017.zip`、`/root/autodl-pub/COCO2017/val2017.zip`。只读取所选 JPEG，不解压整套图像。
- preparation 需要现有 NumPy 和官方 `pycocotools`。如果 base 缺包，由主代理仅在 `/root/autodl-tmp/demo1_sam/runtime/python_packages` 准备必要包；不要覆盖现有 NumPy/PyTorch。GPU runner 不依赖 pycocotools，使用现有 PyTorch、Pillow、官方 SAM 导入所需 torchvision。

所有输出、临时文件在数据盘：

```bash
cd /root/autodl-tmp/demo1_sam
export PYTHONPATH="$PWD/runtime/python_packages${PYTHONPATH:+:$PYTHONPATH}"
bash tools/with_data_disk.sh /root/miniconda3/bin/python \
  research/quality_mechanisms/prepare_coco_subset.py \
  --annotations-zip /root/autodl-pub/COCO2017/annotations_trainval2017.zip \
  --images-zip /root/autodl-pub/COCO2017/val2017.zip \
  --output-dir assets/coco_quality_seed2027_v1 \
  --seed 2027 --num-images 24 --max-instances 4 --min-area 1024
```

输出 `manifest.json`、`selected_annotations.json`、最多24个 JPEG、每图 GT NPZ。抽样只依赖标注，GT 点与框在 manifest 中固定。每图最多4个 non-crowd、annotation area>=1024、可有效解码的实例；因此不是承诺96实例，实际数量由 manifest记录。central 正点来自最后非空3×3 erosion；near-boundary 正点来自一次erosion删去的合法前景边界带。框使用解码mask tight XYXY，xmax/ymax为exclusive坐标。不同对象不合并。

已有manifest不覆盖；如果需要新协议，创建新目录。`--limit-images 2`只限制run，不重新抽样，所以smoke和完整诊断使用相同预固定子集。

## 有卡两图 smoke

```bash
bash tools/with_data_disk.sh /root/miniconda3/bin/python \
  research/quality_mechanisms/run_real_sam.py \
  --subset-dir assets/coco_quality_seed2027_v1 \
  --checkpoint assets/checkpoints/sam_vit_b_01ec64.pth \
  --output-dir results/real_quality_smoke_v1 \
  --device cuda:0 --limit-images 2 --microbatch 4 \
  --attention explicit --diagnostic-on-failure
```

这会运行 official / cached / dense_assoc / factor_projected 四种完整decoder，保留四个mask与四个IoU； FP32、TF32 matmul/cudnn关闭、不使用autocast。每图只执行一次 image encoder，每个原对象提示独立。三个regime间，当dense prompt和token长度相同，复用每图cache。dense_assoc默认auto按FLOPs选序；主代理可按已实测最强路线显式传 `--assoc-order associated|projected`、`--assoc-write-output dense|sparse`，报告保存计划，不把auto称为最强实测计划。

缓存方法 `--attention sdpa` 可作为另一轮诊断，但official保留原始explicit，factor的write attention仍explicit；调用SDPA不证明FlashAttention。第一轮explicit可直接隔离图改写带来的数值影响。

历史真实数据已出现约1e-3 logit差异，故示例显式开启 `--diagnostic-on-failure`。任一候选超过原 **5e-5** mask-logit或IoU预测门槛后，继续导出质量诊断，整体状态固定 `FAILED_DIAGNOSTIC_OUTPUTS_ONLY`；这不代表数值通过。省略该选项时完成当前regime四方法检查后停止，状态 `FAILED_STOPPED_AT_NUMERIC_GATE`，返回码2。没有可放宽门槛的CLI。

## 完整24图诊断

确认smoke图像、GT、独立提示及四mask尺寸正确后运行，输出新目录保留smoke：

```bash
bash tools/with_data_disk.sh /root/miniconda3/bin/python \
  research/quality_mechanisms/run_real_sam.py \
  --subset-dir assets/coco_quality_seed2027_v1 \
  --checkpoint assets/checkpoints/sam_vit_b_01ec64.pth \
  --output-dir results/real_quality_full_v1 \
  --device cuda:0 --limit-images 24 --microbatch 4 \
  --attention explicit --diagnostic-on-failure
```

已有 `report.json` 不覆盖。发生OOM时主代理可在新输出目录降低共同microbatch，保持四方法同批调度；脚本不偷偷改批次或退回CPU。普通错误写`ERROR`和traceback后抛出，不伪装负研究结论。

## 输出与解释

- `report.json`：每图、每regime、每方法的原数值状态、低分辨率logit/IoU最大误差、全分辨率四mask逐prompt像素变化、1–3 IoU argmax变化、缓存计划、独立诊断段时间和质量摘要。
- `image_<id>/<regime>_<method>.npz`：官方完整postprocess后的 `masks[P,4,H,W]` bool、全部 `low_resolution_logits[P,4,256,256]`、`iou_prediction[P,4]`、`gt[P,H,W]`、annotation_ids。全分辨率浮点logits只临时用于官方postprocess，不长期保存，避免占磁盘/内存；它们不是上述低分辨率数值门槛的检查范围。
- `<regime>_<method>_quality.json`：single token0、multimask1–3 IoU-head选择、SAM2本地snapshot动态选择**规则**（不是预训练SAM2）、1–3 / all4两个GT-IoU oracle、Boundary IoU、逐对象/提示/候选结果。原SAM四输出不能冒充SAM2质量。GT oracle不能作为部署方法。
- global `quality_summary`按regime/method分别统计所有选择policy，避免在box regime只报告multimask弱baseline；各policy与oracle都保留。
- decoder/postprocess段使用同步单次计时；encoder、promptencoder、缓存构建、完整图像诊断wall也分开记录。**没有充分预热或重复测量，诊断wall包含误差统计、CPU拷贝、评分、压缩I/O，不能用于性能胜负或端到端提速声明。** 公平性能仍由主benchmark负责。
- 图像采样不能按模型表现挑选；首轮是小量开发诊断，不是最终SOTA评测。正式质量实验应补SAM2/HQ-SAM/针对性强baseline、按image分训练/测试与image-cluster不确定性。

## 本地已验证范围

```bash
python research/quality_mechanisms/check_real_pipeline.py
```

`real_pipeline_contract_results.json`记录：用显式合成COCO ZIP与注入mask decoder验证抽样/选中JPEG提取；再用微型合成image encoder + 官方SAM PromptEncoder/MaskDecoder/Postprocess执行完整run harness，三个regime×四方法、共同microbatch3包含尾批1，检查四maskNPZ尺寸、encoder仅一次及跨regime cache复用。无预训练权重、真实COCO或GPU；本地pycocotools不存在，因此官方COCO解码需在主代理真实prep smoke中验证。

## 新路径：GPU生成与CPU评分解耦

用户已切无卡模式时只准备代码和资产，**不要运行本节GPU命令**。已固定的seed2027子集无需重新抽样；下一次明确有卡后，可使用以下生成参数，把质量评分和压缩移出GPU阶段：

```bash
bash tools/with_data_disk.sh /root/miniconda3/bin/python \
  research/quality_mechanisms/run_real_sam.py \
  --subset-dir assets/coco_quality_seed2027_v1 \
  --checkpoint assets/checkpoints/sam_vit_b_01ec64.pth \
  --output-dir results/real_quality_export_v2 \
  --device cuda:0 --limit-images 24 --microbatch 4 \
  --attention explicit --diagnostic-on-failure \
  --defer-quality --npz-compression none --save-encoded-inputs
```

`--defer-quality`只做官方encoder/promptencoder、四decoder、原数值对照、官方postprocess、CPU拷贝和NPZ/报告导出，不计算IoU/Boundary IoU/GT oracle或bootstrap。`--npz-compression none`保存ZIP_STORED NPZ，避免deflate阻塞；它会增加磁盘用量，全部输出应仍在数据盘。二者都不是性能benchmark开关：数值误差统计、I/O和同步单次计时仍在诊断流程。

旧默认行为不变：不传defer则在线评分，不传compression则压缩。发生数值失败时无论在线还是延迟评分，`status`仍为`FAILED_DIAGNOSTIC_OUTPUTS_ONLY`（诊断继续）或`FAILED_STOPPED_AT_NUMERIC_GATE`；新增`numeric_status`明确为`FAILED`。成功延迟路径用`PASSED_NUMERIC_OUTPUTS_EXPORTED_QUALITY_DEFERRED`，`quality_status=DEFERRED`，不会声称已经完成质量验证。

离线CPU评分无需PyTorch、CUDA、完整SAM权重、COCO ZIP或原JPEG，只需生成run目录的`report.json`和其中全部NPZ。它写新目录，保持源报告字节和源数值状态不变：

```bash
python research/quality_mechanisms/score_saved_outputs.py \
  --run-dir results/real_quality_export_v2 \
  --output-dir results/real_quality_export_v2_cpu_scores
```

可在本机或无卡环境CPU执行评分；不能把大量后处理质量评分放到低CPU实例当作GPU作业延续。`score_report.json`分别保留`source_run_status`、`source_numeric_status`及每method原numeric chunks；评分完成状态`CPU_QUALITY_COMPLETED`只意味着计算了质量指标，不会将原失败改成通过。兼容旧schema1报告和新schema2报告；旧两图smoke若只同步了JSON，没有原NPZ，则不能离线重评分，也不能从JSON构造新mask。

### 编码输入与小decoder权重接口

`--save-encoded-inputs`在本run首次模型初始化后**仅保存一次**：

- `<run-dir>/mask_decoder_state.pt`：`torch.save`字典，`state_dict`为全部CPU FP32 mask decoder参数；另含`architecture={transformer_dim:256,num_multimask_outputs:3,depth:2,num_heads:8,mlp_dim:2048}`、`source_revision`、`model`、`dtype`。独立decoder权重约16MB，后续decoder验证/性能队列应加载它，跳过完整375MB权重/encoder初始化。生成过程不重复hash完整checkpoint。
- `<run-dir>/image_<id>/encoded_inputs.npz`：`image_embeddings[1,256,64,64]`、`image_pe[1,256,64,64]`、`dense_nomask[1,256,64,64]`、`sparse_central[P,2,256]`、`sparse_near_boundary[P,2,256]`、`sparse_box[P,2,256]`（全部FP32），加`annotation_ids[P]`、`original_size[2]`、`input_size[2]`、`image_id`标量与`mask_threshold`标量。输入不编码GT，只包含真实encoder与原独立提示的promptencoder输出。
- 同图`encoded_input_metadata.json`存上述NPZ相对run-dir路径、decoder_state相对run-dir路径、subset manifest SHA、原prompt坐标/annotation_ids，以及`regimes_exported`。数值gate早停时只可能导出已完成的部分regime，必须读这个字段；正常diagnostic continuation导出三regime。

在CPU合同测试中，已验证新defer路径绝不调用评分；压缩和未压缩原始输出数组一致；离线与在线所有质量摘要一致；源report不变；CPU权重+导出输入可重放全部四mask/IoU；注入数值失败在在线、defer、offline中保持相同FAILED状态。测试使用微型合成image encoder和官方SAM组件，**没有真实预训练导出或GPU新路径结果**。

下一次有卡仍需的最短残余检查：两图新参数smoke，确认真实`encoded_inputs`形状/FP32和三regime、仅一份decoder PT、数值失败状态如实保留；用新导出接独立decoder gate/性能工具验证接口。CPU离线评分可在GPU后续纯decoder队列运行期间于本机执行。新NPZ格式本身无需GPU验证，但真实encoder与promptencoder的导出接口尚未在新路径运行过。
