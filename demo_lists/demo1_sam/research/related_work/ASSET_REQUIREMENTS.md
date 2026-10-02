# 下一轮强基线资产清单（2026-10-01）

本记录只访问官方/作者来源、HTTP HEAD 与公开模型元数据；未下载模型、接受许可证、使用账户令牌、SSH、GPU 或安装依赖。大小是当天HEAD/官方HF元数据，不按参数量猜测。机器可读下载清单见 `asset_download_manifest_20261001.json`；来源证据见 `asset_source_audit_20261001.json`。

## 本轮必须备齐：公开资产约3.05GB，SAM3权限单列

保留已有SAM ViT-B与COCO小集；先查已有cache。**下一轮直接需要SAM2.1 small（快速机制/移植）、SAM2.1 large（强质量参考）、HQ-SAM2 large（边界质量参考）与COCO2017 val/annotations**。large与HQ用同Hiera-large级别，避免只打败small就谈质量SOTA。SAM3源码可先备齐；权重是gated资源，未取得正式access前不能宣布完整强基线准备完成。

| 资产 | 官方/作者权重入口与目标文件 | 精确bytes / 约GiB | 状态与优先级 |
|---|---|---:|---|
| SAM2.1 small | [官方直链](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)，`sam2.1_hiera_small.pt` | 184,416,285 / 0.172 | public、HEAD200；下一轮必须 |
| SAM2.1 large | [官方直链](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)，`sam2.1_hiera_large.pt` | 898,083,611 / 0.836 | public、HEAD200；下一轮必须 |
| HQ-SAM2 large（官方beta） | [作者直链](https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt)，`sam2.1_hq_hiera_large.pt` | 898,844,313 / 0.837 | public、HEAD200、HF gated=false；下一轮必须。链接由SysCV官方README/download script指定，不是第三方镜像 |
| COCO2017 annotations | [官方bucket HTTPS路径](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip)，`annotations_trainval2017.zip` | 252,907,541 / 0.236 | public、HEAD200；只需使用其中`annotations/instances_val2017.json` |
| COCO2017 val images | [官方bucket HTTPS路径](https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip)，`val2017.zip` | 815,585,330 / 0.760 | public、HEAD200；5000张，允许短测先使用冻结24张manifest，随后扩展同一val集。比逐步临时补图更便于有卡时持续推进 |
| SAM3 visual/interactive | [官方HF gated入口](https://huggingface.co/facebook/sam3)，目标`sam3.pt`，授权后地址`https://huggingface.co/facebook/sam3/resolve/main/sam3.pt` | 3,450,062,241 / 3.213 | **manual gated**。缺失资源为正式批准的HF账号access/合法已缓存权重；未核验用户账号状态。当前不下载、不代签、不用镜像 |

SAM3官方builder加载`sam3.pt`；不要再重复下载`model.safetensors`（3,439,938,512bytes，属另一路生态）。本轮不需要SAM3.1 multiplex权重；它是视频任务，不能替代SAM3 visual-prompt image质量基线。

COCO官网地址是`http://images.cocodataset.org/...`，本次HEAD亦200；其同域HTTPS证书hostname不匹配。本表使用同一个官方S3 bucket的HTTPS路径，HEAD/Content-Length/ETag与原地址一致，不绕过TLS。公开5个大文件总计**3,049,837,080bytes**；该数字不含已有SAM、源码、解压后的数据、输出与备用空间。

## 官方源码与应保留的配置/资源

| repo | 必需文件/配置 | 当前工作用途 |
|---|---|---|
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | `sam2/configs/sam2.1/sam2.1_hiera_s.yaml`、`sam2.1_hiera_l.yaml`；`sam2/build_sam.py`、`sam2/sam2_image_predictor.py`、`sam2/modeling/sam/mask_decoder.py`；LICENSE/README/setup.py | small+large共享同源，固定一次Git commit。不要运行`checkpoints/download_ckpts.sh`默认下载四型号，只下载small/large |
| [SysCV/sam-hq](https://github.com/SysCV/sam-hq)，使用子目录`sam-hq2/` | `sam-hq2/sam2/configs/sam2.1/sam2.1_hq_hiera_l.yaml`；`sam2/modeling/sam2_hq_base.py`、**`sam2/modeling/sam/mask_hq_decoder.py`**；`demo/demo_hqsam2.py` | 配套HQ2源码。`mask_decoder.py`是普通SAM2模块，不能误用它当HQ模型实现。无需同时下载SAM-HQ v1的全部backbones |
| [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | `sam3/model_builder.py`、`sam3/model/sam3_image.py`、`sam3/model/sam3_image_processor.py`、`sam3/model/sam1_task_predictor.py`、`sam3/sam/mask_decoder.py`；**`sam3/assets/bpe_simple_vocab_16e6.txt.gz`** | public source可备齐。BPE原始文件1,356,917bytes，[官方raw](https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz)。默认builder以pkg资源找此文件；不要误用root`assets/`的不存在路径 |
| [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) | `PythonAPI/pycocotools`或已装官方`pycocotools` | 只做annotation mask/RLE解码与COCO指标，复用已有安装；无需额外detector或训练数据 |

下载/克隆由主代理执行，全部置数据盘。源码checkout与权重/data/cache/results分开；记录源commit、匹配配置与权重来源。SAM2和HQ-SAM2都导出`sam2`包，**不能在同一进程把两套PYTHONPATH或editable安装混用**；独立runner进程只给各自source root，输出`sam2.__file__`核验实际来源。文件齐全不代表依赖或GPU运行已验证。

## 必要依赖：先查现有环境，再补缺项

- **SAM2当前main**：README/Python要求≥3.10，torch≥2.5.1、torchvision≥0.20.1；setup基础依赖numpy≥1.24.4、tqdm≥4.66.1、hydra-core≥1.3.2、iopath≥0.1.10、Pillow≥9.4.0。静态图评测补已有/缺失的pycocotools即可，无需notebooks/train/dev extras。CUDA connected-component extension可选，setup支持`SAM2_BUILD_CUDA=0`；只能在GPU阶段根据评测postprocess合同决定是否需要，不在无卡模式构建。
- **HQ-SAM2 beta**：Python≥3.10、torch≥2.3.1、torchvision≥0.18.1；基础numpy/tqdm/hydra/iopath/Pillow同上。setup额外列matplotlib≥3.9.1、opencv-python≥4.7.0，属于其声明依赖；只备缺失包，勿按README重新建全套。与SAM2同包名的问题由隔离source解决。
- **SAM3**：README支持配置为Python≥3.12、torch≥2.7、CUDA≥12.6；当前README示例torch2.10/cu128仅为示例，不要求重复替换已有兼容torch。pyproject却写Python≥3.8，**元数据不等于已验证支持范围**；按README≥3.12准备独立环境资源。基础依赖timm≥1.0.17、**numpy≥1.26,<2**、tqdm、ftfy==6.1.1、regex、iopath≥0.1.10、typing_extensions、huggingface_hub；实际builder也使用torch/torchvision、pkg_resources。已有numpy2环境不得全局降级，SAM3后续隔离。einops按源码导入需要补；不为静态图评测下载jupyter/训练框架全套。
- SAM3 README把`flash-attn-3`和`cc_torch`列为可选加速依赖；**RTX4080 Ada不适用FA3 Hopper加速**，无须现在装FA3。connected-components默认hole filling会影响最终mask；若缺后端而官方catch warning跳过，必须记录为postprocess未完整运行，不能悄悄与完整官方设置混作质量基线。
- 首轮FP32/TF32状态在**模型import之后**再次设置并记录：SAM3 `model_builder.py`的import-time `_setup_tf32()`会自动开启TF32；不能只在import前关闭。

## 最小COCO点/框评测的官方入口与原始输出范围

没有核验到三个repo共用的“一条命令COCO点/框”官方runner；勿编造官方benchmark命令。最小封装使用下面官方推理API，在同一冻结manifest上用COCO API评分；这是**本项目统一协议**，不是论文作者原评测的复现。GT-tight-box/GT内点与detector-box结果分开，当前无需下载detector。

### SAM2.1 small/large

入口：[官方README](https://github.com/facebookresearch/sam2#image-prediction)、[sam2_image_predictor.py](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)。`build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',checkpoint)`或l配置→`SAM2ImagePredictor`→每图`set_image(RGB)`一次→`predict(point_coords,point_labels,box,multimask_output,return_logits)`。输入坐标原图XY像素、框XYXY；独立提示可用B×N×2点/B×4框。

公开输出：`masks[B,C,Horig,Worig]`、`iou_predictions[B,C]`、`low_res_logits[B,C,256,256]`，单query返回可squeeze batch；C为1或3。`multimask_output=False`可能按配置dynamic stability回退，不总是raw token0。`return_logits=True`返回原图分辨率logits；默认bool threshold0。低分辨率返回值被clamp至[-32,32]，**不是unclamped raw decoder logits**。

完整decoder合同来自`sam_mask_decoder.predict_masks`：4个raw masks、4个IoU、mask token表征、object-score logits，含high_res_features。若数值验证，取此未slice/未clamp层；不能用公开API的1/3输出冒充“四个raw outputs全部验证”。

### HQ-SAM2

入口：[作者demo](https://github.com/SysCV/sam-hq/blob/main/sam-hq2/demo/demo_hqsam2.py)、[HQ2 predictor](https://github.com/SysCV/sam-hq/blob/main/sam-hq2/sam2/sam2_image_predictor.py)。相同API另有`hq_token_only=False`；作者明确COCO/LVIS等定量使用False。加载`configs/sam2.1/sam2.1_hq_hiera_l.yaml`与配套HQ权重。

**原始内部为5个masks + 4个IoU**：4个SAM masks与1个HQ mask。公开API先选SAM的1/3，再加共享HQ logit residual（False）；True只输出HQ mask，但不用于本轮COCO对照。预测IoU不是独立第五个HQ mask的IoU；不得虚构第五分数。最终公开outputs是1/3 refined masks/IoU、256² clamped logits，threshold/postprocess与其官方predictor走一致路径。保留原始5个与refined版本可做诊断，但质量报告以官方refined协议为准。

### SAM3 interactive/PVS（access满足后）

入口：[官方model_builder](https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py)、[sam3_image.predict_inst](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image.py)、[sam1_task_predictor](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam1_task_predictor.py)。使用`build_sam3_image_model(checkpoint_path=...,load_from_HF=False,enable_inst_interactivity=True,bpe_path=正确资源路径)`；`Sam3Processor(model).set_image(PIL_RGB)`生成缓存state，再`model.predict_inst(state,point_coords=...,point_labels=...,box=...,multimask_output=True,return_logits=True)`。

**不要直接对默认builder的`inst_interactive_predictor.set_image`调用**：默认`enable_inst_interactivity=False`；打开后其tracker可能没有独立backbone，官方外层`predict_inst`将processor的`sam2_backbone_out`接进去才是当前source支持路径。`Sam3Processor.set_text_prompt/add_geometric_prompt`的PCS detection outputs与这里PVS不相同，不能替代点/框单对象协议。

PVS公开输出与SAM2样式同为1/3 masks、IoU、clamped低分辨率logits；实际source `_bb_feat_sizes=(288,144,72)`，4×upsampler可输出**288×288**，部分docstring仍写256。因此以真实shape日志为准，不能硬编码所有模型256²。raw`sam3.sam.mask_decoder.predict_masks`有4 masks/4IoU/token/object-score；preserve high-res skips。最终在同原图分辨率评分即可，不强改其native image preprocessing来装成同输出函数。

SAM3默认interactive postprocess hole-area256而SAM2默认可不同；原生质量线保留并完整记录。额外统一无hole-fill线若做，须明列不同协议，不能把可选后端缺失当作native完整成绩。

## 此前三条2026论文的二次核验与下载优先级

同一轮访问arXiv HTML `citation_title/citation_date`，并以官方export API的id-list交叉核验，三个条目标题/日期均一致。API链接：[官方id-list](https://export.arxiv.org/api/query?id_list=2607.08688,2609.33996,2607.20705)。

| 条目 | 已核验题名/首发日期/官方repo | 下一轮是否需要资产 |
|---|---|---|
| [2607.08688](https://arxiv.org/abs/2607.08688) | SAM-MT: Real-Time Interactive Multi-Target Video Segmentation；2026-07-09；arXiv指向[作者project](https://henghuiding.com/SAM-MT/)，project与[官方FudanCVL/SAM-MT](https://github.com/FudanCVL/SAM-MT)相互对应、同作者/题名/arXiv。README已发布checkpoint/inference，training未发布 | **后续video/multi-target阶段必需**；当前静态COCO点/框诊断不下载SAM-MT权重、MOSE/LVOS视频。可以记repo，不扩增本轮大资产 |
| [2609.33996](https://arxiv.org/abs/2609.33996) | UnfoldCRF: Structured Mask Refinement with Image-Conditioned Latent Regions；2026-09-27；摘要说code/supporting materials将发布，arXiv没有作者repo链接。本次不能确认现成官方实现 | **质量refiner理论/容量对照需考虑；本轮模型准备不下载**。不存在可核验官方URL时不要猜repo或用同名第三方实现 |
| [2607.20705](https://arxiv.org/abs/2607.20705) | U-CFR: Uncertainty-Guided Cascade Forward Refinement for Interactive Segmentation；2026-07-22；arXiv comment ICPR2026；arXiv没有作者repo链接，本次未确认官方代码 | **后续自动pseudo-click/cascade路线相关**；当前不下载猜测权重或额外数据 |

下一轮GPU工作不依赖这三个额外模型，依赖的是三份公开SAM2/HQ权重+统一真实数据协议。SAM3是当前强质量竞争的重要资源，但可在未取得access时先推进公开基线/机制实验；结论须明确SAM3尚未比较，不称已挑战当前SOTA。
