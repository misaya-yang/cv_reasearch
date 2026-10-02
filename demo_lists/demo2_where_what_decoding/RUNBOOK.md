# 服务器操作手册（给执行实验的 AI）

目标读者：接手跑实验的 AI。这里只写**在这台机器上实测过**的事实和踩过的坑。

## 1. 机器与路径

```bash
ssh -p 57510 root@connect.westc.seetacloud.com
```

| 项 | 值 |
|---|---|
| GPU | 单卡，32 GB 显存（`nvidia-smi` 显示为 RTX 4080）。**与 demo1、demo3 的任务共用**，见第 9 节 |
| CPU / 内存 | 容器配额 **25 核 / 90 GB**（`nproc` 和 `free` 显示的 128 核 / 503 GB 是宿主机的，不是你能用的） |
| Python | `/root/miniconda3/bin/python`（3.12，torch 2.12.1+cu130）。`python3`、`pip` 不在 PATH 里，必须写全路径 |
| 数据盘 | `/root/autodl-tmp`，共 50 GB |
| 系统盘 | `/`，共 30 GB |
| 公共数据集 | `/root/autodl-pub`（只读压缩包） |

demo2 在服务器上的位置：

| 路径 | 内容 |
|---|---|
| `/root/autodl-tmp/demo2_where_what_decoding` | 代码（与本地本目录同构：`wwd/ mdecode/ scripts/ tests/ tools/ pilots/`），以及 `results/`、`logs/` |
| `/root/autodl-tmp/demo2_dev` | 代码的测试副本。**改了代码先同步到这里跑小样本，通过后再同步到上一行的目录**（排队任务从那里读脚本） |
| `/root/autodl-tmp/demo2_pilot` | 先导实验的工作目录：`env/`（隔离依赖）、`data/`、先导结果与日志；`models` 是指向系统盘的软链接 |
| `/root/demo2_cache` | 系统盘上的大文件：`models/`（全部权重）、`ADEChallengeData2016/`（训练集）、`coco/`、`rf/`（区域特征）、`rec/`（识别器）、`pix/`（逐 patch 头） |

**不要碰 `/root/autodl-tmp/demo1_sam`。** 那是另一个对话的工作区，可以只读它的资产（SAM 权重、COCO 小集）。

## 2. 磁盘：当前最大的风险

数据盘是和 demo1 的对话共用的，而它会写很大的结果文件。2026-10-02 凌晨一小时内实测到的占用：
84% → 94%（只剩 3.5 GB）→ 57%。也就是说可用空间随时可能被吃光，也可能被对方清掉。

- 每次启动任务前先 `df -h /root/autodl-tmp /`。
- 新的大文件优先放系统盘 `/root/demo2_cache`，再软链接到工作目录。
- 不要把概率图或 logits 写盘。评测脚本把 1/4 分辨率的模型输出以 float16 缓存在**内存**里
  （ADE20K 2000 张约 13–20 GB，内存充足）。
- 下载模型时，`hf_fetch.py` 合并分片期间会短暂多占一倍空间。

`/root/demo2_cache` 里现有的文件：

| 文件 | 内容 |
|---|---|
| `ADEChallengeData2016/{images,annotations}/training` | ADE20K 训练集 20210 张 |
| `regfeat_dinov2l_{training,validation}.pt` | 真值类区域的 DINOv2-L 池化特征（165120 / 16909 个区域） |
| `regclf_dinov2l.pt` | 区域识别器（MLP）权重与特征归一化统计量 |
| `patchhead_dinov2l.pt` | 对照用的逐 patch 线性分割头 |

## 3. 依赖：隔离安装，不改基础环境

基础环境只有 torch / torchvision / numpy / PIL。其余都装在独立目录里，并且**追加**到 `sys.path` 末尾，
这样基础环境的 numpy 和 torch 优先：

```bash
/root/miniconda3/bin/pip install --target /root/autodl-tmp/demo2_pilot/env transformers huggingface_hub safetensors scipy
/root/miniconda3/bin/pip install --no-deps --target /root/autodl-tmp/demo2_pilot/env rankseg
```

已装版本：transformers 5.17.0、huggingface_hub 1.33.0、scipy 1.18.1、rankseg 0.0.7。
脚本通过环境变量 `DEMO2_ENV=/root/autodl-tmp/demo2_pilot/env` 找到它。pip 源已配置为阿里云镜像。

## 4. 网络与下载

| 站点 | 状态 |
|---|---|
| `hf-mirror.com` | 可用 |
| `huggingface.co` | 不通 |
| `pypi.org`、阿里云 pip 镜像 | 可用 |
| `github.com` | 可用 |
| `dl.fbaipublicfiles.com`、`download.openmmlab.com`、`data.csail.mit.edu` | 根路径返回 403，具体文件需逐个试 |

**不要用 `from_pretrained("org/name")` 直接从 hub 拉大文件**：默认通道会在一百多 MB 处停住。
用本目录的分段下载脚本，再从本地目录加载：

```bash
cd /root/autodl-tmp/demo2_pilot
/root/miniconda3/bin/python hf_fetch.py facebook/mask2former-swin-large-ade-semantic models/m2f-swin-large-ade \
    config.json preprocessor_config.json "model.safetensors|pytorch_model.bin"
```

`"a|b"` 表示先试 a，没有再试 b。实测 12 路并行约 4 MB/s。如果最后一个分片卡住，杀掉那一个 curl，
脚本会自动用新连接续传：

```bash
pkill -f "[c]url -sL --connect-timeout 15 -m 300"
```

已下载到 `models/` 的权重：

| 目录 | 来源仓库 | 用途 |
|---|---|---|
| `segformer-b0-ade`、`segformer-b2-ade`、`segformer-b5-ade` | `nvidia/segformer-b{0,2}-finetuned-ade-512-512`、`…-b5-…-640-640` | 分割器 |
| `m2f-swin-tiny-ade`、`m2f-swin-large-ade` | `facebook/mask2former-swin-{tiny,large}-ade-semantic` | 分割器 |
| `upernet-convnext-large` | `openmmlab/upernet-convnext-large` | 分割器 |
| `dinov2-large` | `facebook/dinov2-large` | 存在性识别器的冻结编码器 |

## 5. 数据

| 数据集 | 位置 | 说明 |
|---|---|---|
| ADE20K 验证集 | `demo2_pilot/data/ADEChallengeData2016/{images,annotations}/validation` | 2000 张，已解压 |
| ADE20K 训练集 | 同上 `…/training`，软链接到 `/root/demo2_cache` | 20210 张，已解压 |
| Cityscapes | `/root/autodl-pub/cityscapes/*.zip` | 未解压；只需 val 的 500 张和 `gtFine` |
| COCO 2017 | `/root/autodl-pub/COCO2017/*.zip` | 未解压；COCO-Stuff / 全景标注需另行下载 |
| VOC | `/root/autodl-pub/VOCdevkit/*.tar.gz` | 未解压 |

从压缩包里只解出需要的部分，例如：

```bash
unzip -q -o /root/autodl-pub/ADEChallengeData2016/ADEChallengeData2016.zip \
    "ADEChallengeData2016/images/validation/*" "ADEChallengeData2016/annotations/validation/*" -d data
```

## 6. 跑任务的方式

SSH 连接会被远端断开，长任务必须脱离会话，并且**把命令写进脚本文件再执行**：

```bash
ssh -f -p 57510 root@connect.westc.seetacloud.com \
  'cd /root/autodl-tmp/demo2_pilot && nohup sh my_job.sh > my_job.log 2>&1 < /dev/null &'
```

三个实际踩过的坑：

1. **`pkill -f` 会杀掉自己。** 远端 shell 的命令行里包含你写的模式串，所以 `pkill -f run_queue.sh`
   会先把执行这条命令的 shell 杀掉，后面的命令全部不执行，而且没有任何报错。
   要么用方括号技巧 `pkill -f "[r]un_queue.sh"`，要么先 `pgrep` 拿 PID 再 `kill`。
2. `ls a b` 只要有一个文件不存在就返回非零，不能用来判断"两者之一存在"。用 `[ -f a ] || [ -f b ]`。
3. `subprocess.run(..., text=True)` 会把 `\r\n` 折叠成 `\n`，解析 HTTP 头时不要按 `\r\n\r\n` 切分。

GPU 是和 demo1 的对话共用的。启动前看一眼 `nvidia-smi`；本项目的任务都是推理，显存占用 2–12 GB。

## 7. 评测脚本

```bash
cd /root/autodl-tmp/demo2_where_what_decoding
D=/root/autodl-tmp/demo2_pilot
DEMO2_ENV=$D/env /root/miniconda3/bin/python scripts/eval_decoding.py \
    --dataset ade20k --data-root $D/data/ADEChallengeData2016 \
    --family segformer --model-dir $D/models/segformer-b5-ade --short-side 640 \
    --out results/ade20k/segformer-b5.json
```

- 已有输出文件时脚本拒绝覆盖。
- 预处理按 mmseg 的测试流程实现（保持长宽比缩放、尺寸取 32 的倍数、整图推理）。
  已核对的基线：SegFormer-B0 37.43、B2 46.62、B5 51.00、UperNet ConvNeXt-L 53.22、
  Mask2Former Swin-T 47.99、Swin-L 56.05，均与各自论文的单尺度数字一致（差距 ≤ 0.3）。
  **任何新模型接入后，先核对 argmax 基线能否复现论文数字，再看别的。**
- 单元测试（纯 CPU，约 1 分钟）：`/root/miniconda3/bin/python tests/test_core.py`。

## 8. 负载下的耗时参考

| 任务 | 耗时 |
|---|---|
| SegFormer-B0 / B5 推理 ADE20K 验证集 | 61 s / 144 s |
| Mask2Former Swin-L 推理 | 293 s |
| 对 2000 张缓存结果做一遍解码扫描 | 40–60 s |

## 9. 共用 GPU 下的做法（10-02 阶段 A 执行时实测）

这台机器同时跑着 demo1、demo3 的任务。GPU 按进程分时，N 个忙进程各拿约 1/N；显存经常只剩 5 GB 左右。

- **给自己的进程设显存上限。** `wwd/common.py` 在导入时调用 `torch.cuda.set_per_process_memory_fraction`，
  默认 0.2（约 6.3 GB），用环境变量 `DEMO2_GPU_FRAC` 调。评测队列用 0.35。上限保护的是别人的任务：超了只会让自己的进程报错。
- **同时最多 3–4 个 GPU 进程。**
- **减少 GPU 到主机的同步。** 在分时的 GPU 上，每一次 `.unique()`、布尔索引、`.item()`、`.cpu()` 都要等一轮调度。
  `eval_fusion.py` 和 `extract_regions.py` 已经按这个原则写（直方图用带权 `bincount`，区域统计攒到最后一次取回）。
  改写前后，同一个评测从"500 张要 20 分钟以上"降到"500 张 5–6 分钟"。
- **验证集里有一张 1600×1600 的图**（`ADE_val_00001157`）。150 类的后验在它上面每份 1.4 GB。
  第二意见（另一个分割器、其他视角）必须一个一个地算，不能先全部算好放在列表里。
- **排队用 PID 串联**：`sh scripts/queue_x.sh <要等的PID>`，脚本开头 `while kill -0 $1; do sleep 10; done`。
  不要用 `pgrep -f` / `pkill -f` 找自己的队列：远端 shell 的命令行里只要出现过那个名字，就会匹配到自己并被杀掉，方括号技巧也救不了。
- 评测脚本对已存在的输出拒绝覆盖；队列里的 `run()` 对已存在的结果直接跳过，所以队列可以随时杀掉重排。

## 10. 阶段 A 的脚本

| 脚本 | 作用 |
|---|---|
| `wwd/common.py` | 数据集（ADE20K、COCO 全景转 133 类语义）、分割器封装（SegFormer / UperNet / Mask2Former / EoMT / EoMT-DINOv3）、冻结编码器（DINOv2、SigLIP）、区域池化、识别器、逐 patch 头、混淆矩阵与逐图统计 |
| `scripts/extract_regions.py` | 区域特征：真值类区域 + 分割器预测的类区域；来源 = 分割器自己的骨干（`own`）和 / 或外部编码器。`--fast` 用于训练集 |
| `scripts/train_recogniser.py` | 区域识别器（MLP）。`--kinds gt,pred`、`--crossfit`（在验证集上 2 折）、`--concat`（多个编码器拼接） |
| `scripts/train_patch_head.py` | 对照：同一冻结编码器上的逐 patch 分割头 |
| `scripts/eval_fusion.py` | 一次遍历算完：诊断量、四个上限、任意多个识别器的全部融合变体、像素级对照、第二个分割器、同一分割器的其他视角、掩码分类的无标签读出规则；同时存逐图统计 |
| `scripts/bootstrap.py` | 对逐图统计做配对自助法，给出变体之间 mIoU 差的 95% 区间 |
| `scripts/summarize.py` | 从结果 JSON 生成 Markdown 表。**文档里的表都由它生成** |
| `scripts/region_report.py`、`scripts/stack_report.py` | 区域级准确率按面积 / 纯度分层；多来源堆叠识别器（只做分析，在验证集上交叉拟合） |
| `scripts/queue_*.sh` | 实际跑过的队列，保留作记录 |
| `tools/prep_coco_panoptic.py` | COCO 全景标注转语义标签图 |

新增下载的权重（都在 `/root/demo2_cache/models`）：`eomt-large-ade`（`tue-mps/ade20k_semantic_eomt_large_512`）、
`eomt-dinov3-large-ade`（`tue-mps/eomt-dinov3-ade-semantic-large-512`）、`m2f-swin-small-ade`、`m2f-swin-base-in21k-ade`、
`m2f-swin-large-coco`、`m2f-swin-tiny-coco`（`facebook/mask2former-swin-{large,tiny}-coco-panoptic`）、
`siglip2-so400m-p16-512`（`google/siglip2-so400m-patch16-512`）。`facebook/dinov3-*` 需要人工授权，没有下载。

## 11. 10-02 清理后的状态

`$DEMO2_CACHE` 里只剩 ADE20K 训练集、`models/dinov2-large`、`models/eomt-large-ade`、`models/m2f-swin-tiny-ade`；`rf/`、`rec/`、`pix/`、`coco/` 和其余权重已删除。
重跑第 10 节的任何脚本之前，要先重新下载对应权重并重新提取特征（`scripts/extract_regions.py`、`scripts/train_recogniser.py`、`scripts/train_patch_head.py`）。
