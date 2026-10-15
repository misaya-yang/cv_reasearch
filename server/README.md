# 服务器确认实验

本目录放在专用服务器（RTX 4080 Super 32GB）上做的确认实验。本机（M4 Max）只用来找方法；全量运行、重新编码和正式计时都在服务器上做。

- 实验计划：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- 当前状态：只有计划，代码还没写。写代码时按计划第 6 节的移植说明进行，并先通过 E0 的一致性门槛。
- 被确认的方法：FoRIS-free 头 SH，定义和本机证据见 [W23_simple_pipeline.md](../docs/research/x1012_tasks/W23_simple_pipeline.md)。

## 目录约定（与 `docs/research/CONTINUING_ELSEWHERE.md` 一致）

```text
paper_project/
  cv_reasearch/          # Git 克隆（本分支）
  cv_data/               # 从本机复制，路径相对它解析
    a/server_<run>/      # 服务器上的新输出，不覆盖复制过来的目录
```

## 需要复制到服务器的资产

| 用途 | cv_data 下的路径 | 大小 |
|---|---|---:|
| DINOv3-L/16 权重 | `demo4_cache/models/dinov3-vitl16-timm/` | 1.1 GB |
| 位置去偏基（FoRIS 与 SH 共用，SHA 前缀 9b9b2075） | `native_assets/positional_basis.pt` | 2 MB |
| FoRIS 官方源码（commit 1aa02a1）、CRF 源码 | `third_party/foris_official/`、`third_party/crf_source/` | < 1 MB |
| COCO | `demo4_cache/data/COCO2014/` | 6.5 GB |
| LVIS、PASCAL-Part、PACO-Part、SUIM、iSAID、ISIC、Lung、Fundus | `datasets/ics/{LVIS,Pascal-Part,PACO-Part,SUIM,iSAID,ISIC,LungSegmentation,Fundus}/` | 共约 24 GB（ISIC 11 GB） |
| 官方协议 manifest 与协议 | `a/x1013_official/{manifest.json,protocol.json,masks/}` | 168 MB |
| 官方协议用到的掩码与 Road 输入 | `a/{official_s0_manifest_20261008,isaid_seed0_full_20261013,extended_benchmarks_seed0_20261010_v3,chest_seed0_manifest_20261010,fundus_seed0_manifest_20261010,deepglobe200_preparation_20261009}/` | 约 830 MB |
| 新照片留出 manifest 与掩码 | `a/x1013_photo_holdout/{manifest.json,masks/}` | 53 MB |

合计约 34 GB。

- 官方协议共 60,981 例、36,812 张不同图像；新照片留出共 3,211 例、5,034 张图像。
- 数据集目录里可能有符号链接，复制时请用 `rsync -aL`。
- 不需要复制 raw 特征缓存（`a/paco_mean200_*/raw_cache`），服务器上重新编码。

## 本机特有、必须替换的部分

1. **CRF**：本机用 M4 原生 permutohedral 扩展（`cv_data/runtime/macos/crf/*.so`，Linux 不可用）。服务器改用 FoRIS 官方的 `utils/refinement.init_crf`，即 `CRF.DenseGaussianCRF` 加 CUDA `PermutohedralFiltering`，需在 `third_party/crf_source` 编译。
2. **编码器**：本机在 MPS 上运行；服务器在 CUDA 上用 FP32，并关闭 TF32（`ics.foris.build_host` 已这样设置）。
3. **路径**：manifest 中 RGB 路径相对于 cv_data，但掩码路径是本机绝对路径（`/Users/.../cv_data/...`）。读取时把这个前缀换成服务器的 cv_data 根；`ics.official_data.load_inputs` 的 RGB 和 mask 哈希校验会兜底。
4. **运行时目录**：`cv_data/a/x1013/w02/runtime` 是本机拼装的冻结副本，服务器实现不应依赖它。见计划第 6 节。

## 带回本机的东西

- 每例每臂的 I/U 整数（jsonl.gz）、汇总 JSON、计时 JSON；
- 环境快照（`pip freeze`、`nvidia-smi`、`torch.version.cuda`）；
- 封存记录：代码 SHA、manifest SHA、开始与结束时间；
- E0 一致性报告。

放到 `evidence/research/server_*`。掩码本身不带回；只有论文案例图需要的少数例另行导出。
