# 局部验证实验（开发中）

本目录保存 FoRIS 对照、独立 episode 清单生成、特征提取及局部验证器训练代码。当前结果属于开发快照，不是完成的方法结论。

- `foris_control.py`：FoRIS 强对照；当前脚本使用服务器缓存路径。
- `prepare_fresh_manifest.py` 与 `fresh800_seed2040_manifest.json`：新 episode 的显式清单；清单不包含图像像素。
- `extract.py`：局部观测/特征准备。
- `train.py`：训练和评估入口，`--data` 与 `--out` 指定输入及输出。
- `crf_runtime.py`：CRF 运行时，依赖本地第三方源码和编译工具。

第三方完整 checkout 不上传到本仓库。原实验使用：

| 依赖 | 来源 | revision |
|---|---|---|
| FoRIS | https://github.com/Xi-Mu-Yu/FoRIS | `1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e` |
| CRF | https://github.com/netw0rkf10w/CRF | `13a123f7cd3ea1f975e6c483b1e7c52112d7c151` |

复现时需按上游许可自行准备 `foris_source/`、`crf_source/`、模型、COCO 数据及环境，并调整 `/root/...` 路径。完整 report 留在本地；精简快照和哈希见仓库 `evidence/manifest.json`。本次仓库发布未重新运行这些实验。
