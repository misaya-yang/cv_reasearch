# 代码

| 用途 | 入口 |
|---|---|
| 现行方法 SH | [x1012/w23_simple_pipeline.py](x1012/w23_simple_pipeline.py)`::simple_head`。依赖 `r5_candidate.py`、`w19_simple_head.py`、`w09_integration_lambda.py`、`lab.py`，都在 `x1012/` |
| 10-13 之后的实验 | `x1012/w01_*.py`～`w30_*.py`。每个文件的 docstring 写明动机和读法；计划与报告在 [docs/research/x1012_tasks/](../docs/research/x1012_tasks/) |
| 官方协议评分 | [x1012/w12_official.py](x1012/w12_official.py) |
| FoRIS、MFU 与 x1013 链（冻结） | [x1013/](x1013/) |
| 新照片留出（冻结） | [x1012s/photo_holdout.py](x1012s/photo_holdout.py) |
| 方法库与数据加载 | [../src/ics/](../src/ics/) |
| 仓库检查 | [check_repository.py](check_repository.py)：防止 GB 级生成物进入仓库 |

**历史代码：**
- 本目录顶层的 75 个脚本，以及 `x1012/` 中除 `w*.py` 和第一行列出的依赖以外的脚本，是 10-05～10-13 的实验与负结果；
- 它们原地保留，因为内部的导入路径和冻结 SHA 依赖当前位置；
- 它们不是当前入口，结论见 [archive](../docs/research/archive/README.md) 中对应的报告。

模型、图像、mask、缓存的 O24 和位置基都在仓库外的 `../cv_data/`。新的运行输出写到 `../cv_data/a/<run>/`。
