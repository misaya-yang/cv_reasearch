# 环境与资产

最新实测端点：`ssh -p 48002 root@connect.westd.seetacloud.com`，E58实例`rs9ve2du7q-61300576`。
2026-10-07本轮已按用户要求完成无卡准备→关机→Chrome启动GPU→实验/分析→关机；控制台已核实“已关机”。
GPU模式实测12核、RTX4080 SUPER 32760MiB。两小时备用关机自动化已取消，未保留待执行队列。
报告、代码、日志在`/root/autodl-tmp/astra_granularity_20261008/`；[本地报告](../../evidence/local/astra_granularity_20261008/REPORT.md)、[关机确认](../../evidence/local/astra_granularity_20261008/server_stopped.png)。以下旧资产记录不代表当前在线或新授权。

| 资产 | 最新已记录位置/状态 |
|---|---|
| 本轮PACO599/COCO fresh600 FP16特征 | `/root/autodl-tmp/astra_granularity_20261008/cache/`，1199例、20.12GB，保留；缓存完成后剩7.76GB |
| 约19GB、1200唯一例FP16特征 | `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/confirm1200_conditional_v1/inputs`；目录1201项不等于唯一例数 |
| fresh600与固定600 | 重叠305例；固定清单`/root/autodl-tmp/cpu100_20261006_01a1100b/fixed600_evaluation_rows.json` |
| fresh600缓存 | `/root/claude_store/fresh600_cache`已换符号链接；原始固定600的FP32缓存已删 |
| 共享FoRIS依赖 | `/root/autodl-tmp/demo9_extent`，原记录约4.1GB，仍为运行依赖 |
| 数据/权重 | `/root/demo4_cache/data/COCO2014`、`/root/autodl-tmp/datasets/ics`、`/root/demo4_cache/models` |
| Python/包 | `/root/miniconda3/bin/python`、`/root/demo4_cache/env` |
| INSID3/FoRIS/CRF | `/root/autodl-tmp/demo4/INSID3`；`/root/autodl-tmp/demo8_local_verification/{foris_source,crf_source/src,runtime/extensions}` |
| A/B/C字段与冻结源 | `/root/autodl-tmp/evidence_bench_step3_20261007_25142/`；本地[D汇报](../../evidence/local/pro_cards_20261008/D_report.md) |

BF16批组成可改变聚类；两套源码有同名`models/utils`。INSID3无CRF已有完整600结果，CPU CRF曾有malloc错误，未验证完整CRF分支。
主线固定残差转移仍需外部`components/rcg_readout/hypothesis_source_contrast`，本机没有该目录或输出数组。
历史队列及缺失state不构成当前任务；旧环境详情查[清理前SERVER](https://github.com/misaya-yang/cv_reasearch/blob/82df9169a23c5f58ecf1ec57de237cddecc45ec4/docs/harness/SERVER.md)。
