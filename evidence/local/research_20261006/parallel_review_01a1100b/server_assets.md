# 新服务器资产核对：ProM1 / M4 / M5

核对日期：2026-10-06。只使用新端点 `ssh -p 46466 root@connect.westb.seetacloud.com`；主机返回 `autodl-container-5a5e438f59-b729688e`，首次远端时间 `2026-10-06 22:42:45 UTC`。只读检查、导入和 meta 设备模型结构核验；没有编码/推理、GPU作业、安装、下载、远端写入、进程管理或历史队列复用。下列文件存在不代表当前运行、新实验结果或数学/官方数值等价已经验证。

## 结论

现有 DINOv3-L/16 timm 权重、真实骨干源码、RGB、FoRIS 和 native 位置基底可离线使用。**当前没有可见 GPU；容器限制为32 CPU、60GiB内存。DEV241历史缓存全部缺 raw H20/QKV，不能直接运行 M1/M5。** M4 同样需要新的八张移植图真实编码。CPU真实推理的墙钟/峰值内存尚未测；不能据代码和算术工作量承诺完成时限。

## 硬件与运行时

- `nproc` → 32；`/sys/fs/cgroup/cpu.max` → `3200000 100000`。
- `/sys/fs/cgroup/memory.max` → `64424509440` bytes = **60GiB**。`free -h` 展示宿主377GiB，不能作为容器配额。
- `nvidia-smi -L`、`nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader` 均无GPU行；`/dev/nvidia*`不存在。
- `/root/miniconda3/bin/python` + `PYTHONPATH=/root/demo4_cache/env`：torch `2.12.1+cu130`，`torch.cuda.is_available()=False`，`device_count()=0`；timm `1.0.30`；numpy `2.4.6`；scipy `1.18.1`；PIL `12.2.0`。
- timm导入伴随 `Error importing huggingface_hub.hf_api: No module named 'httpcore2'`。这次 timm 导入及 `pretrained=False` 的 meta 模型构造均成功；不把该信息误报为推理已通过或要求下载修依赖。

## 权重与真实骨干

权重目录：`/root/demo4_cache/models/dinov3-vitl16-timm`。

| 资产 | SHA256 |
|---|---|
| `config.json` | `a71f705b0074e173540d0bdbd3aa940fa8d7d3c6c7f020a683004c46ca605b24` |
| `model.safetensors` | `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941` |
| `/root/demo4_cache/env/timm/models/eva.py` | `a8c9807ef5e8dabc725c1e2a89439760a900d6af6cdafefea18fc121173643a2` |
| `/root/demo4_cache/env/timm/layers/pos_embed_sincos.py` | `99dc063e449d1f941373523795aeae6538d1de979310168f55d4e2d77c7a8af5` |

config的 `architecture=vit_large_patch16_dinov3`、`pretrained_cfg.tag=lvd1689m`。config中预训练输入为256/bicubic；实验宿主实际变换是1024方形RGB Resize + ImageNet归一化，必须绑定后者，不能由预训练config替代。

无需加载数值权重的核验命令：

```python
# PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTHONPATH=/root/demo4_cache/env
# /root/miniconda3/bin/python
m = timm.create_model('vit_large_patch16_dinov3', pretrained=False,
                     num_classes=0, device='meta')
# safetensors.safe_open仅读取键与shape，比较m.state_dict：
# missing_keys=[], unexpected_keys=[], shape_mismatches=[]
```

318个权重键完全匹配结构与shape，未做 `load_state_dict` 数值加载或前向。模型为 `timm.models.eva.Eva` / `EvaBlock` / `EvaAttention`，24块，D=1024，16头×64维，CLS+4register共5 prefix。源码 `eva.py:3135-3158`：patch16、dynamic_img_size、无QKV bias、LayerScale初始化1e-5、DINOv3 RoPE温度100、rotate_half=True、LN eps1e-5。meta结构观察到Q/K norm和attention输出norm都是Identity、fused_attn=True。

M1/M5实现应以该真实源码为基准：`EvaAttention.forward`（约225-291行）含QKV投影、可选Q/K norm、只对patch施RoPE、SDPA/softmax、输出投影；`EvaBlock.forward`（435-448行）含norm1、gamma_1、残差、norm2/MLP、gamma_2、残差；`Eva._pos_embed`（946行起）提供真实prefix/rope接口。M1的H20是`blocks[19]`原始残差输出；后缀是`blocks[20:24]`，不能错用归一化中间层。

指定 `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/external` 只有 `astra_emd` 模块，`/root/demo4_cache/models`只有上述timm模型资产；在这些限定路径内没有官方facebookresearch/dinov3 checkout。未扫全服务器，因此不宣称任意路径都不存在官方源码。**官方模型数值等价、官方权重转换链和源码commit未核验**；本次绑定的是实际timm源码哈希。

## 已有缓存与宿主资产

| 资产 | 当前观察 |
|---|---|
| `/root/autodl-tmp/demo9_extent/cache/evidence_v1/feat` | 241个`.pt`；用`torch.load(weights_only=True,mmap=True,map_location='cpu')`读取全241的顶层键，唯一schema为`('debiased','q','r')`。 |
| 上述缓存样例`0_0_72.pt` | `q/r`均`4096×1024,float16`，另有bool `debiased`；没有H20、Q/K/V或prefix原生状态。 |
| `/root/autodl-tmp/demo9_extent/results/extent_v1/episodes.json` | 现有manifest；只读取数据/源码/基底路径与资产映射，不读历史成绩。 |
| `/root/demo4_cache/data/COCO2014` | manifest指定RGB目录；首例support/query RGB均存在。 |
| `/root/autodl-tmp/datasets/ics/COCO2014/annotations` | manifest指定标注目录；首例参考标注存在。未读取query GT。 |
| `/root/autodl-tmp/demo8_local_verification/foris_source` | 完整FoRIS源码存在，可导入；目录不是git checkout，不能报commit。 |
| `/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt` | `state=NATIVE_BASIS_FROZEN`、`source_input=normalized_black_image`、`basis=1024×500,float32`；只核metadata，未重算SVD或重正交化。 |

FoRIS源 `models/foris.py` SHA256：`982cf4aa91ac90370151c353efb37240d7d4d5c35a0dfe9cc9509cd554c85fa9`。
宿主变换 `utils/data.py` SHA256：`37211b2f55391694023cad806cdc9ec210944d32346ac1de0f9502df39731b53`。
位置基底 SHA256：`9b9b20755a796cbda11bb7220d246ee540106e40cb024e249a5f63f884b6a116`。

本地当前 `src/ics/data.py:13-22` 包装器以 `norm=True` 返回中间层；`src/ics/methods/layer_extract.py`抽取block16/24的归一化特征，不是raw H20。旧单块attention intervention也不是M1/M5 suffix。末层缓存不能逆推出所需原生状态。除上述限定cache命名空间外未进行全历史结果搜索。

## 无下载的可用入口与明确缺口

共用入口仍是本地将上传的 `src/ics/data.py:TimmDINOv3` 与 `src/ics/foris.py:build_host`，使用CPU且读取上述现存权重/manifest/basis。`build_host`当前固定`torch.set_num_threads(2)`，即使有32CPU配额也不会自动用满；线程和并发由根代理统一调度，不由审计代理更改。

完整FoRIS的CRF是原Frank-Wolfe DenseGaussianCRF，不依赖pydensecrf的主路径：

```text
PYTHONPATH=/root/demo4_cache/env:
 /root/autodl-tmp/demo8_local_verification/crf_source/src:
 /root/autodl-tmp/demo8_local_verification/runtime/extensions
Python=/root/miniconda3/bin/python
```

`CRF`及`models.foris`导入成功；`Permutohedral.cpython-312-x86_64-linux-gnu.so`与`Permutohedral_gpu...so`均存在。先import torch再导入扩展，避免`libc10.so`未加载报错。源码依据CUDA可见性选择CPU/GPU滤波，本端点将选择CPU；未执行滤波，完整CPU CRF时延和数值等价尚待实测。

| 方法 | 可直接复用 | 仍需真实实现/获取；不得以现存末层cache代替 |
|---|---|---|
| ProM1 | timm blocks/RoPE、宿主RGB变换、末层角色选择、native basis和FoRIS | 原生R/Q raw H20、Q后四块native K/V、共同查询位置分块续算；无排除审核模式的native自探针一致性；ctx/plain/mean-unit/uniform完整输出与成本。 |
| ProM4 | 原R/Q RGB、完整参考mask、同冻结timm编码器、FoRIS回退 | P+/P−矩形、四条件同canonical点映射、八张干预图末层真实编码、paired环境协方差及完整min-cut。`maxflow`模块不存在，但`networkx`和scipy现成；现有算法库并非必须安装新依赖。严禁以不存在的干预cache宣称已编码。 |
| ProM5 | timm blocks/RoPE、末层ROI构造、宿主和basis | 原生R/Q raw H20及四块native Q/K/V、固定native QK的局部V更新/消息重分配、β=1一致性；.75/0及native-region/zero完整输出与成本。 |

安全运行资源估计仅是Pro给出的张量预算：M1四层Q K/V FP32约128MiB、R/Q H20约32MiB；M5四层双图QKV约384MiB、H20约32MiB，不计模型/宿主/临时attention。64/128位置分块，禁止物化全9N×N×heads或四层全N² attention；该限制不会证明CPU时延可接受。

## 证据命令边界

本次仅`ssh`只读shell、文件哈希、必要限定目录清单、只读safetensors/tensor metadata、包导入与meta模型结构核验。没有检查当前作业PIDs，也没有据旧缓存推断运行状态。下一步真实编码/资源试验由根代理依据本次用户授权启动；本文件不创建新待办或新授权。
