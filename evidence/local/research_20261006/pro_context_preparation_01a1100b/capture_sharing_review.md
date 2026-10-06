# 下一cohort原生capture共享边界（不改活跃代码）

已按根代理要求与`audit_4_6`直接发送对齐消息。M1当前真实首4运行保持原模块不变；本页仅是共享合同核对，不自动启动下一队列。

## 实际已对齐项

M1 `run_real_cpu.py`与M5 `audit_worker.py`均取`bound600_v2/smoke4.json`，首例`0_0_72`，RGB目录`/root/demo4_cache/data/COCO2014`；PIL RGB双线性到1024，除255，ImageNet(.485,.456,.406)/(.229,.224,.225)，FP32、冻结eval、无autocast/TF32。参考标签来自同一support annotation按c+1取完整二值mask，然后最近邻1024、area64。M1不以query c/fold作推断输入。

骨干同timm1.0.30 Eva；权重SHA45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941、config SHAa71f705b0074e173540d0bdbd3aa940fa8d7d3c6c7f020a683004c46ca605b24、eva.py SHAa8c9807ef5e8dabc725c1e2a89439760a900d6af6cdafefea18fc121173643a2。prefix5、patch4096、D1024、16heads×64。四层actual SDPA Q/K/V已经包含真实QKnorm/RoPE，raw H20是blocks[20]输入。Pi gate均取每例现存feature的`debiased`，原basis SHA9b9b20755a796cbda11bb7220d246ee540106e40cb024e249a5f63f884b6a116；Pi数值表达式同`x-(x@U)@U.T`，不QR/重开gate。

**尚不能把完整宿主场等同。** M1新图是实际原生末层LN→固定Pi→unit的FP32 query特征，rho1；M5宿主是已封存九项入口的旧FP16 conditional-cache MEAN，rho.25。共享原生producer不改变这两份现有完整读出或宿主合同。

来源：M1 `src/ics/methods/pro_common_context.py:85,158,270`与本目录`run_real_cpu.py`；M5 `src/ics/methods/pro_message_extrapolation.py:384-469`、`evidence/local/research_20261006/pro_message_preparation_01a1100b/audit_worker.py`。

## 当前两类cache不能无检查混拼

M1保存双图四层实际调用args/kwargs与raw输入、raw H20、**仅Q**四层postRoPE K/V、双图未投影raw final LN。M5保存双图raw H20、双图四层完整Q/K/V、双图四层native block输出、实际rope及scale，以及projected未unit的final patch特征。当前M1缺R QKV及native输出；M5当前公开对象没有完整actual args/kwargs和独立raw finalLN出口。

下一次联合capture可从**一次actual paired forward**生成这两类视图：保存原始block输入/输出、actual调用参数、双图QKV与完整raw finalLN；逐字段绑定RGB/preprocess/weight/source/basis/gate/patch位置/case hash。禁止用norm=True中间层作H20，禁止把M1自己的probe状态或M5干预分支写回native memory。M5 adapter必须保留其β1逐层核验；M1保留无排除self核验，两个容差/审核含义不互换。

以真实4101×1024 FP32计，每图每层state=16.01953125MiB：

| 统一存储项 | bytes | MiB |
|---|---:|---:|
| 双图四层QKV | 403144704 | 384.46875 |
| 四层paired raw inputs | 134381568 | 128.15625 |
| 四层paired native outputs | 134381568 | 128.15625 |
| 双图raw final LN | 33595392 | 32.0390625 |
| 不别名的上述合计 | 705503232 | 672.8203125 |
| 下一层input与上一层output别名，只保留H20..H24五个paired raw状态的联合合计 | 604717056 | 576.703125 |

这是张量算术，不是RSS实测；另计RoPE/args、projected/unit特征、模型(約1.2GB权重)、宿主、attention及分块临时峰值。需要公开paired原始张量父对象/视图，不能依赖私有`Tensor._base`拼回。采用别名必须严格只读；不能丢R状态来压成仅Q的128MiB。模型若只在一个进程加载可节省重复实例，跨进程并行复用则应使用带不可变identity的共享CPU storage/文件，不能把pickle callable伪装成可移植状态包。

## M4原Q可复用的精确对象

M4 `FrozenTimmRGBEncoder`用`get_intermediate_layers(norm=True,n=1)`取得未投影末层LN的64×64×1024。当前真实Eva源`eva.py:1030-1056`和`1120-1140`两接口调用同patch embed/pos/blocks与最终`self.norm`；前者在`1059-1066`移除prefix并reshape。因此在同图像/FP32/source下，共享的**unprojected final[Q,5:]**具有同计算定义，reshape可直接提供M4原Q图。

严格数值还需一次同producer核对：M1 paired batch2与M4当前单Q batch1的BLAS路径不保证逐位一样。这个核验未在当前首4进行，不能先承诺bitwise；M4仍需八张edited Q，ref-canvas控制另八张edited R，其17次编码行最多可复用原Q一次，不把八/十六张新干预编码省掉。不得将M1/M5 projected、unit或FP16 cache喂成M4的raw finalLN。
