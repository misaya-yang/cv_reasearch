# 参考 RGB 径向功率纹理 v1

独立实现、物理 RGB 正负例与完整独立 CLI 已完成，可登记一个可运行候选定义；科学原创性、真实收益未成立。尺度、RGB通道及控制不多计。主合同接受后的两个歧义已固定：**环内SUM功率／全部非DC SUM功率**；强单频控制包含每窗最大份额环带的真实编号/3与该份额，平手最低编号。

实现：[reference_texture.py](../../../../src/ics/methods/reference_texture.py:1)；独立入口：[run_reference_texture.py](../../../../scripts/run_reference_texture.py:1)；[合同和历史核查](contract.md)。主场仍是MEAN+g/8，没有自动重开参数/尺度变体。

## 可复核本地结果

[math_check.json](math_check.json)绑定源码SHA `779425b5faf11576d3401b88cd3d26f567831fab166c865d59d72c713c55d031`。检查使用原uint8 RGB1024、完整参考mask、实际PIL1024→256及正常FT/所有控制/两级renderer；不注入描述子/原型。完整物理条纹的颜色总量、转移数相同，仅run长排列不同；实际同ROI颜色/方差原型差均为0。

| 合成完整1024行 | IoU |
|---|---:|
| 固定纹理主场 | 0.9978867 |
| strong dominant-band index+energy | 0.9981403 |
| standalone .5+.5g | 0.9367924 |
| color histogram、variance、zero/MEAN | 0.2327139 |

**强peak也解开且略优于主法，未证明多环带必要性。** 正例只证明给定合法RGB中存在颜色/方差以外的空间组织量，不说明跨图语义迁移。

完整负例未改方法：同纹理错类别使主IoU从正确base1.0降到0.60531、新增341774假像素；目标纹理改变使1.0降到0.21346、新增338902假像素并漏340028真像素。不得将有界残差解释为IoU保障。

Fourier功率份额守恒与Parseval方差、固定RGB256的90°径向描述子不变、flat patch tie=band0/energy0、zero场完整等价、缺纯参考patch与standalone显式base回退均通过。旋转原图后PIL两轴uint8舍入可改变1级RGB，观察完整场最大差6.93e−5，不能声称整个原图管线精确旋转不变。

两worker独立CLI预测通过，NPZ故意含object queryGT但未读取，完整1024与原尺寸预测全部封存；合成cost含全部检查/子进程启动约3.6秒，不当作真实运行时间。

## 真实 smoke4 尚未执行

此前root曾授权同一CPU端点的一次first4、4CPU/8GiB；主法、color、variance、dominant、zero、standalone在真实封存前已固定。关机前只读核查过现存M4清单中的四张原RGB和完整参考mask路径（不是cov64上采样），计划由本地bind_actual4.py复用并验证原SHA，查询annotation只留下评测路径、不读字节。

独立source snapshot已在本地产生，摘要见 [deployment_source_manifest.json](deployment_source_manifest.json)。服务器关闭前连接已中断；没有确认上传成功，没有启动infer/score，没有真实结果。用户随后确认关机并停止远端工作，因此此前授权已结束，不能自动重连或恢复；继续需要新的明确请求。不能将此记作质量负例或完成smoke。

原计划的infer只读q/r/cov/score重建MEAN、原R/Q RGB和完整R mask；不新编码。每臂保存field、packed1024、packed原尺寸mask与SHA；完整seal后score才打开packet truth/native与原query annotation。成本单列MEAN重建、RGB读入、FFT描述子/所有renderer、全进程峰值RSS，输出历史1024与原尺寸两套完整class-summed报告。score会标明4个已暴露例的活动/成本检查，不是总体增益确认，也不是当前待执行队列。


此前runner合同只允许≤2worker×2thread、≤8 decimal GB（比8GiB更保守），不含GPU、新encoder或下载。用户关机后不再执行；任何实际smoke或600复测均需新的明确请求。
