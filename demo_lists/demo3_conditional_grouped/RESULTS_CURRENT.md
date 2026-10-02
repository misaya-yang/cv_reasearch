# 当前短测与下一项

七分支真实权重FP32/TF32关闭、1024输入、各128步已完成。原始报告在 results/pilot_v1，单训练seed/64图像/三个固定时间档；是上采样COCO受控诊断，不是原生1024或SOTA结果。

候选在t=.2/.5较original小幅改善，在t=.8退化；所有三档均落后group32。固定时间基小幅改善所有档。6×6邻域在中高噪声最好，但低噪声退化。完整stem中高噪声改善，低噪声略退化。动态读取首轮未稳定，不因它较弱就称候选胜过强基线。

发现原时间采样128次中只有2次t>=.65、1次t>=.75。下一项从同一128步checkpoint分叉：candidate native继续 vs 分层时间继续，并与fixed/6×6/original同额度分层继续比较。训练图像/噪声种子相同，采样分布变更明确记录，评价仍是原固定三档；留出结果已用于开发决定，后续最终验证要另用未触碰图像。

已完成工程改法：单patch融合正交旋转，逆变换复用原角，反向通过逆恢复中间状态。GPU随机布局/正逆/一阶梯度和当前预训练训练后fixture通过。局部分析+tanh+合成+反向从14.41ms到0.528ms；完整模型forward/backward当前中位数0.1529s到0.1466s，约4.1%，不含optimizer/数据IO/文本编码，共享GPU条件，不作独占吞吐或质量收益。

原结果和model_v1/pilot_v1源码在results/evidence保留。清理回执在results/disk_cleanup.json；没有删除实验结果或有效恢复点。


## 2026-10-01 本轮续接

从同一 neighborhood-stratified 第256步恢复点出发，两分支均重新初始化AdamW，FP32/TF32关闭，分层时间、256次更新、相同图像/噪声/时间流。静态邻域对照已完成，语义门控只调节邻域补充特征，保留own RGB与原预测路径。默认1e-5门控收益约0.001%-0.003%；仅门控LR=1e-3、其余仍1e-5的诊断使 t=.2/.5/.8 MSE 分别比静态对照降低0.717%/1.997%/0.293%。复用同一个静态报告，没有重跑对照。三档图像配对bootstrap区间均低于0，但单训练seed，64开发图已用于方法选择，不代表最终泛化。原报告在 context_gate_v1 与 context_gate_lr；配对统计在 context_gate_paired.json。

官方25步AdamLM/order2/timeshift3/CFG4生成已完成4个固定caption+noise，各pretrained/neighborhood/neighborhood_gated(默认LR)分支，统一FP32调用vendor _impl_sampling，绕开其BF16装饰器；图片与记录在rollout_v1。目视保持可生成，但不能据此称质量提升，当前没有FID/GenEval成绩。pretrained与finetuned比较包含训练；neighbor与gate才是匹配比较。

LR诊断首步OOM日志保留在 logs/context_gate_lr.log；同期其他实验增加显存占用。自动重试1已成功，日志 context_gate_lr_retry1.log。当前 run_combo.sh 检验邻域+full-stem（CPU非零邻域恢复误差2.98e-7、梯度通过，GPU warmstart也检查），相同第256步起点、重置优化器、256更新，与已完成静态对照比较。随后 run_seed_probe.sh 配对复核门控 LR1e-3，seed2028各256步；只变化本段训练随机性，共同的前256步已训练权重保持，不能称两个完整独立训练seed。

预算：本轮 static256 + gate默认256 + gateLR诊断256 已完成768更新；combo256进行中；第二seed配对512排队。后两阶段代码暂存staging后顺序安装，避免运行途中改变源码hash。控制器按实时显存等待，有OOM最多重试3次并保留日志；不会抢占其他实验。GPU当前整体100%，不因等待本项显存而填充无价值作业。数据盘最近13GiB余量；至少保留5GB。重复分块清理2.24GiB、文本cache无损紧缩1.09GiB已完成，成品权重与重要结果仍保留。


最新：combo已完成，GPU预训练初始恢复max1.91e-6，已训练邻域warmstart恢复max2.15e-6，原1e-5阈值当前fixture通过。相对匹配static，t=.2/.5/.8 MSE变化-0.768%/-0.389%/+0.384%；低噪声退化，故不升级这条组合路线。原始结果context_stem_v1与context_stem_paired.json保留。第二seed控制器42942仍有效，最近整卡100%/剩余10.5GB，按14GB准入等待其他实验释放显存。无需也不会停其他实验。CGD实验续接heartbeat已创建(id=cgd)，每5分钟在本线程检查并依据结果继续，状态不变保持安静。


## Heartbeat 2026-10-01 19:52 UTC 续接

第二续训seed2028已完成：从同一前256步权重出发、各再256步，semantic-gate LR1e-3相对静态邻域 t=.2/.5/.8 MSE -0.396%/-2.726%/-2.435%；图像bootstrap三档均低于0。是两次续训随机性，不是两个从头独立seed。raw: results/seed_probe，paired_summary.json。

补齐两种更强门控控制，各256更新（本次新训练预算合计512）：固定64通道可学增益、仅时间输入的同结构门控，新增参数统一LR1e-3，原预测头1e-5；同neighborhood256起点、重新AdamW、seed2027、完全相同图像/噪声/t流。CPU非零邻域初始化逐位恢复、门控梯度通过，GPU旧权重warmstart原门槛检查通过。固定增益 vs static 的三档MSE +0.00259%/+0.00444%/+0.02434%；时间门控 -0.67672%/-0.74765%/+0.07404%。相对semantic，time在中噪声高1.275%（semantic较time低1.259%），低噪声高0.368%，这两档图像CI排除0；高噪声差异CI包含0。不能凭开发图就称语义适应最终成立。raw: gate_controls，gate_controls_paired.json。

新64张官方COCO val预先选定（排除全部既有320图与可识别demo1 IDs），manifest SHA ccaec2f1ae74a39b17dbdf7c0863621fd81fbe34c14a1401026455d427948810；metadata副本results/unseen_val_manifest.json。仍上采样1024，不是native1024。run_unseen_eval.sh已启动控制器47743，Qwen缓存64张已完成；正在依序评估static/semantic/static-gain/time以及seed2028配对，共6×64×3=1152次batch1前向，无训练/新checkpoint。只读评价同时保留velocity与error-gradient诊断，保持固定图像噪声时间；随后只生成semantic LR1e-3的4张25步图，复用rollout_v1静态对照，不重复已有图。脚本加OOM重试和显存准入，失败日志保留。新sample CLI参数错误已在CPU --help发现并修复，GPU生成尚未调用这个错误版本；evaluate CLI在服务器确认。

累计受控训练更新3584（首轮896+时间probe640+context768+combo256+seed512+门控控制512），另有28次资源试跑，不归入方法训练额度。本次heartbeat新增512已经用完，后续只读评价/生成，勿无条件再训练。最新数据盘15GiB余量是其他任务释放空间后的实测，不能记作本任务清理收益。整体GPU100%。

CPU固定投影碰撞反例results/gate_structure_cpu.json：旧混合输入误差5.33e-15，新context-gated表示L2差0.1267。它只说明固定旧投影丢失的信息不能被同h的后处理恢复；固定gain可吸收到可训练邻域投影；不证明完整可重训模型的表达优势或实际质量。新对照源码及此前源码保留在results/evidence。


新64图第一组配对完成：semantic LR1e-3 vs static（seed2027）t=.2/.5/.8 MSE -0.581%/-1.886%/-0.428%，图像bootstrap三档均低于0。data SHA与预先选择manifest一致，结果results/unseen_v1/{static,semantic,first_pair}.json。其余控制和seed2028仍在同一只读队列依序评价，勿重复启动。这个独立图像批次验证的是MSE泛化；仍不能替代真实生成质量评价。

复现新增步骤：bash tools/run_gate_controls.sh（仅首次各256训练，同步已有report会跳过）；python tools/prepare_unseen_val.py；bash tools/run_unseen_eval.sh（只读评价六分支，随后4张semantic LR1e-3官方25步生成）。PYTHONPATH/runtime/data盘环境都写在脚本里，不能在本机无CUDA执行GPU队列。evaluate_mse.py拒绝覆盖旧结果、检查adapter参数契约/Caption SHA/finite；CPU缺torchvision未额外安装，服务器已有官方torchvision，CLI检查通过。CPU两个新门控恢复+梯度结果gate_controls_cpu_check.json。


## Heartbeat 2026-10-01 20:14 UTC

新64图全部6分支完成：time vs semantic在t=.2差异CI包含0，在t=.5/.8分别高1.154%/0.498%（semantic较time低1.141%/0.496%），这两档图像CI排除0。semantic seed2028 vs static_seed2028分别-0.311%/-2.644%/-2.550%，三档图像CI低于0。raw unseen_v1，controls_summary.json、seed2028_summary.json。

同权重门控位置干预（不再训练）：shifted将每图1024个patch门控循环移动512位置，保留同图门控值分布、文本/时间和正确的原hypernetwork条件，仅破坏门控与读取位置对应；image_mean只保留每图平均channel gain。相比aligned，shifted三档MSE+0.490%/+2.662%/+7.551%，image_mean+0.369%/+1.541%/+3.182%，全CI高于0。raw gate_placement、summary.json；CPU零门控逐位恢复/非零干预输出变化/同图分布和均值守恒已通过 placement_cpu_check.json。这里只说明已训练函数依赖位置对齐，不证明所有重新训练的全局门控都较弱。未追加训练。

semantic LR1e-3官方25步/CFG4/FP32生成4张已完成，同captions/noise/static512权重对照可视对比见rollout_gate_lr/paired_preview.jpg。目视整体相近，没有明显生成质量胜出；两分支的sign提示都没有箭头，不能称文字/构图改善。保存完整输出及原始时间记录，不以共享GPU约20s/图当独占性能。

下一项 run_native_probe.sh 已部署并启动：16张DIV2K validation原生1024 center crop，所有比较相同官方Qwen空文本，保持权重、噪声、t/FP32/TF32关闭；只读static/semantic/time及seed2028配对，5×16×3=240前向，前两图保存x0=xt+(1-t)*vhat重建。无缩放。没有训练和新checkpoint。本次heartbeat新增更新0，累计受控训练3584+另28资源试跑。相较COCO同时换图像域与空文本，失败不能单独归因于上采样；这是native-photo扩展诊断，不是T2I生成指标/标准超分辨率/SOTA。

资产：服务器/root/autodl-pub/DIV2K/HighResolution/DIV2K_valid_HR.zip已有官方archive，CRC全条目检查，16裁剪按预声明SHA文件顺序与min尺寸>=1024筛选，不依结果选择。开始检查时远端rg不可用，已改用ls发现共享DIV2K；误启动的官方重复下载进程55640已停止，361MB partial待prepare_native_probe核对共享ZIP前512KB后删除，不将这个自造临时文件删除记为额外历史空间贡献。只保留16实际原生crop，空文本cache使用同inode硬链接，无重复权重安装。源https://data.vision.ee.ethz.ch/cvl/DIV2K/，academic research only，引用Agustsson&Timofte CVPRW2017及NTIRE2017报告。不会修改或删除共享数据盘文件。

模型和只读评价源变更分阶段staging安装，等待上一控制器退出，原源码已保留results/evidence；评价source SHA在启动时记录，防止后续文件写入错记。新的native-crop/重建选项不改变默认aligned评价函数。最新私有数据盘约14-15GiB，保护5GB余量。heartbeat prompt已更新为当前阶段，每5分钟继续核查并选择有效下一项。


原生16照片诊断五分支全部完成。semantic vs static seed2027三档MSE -0.642%/-0.778%/+0.571%；仅高噪声档图像CI排除0。seed2028配对 -0.596%/-1.407%/+0.014%，各档CI包含0。time vs static -0.784%/-0.464%/+0.132%（三个图像CI排除0，低噪声退化）。不把原生结果说成通过/失败的单一阈值：均为质量估计与置信区间。raw native_probe_v1/{static,semantic,time,static_seed2028,semantic_seed2028}.json及first_summary/second_seed_summary。重建公式CPUoracle误差2.38e-7只是代数检查，不是质量成绩。全native诊断240前向，新增训练0。

原生crop准备已实际完成，manifest SHA0afb517e93f41cd709fb205cb1043c717d4c8158617f4496fc0a8b950f6209ef；所有选用photo尺寸>=1024，直接裁剪。共享validZIP全条目CRC通过，取消的私有重复partial已删除，结果native_asset_receipt.json。原生域和空文本同时变化，不能把不稳定收益只归因于上采样。

最有判别力的下一项已接续run_native_adapt.sh：从相同static512权重/函数开始，重置AdamW，seed2029，两方static邻域与semantic-gate各256更新；原预测头1e-5，新增门控1e-3。64个DIV2K_train原生1024 crops +31个原COCO_train回放，95条训练长度与3时间档互质，避免每图始终落同一噪声档。新32个DIV2K_val原生crop排除已诊断16图，仅作留出；另16个COCO开发哨兵监测遗忘。选择规则SHA filename/ID在评分前固定。只CRC验证所读取archive条目，不再扫描全部训练ZIP；共享archive不复制/不删除。COCO图片symlink、text cache hardlink避免重复数据和权重。图片和文本与两臂完全一致、FP32/TF32关闭，冻结semantic骨干，原始完整field可训。

本heartbeat新训练预算512（前面的只读干预/原生诊断0训练），完成后累计4096受控更新（当前进行中，不能提前计作已完成），另28资源试跑。训练控制器不自动重复中途失败作业；需要先审计消耗与保留日志。新报告按manifest.dataset分别统计原生留出和COCO开发哨兵，不能合并数据域报出更好均值；native32之后用于决定也需记开发状态。私有数据盘最近15GiB，临时写checkpoints保护6.5GB准入以保留5GB。代码tools/{prepare_native_adapt,run_native_adapt}可复现。heartbeat已更新为这一阶段。


native-adapt已实际进入GPU训练，static分支已完成256更新及48图验证，semantic-gate接续运行。训练/验证manifest SHA f93125fe41cc422d5cac0182083a06a98f416bb7fb5755fdb73ec4a8252317af，副本results/native_adapt_manifest.json；按域汇总工具tools/summarize_domains.py已部署（--manifest可用于本地metadata副本），会核对共享初始loss/time流和manifest SHA，分别报native32与COCO16。最近GPU整体100%，私有数据盘约14GiB，实际累计更新待两分支report完成后记账。


## Heartbeat 2026-10-01 20:56 UTC

mixed native-adapt两方256更新已完成，无失败/重跑。common-start初始loss/time流/manifest SHA检查通过。原生32新留出 semantic vs static t=.2/.5/.8 MSE -0.516%/-1.427%/+0.205%，前两档图像CI低于0，低噪声CI包含0。COCO16开发哨兵 -0.594%/-1.816%/+1.354%，三档CI排除0（低噪声为退化）。完整原始结果native_adapt_v1/{neighborhood,neighborhood_gated}/report.json，按域domain_summary.json。累计已完成4096受控更新，另28资源更新。本heartbeat新增训练0。

真实生成已接续并完成两方各8张（16张总），选之前固定coco_unseen_v1前8 captions，same seeds31000..31007、官方AdamLM25步/order2/timeshift3/CFG4，统一FP32/TF32关闭，无模型缩减。训练后static768 vs semantic768同预算。raw native_generations_v1，预览paired_preview.jpg。目视整体接近，没有明显质量胜出，不能说改善了FID/GenEval。共享GPU约17-20s/图是日志事实，不是独占性能或方法加速。首版接口扩展已CPU --help核对，adapter参数key契约检查加入加载路径。

下一项只读posthoc envelope诊断已启动run_envelope_probe.sh，同一个已训练semantic768，48张已使用验证图，不追加训练：sigma把额外gate乘1-t，给同branch的clean极限zero-gate边界；reliability使用6×6原始RGB观测方差与已知单位Gaussian噪声σ²=(1-t)²。Replicate padding会重复噪声像素，所以把局部加权方差除1-sum(w_j²)修正，而不是误用36个独立样本。rho=clamp(σ²/(corrected_observed_variance+1e-6),0,1)，mask=(1-t)+t*rho；调节的是额外semantic-gate，旧own RGB/位置/邻域路径都保留。同图不同位置可靠性有别，噪声占主导时接近1、细节方差高时趋向σ。这个mask是启发式，不是推导出最优后验。

CPU非零参数端点：sigma与reliability在t=0等于full gate、t=1等于同一branch当前field取消gate，均逐位相同，gate梯度>0。它不保证与另外已训练static或官方pretrained同输出。2048批IID Gaussian MonteCarlo检查replicate噪声方差修正：corner原方差0.893→1.004，center0.976→1.004，另一corner0.757→1.009；因k6 offset-2..3边界重复权重不对称，corrected值都接近1。证据envelope_cpu_check.json/reliability_cpu_check.json。

两个posthoc读取各48×3=144前向（共288），按native32/COCO16分别汇总，基线复用刚完成semantic/static768真实报告。当前32原生留出已被用于诊断选择，后续不能再称最终未触碰测试。绝不把posthoc失败当成所有重新训练noise-aware门控失败。源码在evidence保留，生成控制器完成后才安装staging/envelope，旧模型default none函数保持，本阶段FP32，不宣称BF16/Flash/compile支持与加速。当前GPU整体100%、私有盘约14GiB、无新增下载与重复依赖。


只读envelope首次在加载GPU模型前被混合manifest断言拦住（原evaluate_mse仅接受纯val文件，此次manifest含95train+48val）。没有完成前向或更新；日志已保留为logs/envelope_sigma_failed_mixed_manifest_20261001.log。已修为显式筛选split=val，CPU核对只选32native+16COCO且排除95训练条目；纯val旧协议函数不变。修复代码已同步staging并重启控制器，sigma/reliability评价继续。不能把这个文件选择错误当方法失败。当前生成16张已全部完成，CPU修正不是GPU性能/质量成绩。


envelope两种GPU只读评价修复后完成。按域对已训练full-gate：sigma三档COCO +0.166%/+1.350%/+1.020%，native +0.158%/+1.192%/+0.744%，均退化；reliability COCO +0.007%/+0.078%/-0.135%（低噪声CI含0），native +0.007%/+0.090%/+0.080%（均CI高于0）。因此没有证据直接套减幅可解决已训练函数的低噪声问题；不是对重新训练envelope方法的否定。raw envelope_probe_v1/domain_summary.json与单图逐档保留。

16张生成的裁剪统计：同8提示、noise、官方CFG4，static平均clipped_fraction0.01969/max0.04968，semantic0.05242/max0.17303（Marine提示index4）。裁剪是overshoot线索，不是单独质量标准，不能等同FID/GenEval胜负。

下一项run_cfg_probe.sh只读两个诊断案例：固定index0+由过裁剪选出的index4（明确属于诊断选择，非独立评测），复用已有independent CFG4参考图。新两变体：shared_uncond CFG4，额外gate只取官方CFG batch前半unconditional image-state的Gamma，重复用于两半；原conditional/unconditional semantic编码与条件化NeRF field仍保留。减少额外Gamma_cond-Gamma_uncond直接CFG放大这一项，不保证消除所有引导效应；并比较原independent gate把整体CFG调低为3这一强控制。sample_indices保留原index与seed31000/31004、相同caption，未重新编号噪声。总4张25步新生成，零更新。CPU非零attention/gate检测shared gate在两半相同、unconditional输出逐位保留、conditional输出改变，cfg_gate_cpu_check.json；不是GPU/质量成绩。源等待上一控制器退出后staging/cfg安装，不改旧报告文件。


## 2026-10-01 用户暂停GPU（最高优先级）

用户明确要求“任务计划可以慢慢排，不要启动GPU了，避免竞争”。本实验GPU授权撤回，后续只CPU方案/代码/资产/已完成结果整理，直到用户再次明确授权。heartbeat cgd已PAUSED。检查本实验精确cwd+工具/控制器进程，暂停时没有存活作业（CFG诊断已完成），未停止其他项目。远端暂停回执results/GPU_PAUSED_BY_USER_20261001.json。

本机及远端GPU_PAUSED标记加入；所有run*.sh在启动前拒绝执行，8个GPU Python入口在Torch/GPU导入前检查marker。无自动GPU续接/预热/编译/采样/训练。不会仅因显存空闲或旧heartbeat指令移除标记。保留全部已完成原始结果及失败日志。
