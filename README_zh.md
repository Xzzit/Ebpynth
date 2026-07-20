简体中文 | [English](./README.md)

Ebpynth/
│
├── stylize.py                # 🕹️ Python 总控制器 —— ✅ 全线贯通，可直接当 CLI 使用
│
├── arguments/
│   └── parser.py             # ✅ 任务 A：argparse CLI 解析 + 校验
│
├── utils/
│   ├── image_io.py           # ✅ 任务 B + 任务 K：图像 ⇄ contiguous CUDA (H, W, C) uint8 张量
│   ├── guide_merge.py        # ✅ 任务 C：torch.cat 通道合并 + 分辨率/通道对齐/上限检查
│   └── pyramid_plan.py       # ✅ 任务 D：金字塔层数 + 每层控制数组 + 权重向量
│
├── synthesis/                # 🔥 纯 PyTorch 合成引擎 —— 第二阶段任务 F~J 在这里逐步生长
│   ├── nnf.py                # ✅ 任务 F ①：init_random_nnf 随机初始化最近邻场
│   ├── vote.py               # ✅ 任务 F ②③：gather_image 简化成像 + vote_image 真投票
│   ├── cost.py                # ✅ 任务 G ①：patch_cost 加权 SSD 代价函数（style+guide 合并通道）
│   ├── propagate.py           # ✅ 任务 G ②：propagate 跳泛洪传播（jump 4→2→1）
│   ├── random_search.py       # ✅ 任务 G ③：random_search 指数增长半径随机搜索
│   ├── patchmatch.py          # ✅ 任务 G ④ + 任务 I 接入点：run_patchmatch 单层主循环
│   ├── pyramid.py             # ✅ 任务 H + 任务 J：run_pyramid 由粗到细 + extrapass3x3 收尾
│   └── uniformity.py          # ✅ 任务 I：Uniformity 均匀性占用惩罚（Omega 占用表）
│
├── ebpynth/                  # 📚 已降级为纯参考资料（不再编译！原生扩展路线已废弃）
│   ├── setup.py              # (废弃) 原 AOT 编译脚本，仅作历史保留
│   ├── include/
│   │   └── ebsynth.h         # 常量出处：EBSYNTH_MAX_STYLE_CHANNELS=8, MAX_GUIDE_CHANNELS=24
│   └── src/                  # 原版 C++/CUDA 源码未修改副本，对拍算法语义时查阅
│       ├── ebsynth.cpp           # 原版 CLI main()：任务 A~E 的语义基准
│       └── ebsynth_cuda.cu       # 原版 PatchMatch 内核：任务 F~J 重写时的算法教科书
│
└── examples/video/           # 测试素材：video_frames/ (原始帧) + output_frames/ (风格化结果)


🐍 第一阶段：纯 Python 统治区（备菜——把所有食材摆进显存盘子）
每个任务做成独立模块（放在 arguments/、utils/ 等包里），各自带 __main__ 沙盒自测，
最后统一在 stylize.py 里拼装调用：

任务 A：参数接收与海关安检（替代原 C++ 的 tryToParseArg 逻辑）✅ 已完成 → arguments/parser.py
用 Python 的 argparse 读入用户输入的 -style、-guide、-patchsize 等参数，做安全范围检查
（比如 patchsize 必须是不小于 3 的奇数）。注意复刻原版 -weight 的"级联绑定"语义：一个裸的
-weight 作用于它紧前面声明的那个 -style 或 -guide。

任务 B：图像秒开直通显存（平替原 C++ 的 tryLoad 逻辑）✅ 已完成 → utils/image_io.py
利用 torchvision.io.read_image 把硬盘里的图片读进来，直接执行 .cuda()，让风格图、导引图在
第一秒钟就直接躺进显卡的 VRAM 里。两个实战里踩过的坑，必须守住：
  ① .permute(1,2,0) 只改 stride 不搬内存，必须紧跟 .contiguous() 再 .cuda()；
  ② 要复刻原版 evalNumChannels 的通道折叠：不透明灰度 → 1 通道、灰度+有效透明 → 2 通道、
     不透明 RGBA → 3 通道等，连"原生 2 通道但 alpha 全不透明也要折叠成 1 通道"这种边角
     情况也要对齐。

任务 C：特征拧麻花合并（一行 torch.cat 干掉原 C++ 几百行的多层大循环）✅ 已完成 → utils/guide_merge.py
利用 torch.cat(..., dim=-1) 把多路导引图的通道合并成"超级特征张量"。注意产出是【两个】
张量，不是一个：
  source_guides = torch.cat([每路 guide 的 source 图], dim=-1)  # (H_style, W_style, ΣC)
  target_guides = torch.cat([每路 guide 的 target 图], dim=-1)  # (H_target, W_target, ΣC)
风格图本身不参与拼接，它是独立的一路。三个 torch.cat 替你查不了的检查要自己做：
  ① 分辨率校验：所有 source guide 必须等于风格图分辨率；所有 target guide 彼此一致
     （target 的分辨率就是最终输出的分辨率）；
  ② 同一路 guide 的 source/target 折叠出的通道数可能不同（比如一边恰好是纯灰度），原版取
     std::max 后把两边对齐到相同通道数，拼接前必须复刻，否则两个大张量的通道会错位；
  ③ 通道总数上限：ΣC ≤ EBSYNTH_MAX_GUIDE_CHANNELS (24)，风格通道 ≤ MAX_STYLE_CHANNELS (8)。
拼接完成后各张量保持 uint8 / (H, W, C) / CUDA / contiguous 的统一约定。

任务 D：金字塔层数控制规划（平替原 C++ 尾部的数学计算）✅ 已完成 → utils/pyramid_plan.py
在 Python 里用简单的数学公式完成两件事（纯标量计算，结果留在 CPU 侧，不进显存）：
  ① 层数：若用户没指定 -pyramidlevels（即 -1），自动算最大可用层数——不断折半，直到最小边
     （style 和 target 四个尺寸里最小者）放不下一个 (2*patchsize+1) 为止；用户显式指定的层数
     也会被静默钳制到这个上限。注意每层的具体高宽不用算——合成引擎拿到层数后自己推导；
  ② 控制数组：构建每层的 search_vote_iters、patch_match_iters、stop_threshold 数组（CLI 语义
     下就是同一标量复制 L 份），以及归一化后的权重向量——styleWeights[i] = styleWeight / 风格总
     通道数（styleWeight 缺省 1.0）；每路 guide 权重缺省 1.0/guide 路数，再除以该路的通道数摊
     到每个通道上。

任务 E：开辟输出画布 ✅ 已完成 → stylize.py
在显存里原地创建一个空张量准备接招：
  output_image = torch.zeros((H_target, W_target, C_style), dtype=torch.uint8, device="cuda")
注意形状用 target guide 的分辨率 × 风格图的通道数，且天生 contiguous，不要再 permute 它。


⚡ 第二阶段：纯 PyTorch 合成引擎（重写 PatchMatch 本体——当前战场）

┌──────┬──────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┐
│ Task │                                 Content                              │                        Goal                          │
├──────┼──────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ F    │ 随机初始化 NNF + 投票成像（先 gather 简化版，再升级真投票）              │ 输出一张 style 像素乱拼的"花屏"——数据流闭环             │
├──────┼──────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ G    │ PatchMatch 迭代：加权代价函数 + 并行传播 + 随机搜索，match/vote 交替     │ 单层合成出可辨认的风格化画面                           │
├──────┼──────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ H    │ 金字塔由粗到细：逐层缩小、NNF 放大 ×2 传递                              │ 大结构稳定，质量接近原版                               │
├──────┼──────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ I    │ uniformity 一致性项（omega 图）+ stopthreshold 提前收工                │ -uniformity/-stopthreshold 参数真正生效               │
├──────┼──────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ J    │ extrapass3x3 终局抛光                                                 │ ebsynth_run() 完全体，拆掉 backend = None 关卡        │
├──────┼──────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┤
│ K    │ 导出落盘                                                              │ ✅ 已完成（save_image_from_vram，stylize.py 已接好）  │
└──────┴──────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┘

【路线变更说明】原计划保留原版 CUDA 内核、用 pybind 桥接调用（所谓 D2D 改造路线），现已
放弃，改用纯 PyTorch 张量运算重写整个合成算法。代价与收益：速度慢一个数量级左右（PyTorch
算子本身仍跑在 GPU 上，不会退化成 CPU 速度），换来零编译、零裸指针、可单步调试、可随时把
中间结果存成图看。ebpynth/ 目录从此降级为纯参考资料。另外注意：PatchMatch 带随机性，且并
行传播顺序与原版不同，输出不会和原版逐字节一致——对拍标准是"视觉等价"，不是"字节相等"。

算法核心是维护一张 NNF（最近邻场，Nearest-Neighbor Field）：形状 (H_target, W_target, 2)
的整型张量，给 target 每个像素记录"它该抄 source 的哪个坐标"。整个第二阶段就是想方设法把
这张表从瞎蒙优化到收敛。新代码统一放在 synthesis/ 包里，按任务逐模块生长，最后汇成一个
ebsynth_run() 函数，拆掉 stylize.py 里 backend = None 的关卡。

任务 F：随机瞎蒙 + 投票成像（先闭环，出一张"噪声马赛克"）✅ 已完成 → synthesis/nnf.py + vote.py
随机初始化 NNF（坐标限制在合法 patch 范围内），然后实现"成像"：先用最简单的逐像素 gather
（每个 target 像素直接抄 NNF 指向的那个 style 像素）跑通全流程，再升级成真正的投票 vote——
每个 target 像素被 patchsize² 个 patch 覆盖，把这些 patch 各自指向的 style 像素平均起来
（scatter_add / F.fold）。里程碑：输出一张由 style 像素乱拼的花屏，证明数据流全线贯通。

任务 G：PatchMatch 迭代（花屏开始收敛成画面）✅ 已完成 → synthesis/cost.py + propagate.py + random_search.py + patchmatch.py
先实现加权 patch 代价函数（数学上 style 项和 guide 项就是把通道和权重拼在一起算的同一个加权
SSD，不用分两次算）：
  cost = Σ (style_weights ⊕ guide_weights)·((source_style ⊕ source_guide) patch
                                            − (当前合成结果 ⊕ target_guide) patch)²
（任务 D 那两个权重向量的用武之地。）然后每轮迭代做两件事：
  传播——原版实际用的不是简单的"上下左右挪 1 格"，而是跳跃步长 4→2→1 的跳泛洪：所有像素并行
        "抄跳 r 格远的邻居的答案"，谁代价低听谁的。这不是简化，是照抄原版本身的设计——原版
        CUDA 内核也是全并行执行，同样面临"没有串行扫描线、如何让信息跨越整张图"的问题，
        4→2→1 跳跃正是它给出的答案，我们直接复用；
  随机搜索——半径从 1 倍增到源图一半，每轮在当前答案周围撒一个随机候选，跳出局部最优。
match 与 vote 交替执行 searchvoteiters 轮，每轮 patchmatchiters 次 传播+随机搜索。
里程碑：单层（无金字塔）、全分辨率下跑 3 轮外层×3 轮内层，1 秒内从噪声收敛出可辨认的画面
（examples/video/temp/task_g_result.png）。
⚠️ 图像边界 patch_size//2 那圈像素天生达不到零误差（复制填充造出的边缘内容在合法 source
区域里根本找不到对应），这是所有基于 patch 的算法的固有边界伪影，不是 bug；沙盒测试里
"全图平均代价"和"内部区域精确恢复率"分开断言，就是为了不被这层伪影误导。

任务 H：金字塔由粗到细（大结构不再错位）✅ 已完成 → synthesis/pyramid.py
把任务 D 算好的层数用起来：每一层都用 F.interpolate 从【原始全分辨率】图像直接双线性缩放
到该层尺寸（不是逐层从上一层再缩小——原版也是每层都从最细层重新采样，避免误差累积）。
最粗层用随机 NNF 起步；之后每层收敛完，把 NNF 坐标 ×2 再加上 (x%2, y%2) 的抖动交给下一层
当初始值——抖动是为了让放大后同一个粗格子对应的 2×2 个子像素不会全部挤在同一个起点上，
白送下一层的搜索一点初始多样性。层与层之间复用任务 G 的 run_patchmatch 原样跑一遍，只是
分辨率和迭代次数（来自 plan_pyramid 的 per-level 数组）不同。
里程碑：540×960、6 层金字塔、默认超参数，10 秒内跑完，画面结构（猞猁的脸、眼睛、耳朵、
胡须）清晰连贯，背景呈现出连续笔触感，相比任务 G 的单层版有质的提升
（examples/video/temp/task_h_result.png）。
⚠️ 原版 nnfUpscale 放大后钳制到 [patchSize, size-1-patchSize]，比我们自己"[r, size-1-r]"
（r=patchSize/2）的不变量更严——和它自己的随机初始化用的边界都对不上，是原版自身不一致
的地方。这里没有照抄，而是延续本项目统一的 [r, size-1-r] 不变量（更严的原版范围只是它的
子集，不影响正确性，纯粹是原版的小瑕疵，不值得为了"字节对齐"引入不一致）。

任务 I：uniformity 一致性项 ✅ 已完成 → synthesis/uniformity.py
用 scatter_add 维护一张 Omega 占用表（source 每个像素当前被多少 target patch 引用），把
"过度使用"计入代价——这就是 -uniformity 参数（缺省 3500）的含义。核心是把原版 tryPatch 的
"cost + lambda·占用率"决策公式搬过来：match/random search 每次接受一个候选时，不再只比较
纯代价，而是比较"代价 + 权重×占用率"的综合分，接受后再用 scatter_add 把占用表从旧位置搬到
新位置。Omega 的生命周期是"整个金字塔层"——同一层内所有 vote/patchmatch 迭代共用同一张、
持续累积的占用表，换层才重新统计。
里程碑：真实图片、默认超参数（uniformity_weight=3500）跑完整金字塔，画面细节（眼睛、胡须、
毛发纹理、前景草叶）比任务 H 更锐利、更少重复感（examples/video/temp/task_i_result.png）；
小场景定量验证：source 明显小于 target 时（16×16 vs 40×40 无关噪声，逼出鸽笼效应式的强制
复用），加权重后 Omega 的方差从 ~40000 降到 ~27000、峰值使用次数从 884 降到 723，而总使用量
（均值）分毫不变——证明这一项只是把"用得太狠"的地方摊薄，没有凭空增删使用总量。
⚠️ 原版 stopthreshold 对应的 mask/dilate 跳过机制，本项目【有意不实现】：那是 CUDA 每线程
独立执行模型下的性能优化（跳过重新计算已收敛像素），在我们全向量化的写法里，张量运算不会
因为"跳过某些像素"而变快（没有真正的逐像素分支），而重新评估一个已经最优的像素也完全无害
（传播/随机搜索只在严格更优时才替换，最优像素重新算一遍还是原地不动）——照抄这部分只会
增加代码复杂度、不会带来任何实际收益，故不迁移。

任务 J：extrapass3x3 终局抛光 ✅ 已完成 → synthesis/pyramid.py (run_pyramid 内)
复刻原版的可选收尾（-extrapass3x3 开启才生效）：金字塔正常跑完后，如果开启这个开关，就用
最细层的收敛结果（nnf、target_style 都不重新初始化）在同一分辨率上再跑一轮 run_patchmatch——
只是这次 patch_size 强制改成 3（比用户设的patchsize更细，专门找小细节）、uniformity_weight
强制归零（3×3 的小 patch 拼均匀性没有意义）。原版是把同一个 for 循环的 level 计数器减一、
再走一遍最细层的循环体来实现这个"回头再抛光一次"，我们直接对最细层的 (nnf, target_style)
再调一次 run_patchmatch，效果等价、可读性更好。
里程碑：开启 vs 不开启 extrapass3x3 分别跑一遍真实图片，用拉普拉斯方差（一种常见的"锐度"
量化指标）验证细节确实变锐利了，而不只是主观感觉（examples/video/temp/task_j_extrapass.png
vs task_j_noextrapass.png）。⚠️ 调试时发现这个对比测试如果不固定随机种子，两次跑其实是两次
完全独立的随机合成（每层随机 NNF、每次随机搜索的抽样都不同），extrapass3x3 带来的锐度提升
会被运行间的自然波动淹没——已加 torch.manual_seed 让两次跑走完全相同的粗到细轨迹、只在
extra pass 这一步分叉，测试才稳定。
🕹️ stylize.py 直接调用 synthesis.run_pyramid(...)（原本多加了一层 ebsynth_run(config, plan)
包装，后来觉得太单薄、纯粹是转发调用，已经去掉，config/plan 字典直接在 stylize.py 里解包
传给 run_pyramid），backend = None 的关卡正式拆除；Task E 预分配的
output_image 从"预留写入缓冲区"变成了纯粹的形状校验（assert）——本项目引擎是纯函数式风格
（每一步都返回新张量，不做原地写入），这一点和 CUDA 版"直接把结果写回同一块显存"的方式不同，
但对外行为完全等价。至此全项目命令行首次端到端跑通：
  python stylize.py -style <风格图> -guide <src> <tgt> -output <输出路径> [-extrapass3x3]


🐍 第三阶段：重回 Python 统治区（算完收网）

任务 K：导出落盘 ✅ 已完成 → utils/image_io.py (save_image_from_vram)
合成完的 output_image（uint8 / (H, W, C)、依然躺在 GPU 上）用 torchvision.io.write_png 落盘
（permute 回 CHW + .cpu()；2/4 通道 torchvision 编码器不支持，自动走 PIL 兜底写 LA/RGBA）。
输出路径来自 -output 参数。stylize.py 里的调用已就位，合成引擎一接上即自动生效。
⚠️ 不要用 torchvision.utils.save_image——它期望 [0,1] 浮点 CHW 张量，喂 uint8 会全白报废。


导师大白话小结
现在整个项目是 100% Python，A~K 全部完成，stylize.py 可以当成 ebsynth 原版 CLI 的平替直接
使用了。第一阶段 A~E 备菜，第二阶段 F~J 用 PyTorch 张量运算重写了整个 PatchMatch 本体
（随机 NNF → 加权代价+传播+随机搜索 → 金字塔粗到细 → 均匀性约束 → extrapass3x3 收尾），
第三阶段 K 打包上桌。你没有写任何一行 C++/CUDA——原版内核 ebsynth_cuda.cu 全程只当算法
教科书查阅，从未参与编译；ebpynth/ 目录里的原生扩展骨架也就此彻底作废,可以随时删掉。
