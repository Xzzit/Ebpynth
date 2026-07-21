简体中文 | [English](./README.md)

# Ebpynth

一个用 **纯 Python + PyTorch** 从零重写的 [ebsynth](https://github.com/jamriska/ebsynth)。

原版 ebsynth 是 C++/CUDA 写的基于样例的图像合成工具（example-based image synthesis），核心算法是
PatchMatch。这个项目的目标不是做出一个更快的 ebsynth，而是**把整条流水线——参数解析、图像读写、
guide 合并、以及 PatchMatch 合成引擎本身——全部翻译成读得懂、能单步调试的 PyTorch 张量运算**，
方便自己把这个算法彻底吃透。全项目不编译、不链接任何一行 C++/CUDA 代码；速度比原生内核慢一个
数量级左右（但运算仍然跑在 GPU 上，不会退化成 CPU 速度），这是刻意接受的代价，换来的是：想看哪个
中间结果，直接 `print`/存图看；想改哪一步的逻辑，改的是普通 Python 函数，不用碰 nvcc。

前半段准备工作（CLI 解析、图像加载、guide 合并、金字塔规划）要求和原版语义上完全一致（校验规则、通道折叠逻辑、
默认值、报错文案）；但合成引擎本身**不追求逐字节对齐**——PatchMatch 带随机性，并行传播的执行顺序
也和原版不同，所以两边的输出不会完全一样，评判标准是"看起来是不是同一回事"（视觉等价），而不是
"两个文件 diff 出来是不是空的"。

---

## 项目结构

```
Ebpynth/
│
├── stylize.py                   # 总入口：命令行 -> 解析 -> 加载 -> 合成 -> 落盘，一条龙跑完
│
├── arguments/
│   └── parser.py                # 用 argparse 读 CLI 参数，复刻 -weight 的"级联绑定"语义
│
├── utils/
│   ├── image_io.py               # 图片 <-> 显存里的 (H, W, C) uint8 CUDA 张量，负责读入和落盘
│   ├── guide_merge.py            # 把多路 guide 图的通道拼成两张"超级特征张量"（source 侧/target 侧）
│   └── pyramid_plan.py           # 纯 CPU 标量数学：算金字塔层数 + 每层迭代次数 + 归一化权重向量
│
├── synthesis/                    # 合成引擎本体：PatchMatch 的纯 PyTorch 重写，全项目的核心
│   ├── nnf.py                    # 随机初始化 NNF（最近邻场）——整个引擎唯一要优化的"状态"
│   ├── vote.py                   # 拿着 NNF 生成图像：gather（抄一个像素）/ vote（patch 平均投票）
│   ├── cost.py                   # 算一个 NNF 好不好：加权 patch SSD 代价函数
│   ├── propagate.py              # 让"好答案"在图上并行扩散——跳泛洪传播（jump-flood, 4→2→1）
│   ├── random_search.py          # 让每个像素在自己当前答案附近随机探索，跳出局部最优
│   ├── patchmatch.py             # 单一分辨率下的 match/vote 主循环，把上面几个模块串起来
│   ├── pyramid.py                # 由粗到细金字塔调度 + 可选的 3x3 收尾抛光（extrapass3x3）
│   └── uniformity.py             # 均匀性惩罚：不让某几个 source patch 被无限复用
│
└── examples/video/                # 测试素材：video_frames/(原始帧) + output_frames/(风格化关键帧)
```

---

## 环境准备

- Python 3.9+
- PyTorch（CUDA 版本，本项目所有张量都常驻显存）
- torchvision（图像读写）
- Pillow（PIL，作为 2/4 通道图片写盘时 torchvision 编码器不支持的兜底方案）

本项目开发时用的是 conda 环境 `ezsynth`（`torch.cuda.is_available()` 为 True）。

## 如何运行

最基本的调用形式：

```bash
python stylize.py -style <风格图> -guide <source导引图> <target导引图> [-guide ...] [其他可选参数]
```

真实示例——用第 000 帧的风格化结果作为 style，去风格化第 001 帧（视频逐帧风格化的典型用法）：

```bash
python stylize.py \
  -style examples/video/output_frames/000.png \
  -guide examples/video/video_frames/000.jpg examples/video/video_frames/001.jpg \
  -output result.png -extrapass3x3
```

### 参数详解

| 参数 | 默认值 | 含义 |
|---|---|---|
| `-style <path>` | 必填 | 风格关键帧。**最终输出图像的每一个像素颜色，都只从这张图里"抄"来**，guide 只负责决定"抄哪个位置"。 |
| `-guide <source> <target>` | 至少填一组 | 一组导引图：`source` 要和 style 图像素对齐（比如 style 帧对应的原始视频帧），`target` 要和"你想要的输出"像素对齐（比如下一帧的原始视频帧）。可以重复 `-guide` 传多组（例如同时给"颜色相似度" + "边缘" + "光流"当作多重线索）。 |
| `-weight <value>` | style: `1.0`；guide: `1/组数` | 给"紧跟在它前面声明的那个 `-style` 或 `-guide`"设置权重，级联绑定，不是全局参数。权重越大，这个 guide/style 在代价函数里的话语权越重。 |
| `-output <path>` | `output.png` | 输出图像路径。 |
| `-uniformity <value>` | `3500.0` | 均匀性惩罚力度：数值越大，越不允许某个 source 像素被过度重复引用（避免大片同一块纹理被无脑复制粘贴）。设成 0 等于关闭这一项。 |
| `-patchsize <奇数, ≥3>` | `5` | 匹配用的正方形 patch 边长。越大，匹配越"看整体结构"，越小，越能抠细节但也更容易出噪点。 |
| `-pyramidlevels <int>` | `-1`（自动） | 金字塔层数。`-1` 时按图像尺寸和 patchsize 自动推导最大可用层数；显式指定的值也会被静默钳制到这个上限，不会报错。 |
| `-searchvoteiters <int>` | `6` | 每个金字塔层里，"match 一遍再 vote 一遍"这个大循环要跑几轮。 |
| `-patchmatchiters <int>` | `4` | 每一轮 match 内部，"传播 + 随机搜索"要跑几次。 |
| `-stopthreshold <int>` | `5` | 为了兼容原版命令行而保留，**本项目故意没有实现它的效果**——原因见下面"设计取舍"一节。 |
| `-extrapass3x3` | 关闭 | 金字塔跑完之后，再加一轮 patch_size 强制为 3、且关闭均匀性惩罚的收尾精修，专门找细节。 |

---

## 核心概念：NNF（最近邻场，Nearest-Neighbor Field）

整个合成引擎自始至终只维护和优化**一张表**：NNF。它是一个形状为 `(H_target, W_target, 2)` 的整型
张量，`nnf[y, x] = (sy, sx)` 的意思是——"输出图像里 (y, x) 这个位置，应该长得像 style 图里 (sy, sx)
周围那一小块 patch"。

- **合成 = 优化这张表。** 一开始 NNF 是完全随机瞎猜的（`nnf.py`），画面自然是一团马赛克；PatchMatch
  算法（`propagate.py` + `random_search.py`）不断把 NNF 里"匹配得差"的条目替换成"匹配得更好"的，
  代价函数（`cost.py`）负责量化"好不好"；每优化几轮，就拿当前的 NNF 去"投票"生成一次图像
  （`vote.py`）——这就是我们真正想要的最终输出。
- **有一个不变量贯穿全程：** NNF 里的每个坐标 `(sy, sx)` 都保证落在 `[r, size-1-r]` 范围内
  （`r = patch_size // 2`）。这保证了"以 (sy, sx) 为中心取一个 patch_size×patch_size 的 patch"永远
  不会越界，所以后面所有需要读 patch 的地方（成像、代价函数、传播、随机搜索）都可以放心地做静态
  切片/gather，完全不用写任何逐像素的边界检查分支——这是让整个引擎能"纯张量化、不写 for 像素循环"
  的关键设计。

---

## 端到端 Workflow 详解

下面按 `stylize.py` 里真实的调用顺序，从命令行敲下回车开始，一步一步讲到图片落盘为止。每一步都
说清楚"这一步在干什么、为什么要这么干"。

### 阶段 0：解析命令行参数 —— `arguments/parser.py`

用 `argparse` 把 `-style`、`-guide`、`-patchsize` 等字符串参数，解析成一个 Python 字典 `config`，
同时做安全范围校验（比如 patchsize 必须是不小于 3 的奇数）。这里最不平凡的一点是 `-weight` 的
"级联绑定"语义：`-weight` 本身不带自己的目标，它默认作用于**命令行里紧挨在它前面出现的那一个**
`-style` 或 `-guide`。用自定义的 `argparse.Action`（`StyleAction`/`GuideAction`/`WeightAction`）
搭配一个 `namespace._last_added` 指针来记住"刚刚加的是谁"，实现了这套绑定逻辑。

### 阶段 1：把图像读进显存 —— `utils/image_io.py`

`load_image_to_vram` 用 `torchvision.io.read_image` 从硬盘读图，然后 `.permute(1,2,0)` 把
`(C,H,W)` 转成本项目统一约定的 `(H,W,C)`，再 `.contiguous()`（因为 permute 只改 stride 不搬内存，
不加这一步后面很多张量操作会在错误的内存布局上工作）之后 `.cuda()`，让图像从第一秒起就常驻显卡
VRAM，全程不用来回搬。

还要复刻原版 `evalNumChannels` 的"通道折叠"规则：比如一张灰度图折成 1 通道，灰度+有效透明折成
2 通道，不透明的 RGBA 折成 3 通道——甚至"原生是 2 通道、但 alpha 通道整张图全是不透明"这种边角情况
也要一并识别、折叠成 1 通道，和原版行为逐条对齐。

### 阶段 2：合并所有 guide —— `utils/guide_merge.py`

如果传了多组 `-guide`（比如同时给了颜色相似度 + 边缘检测两条线索），`merge_guides` 用
`torch.cat(..., dim=-1)` 把它们的通道拼接成两张大的"特征张量"：

```python
source_guides = torch.cat([每组 guide 的 source 图], dim=-1)  # (H_style,  W_style,  ΣC)
target_guides = torch.cat([每组 guide 的 target 图], dim=-1)  # (H_target, W_target, ΣC)
```

风格图本身不参与这次拼接，它是独立的一路，只在最后"成像"那一步才会用到。`torch.cat` 自己不会
帮你查的几件事，必须手工校验：

1. 所有 source guide 的分辨率必须等于 style 图；所有 target guide 的分辨率必须彼此一致（这也就
   是最终输出图像的分辨率）。
2. 同一组 guide 的 source 图和 target 图折叠出来的通道数可能不一样（比如刚好一边是纯灰度），原版
   取两边的 `max` 强行对齐，否则拼接后两张大张量的通道会互相错位——必须复刻这一步。
3. 总通道数上限：所有 guide 通道数之和 ≤ 24，style 通道数 ≤ 8（来自原版 `ebsynth.h` 里的常量）。

拼完之后所有张量继续保持 `uint8 / (H, W, C) / CUDA / contiguous` 这套统一约定。

### 阶段 3：规划金字塔 —— `utils/pyramid_plan.py`

这一步全是标量数学，不碰任何显存里的张量，结果留在 CPU 上：

- **金字塔层数：** 如果用户没指定 `-pyramidlevels`（即 `-1`），就不断把最小边长（style 和 target
  四个尺寸里最小的那个）折半，直到它连一个 `2*patchsize+1` 的搜索窗口都放不下为止——能撑住的最大
  折半次数就是层数。用户显式指定的层数如果超过这个上限，也会被静默地钳制下来，不报错。注意这里
  **不用算每一层具体多宽多高**——那是 `synthesis/pyramid.py` 里 `level_size` 自己的活。
- **每层的迭代次数数组：** `search_vote_iters_per_level`、`patch_match_iters_per_level`——原版内核
  API 允许每层用不同的值，但 CLI 语义下每层都是同一个数字，所以就是把用户传的标量复制层数份。
- **归一化权重向量：** style 这边，`style_weight`（缺省 1.0）平均分给每个 style 通道；guide 这边，
  每一组 guide 的权重缺省是 `1/组数`，再把这个权重平均摊到这一组 guide 自己的每个通道上。这两个
  向量会一路传到 `synthesis/cost.py`，是代价函数里每个通道到底"该被多在乎"的依据。

### 阶段 4：把一切交给合成引擎 —— `stylize.py` → `synthesis.run_pyramid`

到这里，`stylize.py` 已经手握 `source_style`、`source_guides`、`target_guides` 三张显存里的张量，
和一份 `plan`（层数、迭代次数、权重）。接下来直接把它们喂给 `synthesis.run_pyramid(...)`——这是
整个引擎唯一对外暴露的入口函数，剩下的复杂度全部封装在 `synthesis/` 包内部。这也是全项目真正的
核心战场，展开讲：

#### 4.1 由粗到细的金字塔调度 —— `synthesis/pyramid.py: run_pyramid`

直接从最粗的一层跑到最细（原始分辨率）的一层，每一层都调用一次单分辨率的 `run_patchmatch`
（见 4.2），层与层之间只交接一件事：NNF。

- `level_size(base_h, base_w, num_levels, level)`：算出某一层具体的宽高。做法是把原始分辨率乘上
  `2^-(num_levels-1-level)` 再取整——**用浮点数缩放再截断**，而不是简单的整数右移，这是为了和原版
  `pyramidLevelSize` 的取整方式逐位对齐（整数右移在某些尺寸下会差 1 个像素）。
- `resize_image`：每一层的 style/guide 图像，都是**直接从最原始的全分辨率图**用双线性插值缩放
  下来的，而不是"上一层缩小的结果再缩小一次"——原版也是这么做的，这样可以避免多次连续降采样带来
  的误差累积。
- **最粗层**：用 `init_random_nnf` 完全随机初始化 NNF——反正这一层分辨率很小，随机瞎猜也很快能被
  PatchMatch 收敛掉。
- **更细的层**：用 `upscale_nnf` 把上一层收敛好的 NNF"放大"过来当初始猜测，而不是重新随机初始化：
  每个新像素找到自己在上一层里的"父像素"（坐标除以 2），直接继承父像素的匹配坐标乘以 2，再加上
  `(x%2, y%2)` 的一点点抖动——抖动的意义是：如果不抖动，同一个粗格子对应的 2×2 四个子像素会全部
  挤在完全相同的起点上，加上抖动后这四个子像素起手就有一点点差异，等于白送下一层的随机搜索一些
  初始多样性，收敛更快。
- 每一层结束后，把这一层收敛好的 (NNF, 当前重建图像) 交给下一层继续精修，直到跑完最细（原始分辨率）
  的一层——它的输出就是最终结果，除非用户开了 `-extrapass3x3`（见 4.7）。

> ⚠️ 原版 `nnfUpscale` 放大后把坐标钳制到 `[patchSize, size-1-patchSize]`，比本项目统一用的
> `[r, size-1-r]`（`r = patchSize/2`）更严格——这其实是原版自己前后不一致的地方（它自己的随机初始化
> 用的就是 `[r, size-1-r]`）。本项目没有照抄这处不一致，而是全程统一用一套边界不变量（原版更严的
> 范围只是这套不变量的一个子集，不会破坏正确性），这是"视觉等价、不追求字节对齐"这条设计原则下的
> 一处刻意选择。

#### 4.2 单一分辨率下的主循环 —— `synthesis/patchmatch.py: run_patchmatch`

这是"由粗到细"金字塔里，每一层实际要重复执行的算法本体，结构完全照抄原版：

```text
成像一次，得到初始的重建图像
（如果开启了 uniformity，从这一层的起始 NNF 建一张 Omega 占用表）
for 第 i 轮 (共 num_search_vote_iters 轮):
    用当前重建图像 vs source，重新算一遍每个像素的 patch 代价
    for 第 j 次 (共 num_patch_match_iters 次):
        传播一次（propagate）
        随机搜索一次（random_search）
    用刚刚优化过的 NNF 重新投票（vote），刷新重建图像
```

外层循环叫 search-vote（先搜索优化 NNF，再投票出图），内层循环叫 patch-match（每轮先传播、再随机
搜索）。理解这个双层循环结构，就理解了这个引擎"怎么从一堆随机数变成一张风格化图片"的全部套路。

#### 4.3 NNF 初始化 —— `synthesis/nnf.py`

`init_random_nnf` 给每个 target 像素随机分配一个 source 坐标，唯一的约束是前面讲的边界不变量
`[r, size-1-r]`。全靠 `torch.randint` 一次性对整张图采样，没有任何逐像素循环。

#### 4.4 成像：拿 NNF 生成图片 —— `synthesis/vote.py`

有两种"把 NNF 变成一张图"的方式：

- `gather_image`：最简单粗暴的版本——每个 target 像素直接抄它 NNF 指向的**那一个** style 像素。
  只是用来在开发早期验证"数据能不能全流程跑通"的训练轮子，最终不会被使用。
- `vote_image`：真正会被使用的版本，均匀平均投票（`EBSYNTH_VOTEMODE_PLAIN`）。核心洞察是：target
  上的每个像素 q，其实被 `patch_size²` 个不同的 patch"共同认领"过——凡是中心在 q 周围 patch_size×
  patch_size 范围内的每个 patch p，都会"声称" q 应该长得像 `source[nnf[p] + (q - p)]`。把所有这些
  声称取平均，就是 q 最终的颜色。代码把这个"谁认领了我"的问题，反过来变成"对于固定的偏移量 d，
  center 恰好是 `q - d` 的那些像素"——对每个 `d` 这就是一次静态切片 + gather，一共
  `patch_size²` 次，完全不需要用 `scatter_add`。

#### 4.5 代价函数：一个 NNF 好不好，怎么打分 —— `synthesis/cost.py`

原版把 style 和 guide 的代价分开算再相加：`Σ styleWeight·(styleDiff)² + Σ guideWeight·(guideDiff)²`。
这其实就是"加权平方差求和"，只要把 style 通道和 guide 通道**拼接成同一个通道轴**、把两边的权重也
拼成同一个权重向量，就能把整个代价函数收拢成**一次加权 SSD**，数学上完全等价，代码却简单得多：

```python
cost = Σ_channel  weight[c] · (target_patch[c] - source_patch[c])²
```

其中 `target_patch` 是"当前重建图像 + target guide"拼在一起的那个 patch，`source_patch` 是
"style 图 + source guide"拼在一起、由 NNF 指向的那个 patch。为了让图像边缘的 patch 也能像内部
patch 一样直接做静态切片（不用为每个像素单独判断"会不会越界"），target 一侧会先用
`F.pad(..., mode="replicate")` 复制填充 `r` 圈；source 一侧则完全不需要填充，因为阶段 4.3 讲的
NNF 边界不变量已经保证它永远不会越界。

> ⚠️ 图像最外面 `patch_size//2` 那一圈像素，代价天生不可能降到 0——复制填充"编"出来的边缘内容，
> 在合法的 source 区域里根本找不到真正对应的东西。这是所有基于 patch 的合成算法共有的边界伪影，
> 不是 bug；代码里的自测特意把"全图平均代价"和"内部区域精确恢复率"分开断言，就是为了不被这层
> 伪影误导。

#### 4.6 传播：让好答案在图上扩散 —— `synthesis/propagate.py`

PatchMatch 的核心洞察之一是："如果我的邻居找到了一个很好的匹配，那我很可能也适合用差不多的匹配"
（因为自然图像里相邻像素的最佳匹配往往也是相邻的）。原版 CUDA 用的是**跳跃步长 4→2→1** 的"跳泛洪"
式传播，而不是简单地"每次只看紧挨着的上下左右 1 格"——这不是本项目自己的简化，而是原版本身的设计：
原版内核也是全像素并行执行的，没有办法像串行扫描线那样"一遍下来信息就能跨越整张图"，用递减的跳跃
步长（先跳得远、再跳得近）能让信息在几轮之内就传遍全图，本项目原样照抄了这个方案。

对每个跳跃距离 `r`（依次是 4、2、1）和四个方向（上、下、左、右各 `r` 格远）：
"如果我 `r` 格远的邻居正在用 source 位置 s 给自己用，那么 `s - 偏移量` 就是我如果也采用邻居那套
对齐方式，应该去尝试的候选"——比较这个候选的代价是不是比我现在的答案更低，低就换。四个方向是
**依次**处理、立刻更新当前最优解的（不是一次性批量比较四个候选取最优），这样每一个方向的比较
（以及如果开了 uniformity，它的占用表记账）都能看到前面几个方向已经生效的结果，和原版一次次调用
`tryNeighborsOffset` 的顺序语义保持一致。

#### 4.7 随机搜索：跳出局部最优 —— `synthesis/random_search.py`

只靠传播，一个好答案只能在"已经存在于图里的某个匹配"之间扩散，永远发现不了全新的、更好的匹配。
随机搜索负责填这个坑：搜索半径从 1 开始，不断翻倍（1、2、4、8、……）直到达到 source 图最大边长的
一半为止；每一轮，每个像素都在**自己当前答案的周围** `±半径` 范围内，随机撒一个候选点，代价更低
就采纳。"半径从小到大翻倍"兼顾了"在当前答案附近精细微调"和"有机会跳到很远的地方发现全新匹配"。

#### 4.8 均匀性惩罚 —— `synthesis/uniformity.py`

如果放任不管，PatchMatch 很容易让一大片 target 区域都去抄 source 里同一小块特别"讨喜"的纹理，
造成视觉上明显的重复感。`-uniformity` 参数就是用来压制这种现象的：

- 维护一张和 source 图同样大小的 **Omega 占用表**：source 上每个像素，当前被多少个 target patch
  引用着。用 `scatter_add_` 实现——这是和代价函数（4.5）刚好相反的方向：代价函数问的是"这个 target
  patch 该从 source 的哪里去读"（gather），而占用表问的是"这个 NNF 一共往 source 的哪些位置写入了
  引用"（scatter），所以必须用 scatter_add，不能像 vote_image 那样用固定切片。
- 同时算出一个"理想占用值"：如果 target 对 source 的引用是完全均匀分布的，每个 source patch
  平均应该承担多少次引用——这是一个用面积比推出来的理论基准值。
- 传播和随机搜索在决定"要不要采纳一个新候选"时，不再只比较原始代价，而是比较
  `代价 + uniformity_weight × (当前占用 / 理想占用)` 这个综合分数——占用越超标，这个候选就越"不划算"，
  哪怕它原始代价更低也可能被拒绝。一旦某个像素的匹配真的换了，占用表要跟着把"占用名额"从旧位置
  搬到新位置（同样是 scatter_add_，处理多个像素同时改动、又可能落在同一个 source 像素上的情况）。
- **Omega 占用表的生命周期是"一整个金字塔层"**：同一层内，无论 search-vote 循环跑了多少轮、
  patch-match 内层跑了多少次，都共用同一张持续累积的占用表；只有开始跑下一个金字塔层时才会重新
  统计一遍——这和原版 Omega 的生命周期完全一致。

> ⚠️ **关于 `-stopthreshold` 的设计取舍：** 原版这个参数驱动的是一套"跳过已经收敛、不用再算的
> 像素"机制（配合 mask/dilate），这本质上是 CUDA 每线程独立执行模型下的一种**性能优化**：省掉对
> 已经不会再变的像素重复计算。但在本项目全向量化的写法里，张量运算是整张图一起算的，没有真正意义
> 上的"逐像素分支"可以省——重新计算一个已经最优的像素完全无害（传播/随机搜索只在严格更优时才会
> 替换掉旧答案，最优像素重算一遍原地不动），跳过它不会让张量运算变快。所以这里是**故意**没有实现
> 这个机制，而不是遗漏，`stop_threshold` 只是为了兼容原版命令行才被解析和保留下来的。

#### 4.9 extrapass3x3：终局抛光 —— `synthesis/pyramid.py: run_pyramid` 内

`-extrapass3x3` 开启时，金字塔正常跑完最细一层之后，**不重新初始化任何东西**，直接拿最细层收敛好
的 `(nnf, target_style)`，在同一分辨率上再跑一轮 `run_patchmatch`——只是这次强制把 `patch_size`
改成 3（比用户设置的 patchsize 更细，专门用来抠小细节）、并强制把 `uniformity_weight` 归零（3×3
这么小的 patch，谈"是否被过度复用"已经没有意义）。原版的做法是把外层 for 循环的层数计数器减一，
让代码重新走一遍最细层的循环体来实现这次"回头再抛一层光"；本项目直接对最细层自己的
`(nnf, target_style)` 再调用一次 `run_patchmatch`，效果等价，可读性更好。

### 阶段 5：落盘 —— `utils/image_io.py: save_image_from_vram`

合成引擎返回的 `output_image`（`uint8`、`(H, W, C)`，仍然待在 GPU 上）先 `.permute` 回
`torchvision` 要的 `(C, H, W)` 再 `.cpu()`，用 `torchvision.io.write_png` 写盘；如果通道数是 2 或 4
（带 alpha），因为 `torchvision` 的编码器不支持，会自动改用 PIL 写成 LA/RGBA 格式兜底。

> ⚠️ 不要直接用 `torchvision.utils.save_image`——它期望的是 `[0,1]` 范围的浮点 `(C,H,W)` 张量，
> 直接喂 `uint8` 张量进去会得到一张全白的废图。

---

## 设计取舍小结

- **放弃了 CUDA 桥接方案。** 最早的计划是保留原版 CUDA 内核，用 pybind11 桥接调用（"D2D 改造"路线）；
  后来改成了现在这套纯 PyTorch 重写方案，图的是零编译、零裸指针、可单步调试、随时能把中间结果存图
  查看。代价是速度慢了一个数量级左右（但仍然是 GPU 计算，不是退化成 CPU）。
- **`stylize.py` 直接调用 `synthesis.run_pyramid(...)`。** 早期版本多包了一层 `ebsynth_run(config, plan)`
  函数，后来发现它纯粹是转发调用、没有自己的逻辑，判定为多余而删除了——`config`/`plan` 字典现在
  直接在 `stylize.py` 里解包传给 `run_pyramid`。
- **合成引擎是纯函数式风格。** 每一步都返回一个新张量，不做原地写入——这和原版 CUDA "直接把结果
  写回同一块显存缓冲区"的方式不同，但对外行为完全等价。
- **`-stopthreshold` 有意不实现**（见 4.8 结尾），**`nnfUpscale` 的边界钳制刻意不追随原版的不一致
  行为**（见 4.1 结尾）——这两处都在代码注释和上文正文里做了明确标注，不是疏漏。
- **输出不会和原版逐字节相同。** PatchMatch 带随机性，并行传播的执行顺序也和原版不同；判断"对不对"
  的标准是视觉上是否等价，而不是 diff 是否为空。

---

## 一句话总结

命令行参数进来 → 图像和 guide 全部搬进显存、按通道拼接成两张"超级特征张量" → 根据图像尺寸规划好
金字塔层数和每层的权重/迭代次数 → 从最粗的一层开始，每层都用 PatchMatch（传播 + 随机搜索，靠加权
patch 代价函数打分，可选叠加均匀性惩罚）不断优化 NNF 这张"每个输出像素该抄 source 哪里"的表，再
拿它去投票生成图像，然后把 NNF 放大交给下一层继续精修 → 最细层跑完后可选再来一轮 3×3 小 patch 的
抛光 → 把最终图像从显存写回硬盘。全程没有一行 C++/CUDA，也没有任何一步逐像素的 Python for 循环——
每一步都是一次张量运算。
