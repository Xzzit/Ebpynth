简体中文 | [English](./README.md)

# Ebpynth

一个用 **纯 Python + PyTorch** 从零重写的 [ebsynth](https://github.com/jamriska/ebsynth)。

原版 ebsynth 是基于 C++/CUDA 实现的 Example-based Image Synthesis 工具，底层核心是 PatchMatch 算法。
这个项目并不打算做一个更快的 ebsynth，而是把从 CLI 解析、图像读写、
Guide 合并到 PatchMatch 合成 Engine 的整条 Pipeline，全部重写成易读、可单步调试的纯 PyTorch Tensor 运算，方便把算法彻底吃透。

全项目不需要编译或链接任何 C++/CUDA 代码，虽然运行速度比原生 CUDA Kernel 慢一个数量级，
但换来的是极高的透明度：想看哪个步骤的中间结果，随时 print 或存图观察；想改哪里的逻辑，改的都是普通 Python 代码，完全不用碰 nvcc。

---

## 项目结构

```
Ebpynth/
│
├── stylize.py                   # 总入口：从解析参数到保存图片的全部主流程按顺序写在这里，包括金字塔循环和 match/vote 循环
│
├── arguments/
│   └── parser.py                # 用 argparse 读 CLI 参数，权重作为 -style/-guide 的可选尾随参数
│
├── utils/
│   ├── image_io.py               # 图片 <-> 显存里的 (H, W, C) uint8 CUDA 张量，负责读写
│   ├── guide_merge.py            # 把多路 guide 图的通道拼成两张"超级特征张量"（source 侧/target 侧）
│   └── pyramid_plan.py           # 纯 CPU 标量数学：算金字塔层数 + 每层迭代次数 + 归一化权重向量
│
├── synthesis/                    # PatchMatch 的积木函数，全部由 stylize.py 按顺序直接调用
│   ├── nnf.py                    # 随机初始化 NNF（最近邻场）——整个引擎唯一要优化的"状态"
│   ├── vote.py                   # 拿着 NNF 生成图像：gather（抄一个像素）/ vote（patch 平均投票）
│   ├── cost.py                   # 算一个 NNF 好不好：加权 patch SSD 代价函数
│   ├── propagate.py              # 让"好答案"在图上并行扩散——跳泛洪传播（jump-flood, 4→2→1）
│   ├── random_search.py          # 让每个像素在自己当前答案附近随机探索，跳出局部最优
│   ├── pyramid.py                # 金字塔数学：每层尺寸 / 图像缩放 / NNF 放大
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
python stylize.py -style <风格图> [权重] -guide <source导引图> <target导引图> [权重] [-guide ...] [其他可选参数]
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
| `-style <path> [weight]` | path 必填，weight 默认 `1.0` | 风格关键帧，输出图像的颜色全部来自这张图；guide 只决定"抄哪个位置"。weight 是可选的尾随参数，值越大在代价函数中占比越高。 |
| `-guide <source> <target> [weight]` | 至少一组，weight 默认 `1/组数` | `source` 与 style 图像素对齐，`target` 与目标输出像素对齐。weight 是可选的尾随参数。可重复传多组（如颜色 + 边缘 + 光流）。 |
| `-output <path>` | `output.png` | 输出图像路径。 |
| `-uniformity <value>` | `3500.0` | 均匀性惩罚力度，越大越抑制某个 source patch 被过度复用；0 为关闭。 |
| `-patchsize <奇数, ≥3>` | `5` | 匹配用的正方形 patch 边长，越大越偏结构、越小越偏细节但易出噪点。 |
| `-pyramidlevels <int>` | `-1`（自动） | 金字塔层数，`-1` 按图像尺寸和 patchsize 自动推导上限；显式指定值会被静默钳制到该上限。 |
| `-searchvoteiters <int>` | `6` | 每个金字塔层内 match/vote 大循环的轮数。 |
| `-patchmatchiters <int>` | `4` | 每轮 match 内传播+随机搜索的次数。 |
| `-stopthreshold <int>` | `5` | 仅为兼容原版 CLI 保留，本项目未实现其效果，见下文"设计取舍"。 |
| `-extrapass3x3` | 关闭 | 金字塔结束后加一轮 patch_size=3、uniformity=0 的收尾精修。 |

---

## 核心概念：NNF（最近邻场，Nearest-Neighbor Field）

引擎只维护和优化一张表：NNF，形状 `(H_target, W_target, 2)` 的整型张量，`nnf[y, x] = (sy, sx)`
表示输出图 (y, x) 处应该长得像 style 图 (sy, sx) 周围的 patch。合成 = 优化这张表：随机初始化
（`nnf.py`）→ PatchMatch 用传播 + 随机搜索替换较差的匹配（`propagate.py` / `random_search.py`）→
代价函数量化好坏（`cost.py`）→ 每隔几轮用当前 NNF 投票生成图像（`vote.py`），最终这张图就是输出。

不变量：坐标恒落在 `[r, size-1-r]`（`r = patch_size // 2`），保证以其为中心的 patch 不越界，所以
所有 patch 读取都能用静态切片/gather，不需要逐像素边界检查。

---

## 端到端流程详解

下面按 `stylize.py` 的调用顺序进行叙述。

### 阶段 0：解析命令行参数 —— `arguments/parser.py`

`argparse` 把 `-style`、`-guide`、`-patchsize` 等参数解析成字典 `config`，并做范围校验（如 patchsize
必须是 ≥3 的奇数）。`-style`/`-guide` 都用 `nargs="+"` 接收变长 token：`-style` 是 `<path> [weight]`，
`-guide` 是 `<source> <target> [weight]`——weight 是否传入靠 token 数量判断（2 个还是 1 个/3 个还是 2 个），
省略时用 `-1.0` 作为"未设置"哨兵值，交给 `plan_pyramid` 解析成真正的默认值（style 1.0，guide 1/组数）。

### 阶段 1：把图像读进显存 —— `utils/image_io.py`

`load_image_to_vram`：`torchvision.io.read_image` 读图 → `.permute(1,2,0)` 转成 `(H,W,C)` →
`.contiguous()`（permute 只改 stride 不搬内存，必须紧跟这一步）→ `.cuda()`。

复刻原版 `evalNumChannels` 的通道折叠规则：灰度→1 通道，灰度+有效透明→2 通道，不透明 RGBA→3 通道，
含"原生 2 通道但 alpha 全不透明"这类边界情况。

### 阶段 2：合并所有 guide —— `utils/guide_merge.py`

`merge_guides` 用 `torch.cat(dim=-1)` 把多组 guide 的通道拼成两张特征张量：

```python
source_guides = torch.cat([每组 guide 的 source 图], dim=-1)  # (H_style,  W_style,  ΣC)
target_guides = torch.cat([每组 guide 的 target 图], dim=-1)  # (H_target, W_target, ΣC)
```

style 图不参与拼接，独立一路，留到成像阶段使用。`torch.cat` 之外需手工校验：

1. source guide 分辨率须等于 style 图；target guide 分辨率须彼此一致（即输出分辨率）。
2. 同组 source/target 折叠出的通道数可能不同，取 `max` 对齐，否则拼接后通道错位。
3. 总通道数上限：guide ΣC ≤ 24，style ≤ 8（原版 `ebsynth.h` 常量）。

拼接后所有张量仍保持 `uint8 / (H, W, C) / CUDA / contiguous`。

### 阶段 3：规划金字塔 —— `utils/pyramid_plan.py`

纯 CPU 标量计算，不碰张量：

- **层数：** 若 `-pyramidlevels` 为 `-1`，把最小边长（style/target 四个尺寸取最小）不断折半，直到放不下
  `2*patchsize+1` 的窗口为止，能撑住的折半次数即层数；显式指定的层数会被静默钳制到该上限。每层的具体
  宽高由 `synthesis/pyramid.py` 的 `level_size` 计算，这里不算。
- **每层迭代次数：** `search_vote_iters_per_level`、`patch_match_iters_per_level`——CLI 语义下每层相同，
  标量复制层数份。
- **权重向量：** `style_weight`（缺省 1.0）均分给每个 style 通道；每组 guide 权重缺省 `1/组数`，再摊到
  该组自己的通道上。传给 `synthesis/cost.py` 决定每个通道在代价函数里的比重。

### 阶段 4：由粗到细逐层合成 —— `stylize.py`（主循环内联）

金字塔循环和 match/vote 循环直接写在 `stylize.py` 里，对应代码中的 4a/4b/4c 注释；`synthesis/`
只提供每一步调用的积木函数。

#### 4.1 金字塔调度（4a + 4b）

从最粗层跑到最细（原始分辨率）层，层间只传递 NNF：

- 4a 缩放：`level_size` 算本层宽高（原始分辨率乘 `2^-(num_levels-1-level)` 后取整，浮点缩放再截断，
  对齐原版 `pyramidLevelSize`，整数右移会有 1 像素误差）；`resize_image` 每层都从原始全分辨率图
  双线性缩放，不逐层级联缩小，避免误差累积。
- 4b NNF 初值：最粗层 `init_random_nnf` 随机初始化；更细层 `upscale_nnf` 把上一层 NNF 坐标 ×2 并加
  `(x%2, y%2)` 抖动（避免同一粗格子对应的 2×2 子像素挤在同一起点，给下一层搜索一点初始多样性）。
- 最细层输出即最终结果，除非开启 `-extrapass3x3`（见 4.9）。

> ⚠️ 原版 `nnfUpscale` 钳制到 `[patchSize, size-1-patchSize]`，比本项目统一用的 `[r, size-1-r]`
> 更严，且与其自身 `nnfInitRandom` 的边界不一致。本项目不追随这处不一致，全程用同一套不变量
> （原版范围是它的子集，不影响正确性）。

#### 4.2 单层 match/vote 循环（4c）

每层的算法本体，结构照抄原版：

```text
成像一次，得到初始重建图像（若开启 uniformity，从起始 NNF 建 Omega 占用表）
for i in range(num_search_vote_iters):
    重建图像 vs source，重算每像素 patch 代价
    for j in range(num_patch_match_iters):
        propagate（传播）
        random_search（随机搜索）
    用优化后的 NNF 重新投票，刷新重建图像
```

外层 search-vote：搜索优化 NNF 再投票出图；内层 patch-match：传播 + 随机搜索。

#### 4.3 NNF 初始化 —— `synthesis/nnf.py`

`init_random_nnf`：`torch.randint` 一次性对整张图采样，坐标限制在 `[r, size-1-r]`。

#### 4.4 成像 —— `synthesis/vote.py`

- `gather_image`：每个 target 像素直接抄 NNF 指向的单个 style 像素，仅用于早期验证数据流。
- `vote_image`：实际使用的路径。target 像素 q 被 `patch_size²` 个 patch 共同认领——中心在 q 周围
  的每个 patch p 都声称 q 应长得像 `source[nnf[p] + (q - p)]`，取平均即 q 的颜色。实现上把"谁认领
  我"转成"固定偏移 d 下的静态切片"，`patch_size²` 次切片 + gather 即可，不需要 scatter。

#### 4.5 代价函数 —— `synthesis/cost.py`

原版分别计算 style 和 guide 代价再相加：`Σ styleWeight·diff² + Σ guideWeight·diff²`。把 style/guide
通道拼成同一通道轴、权重拼成同一权重向量后，等价于一次加权 SSD：

```python
cost = Σ_channel  weight[c] · (target_patch[c] - source_patch[c])²
```

`target_patch` 取自"重建图像 + target guide"，`source_patch` 取自"style + source guide"（由 NNF
指向）。target 侧用 `F.pad(mode="replicate")` 填充 `r` 圈以支持静态切片；source 侧因 NNF 边界不变量
无需填充。

> ⚠️ 边界 `patch_size//2` 圈像素代价无法归零——填充内容在 source 里没有真实对应，这是 patch 类算法
> 的固有边界伪影，非 bug。自测中"内部恢复率"与"全图均值代价"分开断言，避免被这层伪影误导。

#### 4.6 传播 —— `synthesis/propagate.py`

跳泛洪传播，步长 4→2→1（非简单的单像素挪动）：原版内核全并行执行、无串行扫描线，需要这种递减
跳跃步长让信息在几轮内传遍全图，本项目沿用同一方案。

每个步长 r 和四个方向（上下左右各 r 格）：把邻居当前使用的 source 位置减去偏移量作为候选，代价更
低则替换。四个方向依次处理、立即生效（非批量比较），与原版 `tryNeighborsOffset` 的顺序语义一致。

#### 4.7 随机搜索 —— `synthesis/random_search.py`

半径从 1 倍增到 source 最大边长的一半，每轮在当前答案 `±半径` 内随机采样一个候选，更优则替换。
用于跳出传播无法触及的局部最优。

#### 4.8 均匀性惩罚 —— `synthesis/uniformity.py`

抑制大片 target 区域抄同一块 source 纹理造成的重复感：

- `scatter_add_` 维护 Omega 占用表（source 每像素当前被多少 target patch 引用）；这里必须用 scatter
  而非固定切片，因为问的是"NNF 写入了 source 的哪些位置"，方向与代价函数的 gather 相反。
- 按面积比算出理想占用值（均匀分布下每个 source patch 应承担的引用次数）。
- propagate/random_search 决策时用 `cost + uniformity_weight × (占用/理想占用)` 代替纯 cost；候选
  被接受后，把占用从旧位置转移到新位置。
- Omega 生命周期为一个金字塔层：层内所有 vote/patchmatch 迭代累积共用，换层时重新统计。

> ⚠️ `-stopthreshold` 对应原版的"跳过已收敛像素"优化（mask/dilate），是 CUDA 单线程模型下的性能
> 手段。全向量化实现没有可省的逐像素分支，重算已收敛像素也无害（只在严格更优时才替换），因此故意
> 不实现，仅为兼容 CLI 保留参数。

#### 4.9 extrapass3x3 —— `stylize.py` 阶段 4.9 段

开启后，在最细层收敛的 NNF 上再跑一遍同样的 match/vote 循环：`patch_size` 强制为 3，均匀性惩罚
强制关闭，不重新初始化。原版做法是层计数器减一、重入最细层循环体，效果等价。

### 阶段 5：保存图片 —— `utils/image_io.py: save_image_from_vram`

`output_image` permute 回 `(C, H, W)` → `.cpu()` → `torchvision.io.write_png`；2/4 通道（带 alpha）
因编码器不支持，改用 PIL 写 LA/RGBA。

> ⚠️ 不要用 `torchvision.utils.save_image`——它要求 `[0,1]` 浮点张量，喂 `uint8` 会得到全白废图。

---

## 设计取舍小结

- 放弃 CUDA 桥接方案（原计划 pybind11 调用原版内核），改为纯 PyTorch 重写：零编译、可单步调试，代价
  是速度慢一个数量级（仍是 GPU 计算）。
- 主循环内联在 `stylize.py`：金字塔循环和 match/vote 循环直接按顺序写在入口文件里，`synthesis/` 只留
  积木函数。曾经存在的 `run_pyramid`/`run_patchmatch` 包装层已删除，为的是"看一个文件就能看清数据
  被一步步处理的全过程"。
- 合成引擎是纯函数式风格：每步返回新张量，不做原地写入，与原版直接写回显存缓冲区的方式不同但行为等价。
- `-stopthreshold`（4.8）、`nnfUpscale` 边界钳制（4.1）两处有意不追随原版，均已在正文标注。
- 输出不追求与原版逐字节相同：PatchMatch 带随机性，并行传播顺序也不同，评判标准是视觉等价。

---

## 一句话总结

CLI 解析 → 图像/guide 搬进显存并拼接 → 规划金字塔层数与权重 → 由粗到细，每层用 PatchMatch（传播 +
随机搜索，加权 patch 代价打分，可选均匀性惩罚）优化 NNF 并投票出图，NNF 放大后交给下一层 → 可选
extrapass3x3 抛光 → 保存图片。全程无 C++/CUDA，无逐像素 Python 循环。
