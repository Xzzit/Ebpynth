# Ebpynth

[简体中文](./README_zh.md) | English

A pure Python + PyTorch reimplementation of [ebsynth](https://github.com/jamriska/ebsynth), the example-based
image synthesis tool (PatchMatch + coarse-to-fine pyramid + patch voting). Nothing is compiled — every stage is
plain tensor code you can step through and modify, at roughly an order of magnitude the native kernel's runtime.

## Install

Tested on Windows 11 and Ubuntu 26.04, Python 3.10, PyTorch 2.13.0 (CUDA 13.2), RTX 5070 Ti.
**A CUDA GPU is required** — there is no CPU fallback.

```bash
conda create -n ebpynth python=3.10
conda activate ebpynth
pip install -r requirements.txt
```

## Usage

```bash
python stylize.py -style <style_image> [weight] -guide <source_guide> <target_guide> [weight] [-guide ...] [options]
```

One style image, plus at least one guide **pair**. The source guide is pixel-aligned with the style image; the
target guide is aligned with the output you want. Weights are optional trailing tokens on `-style` and `-guide`.

## Examples

Five self-contained use cases live in `examples/`. The still-image grids read **left to right: guides, then
style**, and **top to bottom: source, then target** — so the bottom-right cell is the synthesized result. Where an
example has several guides they sit side by side in one cell, in the order named in the column header. Each
command writes the `output.png` shown — though PatchMatch is randomized and no seed is fixed, so your run will
differ in fine detail. Equivalent, not identical.

### 1. Video frame

A hand-painted keyframe transferred onto a new frame, guided by the raw frame pair it was painted from.

```bash
python stylize.py \
  -style examples/frame/source_painting.jpg 0.5 \
  -guide examples/frame/source_frame.jpg examples/frame/target_frame.jpg \
  -output examples/frame/output.png \
  -extrapass3x3
```

|  | Guide — `frame` | Style |
|:--:|:--:|:--:|
| **Source** | <img src="examples/frame/source_frame.jpg" alt="source frame" width="390"> | <img src="examples/frame/source_painting.jpg" alt="source painting" width="390"> |
| **Target** | <img src="examples/frame/target_frame.jpg" alt="target frame" width="390"> | <a href="examples/frame/output.png"><img src="examples/frame/output.png" alt="synthesized frame" width="390"></a> |

### 2. Video

A whole clip stylized from a handful of painted keyframes, via `stylize_video.py`. The engine is unchanged; only
the guides differ. Alongside colour and edge, two extra guides carry the time axis: a *positional* guide (an
identity coordinate ramp advected by optical flow) and a *temporal* guide (the previous output warped into the
current frame). Each output feeds the next frame's temporal guide, so synthesis is one chain rather than 100
independent runs — which is what stops it flickering.

```bash
python stylize_video.py \
  -video examples/video/cat_full.mp4 \
  -styledir examples/video/style \
  -output examples/video/cat_styled.mp4
```

|  | 100 frames, 960x540, 20 fps |
|:--:|:--:|
| **Source** | <img src="examples/video/preview_source.webp" alt="source clip" width="480"> |
| **Stylized** | <img src="examples/video/preview_styled.webp" alt="stylized clip" width="480"> |

Keyframes live in `examples/video/style/` as `style<frame>.png`. The frame number is only a hint — each one is
re-matched against the video by edge correlation, since a stylized frame keeps its source's geometry but none of
its palette, and a silent off-by-one would misalign every guide derived from it.

Roughly 7 s per frame, and each frame is synthesized twice (forward from the previous keyframe, backward from the
next) before being crossfaded, so budget ~35 minutes for 100 frames. Add `-maxframes 7` for a quick smoke test.

### 3. Face portrait

A portrait painting's style transferred onto a photo while preserving the subject's identity, using three
facial-landmark-derived guides instead of raw pixels: `Gapp` (target luminance matched to the painting),
`Gseg` (soft face segmentation), `Gpos` (dense warp field mapping target pixels to their source correspondence).

```bash
python stylize.py \
  -style examples/facestyle/source_painting.png \
  -guide examples/facestyle/source_Gapp.png examples/facestyle/target_Gapp.png 2.0 \
  -guide examples/facestyle/source_Gseg.png examples/facestyle/target_Gseg.png 1.5 \
  -guide examples/facestyle/source_Gpos.png examples/facestyle/target_Gpos.png 1.5 \
  -output examples/facestyle/output.png
```

|  | Guides — `Gapp`, `Gseg`, `Gpos` | Style |
|:--:|:--:|:--:|
| **Source** | <img src="examples/facestyle/source_Gapp.png" alt="source Gapp" width="150"> <img src="examples/facestyle/source_Gseg.png" alt="source Gseg" width="150"> <img src="examples/facestyle/source_Gpos.png" alt="source Gpos" width="150"> | <img src="examples/facestyle/source_painting.png" alt="source painting" width="300"> |
| **Target** | <img src="examples/facestyle/target_Gapp.png" alt="target Gapp" width="150"> <img src="examples/facestyle/target_Gseg.png" alt="target Gseg" width="150"> <img src="examples/facestyle/target_Gpos.png" alt="target Gpos" width="150"> | <a href="examples/facestyle/output.png"><img src="examples/facestyle/output.png" alt="synthesized portrait" width="300"></a> |

### 4. Texture by numbers

A photo resynthesized from a hand-painted target segmentation map, from a single guide pair. Wants a smaller
patch and a lighter uniformity penalty than the defaults.

```bash
python stylize.py \
  -patchsize 3 -uniformity 1000 \
  -style examples/texbynum/source_photo.png \
  -guide examples/texbynum/source_segment.png examples/texbynum/target_segment.png \
  -output examples/texbynum/output.png
```

|  | Guide — `segment` | Style |
|:--:|:--:|:--:|
| **Source** | <img src="examples/texbynum/source_segment.png" alt="source segmentation" width="270"> | <img src="examples/texbynum/source_photo.png" alt="source photo" width="270"> |
| **Target** | <img src="examples/texbynum/target_segment.png" alt="target segmentation" width="270"> | <a href="examples/texbynum/output.png"><img src="examples/texbynum/output.png" alt="synthesized texture" width="270"></a> |

### 5. StyLit

A hand-painted, non-photorealistic shading style (an illuminated ball in colored pencil) transferred onto a
3D render, guided by four path-traced lighting passes instead of raw pixel color: `fullgi` (full global
illumination), `dirdif` (direct diffuse), `dirspc` (direct specular), `indirb` (indirect bounce). The four guide
weights sum to 2.0 — the same 2:1 guide-to-style ratio as the original StyLit example.

```bash
python stylize.py \
  -style examples/stylit/source_style.png \
  -guide examples/stylit/source_fullgi.png examples/stylit/target_fullgi.png 0.5 \
  -guide examples/stylit/source_dirdif.png examples/stylit/target_dirdif.png 0.5 \
  -guide examples/stylit/source_dirspc.png examples/stylit/target_dirspc.png 0.5 \
  -guide examples/stylit/source_indirb.png examples/stylit/target_indirb.png 0.5 \
  -output examples/stylit/output.png
```

|  | Guides — `fullgi`, `dirdif`, `dirspc`, `indirb` | Style |
|:--:|:--:|:--:|
| **Source** | <img src="examples/stylit/source_fullgi.png" alt="source full GI" width="130"> <img src="examples/stylit/source_dirdif.png" alt="source direct diffuse" width="130"> <img src="examples/stylit/source_dirspc.png" alt="source direct specular" width="130"> <img src="examples/stylit/source_indirb.png" alt="source indirect bounce" width="130"> | <img src="examples/stylit/source_style.png" alt="source style" width="280"> |
| **Target** | <img src="examples/stylit/target_fullgi.png" alt="target full GI" width="130"> <img src="examples/stylit/target_dirdif.png" alt="target direct diffuse" width="130"> <img src="examples/stylit/target_dirspc.png" alt="target direct specular" width="130"> <img src="examples/stylit/target_indirb.png" alt="target indirect bounce" width="130"> | <a href="examples/stylit/output.png"><img src="examples/stylit/output.png" alt="synthesized render" width="280"></a> |

## Options

For `stylize.py`. `stylize_video.py` accepts the tuning flags below (`-uniformity`, `-patchsize`,
`-pyramidlevels`, `-searchvoteiters`, `-patchmatchiters`, `-extrapass3x3`) plus its own `-video`, `-styledir`,
`-height`, `-maxframes` and `-smallflow`; see `python stylize_video.py --help`.

| Flag | Default | Meaning |
|---|---|---|
| `-style <path> [weight]` | required, weight `1.0` | Style keyframe; its pixels are the only source of output color. |
| `-guide <source> <target> [weight]` | at least one required, weight `1/N` | A guide pair. Repeatable for multiple guides (e.g. color + edges + optical flow). |
| `-output <path>` | `output.png` | Output image path. Always written as PNG. |
| `-uniformity <float>` | `3500.0` | Penalty discouraging the same source patch from being overused. |
| `-patchsize <odd int, >= 3>` | `5` | Side length of the square matching patch. |
| `-pyramidlevels <int>` | `-1` (auto) | Pyramid level count. Derived from image and patch size when `-1`. |
| `-searchvoteiters <int>` | `6` | Match/vote iterations per pyramid level. |
| `-patchmatchiters <int>` | `4` | Propagation + random-search iterations per match/vote round. |
| `-stopthreshold <int>` | `5` | Accepted for CLI compatibility only; this engine has no per-pixel work to skip, so it is unused. |
| `-extrapass3x3` | off | Final 3x3-patch pass that sharpens fine detail. |

## Project structure

```
Ebpynth/
├── stylize.py                   # CLI entry point — the whole pipeline, including the pyramid and match/vote loops, written out step by step
├── stylize_video.py             # Video entry point — keyframe propagation, crossfading, and its own copy of the match/vote loop
│
├── arguments/
│   └── parser.py                # Parses and validates CLI arguments
│
├── utils/
│   ├── image_io.py               # Loads/saves images as CUDA uint8 tensors
│   ├── guide_merge.py            # Concatenates guide image pairs into feature tensors
│   └── pyramid_plan.py           # Computes pyramid level count and per-level hyperparameters
│
├── synthesis/                    # PatchMatch building blocks, called in order by stylize.py
│   ├── nnf.py                    # Random Nearest-Neighbor Field (NNF) initialization
│   ├── vote.py                   # Reconstructs an image from an NNF ("imaging")
│   ├── cost.py                   # Weighted patch-distance (SSD) cost function
│   ├── propagate.py              # Jump-flood neighbor propagation
│   ├── random_search.py          # Randomized local search for better matches
│   ├── pyramid.py                # Pyramid math: level sizes, image resizing, NNF upscaling
│   └── uniformity.py             # Penalizes overused source patches
│
├── video/                        # Video-only building blocks, used by stylize_video.py
│   ├── frames.py                 # Decodes/encodes video as CUDA uint8 frame tensors
│   ├── flow.py                   # RAFT optical flow + the warp used by both time-axis guides
│   └── guides.py                 # Edge guide and the identity coordinate ramp
│
└── examples/                     # Sample style/guide assets, one folder per use case
    ├── frame/                    # Single-frame stylization
    ├── video/                    # Full clip + painted keyframes
    ├── facestyle/                # Face portrait stylization
    ├── texbynum/                 # Texture-by-numbers
    └── stylit/                   # Illumination-guided 3D render stylization
```

Most modules run standalone as a self-test, e.g. `python synthesis/vote.py` (run from the repo root).
