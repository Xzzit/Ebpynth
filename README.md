# Ebpynth

[简体中文](./README_zh.md) | English

A pure Python + PyTorch reimplementation of [ebsynth](https://github.com/jamriska/ebsynth), the example-based
image synthesis tool built on a PatchMatch-style algorithm. The original ships as a C++/CUDA binary; this project
rewrites the entire pipeline — argument parsing, image I/O, guide merging, and the PatchMatch synthesis engine
itself — as readable, debuggable PyTorch tensor code. No C++/CUDA is compiled or required. The trade-off is speed
(roughly an order of magnitude slower than the native kernel) for clarity: every stage is a plain tensor operation
you can step through, inspect, and modify.

## Dependencies

Tested With:

* Windows 11 & Ubuntu 26.04 (Reconmended)
* Python 3.10
* Pytorch 2.13.0 (cuda 13.2)
* 5070 Ti GPU

```
conda create -n ebpynth python=3.10
conda activate ebpynth
pip install -r requirements.txt
```

## Project structure

```
Ebpynth/
├── stylize.py                   # CLI entry point — the whole pipeline, including the pyramid and match/vote loops, written out step by step
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
└── examples/                     # Sample style/guide assets, one folder per use case
    ├── frame/                    # Video frame stylization (a painted keyframe + raw frame pair)
    ├── facestyle/                # Face portrait stylization (appearance/segmentation/position guides)
    ├── texbynum/                 # Texture-by-numbers (a single segmentation-map guide)
    └── stylit/                   # Illumination-guided 3D render stylization (multiple lighting-pass guides)
```

## Setup

Requires Python 3.9+, PyTorch (CUDA build), torchvision, and Pillow.

## Usage

```bash
python stylize.py -style <style_image> [weight] -guide <source_guide> <target_guide> [weight] [-guide ...] [options]
```

Example 1 — video frame stylization: a hand-painted keyframe (`source_painting`) is transferred onto a new
frame, guided by the raw frame pair it was painted from (`source_frame` → `target_frame`):

```bash
python stylize.py \
  -style examples/frame/source_painting.jpg \
  -guide examples/frame/source_frame.jpg examples/frame/target_frame.jpg \
  -output examples/frame/output.png \
  -extrapass3x3
```

Example 2 — face portrait stylization (the "FaceStyle" use case from the original ebsynth): transfers a portrait
painting's style onto a photo while preserving the subject's identity, using three facial-landmark-derived guides
instead of raw pixels — `Gapp` (target's luminance matched to the painting, keeps identity), `Gseg` (soft face
segmentation), and `Gpos` (a dense warp field mapping target pixels to their source correspondence):

```bash
python stylize.py \
  -style examples/facestyle/source_painting.png \
  -guide examples/facestyle/source_Gapp.png examples/facestyle/target_Gapp.png 2.0 \
  -guide examples/facestyle/source_Gseg.png examples/facestyle/target_Gseg.png 1.5 \
  -guide examples/facestyle/source_Gpos.png examples/facestyle/target_Gpos.png 1.5 \
  -output examples/facestyle/output.png
```

Example 3 — texture-by-numbers: a photo is resynthesized from a hand-painted target segmentation map, using a
single guide pair (source segmentation → target segmentation):

```bash
python stylize.py \
  -patchsize 3 -uniformity 1000 \
  -style examples/texbynum/source_photo.png \
  -guide examples/texbynum/source_segment.png examples/texbynum/target_segment.png \
  -output examples/texbynum/output.png
```

Example 4 — StyLit: transfers a hand-painted, non-photorealistic shading style (here, an illuminated ball drawn in
colored pencil) onto a 3D-rendered model, using several path-traced lighting passes as guides instead of raw pixel
color — `fullgi` (full global illumination), `dirdif` (direct diffuse), `dirspc` (direct specular), and `indirb`
(indirect bounce). The four guide weights sum to 2.0, the same 2:1 guide-to-style ratio as the original StyLit
example (which used three guides at 0.66 each):

```bash
python stylize.py \
  -style examples/stylit/source_style.png \
  -guide examples/stylit/source_fullgi.png examples/stylit/target_fullgi.png 0.5 \
  -guide examples/stylit/source_dirdif.png examples/stylit/target_dirdif.png 0.5 \
  -guide examples/stylit/source_dirspc.png examples/stylit/target_dirspc.png 0.5 \
  -guide examples/stylit/source_indirb.png examples/stylit/target_indirb.png 0.5 \
  -output examples/stylit/output.png
```

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `-style <path> [weight]` | path required, weight `1.0` | Style keyframe; its pixels are the only source of output color. Optional trailing weight. |
| `-guide <source> <target> [weight]` | at least one required, weight `1/N` | A guide pair: `source` is pixel-aligned with the style image, `target` is aligned with the desired output. Optional trailing weight. Repeat for multiple guides (e.g. color + edges + optical flow). |
| `-output <path>` | `output.png` | Output image path. |
| `-uniformity <value>` | `3500.0` | Penalty weight discouraging the same source patch from being overused. |
| `-patchsize <odd int, >= 3>` | `5` | Side length of the square patch used for matching. |
| `-pyramidlevels <int>` | `-1` (auto) | Number of pyramid levels. Auto-derived from image size and patch size when `-1`. |
| `-searchvoteiters <int>` | `6` | Match/vote iterations per pyramid level. |
| `-patchmatchiters <int>` | `4` | Propagation + random-search iterations per match/vote round. |
| `-stopthreshold <int>` | `5` | Accepted for CLI compatibility with the original tool; unused here — see note below. |
| `-extrapass3x3` | off | Adds a final 3x3-patch pass that sharpens fine detail. |

**Note on `-stopthreshold`:** in the original, this gates a per-pixel "skip already-converged pixels" optimization
that only pays off in a per-thread CUDA kernel. In a fully vectorized implementation there's no per-pixel work to
skip, and re-evaluating an already-optimal pixel is a no-op — so it's intentionally not implemented.

## Workflow

1. **Parse** CLI arguments into a config.
2. **Load** the style image and every guide image straight onto the GPU as `(H, W, C)` `uint8` tensors.
3. **Merge** all guides' channels into two feature tensors via `torch.cat` — one aligned with the style image
   (source side), one aligned with the desired output (target side).
4. **Plan** the pyramid: how many coarse-to-fine levels to use, how many iterations per level, and normalized
   per-channel weight vectors for the style and guide terms.
5. **Synthesize**, coarse to fine. At each pyramid level:
   - Resize the style/guide tensors to that level's resolution.
   - Initialize the NNF randomly (coarsest level) or upscale it from the previous level.
   - Repeatedly **propagate** matches between neighboring pixels, **randomly search** for better ones (scored by a
     weighted patch-distance cost, optionally penalized for overusing a source patch), and **re-vote** to refresh
     the reconstructed image from the current NNF.
6. **Refine** (optional): one more match/vote round at a smaller 3x3 patch size for sharper detail.
7. **Save** the finished image to disk.

The core state threaded through step 5 is the **NNF (Nearest-Neighbor Field)**: a per-pixel map from each output
position to the style-image position it should copy from. Synthesis is the process of optimizing that map.
