# Ebpynth

[简体中文](./README_zh.md) | English

A pure Python + PyTorch reimplementation of [ebsynth](https://github.com/jamriska/ebsynth), the example-based
image synthesis tool built on a PatchMatch-style algorithm. The original ships as a C++/CUDA binary; this project
rewrites the entire pipeline — argument parsing, image I/O, guide merging, and the PatchMatch synthesis engine
itself — as readable, debuggable PyTorch tensor code. No C++/CUDA is compiled or required. The trade-off is speed
(roughly an order of magnitude slower than the native kernel) for clarity: every stage is a plain tensor operation
you can step through, inspect, and modify.

## Project structure

```
Ebpynth/
├── stylize.py                   # CLI entry point — wires every stage below into one pipeline
│
├── arguments/
│   └── parser.py                # Parses and validates CLI arguments
│
├── utils/
│   ├── image_io.py               # Loads/saves images as CUDA uint8 tensors
│   ├── guide_merge.py            # Concatenates guide image pairs into feature tensors
│   └── pyramid_plan.py           # Computes pyramid level count and per-level hyperparameters
│
├── synthesis/                    # The PatchMatch engine, pure PyTorch
│   ├── nnf.py                    # Random Nearest-Neighbor Field (NNF) initialization
│   ├── vote.py                   # Reconstructs an image from an NNF ("imaging")
│   ├── cost.py                   # Weighted patch-distance (SSD) cost function
│   ├── propagate.py              # Jump-flood neighbor propagation
│   ├── random_search.py          # Randomized local search for better matches
│   ├── patchmatch.py             # Single-resolution match/vote optimization loop
│   ├── pyramid.py                # Coarse-to-fine driver + optional 3x3 refinement pass
│   └── uniformity.py             # Penalizes overused source patches
│
└── examples/video/               # Sample style/guide frames for testing
```

## Setup

Requires Python 3.9+, PyTorch (CUDA build), torchvision, and Pillow.

## Usage

```bash
python stylize.py -style <style_image> -guide <source_guide> <target_guide> [-guide ...] [options]
```

Example — stylize frame 001 using frame 000's stylization as the style keyframe:

```bash
python stylize.py \
  -style examples/video/output_frames/000.png \
  -guide examples/video/video_frames/000.jpg examples/video/video_frames/001.jpg \
  -output result.png -extrapass3x3
```

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `-style <path>` | required | Style keyframe. Its pixels are the only source of output color. |
| `-guide <source> <target>` | at least one required | A guide pair: `source` is pixel-aligned with the style image, `target` is aligned with the desired output. Repeat for multiple guides (e.g. color + edges + optical flow). |
| `-weight <value>` | style: `1.0`, guide: `1/N` | Sets the weight of the `-style`/`-guide` declared immediately before it. |
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
