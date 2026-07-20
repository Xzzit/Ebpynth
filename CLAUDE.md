# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A from-scratch, **100% Python + PyTorch** reimplementation of [ebsynth](https://github.com/jamriska/ebsynth) (the
classic C++/CUDA example-based image synthesis tool). The original plan kept the upstream CUDA kernel behind a
pybind11 bridge; that route has been **abandoned** — the PatchMatch synthesis itself is being rewritten in pure
PyTorch tensor ops (an order of magnitude slower than the native kernel, still GPU-resident, chosen deliberately for
readability and zero compilation). No C++/CUDA is ever compiled in this project anymore.

The sibling directory `../ebsynth` is the **unmodified upstream C++/CUDA project** kept as the behavioral ground
truth. For the preparation pipeline (Tasks A–E below) Python must match its semantics exactly (validation rules,
channel-collapsing logic, default values, error message wording). For the synthesis engine rewrite, outputs will
**not** match byte-for-byte — PatchMatch is randomized and the vectorized propagation order differs — so the bar
there is visual equivalence against the original binary's output, not byte equality. When in doubt about semantics,
read `../ebsynth/src/ebsynth.cpp` (CLI/prep) or `../ebsynth/src/ebsynth_cuda.cu` (algorithm) first.

## Current status

- **Done — the entire prep/IO side (Tasks A–E, K):** `arguments/parser.py` (CLI parsing with cascading `-weight`),
  `utils/image_io.py` (`load_image_to_vram` + `save_image_from_vram`), `utils/guide_merge.py` (`merge_guides`),
  `utils/pyramid_plan.py` (`plan_pyramid`), and `stylize.py` wired end-to-end: it runs A→E, prints the same
  hyper-parameter roll call as the original CLI, then exits at a `backend = None` gate. The Task K save call already
  sits after the gate and goes live the moment a backend exists.
- **Not started — the synthesis engine:** a `synthesis/` package implementing PatchMatch in pure PyTorch, staged as
  Tasks F–J in `README` (random NNF + vote imaging → PatchMatch propagation/search iterations → coarse-to-fine
  pyramid → uniformity term + stopthreshold → extrapass3x3), converging on one `ebsynth_run()` that replaces the
  gate in `stylize.py`.
- **Reference-only, never build:** `ebpynth/` holds unmodified upstream source copies plus `include/ebsynth.h`
  (origin of the channel-limit constants). Its `setup.py` belongs to the abandoned native-extension route — do not
  run it. Likewise `../ebsynth/pyebsynth.cpp` + `run_test.py` (the old JIT bridge prototype) are only useful for
  sanity-running the original kernel to produce reference output.

Follow the project's staged plan in `README` (in Chinese, Tasks A–K). Work proceeds **one task at a time with user
review in between** — don't jump ahead and write later tasks unless asked.

## Commands

Dev environment: conda env `ezsynth` (torch with CUDA, `torch.cuda.is_available()` is True).

No test suite or lint config; each module ends with an `if __name__ == "__main__"` sandbox check with asserts.
Run everything from the repo root:

```bash
python arguments/parser.py      # parser sandbox (mock CLI invocation)
python utils/image_io.py        # load + save/load round-trip test (uses examples/)
python utils/guide_merge.py     # merge + channel-alignment asserts (uses examples/, needs CUDA)
python utils/pyramid_plan.py    # pyramid/weight math asserts (pure CPU)

# Full pipeline — runs A→E then exits 1 at the backend gate until synthesis/ exists:
python stylize.py -style examples/video/output_frames/000.png \
                  -guide examples/video/video_frames/000.jpg examples/video/video_frames/001.jpg \
                  -output /tmp/001.png
```

## Architecture

Original `ebsynth.cpp` `main()` is one long function; this rewrite splits it along its phases (line numbers refer to
`../ebsynth/src/ebsynth.cpp` = `ebpynth/src/ebsynth.cpp`, byte-identical):

- **`arguments/parser.py`** replaces the `tryToParseArg` CLI loop (~lines 195–304). Custom `argparse.Action`s
  (`StyleAction`/`GuideAction`/`WeightAction`) reproduce the *cascading weight* rule — a bare `-weight` binds to the
  immediately preceding `-style`/`-guide`, tracked via `namespace._last_added`. Returns a plain config dict.

- **`utils/image_io.py`** replaces `tryLoad` + `evalNumChannels` + output write (~lines 134–153, 310–321, 461).
  Loading permutes CHW→HWC and calls `.contiguous()` before `.cuda()` and after channel-slicing. Channel collapse
  must stay behaviorally identical to `evalNumChannels` (opaque gray → 1ch, gray+alpha → 2ch, opaque RGBA → 3ch,
  including a native 2-channel image with fully-opaque alpha collapsing to 1ch). Saving: `write_png` handles 1/3
  channels; 2/4 (alpha-bearing) go through PIL as LA/RGBA since torchvision's encoder rejects them.

- **`utils/guide_merge.py`** replaces the guide load/validate loop and both packing loops (~lines 327–381).
  `merge_guides` concatenates all guide sources and all guide targets into two `(H, W, ΣC)` tensors. Checks that
  `torch.cat` can't do itself: source guides must match the style resolution, target guides must match each other,
  ΣC ≤ 24 and style channels ≤ 8 (`EBSYNTH_MAX_*_CHANNELS`). A single guide's source/target may collapse to
  different channel counts; the original takes `std::max` and packs from the forced-RGBA buffer — crucially taking
  lane R for 1-channel and lanes R+A (not R+G) for 2-channel — replicated by `_expand_channels`, which rebuilds
  RGBA lanes before re-slicing. Returns per-guide channel counts, which `plan_pyramid` needs for weight spreading.

- **`utils/pyramid_plan.py`** replaces the tail-end scalar math (~lines 383–426). Auto level count uses float
  scaling + int truncation to reproduce `pyramidLevelSize`'s `V2f→V2i` rounding exactly (an integer shift can be
  off by one); explicit `-pyramidlevels` values are silently clamped to the derived max, like the original. Per-level
  iteration arrays are one scalar replicated (the kernel API allows per-level values; the CLI never varies them).
  Weight vectors: `style_weight/C_style` per style channel; each guide defaults to `1/numGuides` then spreads over
  its own channels. All outputs are plain CPU-side Python lists/ints.

- **`stylize.py`** — the top-level orchestrator (the README's staged plan builds toward it; note it was called
  `pipeline.py` in very early drafts). Chains parse → load style → merge guides → plan pyramid → allocate the
  `(H_target, W_target, C_style)` uint8 CUDA output canvas → parity printout (~lines 430–436) → `backend = None`
  gate → save. The gate is where `synthesis.ebsynth_run()` will plug in.

- **`synthesis/`** (not yet written) — the pure-PyTorch PatchMatch engine, README Tasks F–J. Core state is the NNF:
  an integer tensor `(H_target, W_target, 2)` mapping each target pixel to a source coordinate. The algorithm
  reference is `../ebsynth/src/ebsynth_cuda.cu` (per-level sizes derived internally at lines ~742–744; the
  192-entry `dispatchEbsynth[24][8]` template table at ~1126 is why the channel limits exist).

Tensor convention throughout: images are `uint8`, shaped `(H, W, C)` (interleaved, not planar), on CUDA, kept
`.contiguous()`. With no raw-pointer boundary left, contiguity is no longer a silent-corruption risk — the
convention is kept for consistency and predictable memory layout.
