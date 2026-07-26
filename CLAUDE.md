# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Ebpynth** — a from-scratch, 100% Python + PyTorch reimplementation of
[ebsynth](https://github.com/jamriska/ebsynth), the example-based image synthesis tool (PatchMatch + coarse-to-fine
pyramid + patch voting). `stylize.py` is a working drop-in replacement for the original binary's basic usage:
single style image + N guide pairs → one synthesized PNG. No C++/CUDA is compiled; everything is PyTorch tensor
ops on CUDA. It is roughly an order of magnitude slower than the native kernel.

**The upstream C++/CUDA source is NOT present in this repo or beside it.** Docstrings throughout cite
`ebsynth.cpp` / `ebsynth_cuda.cu` line numbers (e.g. "replacing krnlPropagationPass, ~line 187"). Those are
**historical provenance notes explaining why the code is shaped the way it is** — do not try to open those files,
and do not treat "match the original" as a live requirement. This is now a standalone learning project.

Two consequences of that standalone status, both settled decisions:
- The bar for the synthesis engine was never byte-equality with the original (PatchMatch is randomized and the
  vectorized propagation order differs) — only visual equivalence. It has met it.
- Where the original was internally inconsistent, this project chose the clean version. The main one: the NNF
  center-bound invariant is `[r, size-1-r]` (r = patch_size // 2) **everywhere**, including `upscale_nnf`, where
  the original used a stricter, inconsistent `[patchSize, size-1-patchSize]`. Don't "fix" this back.

## Project goals (current phase)

1. **Understand the algorithm** — walking the pipeline end to end. `README_zh.md` is the user's primary reading
   copy (Chinese, detailed); `README.md` is the concise public English one. `README_zh.md`'s 阶段 0–5 / 4a-4b-4c
   headings deliberately mirror `stylize.py`'s section comments — **keep them in sync when the pipeline changes.**
2. **Then optimize for speed.** See the Performance section below for measured hotspots — read it before
   proposing any optimization, and re-measure rather than guessing.

Established working rhythm the user expects: **one task/change at a time, with user review in between.**

## Commands

Conda env is **`ebpynth`** (`E:\Miniconda\envs\ebpynth`), torch 2.13.0+cu132 / torchvision 0.28.0, CUDA available
(RTX 5070 Ti). There is no test suite and no lint config — modules self-test via `if __name__ == "__main__"`
sandbox blocks with asserts. Run everything **from the repo root**.

```bash
python utils/pyramid_plan.py    # pyramid level-count / weight-spreading asserts (pure CPU)
python utils/guide_merge.py     # 3-guide merge on examples/facestyle (expect channels [1, 3, 3])
python synthesis/pyramid.py     # level_size + upscale_nnf math checks
python synthesis/vote.py        # identity-NNF exactness + debug mosaics into examples/temp/
python synthesis/propagate.py   # synthetic identity-guide convergence (propagate + random_search together)
python synthesis/uniformity.py  # occupancy-flattening check (expect var ~40000 -> ~27000, peak 884 -> 723)
```

All six pass. `arguments/parser.py` and `utils/image_io.py` intentionally have no sandbox block — they're covered
only by a full pipeline run.

Full pipeline (~25s for the 960x540 `frame` example with defaults + extrapass3x3):

```bash
python stylize.py -style examples/frame/source_painting.jpg -guide examples/frame/source_frame.jpg examples/frame/target_frame.jpg -output examples/frame/output.png -extrapass3x3
```

`examples/` has four use cases, each self-contained with a committed `output.png` for visual comparison:
`frame/` (960x540, 1 guide), `texbynum/` (320x462, 1 guide, wants `-patchsize 3 -uniformity 1000`),
`facestyle/` (810x1000, 3 weighted guides), `stylit/` (1200x912, 4 weighted guides — the slowest). `README.md` has
the exact invocation for each.

**Benchmarking gotcha (learned the hard way):** any A/B comparison of two full syntheses must
`torch.manual_seed(...)` identically before each run. Otherwise you're comparing two independent random
syntheses whose natural variance swamps whatever effect you're measuring.

## Architecture

**Deliberate flattening (user-requested — do not undo).** The entire main flow — prep stages, the coarse-to-fine
pyramid loop, *and* the per-level match/vote loop — is written out sequentially inside `stylize.py:main()`, ~100
lines, so reading that one file gives the whole picture of how data moves. Wrapper functions
`synthesis.run_pyramid` / `run_patchmatch` (and `synthesis/patchmatch.py` entirely) were **removed** for exactly
this reason. `synthesis/` holds only leaf building blocks. **New pipeline steps go inline into `stylize.py`; do not
reintroduce intermediate orchestration wrappers.**

The engine is pure-functional: every step returns a new tensor rather than writing into a shared buffer. The one
exception is `Uniformity`, which is explicitly stateful (see below).

**Tensor convention:** images are `uint8`, `(H, W, C)` interleaved (not planar), on CUDA, `.contiguous()`.

**The central state is the NNF** — an int64 `(H_target, W_target, 2)` tensor in **(y, x)** order mapping each
target pixel to a source patch center, with both coordinates bounded to `[r, size-1-r]`. **Every stage must
preserve that invariant**; it is what lets voting, cost, propagation and search gather `center + offset` with no
bounds checks anywhere.

### Prep / IO

- **`arguments/parser.py`** — CLI → plain config dict. Deliberately diverges from the original's CLI shape:
  `-style <path> [weight]` and `-guide <src> <tgt> [weight]` take weight as an inline optional trailing token
  (`nargs="+"`, length tells you if it was given), instead of the original's separate cascading `-weight` flag
  bound to whatever came before it. No `Action` subclasses or parser state needed. Omitted weights come back as a
  `-1.0` sentinel that `plan_pyramid` resolves to the real default.
- **`utils/image_io.py`** — `load_image_to_vram` / `save_image_from_vram`. Channel collapsing mirrors
  `evalNumChannels`: opaque gray → 1ch, gray+alpha → 2ch, opaque RGBA → 3ch (including a native 2-channel image
  with fully-opaque alpha collapsing to 1ch). Saving: torchvision's PNG encoder rejects 2/4 channels, so
  alpha-bearing outputs go through PIL as LA/RGBA.
- **`utils/guide_merge.py`** — `merge_guides` concatenates all guide sources / all guide targets into two
  `(H, W, ΣC)` tensors. Does the checks `torch.cat` can't: source guides must match style resolution, target
  guides must match each other, ΣC ≤ 24 and style channels ≤ 8. (Those limits exist because the original had a
  192-entry `dispatchEbsynth[24][8]` template table; they're vestigial here but kept for behavioral parity.) A
  single guide's source/target may collapse to *different* channel counts — both are aligned up to `max` via
  `_expand_channels`, which rebuilds RGBA lanes then re-slices, crucially taking lanes **R+A (not R+G)** for the
  2-channel case.
- **`utils/pyramid_plan.py`** — pure CPU scalar math. Auto level count uses float scaling + int truncation to
  reproduce `pyramidLevelSize`'s rounding exactly (an integer shift can be off by one); explicit
  `-pyramidlevels` is silently clamped to the derived max. Per-level iteration arrays are one scalar replicated
  (the kernel API allowed per-level values; the CLI never varies them). Weights: `style_weight / C_style` per
  style channel; each guide defaults to `1/num_guides` then spreads over its own channels.

### `synthesis/` — leaf building blocks

- **`nnf.py`** — `init_random_nnf`. Establishes the NNF invariant described above.
- **`cost.py`** — `patch_cost` is the heart of the engine. Style and guide channels are concatenated into **one**
  weighted SSD (mathematically identical to summing the two terms separately, since both are weighted sums of
  squared per-channel diffs). `pad_target` replicate-pads the *target* side by `r` so border patches need no
  bounds checks; the source side never needs padding thanks to the NNF invariant. Vectorized as `patch_size²`
  iterations where, for a fixed `(dy, dx)`, the source side is one flat-index gather and the target side is one
  static slice.
  - **A 2-pixel border ring can never reach zero cost** — replicate-padded edge content has no true match in the
    valid source region. Inherent to patch-based synthesis, not a bug. This is why tests separate "interior
    recovery rate" (the real correctness bar) from "whole-image mean cost".
- **`vote.py`** — `vote_image` is plain-average voting, done as `patch_size²` *sliced gathers* rather than a
  scatter: "which pixels does patch p cover" flips into "which patch centers cover pixel q", and for a fixed
  offset that's just a shifted slice. (`gather_image` is single-pixel copy, debug only.)
- **`propagate.py`** — **jump-flood at radii 4→2→1, not simple 1-pixel propagation.** The original's serial
  scanline lets a good match crawl across the whole image in one pass; a fully-parallel rewrite can't rely on that
  ordering, so shrinking jump distances restore cross-image information flow in a handful of passes. (The original
  CUDA kernel already does exactly this, for the same reason.) The four directions are tried **sequentially**,
  each immediately updating the running best — so a later direction sees the earlier ones' outcome, including
  their occupancy bookkeeping.
- **`random_search.py`** — doubling radius 1, 2, 4, … up to half the source's largest dimension, one random
  candidate per pixel per radius. This is what escapes local optima; propagation alone can only spread matches
  that already exist somewhere.
- **`pyramid.py`** — `level_size` (float-scale-then-truncate), `resize_image` (**always** bilinear-resamples from
  the ORIGINAL full-res tensor, never progressively level-to-level), `upscale_nnf` (doubles a coarse NNF plus an
  `(x%2, y%2)` jitter so a 2x2 child block doesn't collapse onto one identical starting patch).
- **`uniformity.py`** — the only stateful piece. `Uniformity` bundles an Omega occupancy tensor, an
  ideal-occupancy target, and the weight. `.score(cost, nnf)` implements the original tryPatch's
  `cost + lambda*occupancy` decision formula; `.update(old, new, changed)` scatter-moves occupancy claims on
  acceptance. **One instance per pyramid level**, built in `stylize.py` from that level's starting NNF and threaded
  through every `propagate`/`random_search` call for the whole level — its state accumulates across all
  vote/patchmatch iterations, matching the original Omega's lifetime. Pass `uniformity=None` to disable.

### In `stylize.py` itself

The pyramid loop (4a resize → 4b NNF init/upscale → 4c match/vote loop) and the optional `-extrapass3x3` block,
which re-runs the match/vote loop on the finest level's converged NNF with `patch_size=3` and uniformity off.

**Deliberately NOT ported:** the original's stopthreshold-driven mask/dilate pixel-skip
(`krnlEvalMask`/`krnlDilateMask`). It's a CUDA per-thread performance shortcut with no analog benefit in a
vectorized rewrite — skipping isn't cheaper for a tensor op, and re-evaluating an already-optimal pixel is a no-op
since propagate/random_search only replace on strict improvement. `plan_pyramid`'s `stop_threshold_per_level` is
unused by design; `-stopthreshold` is parsed for CLI compatibility only.

## Performance

Measured on the `frame` example (960x540, 1 RGB guide, defaults + `-extrapass3x3`), RTX 5070 Ti, **25.5s total**,
via CUDA-synchronized wall-clock wrappers:

| component | calls | seconds | % total |
|---|---|---|---|
| `propagate` *(inclusive)* | 168 | 16.02 | 62.8% |
| `random_search` *(inclusive)* | 168 | 8.95 | 35.1% |
| `patch_cost` *(leaf)* | 3210 | 8.67 | 34.0% |
| `Uniformity.score` *(leaf)* | 5328 | 8.53 | 33.5% |
| `Uniformity.update` *(leaf)* | 2664 | 6.94 | 27.2% |
| `vote_image` | 49 | 0.14 | 0.5% |
| `compute_omega`, `pad_target` | 6 / 42 | ~0.02 | ~0.1% |

Leaf rows sum to ~98%. **Everything outside the candidate-evaluation inner loop is noise** — voting, resizing,
I/O, and pyramid planning are collectively under 1%. Optimize the inner loop or don't bother.

**Work model.** Per pyramid level: `searchvoteiters` (6) × [1 full-field `patch_cost` + `patchmatchiters` (4) ×
(12 propagate candidates + ~log₂(max_src_dim/2) random-search candidates)] + 1 `vote_image`. Propagate's 12 =
3 jump radii × 4 directions. Each *candidate evaluation* costs, in full-field passes of `patch_size²` gathers:
1 × `patch_cost` (C channels), plus with uniformity on 2 × `score` (1 channel) and 2 × `patch_size²` scatter_adds
in `update`. That's why **uniformity is ~61% of runtime while contributing no cost-function information** — it is
the single biggest lever.

Concrete leads, roughly by expected payoff (**none of these have been implemented or validated — measure first**):

1. **`Uniformity.score(best_cost, best_nnf)` is recomputed from scratch for every candidate**, in both
   [propagate.py:72](synthesis/propagate.py:72) and [random_search.py:49](synthesis/random_search.py:49), even
   though it's the score of a field that only changes on acceptance. Half of all 5328 `score` calls recompute a
   value that could be carried forward. Careful: the incumbent's score genuinely *does* shift when `update`
   mutates Omega, so this needs thought, not a naive cache.
2. **`patch_cost` re-materializes `combined_source.reshape(-1, channels).float()` on every one of its 3210
   calls** ([cost.py:84](synthesis/cost.py:84)), even though `combined_source` is constant for a whole pyramid
   level. Hoisting the float cast to level scope is nearly free to do.
3. **Kernel-launch overhead is likely a large share.** `patch_cost` issues ~`patch_size²` × several ops ≈ 150
   launches per call; across the run that's on the order of 10⁵–10⁶ tiny launches. `torch.compile` on the leaf
   functions, or restructuring the `(dy, dx)` loop into a batched/unfolded form, targets this directly.
4. **Batching propagate's 4 directions into one `patch_cost` call** would cut launches 4x — but it **changes
   semantics**: all four candidates would be scored against the same starting incumbent instead of cascading.
   Probably acceptable (PatchMatch is stochastic anyway) but it is a real behavioral change, so gate it behind a
   decision with the user and A/B it under a fixed seed.
5. **Precision.** Everything is float32. Costs are sums of squared uint8 diffs; fp16/bf16 accumulation may be
   viable and would halve memory traffic on a bandwidth-bound workload.

When profiling: `synthesis/__init__.py` re-exports `propagate` and `random_search` as *functions*, shadowing the
same-named submodules — `import synthesis.propagate as m` silently hands you the function. Use
`sys.modules["synthesis.propagate"]`. Also, `stylize.py` binds its imports directly (`from synthesis import
patch_cost, ...`), so patching a module attribute won't affect it; patch `stylize.<name>` too.
