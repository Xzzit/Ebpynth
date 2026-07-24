# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A from-scratch, **100% Python + PyTorch** reimplementation of [ebsynth](https://github.com/jamriska/ebsynth) (the
classic C++/CUDA example-based image synthesis tool). The original plan kept the upstream CUDA kernel behind a
pybind11 bridge; that route has been **abandoned** — the PatchMatch synthesis is rewritten in pure PyTorch tensor
ops (an order of magnitude slower than the native kernel, still GPU-resident, chosen deliberately for readability
and zero compilation). No C++/CUDA is ever compiled in this project anymore.

The sibling directory `../ebsynth` is the **unmodified upstream C++/CUDA project** kept as the behavioral ground
truth. For the preparation pipeline Python must match its semantics exactly (validation rules, channel-collapsing
logic, default values, error message wording). For the synthesis engine, outputs will **not** match byte-for-byte —
PatchMatch is randomized and the vectorized propagation order differs — so the bar there is visual equivalence
against the original binary's output, not byte equality. When in doubt about semantics, read
`../ebsynth/src/ebsynth.cpp` (CLI/prep) or `../ebsynth/src/ebsynth_cuda.cu` (algorithm) first.

## Current status

**The project is complete and working end-to-end.** `stylize.py` is a drop-in replacement for the original
`ebsynth` binary's basic usage (single style + N guide pairs -> one synthesized PNG).

**Architecture principle (a deliberate, user-requested flattening):** the entire main flow — prep stages AND the
coarse-to-fine pyramid loop AND the per-level match/vote loop — is written out sequentially inside
`stylize.py:main()`, so reading that one file gives the whole big picture of how data moves. The former
`synthesis.run_pyramid` / `synthesis.run_patchmatch` wrapper functions were **removed** for exactly this reason
(as was `synthesis/patchmatch.py` entirely). `synthesis/` now holds only leaf building blocks. Do not reintroduce
intermediate orchestration wrappers; new pipeline steps go inline into `stylize.py`.

- **Prep/IO:** `arguments/parser.py` (CLI parsing with cascading `-weight`), `utils/image_io.py`
  (`load_image_to_vram` + `save_image_from_vram`), `utils/guide_merge.py` (`merge_guides`), `utils/pyramid_plan.py`
  (`plan_pyramid`).
- **Synthesis building blocks (`synthesis/`):**
  - `nnf.py` — `init_random_nnf`. NNF is int64 `(H_t, W_t, 2)` in (y, x) order with centers bounded to
    `[r, size-1-r]` (r = patch_size // 2) — an invariant every stage preserves so voting/cost/propagation/search
    need no bounds checks.
  - `vote.py` — `gather_image` (single-pixel copy, debug only); `vote_image` (plain-average voting done as
    patch_size² sliced gathers instead of a scatter).
  - `cost.py` — `patch_cost`: style and guide channels concatenated into one weighted SSD (mathematically identical
    to summing them separately); `pad_target` replicate-pads the target side by `r` so border patches need no
    bounds checks; the source side never needs padding thanks to the NNF invariant. A 2-pixel border ring can never
    reach zero cost (replicate-padded edge content has no match in the valid source region) — an inherent
    patch-based-synthesis artifact, not a bug; tests separate "interior recovery rate" (the real correctness bar)
    from "whole-image mean cost" for this reason.
  - `propagate.py` — jump-flood at radii 4→2→1, **not** simple 1-pixel-offset propagation; the original CUDA kernel
    already uses this exact scheme because it too runs fully parallel with no serial scanline dependency. The four
    directions are tried sequentially (each immediately updates the running best), mirroring `tryNeighborsOffset`.
  - `random_search.py` — doubling radius 1,2,4,... up to half the source's largest dimension.
  - `pyramid.py` — pyramid math only: `level_size` (replicates `pyramidLevelSize`'s float-scale-then-truncate
    exactly; an integer shift can be off by one), `resize_image` (always bilinear-resamples from the ORIGINAL
    full-res tensor, never progressively level-to-level, matching the original), `upscale_nnf` (doubles a coarse
    NNF plus an `(x%2, y%2)` jitter so a 2x2 child block doesn't collapse onto one identical starting patch).
    Note: the original's `nnfUpscale` clamps to `[patchSize, size-1-patchSize]` — stricter than, and inconsistent
    with, its own `nnfInitRandom`'s `[r, size-1-r]`. This project intentionally keeps one `[r, size-1-r]`
    invariant everywhere (the original's is a strict subset, so nothing breaks) — a deliberate non-bug-for-bug
    choice consistent with the "visual equivalence, not byte equality" bar.
  - `uniformity.py` — `Uniformity` bundles an Omega occupancy tensor, an ideal-occupancy target, and the weight;
    `.score(cost, nnf)` replaces tryPatch's `cost + lambda*occupancy` decision formula, `.update(...)`
    scatter-moves occupancy claims on acceptance. One instance is built per pyramid level (in `stylize.py`'s
    per-level block, from that level's starting NNF) and threaded through every `propagate`/`random_search` call
    for the entire level — its state accumulates across all vote/patchmatch iterations of that level, matching the
    original Omega's lifetime (ebsynth_cuda.cu ~lines 885-906). `propagate`/`random_search` take `uniformity=None`
    to disable the term.
- **In `stylize.py` itself:** the pyramid loop (resize → NNF init/upscale → match/vote loop, stages 4a/4b/4c) and
  the optional extrapass3x3 block (re-runs the match/vote loop on the finest level's converged NNF with
  `patch_size=3` and uniformity off, replacing the original's `level--; patchSize=3; uniformityWeight=0` re-entry,
  ebsynth_cuda.cu ~lines 1089-1095).
- **Deliberately NOT ported:** the original's stopthreshold-driven mask/dilate pixel-skip
  (krnlEvalMask/krnlDilateMask) — a CUDA per-thread performance shortcut with no analog benefit in a fully
  vectorized rewrite (skipping isn't cheaper for a tensor op, and re-evaluating an already-optimal pixel is a
  no-op since propagate/random_search only replace on strict improvement). `stop_threshold_per_level` from
  `plan_pyramid` is unused by design; `-stopthreshold` is parsed only for CLI compatibility.

Docs: `README.md` (English, concise, public-facing) and `README_zh.md` (Chinese, detailed — the user's primary
reading copy; its 阶段/4a/4b/4c structure mirrors `stylize.py`'s section comments, keep them in sync). If picking
up further work, keep the established rhythm: **one task/change at a time with user review in between.**

## Commands

Dev environment: conda env `ezsynth` (torch with CUDA, `torch.cuda.is_available()` is True).

No test suite or lint config; modules end with an `if __name__ == "__main__"` sandbox check with asserts.
Run everything from the repo root:

```bash
python arguments/parser.py      # parser sandbox (mock CLI invocation)
python utils/image_io.py        # load + save/load round-trip test (uses examples/)
python utils/guide_merge.py     # merge + channel-alignment asserts (uses examples/, needs CUDA)
python utils/pyramid_plan.py    # pyramid/weight math asserts (pure CPU)
python synthesis/vote.py        # identity-NNF exactness check + mosaic milestones
python synthesis/propagate.py   # synthetic identity-guide convergence check (propagate + random_search together)
python synthesis/pyramid.py     # level-size / NNF-upscale math checks
python synthesis/uniformity.py  # occupancy-flattening check (expected: var ~40000 -> ~27000, peak 884 -> 723)

# Full pipeline, end to end (~25-30s at 540x960 with defaults) — this is also the real-image milestone test:
python stylize.py -style examples/video/output_frames/000.png \
                  -guide examples/video/video_frames/000.jpg examples/video/video_frames/001.jpg \
                  -output examples/video/temp/001.png -extrapass3x3
```

Testing gotcha (learned the hard way): any A/B comparison of two full syntheses (e.g. extrapass3x3 on vs off) must
`torch.manual_seed(...)` identically before each run — otherwise the two runs are independent random syntheses
whose natural variance swamps the smaller effect being measured.

## Architecture

Original `ebsynth.cpp` `main()` is one long function; this rewrite splits it along its phases (line numbers refer to
`../ebsynth/src/ebsynth.cpp`):

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
  scaling + int truncation to reproduce `pyramidLevelSize`'s `V2f→V2i` rounding exactly; explicit `-pyramidlevels`
  values are silently clamped to the derived max, like the original. Per-level iteration arrays are one scalar
  replicated (the kernel API allows per-level values; the CLI never varies them). Weight vectors:
  `style_weight/C_style` per style channel; each guide defaults to `1/numGuides` then spreads over its own
  channels. All outputs are plain CPU-side Python lists/ints.

- **`stylize.py`** — the whole pipeline, written out sequentially: parse → load style → merge guides → plan
  pyramid → parity printout (~lines 430–436) → coarse-to-fine pyramid loop with the match/vote loop inline
  (replacing `ebsynthRunCuda` + the per-level body of ebsynth_cuda.cu's main loop, ~lines 828-1095) → optional
  extrapass3x3 → shape assert → save. Section comments (阶段 0-5, 4a/4b/4c) mirror `README_zh.md`'s workflow
  numbering. The engine is pure-functional — every step returns a new tensor rather than writing into one shared
  buffer.

- **`synthesis/`** — leaf building blocks only (no orchestration; the loops live in `stylize.py`). Core state
  throughout is the NNF: an integer tensor `(H_target, W_target, 2)` mapping each target pixel to a source
  coordinate. The algorithm reference is `../ebsynth/src/ebsynth_cuda.cu` (per-level sizes derived internally at
  lines ~742–744; the 192-entry `dispatchEbsynth[24][8]` template table at ~1126 is why the channel limits exist).
  Every module is importable individually for testing, as the `__main__` sandboxes do.

Tensor convention throughout: images are `uint8`, shaped `(H, W, C)` (interleaved, not planar), on CUDA, kept
`.contiguous()`. With no raw-pointer boundary left, contiguity is no longer a silent-corruption risk — the
convention is kept for consistency and predictable memory layout.
