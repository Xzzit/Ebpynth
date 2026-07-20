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

**All of Tasks A–K are done. `stylize.py` is a working end-to-end CLI**, a drop-in replacement for the original
`ebsynth` binary's basic usage (single style + N guide pairs -> one synthesized PNG). No C++/CUDA is compiled or
loaded anywhere; `ebpynth/` (the abandoned native-extension scaffold) can be deleted whenever, it's dead weight.

- **Prep/IO side (Tasks A–E, K):** `arguments/parser.py` (CLI parsing with cascading `-weight`), `utils/image_io.py`
  (`load_image_to_vram` + `save_image_from_vram`), `utils/guide_merge.py` (`merge_guides`), `utils/pyramid_plan.py`
  (`plan_pyramid`).
- **The synthesis engine (Tasks F–J):** a `synthesis/` package implementing PatchMatch in pure PyTorch (random NNF +
  vote imaging → PatchMatch propagation/search iterations → coarse-to-fine pyramid → uniformity term →
  extrapass3x3), converging on `synthesis.ebsynth_run()`.
  - Task F: `synthesis/nnf.py` (`init_random_nnf`, NNF as int64 `(H_t, W_t, 2)` in (y, x) order with centers
    bounded to `[r, size-1-r]` — an invariant all later stages must preserve so voting/cost/propagation/search
    need no bounds checks) and `synthesis/vote.py` (`gather_image` single-pixel copy; `vote_image` plain-average
    voting done as patch_size² sliced gathers instead of a scatter).
  - Task G: `synthesis/cost.py` (`patch_cost` — style and guide channels concatenated into one weighted SSD,
    since that's mathematically identical to summing them separately; `pad_target` replicate-pads the target
    side by `r` so border patches need no bounds checks, source side never needs padding thanks to the NNF
    invariant), `synthesis/propagate.py` (`propagate` — jump-flood at radii 4→2→1, **not** simple 1-pixel-offset
    propagation; the original CUDA kernel already uses this exact scheme because it too runs fully parallel with
    no serial scanline dependency, so it's reused as-is rather than simplified), `synthesis/random_search.py`
    (`random_search` — doubling radius 1,2,4,... up to half the source's largest dimension), and
    `synthesis/patchmatch.py` (`run_patchmatch` — one pyramid level's full match/vote loop, no uniformity term
    and no pyramid yet). A 2-pixel border ring can never reach zero cost (replicate-padded edge content has no
    match anywhere in the valid source region) — this is an inherent patch-based-synthesis artifact, not a bug;
    the sandbox test separates "interior recovery rate" (the real correctness bar) from "whole-image mean cost"
    for exactly this reason. Sandbox: `python synthesis/patchmatch.py` runs a synthetic identity-guide
    convergence check, then a real-image milestone (`examples/video/temp/task_g_result.png` — recognizable
    subject after 3 vote iters x 3 patchmatch iters, ~1s at 540x960 on GPU, no pyramid yet so noticeably rougher
    than the eventual full pipeline).
  - Task H: `synthesis/pyramid.py` (`level_size` replicates `pyramidLevelSize`'s float-scale-then-truncate exactly;
    `resize_image` bilinear-resamples from the ORIGINAL full-res tensor down to each level's size, never
    progressively level-to-level, matching the original; `upscale_nnf` doubles a coarser level's converged NNF
    plus an `(x%2, y%2)` jitter so a 2x2 child block doesn't collapse onto one identical starting patch;
    `run_pyramid` drives coarsest-to-finest, calling `run_patchmatch` once per level with that level's resized
    images and per-level iteration counts from `plan_pyramid`). Note: the original's `nnfUpscale` clamps to
    `[patchSize, size-1-patchSize]` — stricter than, and inconsistent with, its own `nnfInitRandom`'s `[r,
    size-1-r]` margin (r = patchSize/2). This project intentionally keeps one `[r, size-1-r]` invariant
    everywhere instead (the original's is a strict subset, so nothing breaks) — a deliberate non-bug-for-bug
    choice, consistent with this phase's "visual equivalence, not byte equality" bar. Sandbox:
    `python synthesis/pyramid.py` checks level-size monotonicity and NNF upscale correctness, then runs the full
    engine end-to-end with real default hyperparameters (`examples/video/temp/task_h_result.png` — 6 levels,
    ~10s at 540x960, markedly more coherent than Task G's single-level milestone).
  - Task I: `synthesis/uniformity.py` (`Uniformity` — bundles an Omega occupancy tensor, an ideal-occupancy target,
    and the uniformity weight; `.score(cost, nnf)` replaces tryPatch's `cost + lambda*occupancy` decision formula,
    `.update(old_nnf, new_nnf, changed)` scatter-moves the occupancy claim from old to new source position on
    acceptance). One `Uniformity` instance is built once per pyramid level (in `run_patchmatch`, from that level's
    starting NNF) and threaded through every `propagate`/`random_search` call for the entire level — its state
    persists and accumulates across all vote/patchmatch iterations of that level, matching the original Omega's
    lifetime (ebsynth_cuda.cu ~lines 885-906). `propagate`/`random_search` both take an optional `uniformity=None`
    parameter (default preserves Tasks G/H's original pure-cost behavior exactly). Deliberately NOT ported: the
    original's stopthreshold-driven mask/dilate pixel-skip (krnlEvalMask/krnlDilateMask) — it's a CUDA
    per-thread performance shortcut with no analog benefit in a fully vectorized rewrite (skipping isn't cheaper
    for a tensor op, and re-evaluating an already-optimal pixel is a no-op since propagate/random_search only ever
    replace on strict improvement), so `stop_threshold_per_level` from `plan_pyramid` remains unused by design.
    Sandbox: `python synthesis/uniformity.py` — a small forced-reuse scenario (16x16 source vs 40x40 unrelated-
    noise target) shows uniformity_weight=3500 cutting Omega's variance from ~40000 to ~27000 and its peak from
    884 to 723 while the mean (total occupancy) stays exactly conserved; then a real-image full-pipeline milestone
    (`examples/video/temp/task_i_result.png`) with the CLI's actual default uniformity_weight.
  - Task J: `synthesis/pyramid.py`'s `run_pyramid` gained an `extra_pass_3x3` flag — after the normal coarse-to-fine
    loop finishes, if set, it calls `run_patchmatch` one more time on the finest level's own `(nnf, target_style)`
    (not restarted from scratch), forcing `patch_size=3` and `uniformity_weight=0.0` regardless of what the caller
    configured, replacing the original's `level--; patchSize=3; uniformityWeight=0` re-entry into its own finest
    level (ebsynth_cuda.cu ~lines 1089-1095). There's no separate top-level entry point module — `stylize.py`
    calls `run_pyramid` directly (an `ebsynth_run(config, plan)` wrapper existed briefly but was removed as
    redundant pass-through; `stylize.py` just unpacks `config`/`plan` into `run_pyramid`'s arguments itself,
    replacing `ebsynthRunCuda`, ebsynth_cuda.cu ~line 1103, as the call site). Sandbox: `python synthesis/pyramid.py`
    (appended after the Task H milestone) runs the real example with `extrapass3x3` on and off — **seeded
    identically before each run** so both take an identical coarse-to-fine path and only diverge at the extra pass;
    without that seeding the comparison is two independent random syntheses whose natural variance swamps the
    (real but smaller) sharpening effect, which is exactly what happened when this test was first merged in from
    the old `run.py`. Asserts a Laplacian-variance sharpness proxy is higher with `extrapass3x3` on.
- **Reference-only, never build:** `ebpynth/` holds unmodified upstream source copies plus `include/ebsynth.h`
  (origin of the channel-limit constants). Its `setup.py` belongs to the abandoned native-extension route — do not
  run it; it can be deleted. Likewise `../ebsynth/pyebsynth.cpp` + `run_test.py` (the old JIT bridge prototype) are
  only useful for sanity-running the original kernel to produce reference output.

Follow the project's staged plan in `README` (in Chinese, Tasks A–K, all now checked off). If picking up further
work here, keep the established rhythm: **one task/change at a time with user review in between.**

## Commands

Dev environment: conda env `ezsynth` (torch with CUDA, `torch.cuda.is_available()` is True).

No test suite or lint config; each module ends with an `if __name__ == "__main__"` sandbox check with asserts.
Run everything from the repo root:

```bash
python arguments/parser.py      # parser sandbox (mock CLI invocation)
python utils/image_io.py        # load + save/load round-trip test (uses examples/)
python utils/guide_merge.py     # merge + channel-alignment asserts (uses examples/, needs CUDA)
python utils/pyramid_plan.py    # pyramid/weight math asserts (pure CPU)
python synthesis/vote.py        # Task F: identity-NNF exactness check + mosaic milestones
python synthesis/patchmatch.py  # Task G: synthetic convergence check + single-level milestone
python synthesis/pyramid.py     # Task H + J: level-size/upscale checks, full-pyramid milestone,
                                 #             extrapass3x3 sharpness check + before/after milestones
python synthesis/uniformity.py  # Task I: occupancy-flattening check + milestone

# Full pipeline, end to end (takes ~25-30s at 540x960 with defaults):
python stylize.py -style examples/video/output_frames/000.png \
                  -guide examples/video/video_frames/000.jpg examples/video/video_frames/001.jpg \
                  -output examples/video/temp/001.png -extrapass3x3
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
  `pipeline.py` in very early drafts). Chains parse → load style → merge guides → plan pyramid → parity printout
  (~lines 430–436) → `synthesis.run_pyramid()` (called directly, unpacking `config`/`plan` into its arguments) →
  save, with a shape assert standing in for Task E's original "pre-allocate the output canvas" role (the engine is
  pure-functional — every stage returns a new tensor rather than writing into one shared buffer — so there's no
  literal buffer to pre-allocate into anymore).

- **`synthesis/`** — the pure-PyTorch PatchMatch engine, README Tasks F–J, described level by level above. Core
  state throughout is the NNF: an integer tensor `(H_target, W_target, 2)` mapping each target pixel to a source
  coordinate. The algorithm reference is `../ebsynth/src/ebsynth_cuda.cu` (per-level sizes derived internally at
  lines ~742–744; the 192-entry `dispatchEbsynth[24][8]` template table at ~1126 is why the channel limits exist).
  `run_pyramid` (in `pyramid.py`) is the package's public entry point — there's no separate top-level wrapper
  module; everything else is an internal building block importable individually for testing (as every module's
  own `__main__` sandbox does).

Tensor convention throughout: images are `uint8`, shaped `(H, W, C)` (interleaved, not planar), on CUDA, kept
`.contiguous()`. With no raw-pointer boundary left, contiguity is no longer a silent-corruption risk — the
convention is kept for consistency and predictable memory layout.
