# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Ebpynth** — a from-scratch, 100% Python + PyTorch reimplementation of
[ebsynth](https://github.com/jamriska/ebsynth), the example-based image synthesis tool (PatchMatch + coarse-to-fine
pyramid + patch voting). `stylize.py` is a working drop-in replacement for the original binary's basic usage:
single style image + N guide pairs → one synthesized PNG. No C++/CUDA is compiled; everything is PyTorch tensor
ops on CUDA. It is roughly an order of magnitude slower than the native kernel.

**The upstream C++/CUDA source is NOT present in this repo or beside it, and the code no longer refers to it.**
Docstrings used to cite `ebsynth.cpp` / `ebsynth_cuda.cu` kernel names and line numbers as provenance; those were
deliberately stripped, because this is now a standalone learning project and the citations pointed at files nobody
can open. The *reasoning* they carried was kept and rewritten to stand on its own (why jump-flood instead of a
1-pixel step, why the target side is padded, why channels are aligned to `max`). **Do not reintroduce upstream
citations**, and do not treat "match the original" as a live requirement.

Two consequences of that standalone status, both settled decisions:
- The bar for the synthesis engine was never byte-equality with the original (PatchMatch is randomized and the
  vectorized propagation order differs) — only visual equivalence. It has met it.
- Where the original was internally inconsistent, this project chose the clean version. The main one: the NNF
  center-bound invariant is `[r, size-1-r]` (r = patch_size // 2) **everywhere**, including `upscale_nnf`, where
  the original used a stricter, inconsistent `[patchSize, size-1-patchSize]`. Don't "fix" this back.

## Project goals (current phase)

1. **Understand the algorithm** — walking the pipeline end to end. `README_zh.md` is the user's primary reading
   copy (Chinese, detailed); `README.md` is the concise public English one. `README_zh.md`'s 阶段 0–6 / 4a-4b-4c
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
  static slice. **Rank-agnostic in the NNF's leading dims**: a `(B, H, W, 2)` NNF returns `(B, H, W)` for the same
  op count (source gather widens, target slice broadcasts). `propagate` relies on this.
  - **A 2-pixel border ring can never reach zero cost** — replicate-padded edge content has no true match in the
    valid source region. Inherent to patch-based synthesis, not a bug. This is why tests separate "interior
    recovery rate" (the real correctness bar) from "whole-image mean cost".
- **`vote.py`** — `vote_image` is plain-average voting, done as `patch_size²` *sliced gathers* rather than a
  scatter: "which pixels does patch p cover" flips into "which patch centers cover pixel q", and for a fixed
  offset that's just a shifted slice. (`gather_image` is single-pixel copy, debug only.)
- **`propagate.py`** — **jump-flood at radii 4→2→1, not simple 1-pixel propagation.** The original's serial
  scanline lets a good match crawl across the whole image in one pass; a fully-parallel rewrite can't rely on that
  ordering, so shrinking jump distances restore cross-image information flow in a handful of passes. (The original
  CUDA kernel already does exactly this, for the same reason.) A jump's four directions are **stacked and scored in
  one batched `patch_cost` call**, then resolved together with the incumbent by a single `argmin` (incumbent at
  index 0, so ties keep the incumbent — same "strictly better wins" rule as before). Note all four directions read
  the *same* `nnf` either way (it is only reassigned at the end of a jump), so with `uniformity=None` the batched
  argmin is **mathematically identical** to the old sequential cascade; the only behavioral change is that Omega is
  now updated once per jump instead of once per direction.
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

Profiled on the `frame` example (960x540, 1 RGB guide, defaults + `-extrapass3x3`), RTX 5070 Ti, via
CUDA-synchronized wall-clock wrappers. Three optimizations have landed (see **Done** below). The first two left
call counts untouched; the third cuts them, so the table carries both:

| leaf component | calls (orig → now) | original | current | % of current total |
|---|---|---|---|---|
| `patch_cost` | 3210 → **1698** | 8.67 s | **4.54 s** | 63.1% |
| `Uniformity.update` | 2664 → **1368** | 6.94 s | **1.03 s** | 14.2% |
| `Uniformity.score` | 5328 → **2736** | 8.53 s | **0.29 s** | 4.0% |
| `vote_image` | 49 | 0.14 s | 0.15 s | 2.1% |
| `compute_omega`, `pad_target`, `resize_image` | 6 / 42 / 18 | ~0.02 s | ~0.05 s | 0.7% |
| **TOTAL** | | **25.5 s** | **7.20 s** | 3.5x |

**Everything outside the candidate-evaluation inner loop is noise** — voting, resizing, I/O, and pyramid planning
are collectively under 3%. Optimize the inner loop or don't bother.

**By stage** (same run; stages are `stylize.py`'s own numbering). Prep is free, and the level breakdown is the
dispatch-bound story in one table — level 0 is 480 pixels and level 4 is 129,600, yet they cost the same until
batching lands, after which the coarse levels get *much* cheaper and the finest barely moves:

| stage | before batching | after batching |
|---|---|---|
| 0–3 prep (parse, load, merge, plan) | 0.05 s (0.6%) | 0.04 s (0.6%) |
| level 0 (16x30) | 1.24 s (13.9%) | **0.68 s** (9.4%) |
| level 1 (33x60) | 1.27 s (14.2%) | **0.60 s** (8.4%) |
| level 2 (67x120) | 1.11 s (12.4%) | **0.68 s** (9.4%) |
| level 3 (135x240) | 1.12 s (12.5%) | 0.98 s (13.5%) |
| level 4 (270x480) | 1.17 s (13.1%) | 0.99 s (13.8%) |
| level 5 (540x960) | 2.16 s (24.1%) | 2.30 s (32.0%) |
| stage 5 `extrapass3x3` | 0.68 s (7.6%) | 0.79 s (11.0%) |
| stage 6 save | 0.14 s (1.6%) | 0.14 s (1.9%) |

Level 5 not improving is the model working as advertised: at full resolution the work is finally big enough to be
bandwidth-bound, so trading bytes for dispatches buys nothing there. The same effect explains why `stylit`
(1200x912, C=15) gains least from batching and `texbynum` (320x462, C=6) gains most.

**Work model.** Per pyramid level: `searchvoteiters` (6) × [1 full-field `patch_cost` + `patchmatchiters` (4) ×
(3 batched propagate passes + ~log₂(max_src_dim/2) random-search candidates)] + 1 `vote_image`. Each propagate
pass scores 4 candidates in one call (3 jump radii × 4 directions, batched). Every candidate evaluation is one
full-field `patch_cost`; with uniformity on, a propagate pass also costs 2 × `score` and 1 × `update`.

### The binding constraint: op dispatch, not bandwidth

Two measurements pin this down, and they should steer every optimization decision here:

- **CPU submit time ≈ total wall time** for all three hot functions (`patch_cost` 3.11 / 3.75 ms,
  `Uniformity.score` 1.37 / 1.35 ms, `Uniformity.update` 2.64 / 2.65 ms at 540x960). The GPU is idling while
  Python and the PyTorch dispatcher issue work.
- **`patch_cost` costs the same regardless of data size**: 2.27 ms at 16x30 versus 3.86 ms at 540x960 — a 1080x
  difference in pixels for 1.7x the time. That flat ~2.2 ms floor is pure per-op overhead. Consequence: coarse
  pyramid levels are *not* cheap. Levels 0–4 hold 25% of the pixels but burn ~75% of pyramid time.

So the lever is **fewer ops per offset**, not fewer bytes. Two corollaries, both verified the hard way:

- **`torch.compile` is unavailable in this environment** — no Triton on Windows (`Cannot find a working triton
  installation`). Would need the third-party `triton-windows` package; not installed, not a decision made yet.
- **Batching the `patch_size²` offsets into one gather is a large net LOSS.** Tried both a full
  `(H·W, patch_size², C)` unfold (0.36x on frame — 1.5 GB peak, 8.3 GB on stylit) and a per-row variant (0.61x).
  Streaming the loop keeps the working set tiny; batching trades cheap dispatches for expensive memory traffic.
  Don't retry this.

### Done

Cumulative, fixed seed: frame 23.4s → **6.1s** (3.8x), texbynum 9.6s → **1.7s** (5.6x),
stylit 45.4s → **14.7s** (3.1x). Wall time varies ~8% run to run; converged cost does not vary at all.

**1. √weight folding in `patch_cost`** (see `build_channel_scales`): since Σ w·(t−s)² ≡ Σ (√w·t − √w·s)², scaling
both feature tensors once per level removes the per-offset weight multiply. Bundled with hoisting the flat base
index and the uint8→float32 cast out of the loop, and making `pad_target`'s output contiguous. ~13 ops per
offset → ~6. Gains scale with channel count and patch size (stylit's C=15 benefited most; texbynum's patch_size 3
means only 9 offsets, so barely at all).

**2. Box-filtering Omega in `Uniformity`** (see `_box_sum`) — the bigger win, 19x on `score`. The old
`_patch_omega_sum` gathered `patch_size²` times to sum Omega over each candidate's window, but that sum depends
only on the window's *center*, not on which target pixel asked: it is a box filter over Omega, computable for all
source positions in one `avg_pool2d(divisor_override=1)`, after which each target pixel needs a single gather.
Cached and invalidated by `update`, so the two `score` calls within one propagate direction share one filter pass.
Plus: `update`'s `-old`/`+new` scatters fused into one call over a pre-built concatenated index, and
`weight / patch_size² / ideal` collapsed into one constant.

Worth internalising as the general pattern here: **an inner loop indexed by patch offset is often a convolution in
disguise.** `compute_omega` and `vote_image` have the same shape, though both are already off the critical path.

Changes 1 and 2 are pure op-count reductions with no algorithmic change. Quality is unchanged but output is **not**
bit-identical: the ~2e-07 relative perturbation flips a few `cand_cost < best_cost` comparisons, which PatchMatch
amplifies chaotically into a visibly different-but-equivalent NNF (PSNR 29–38 dB against baseline). Verified by
final converged mean cost — frame +0.16%, texbynum +0.57%, stylit −0.16%, i.e. within noise and random in
direction. **When judging a change here, compare converged cost, not pixels.** The run-to-run noise floor is
genuinely zero (same code + same seed reproduces bit-exact, `scatter_add_` notwithstanding), so any pixel
difference at all means the change altered the search trajectory.

**3. Batching propagate's 4 directions into one `patch_cost`** — the first change that touches behavior, though
less than it looks. `patch_cost` is now rank-agnostic in the NNF's leading dims, so a `(4, H, W, 2)` stack costs
one dispatch instead of four; the incumbent joins the stack at index 0 and one `argmin` resolves the pass.
`patch_cost` calls 3210 → 1698, `Uniformity.update` 2664 → 1368.

Because all four directions already read the same `nnf`, this is **mathematically identical** to the old cascade
whenever `uniformity=None` — `propagate.py`'s own sandbox reproduces its previous numbers exactly (275431.6 →
43980.2, 100% interior). The only real change is Omega updating once per jump instead of once per direction, which
is why the uniformity-heaviest example is the only one to move:

| | speedup | converged cost | peak VRAM |
|---|---|---|---|
| frame | 1.30–1.41x | **−0.04%** | 155 → 400 MB (2.6x) |
| texbynum | 1.69–1.74x | **−0.62%** | 45 → 117 MB (2.6x) |
| stylit | 1.19–1.22x | **+1.07%** | 505 → 1288 MB (2.6x) |

stylit's +1.07% is the one number outside the established ±0.6% noise band — a real, small quality cost, taken
knowingly for the speedup, and visually indistinguishable. **The 2.6x peak-VRAM cost is the thing to watch** if
inputs get much larger than `stylit`; it is inherent (B candidates = B times the bytes in flight). Tried and
rejected as a mitigation: `diff.square_()` in place — it did not move peak VRAM at all (the allocator's peak is
set by the gather and the stacked NNFs, not the squared temporary) and made frame slower.

### Tested and rejected: cutting iterations

**There are no wasted iterations to reclaim.** Both ways of running fewer search/vote iterations were measured
against final converged cost, fixed seed, and both just slide along a quality/speed trade-off curve rather than
buying efficiency. Neither is an optimization; do not re-litigate this without new evidence.

Per-level convergence does decay steeply (each level's 2nd iteration takes −20%~−34%, iterations 5–7 only 1–3%
each), which makes the late iterations *look* skippable. They are not: their small per-level gains compound
through every finer level above them.

Fixed schedules, `searchvoteiters` per level — note the two examples **disagree about which end is expendable**,
so there is no portable rule here:

| schedule | frame | stylit |
|---|---|---|
| `[3,3,3,3,6,6]` cut coarse | 1.21x / **+22.9%** cost | 1.21x / +1.6% cost |
| `[6,6,6,6,3,3]` cut fine | 1.35x / +7.8% cost | 1.73x / **+4.2%** cost |
| `[2,3,4,5,6,6]` ramp | 1.28x / +26.9% cost | 1.18x / **+0.6%** cost |

Adaptive early-stop (leave a level when the relative cost drop falls below a tolerance) does not escape the curve
either — on frame, tol=3% gives 1.32x for +7.9%, essentially the same trade as the best fixed schedule's
1.35x for +7.8%:

| tolerance | frame | stylit |
|---|---|---|
| 1% | 1.05x / +0.7% | 1.41x / +2.1% |
| 2% | 1.12x / +2.6% | 1.80x / +3.9% |
| 3% | 1.32x / +7.9% | 1.93x / +4.7% |

Early-stop is still defensible as a **user-facing quality knob** (stylit at 1.41x for +2.1% is a good deal if the
user wants it) — but that would be a new CLI feature, not a speedup, and it is not implemented.

### Remaining leads

**`random_search` is now the top consumer, not `propagate`** — 47.1% of runtime inclusive versus propagate's 45.6%,
after batching flipped the order. In descending order of expected value:

1. **Batch `random_search`'s radii, in chunks.** Same trick as change 3 above, and the obvious next step. Two
   differences that make it a harder call than propagate was: (a) its cascade is **genuine** — each radius samples
   around `nnf` *as updated by the previous radius*, so batching really does change the search, unlike propagate
   where all four directions read the same NNF anyway; (b) the radius count varies by level (4 at level 0, 9 at
   level 5), and batching all 9 would put ~2.4 GB per temporary on stylit, so chunk to 4 at a time (3 calls instead
   of 9 at the finest level) to hold memory at propagate's already-measured level. Expect the quality effect to
   land on top of stylit's existing +1.07%, so A/B all three examples before keeping it.
2. **`Uniformity.update` still scatters over the whole field** (1.03 s, 14%) even though late in a level only a few
   percent of pixels changed; the rest are `delta = 0` atomic adds that still cost traffic and contention.
   Compressing to changed pixels via `nonzero` would cut the work but introduces a device→host sync — measure
   before committing, a stall may cost more than it saves in a dispatch-bound loop.
3. **Level 5 and `extrapass3x3` are now 43% of runtime between them** and are the parts batching did *not* help,
   because at full resolution the work is genuinely bandwidth-bound. Anything further there needs a different
   lever than dispatch reduction — which, given `torch.compile` is unavailable, likely means there isn't one.

**Not worth doing:** fp16/bf16, int32 indices, or anything else that trades ops for bytes. Effective bandwidth is
nowhere near the 5070 Ti's ~896 GB/s ceiling; the workload is dispatch-bound.

When profiling: `synthesis/__init__.py` re-exports `propagate` and `random_search` as *functions*, shadowing the
same-named submodules — `import synthesis.propagate as m` silently hands you the function. Use
`sys.modules["synthesis.propagate"]`. Also, `stylize.py` binds its imports directly (`from synthesis import
patch_cost, ...`), so patching a module attribute won't affect it; patch `stylize.<name>` too.
