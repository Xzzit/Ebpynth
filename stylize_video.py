"""
Keyframe-driven video stylization.

Given a video and a handful of stylized keyframes, this propagates each keyframe's
style along the timeline and crossfades between neighbouring keyframes, producing a
fully stylized clip.

The engine is the same one `stylize.py` drives; only the guides differ. Four guide
pairs per frame carry the work:

  colour      video[k]        -> video[t]              what the scene looks like
  edge        edges(video[k]) -> edges(video[t])       where its contours are
  positional  identity ramp   -> ramp advected by flow which part of the keyframe this is
  temporal    style[k]        -> warp(output[t-1])     what it looked like a frame ago

The temporal pair is what separates this from stylizing frames one by one. Each output
becomes the next frame's temporal target, so the synthesis is chained rather than
independent — without it every frame would pick its own unrelated patches and the
result would flicker violently.

Run `python stylize_video.py --help` from the repo root.
"""
import argparse
import os
import sys
import time

import torch

from utils import plan_pyramid, load_image_to_vram, save_image_from_vram
from synthesis import (
    Uniformity, build_channel_scales, build_combined_source, init_random_nnf,
    level_size, pad_target, patch_cost, propagate, random_search, resize_image,
    upscale_nnf, vote_image,
)
from video import read_frames, write_video, OpticalFlow, warp_u8, edge_guide, identity_ramp

# Channel count and relative weight of each guide pair, in the order they are
# concatenated. Weights follow the ratios established for example-based video
# stylization: colour dominates, position anchors correspondence, and the temporal
# term is deliberately gentle — pushed too hard it freezes the output into the first
# frame instead of merely discouraging flicker.
GUIDE_CHANNELS = [3, 1, 3, 3]
GUIDE_WEIGHTS = [6.0, 1.0, 2.0, 0.5]


def synthesize(style, source_guides, target_guides, config):
    """
    One full coarse-to-fine synthesis — the same pipeline as `stylize.py`'s stages 4
    and 5, operating on in-memory tensors instead of CLI arguments and files.

    Args:
        style: uint8 CUDA tensor, shape (H_style, W_style, 3) — the stylized keyframe.
        source_guides: uint8 CUDA tensor, (H_style, W_style, ΣC), aligned with style.
        target_guides: uint8 CUDA tensor, (H_target, W_target, ΣC), aligned with output.
        config: dict in `plan_pyramid`'s expected shape.

    Returns:
        uint8 CUDA tensor, shape (H_target, W_target, 3) — the stylized frame.
    """
    style_h, style_w, _ = style.shape
    target_h, target_w = target_guides.shape[0], target_guides.shape[1]
    plan = plan_pyramid(config, style.shape, target_guides.shape, GUIDE_CHANNELS)
    num_levels = plan["num_pyramid_levels"]
    patch_size = config["patch_size"]
    scales = build_channel_scales(plan["style_weights"], plan["guide_weights"])

    for level in range(num_levels):
        lvl_style_h, lvl_style_w = level_size(style_h, style_w, num_levels, level)
        lvl_target_h, lvl_target_w = level_size(target_h, target_w, num_levels, level)
        lvl_source_style = resize_image(style, lvl_style_h, lvl_style_w)
        lvl_source_guides = resize_image(source_guides, lvl_style_h, lvl_style_w)
        lvl_target_guides = resize_image(target_guides, lvl_target_h, lvl_target_w)

        if level == 0:
            nnf = init_random_nnf(lvl_target_h, lvl_target_w, lvl_style_h, lvl_style_w, patch_size)
        else:
            nnf = upscale_nnf(nnf, lvl_target_h, lvl_target_w, lvl_style_h, lvl_style_w, patch_size)

        combined_source = build_combined_source(lvl_source_style, lvl_source_guides, scales)
        target_style = vote_image(nnf, lvl_source_style, patch_size)
        uniformity = None
        if config["uniformity_weight"] > 0:
            uniformity = Uniformity(nnf, (lvl_style_h, lvl_style_w),
                                    (lvl_target_h, lvl_target_w), patch_size,
                                    config["uniformity_weight"])

        for _ in range(plan["search_vote_iters_per_level"][level]):
            combined_target_padded = pad_target(
                torch.cat([target_style, lvl_target_guides], dim=-1), patch_size, scales)
            cost = patch_cost(nnf, combined_source, combined_target_padded, patch_size)
            for _ in range(plan["patch_match_iters_per_level"][level]):
                nnf, cost = propagate(nnf, cost, combined_source, combined_target_padded,
                                      patch_size, uniformity)
                nnf, cost = random_search(nnf, cost, combined_source, combined_target_padded,
                                          patch_size, uniformity)
            target_style = vote_image(nnf, lvl_source_style, patch_size)

    if config["extra_pass_3x3"]:
        combined_source = build_combined_source(style, source_guides, scales)
        target_style = vote_image(nnf, style, 3)
        for _ in range(plan["search_vote_iters_per_level"][-1]):
            combined_target_padded = pad_target(
                torch.cat([target_style, target_guides], dim=-1), 3, scales)
            cost = patch_cost(nnf, combined_source, combined_target_padded, 3)
            for _ in range(plan["patch_match_iters_per_level"][-1]):
                nnf, cost = propagate(nnf, cost, combined_source, combined_target_padded, 3)
                nnf, cost = random_search(nnf, cost, combined_source, combined_target_padded, 3)
            target_style = vote_image(nnf, style, 3)

    return target_style


def locate_keyframe(style, frames, named_index, search=4):
    """
    Finds which video frame a stylized keyframe was painted over.

    A stylized image keeps its source frame's geometry but none of its palette, so the
    match runs on downsampled edge maps, not pixels. The filename's number is only a
    hint — off-by-one naming is common and silently misaligns every guide derived from
    that keyframe.

    Args:
        style: uint8 CUDA tensor, (H, W, 3) — the stylized keyframe.
        frames: list of uint8 CUDA tensors, the decoded video.
        named_index: the index parsed from the filename.
        search: how many frames either side of named_index to consider.

    Returns:
        (best_index, correlation_at_best, correlation_at_named).
    """
    def signature(img):
        e = edge_guide(img).float().squeeze(-1)
        e = torch.nn.functional.avg_pool2d(e[None, None], 4).squeeze()
        return (e - e.mean()) / (e.std() + 1e-6)

    target = signature(style)
    lo, hi = max(0, named_index - search), min(len(frames), named_index + search + 1)
    scores = {t: float((target * signature(frames[t])).mean()) for t in range(lo, hi)}
    best = max(scores, key=scores.get)
    return best, scores[best], scores.get(named_index, float("nan"))


def propagate_run(style, key_index, targets, frames, edges, flows, ramp, config, tag):
    """
    Chains synthesis along a run of consecutive frames leaving one keyframe.

    Args:
        style: the keyframe's stylized image, uint8 (H, W, 3).
        key_index: the keyframe's frame index.
        targets: frame indices to synthesize, ordered outward from key_index and each
                 adjacent to the one before it.
        frames, edges: per-frame video tensors and their edge guides.
        flows: dict {(a, b): (2, H, W) float16 CPU tensor} — motion from frame a to b.
        ramp: the identity coordinate ramp (positional guide's source side).
        config: synthesis config dict.
        tag: label for progress output.

    Returns:
        dict {frame index: uint8 CUDA (H, W, 3)}.
    """
    results = {}
    prev_index, prev_output, prev_ramp = key_index, style, ramp

    for t in targets:
        flow = flows[(t, prev_index)].cuda().float()
        # Both the previous output and the accumulated ramp live in prev_index's
        # geometry; this drags them into frame t's.
        warped_output = warp_u8(prev_output, flow)
        warped_ramp = warp_u8(prev_ramp, flow)

        source_guides = torch.cat([frames[key_index], edges[key_index], ramp, style], dim=-1)
        target_guides = torch.cat([frames[t], edges[t], warped_ramp, warped_output], dim=-1)

        t0 = time.perf_counter()
        results[t] = synthesize(style, source_guides.contiguous(),
                                target_guides.contiguous(), config)
        torch.cuda.synchronize()
        print(f"  [{tag}] key {key_index:3d} -> frame {t:3d}   "
              f"{time.perf_counter() - t0:5.1f}s", flush=True)

        prev_index, prev_output, prev_ramp = t, results[t], warped_ramp

    return results


def main():
    parser = argparse.ArgumentParser(description="Keyframe-driven video stylization")
    parser.add_argument("-video", default="examples/video/cat_full.mp4")
    parser.add_argument("-styledir", default="examples/video/style")
    parser.add_argument("-outdir", default="examples/video/out")
    parser.add_argument("-output", default="examples/video/cat_styled.mp4")
    parser.add_argument("-height", type=int, default=540,
                        help="crop decoded frames to this many rows (encoder padding)")
    parser.add_argument("-maxframes", type=int, default=-1,
                        help="stop after N frames — for quick smoke tests")
    parser.add_argument("-uniformity", type=float, default=3500.0, dest="uniformity_weight")
    parser.add_argument("-patchsize", type=int, default=5, dest="patch_size")
    parser.add_argument("-pyramidlevels", type=int, default=-1, dest="num_pyramid_levels")
    parser.add_argument("-searchvoteiters", type=int, default=6, dest="num_search_vote_iters")
    parser.add_argument("-patchmatchiters", type=int, default=4, dest="num_patch_match_iters")
    parser.add_argument("-extrapass3x3", action="store_true", dest="extra_pass_3x3")
    parser.add_argument("-smallflow", action="store_true",
                        help="use raft_small instead of raft_large")
    args = parser.parse_args()

    config = {
        "patch_size": args.patch_size,
        "num_pyramid_levels": args.num_pyramid_levels,
        "num_search_vote_iters": args.num_search_vote_iters,
        "num_patch_match_iters": args.num_patch_match_iters,
        "stop_threshold": 5,
        "style_weight": -1.0,
        "uniformity_weight": args.uniformity_weight,
        "extra_pass_3x3": 1 if args.extra_pass_3x3 else 0,
        "guides": [{"weight": w} for w in GUIDE_WEIGHTS],
    }

    # Stage 1: decode the video
    t_start = time.perf_counter()
    frames, fps = read_frames(args.video, crop_height=args.height)
    if args.maxframes > 0:
        frames = frames[:args.maxframes]
    height, width, _ = frames[0].shape
    print(f"decoded {len(frames)} frames at {width}x{height}, {fps:g} fps")

    # Stage 2: load keyframes and find which frame each was actually painted over
    style_files = sorted(f for f in os.listdir(args.styledir)
                         if f.lower().endswith((".png", ".jpg", ".jpeg")))
    keyframes = {}
    for name in style_files:
        named = int("".join(c for c in name if c.isdigit()) or -1)
        if not 0 <= named < len(frames):
            continue
        style = load_image_to_vram(os.path.join(args.styledir, name))
        if style.shape[:2] != (height, width):
            print(f"  ! {name} is {tuple(style.shape[:2])}, expected {(height, width)} — skipped")
            continue
        best, corr_best, corr_named = locate_keyframe(style, frames, named)
        if best != named:
            print(f"  ! {name}: filename says frame {named} but frame {best} matches better "
                  f"({corr_best:.3f} vs {corr_named:.3f}) — using {best}")
        keyframes[best] = style
    key_indices = sorted(keyframes)
    if not key_indices:
        sys.exit("error: no usable keyframes found")
    print(f"keyframes at {key_indices}")

    # Stage 3: optical flow for every adjacent pair, both directions. Kept on the CPU
    # as float16 — all of them at once would otherwise crowd out the synthesis.
    print("computing optical flow...", flush=True)
    flow_model = OpticalFlow(small=args.smallflow)
    flows = {}
    for t in range(len(frames)):
        for neighbor in (t - 1, t + 1):
            if 0 <= neighbor < len(frames):
                flows[(t, neighbor)] = flow_model.between(
                    frames[t], frames[neighbor]).half().cpu()
    del flow_model
    torch.cuda.empty_cache()
    print(f"  {len(flows)} flow fields in {time.perf_counter() - t_start:.1f}s")

    # Stage 4: per-frame edge guides and the shared identity ramp
    edges = [edge_guide(f) for f in frames]
    ramp = identity_ramp(height, width)

    # Stage 5: propagate each keyframe forward and backward to its neighbours.
    # Every frame between two keyframes gets synthesized twice, once from each side.
    forward, backward = {}, {}
    for a, b in zip(key_indices, key_indices[1:]):
        forward.update(propagate_run(keyframes[a], a, list(range(a + 1, b + 1)),
                                     frames, edges, flows, ramp, config, "fwd"))
        backward.update(propagate_run(keyframes[b], b, list(range(b - 1, a - 1, -1)),
                                      frames, edges, flows, ramp, config, "bwd"))
    # Frames outside the keyframe span are reachable from one side only
    first, last = key_indices[0], key_indices[-1]
    if first > 0:
        backward.update(propagate_run(keyframes[first], first, list(range(first - 1, -1, -1)),
                                      frames, edges, flows, ramp, config, "bwd"))
    if last < len(frames) - 1:
        forward.update(propagate_run(keyframes[last], last, list(range(last + 1, len(frames))),
                                     frames, edges, flows, ramp, config, "fwd"))

    # Stage 6: crossfade the two passes. Near a keyframe its own pass is trusted almost
    # entirely; midway between two, they are averaged.
    os.makedirs(args.outdir, exist_ok=True)
    output = []
    for t in range(len(frames)):
        if t in keyframes:
            frame = keyframes[t]
        else:
            before = [k for k in key_indices if k < t]
            after = [k for k in key_indices if k > t]
            if before and after and t in forward and t in backward:
                alpha = (t - before[-1]) / (after[0] - before[-1])
                frame = (forward[t].float() * (1 - alpha) + backward[t].float() * alpha)
                frame = frame.round().clamp(0, 255).to(torch.uint8)
            else:
                frame = forward.get(t, backward.get(t))
        output.append(frame)
        save_image_from_vram(frame, os.path.join(args.outdir, f"{t:05d}.png"))

    # Stage 7: encode
    write_video(output, args.output, fps)
    print(f"\nwrote {args.output} ({len(output)} frames) "
          f"in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
