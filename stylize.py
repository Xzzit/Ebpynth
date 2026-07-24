import sys

import torch

from arguments import parse_arguments
from utils import load_image_to_vram, save_image_from_vram, merge_guides, plan_pyramid
from synthesis import (
    Uniformity, build_combined_source, build_cost_weights, init_random_nnf,
    level_size, pad_target, patch_cost, propagate, random_search, resize_image,
    upscale_nnf, vote_image,
)


def main():
    # Stage 0: parse CLI arguments into a config dict
    config = parse_arguments(sys.argv)

    # Stage 1: load the style image, (H, W, C) uint8 CUDA tensor
    source_style = load_image_to_vram(config["style_file"])
    style_h, style_w, style_c = source_style.shape

    # Stage 2: merge all guide pairs into source/target feature tensors,
    # (H_style, W_style, ΣC) and (H_target, W_target, ΣC) uint8 CUDA
    source_guides, target_guides, guide_channels = merge_guides(
        config["guides"], source_style.shape, config["style_file"])
    target_h, target_w = target_guides.shape[0], target_guides.shape[1]

    # Stage 3: plan pyramid level count, per-level iteration counts, and per-channel
    # weights (CPU-side scalars/lists only)
    plan = plan_pyramid(config, source_style.shape, target_guides.shape, guide_channels)
    num_levels = plan["num_pyramid_levels"]
    patch_size = config["patch_size"]
    weights = build_cost_weights(plan["style_weights"], plan["guide_weights"])

    # Parity printout, matching the original CLI output (ebsynth.cpp ~430-436)
    print(f"uniformity: {config['uniformity_weight']:.0f}")
    print(f"patchsize: {patch_size}")
    print(f"pyramidlevels: {num_levels}")
    print(f"searchvoteiters: {config['num_search_vote_iters']}")
    print(f"patchmatchiters: {config['num_patch_match_iters']}")
    print(f"stopthreshold: {config['stop_threshold']}")
    print(f"extrapass3x3: {'yes' if config['extra_pass_3x3'] else 'no'}")

    # Stage 4: coarse-to-fine synthesis
    for level in range(num_levels):
        # 4a. resize style/guide tensors to this level's resolution — always resampled
        # from the original full-resolution source, never level-to-level
        lvl_style_h, lvl_style_w = level_size(style_h, style_w, num_levels, level)
        lvl_target_h, lvl_target_w = level_size(target_h, target_w, num_levels, level)
        lvl_source_style = resize_image(source_style, lvl_style_h, lvl_style_w)
        lvl_source_guides = resize_image(source_guides, lvl_style_h, lvl_style_w)
        lvl_target_guides = resize_image(target_guides, lvl_target_h, lvl_target_w)

        # 4b. seed this level's NNF: random on the coarsest level, upscaled from the
        # previous level's converged NNF otherwise
        if level == 0:
            nnf = init_random_nnf(lvl_target_h, lvl_target_w, lvl_style_h, lvl_style_w, patch_size)
        else:
            nnf = upscale_nnf(nnf, lvl_target_h, lvl_target_w, lvl_style_h, lvl_style_w, patch_size)

        # 4c. PatchMatch for this level: search-vote outer loop x patch-match inner
        # loop. combined_source is built once (constant within the level); the Omega
        # occupancy table (if uniformity is on) lives for the whole level too.
        combined_source = build_combined_source(lvl_source_style, lvl_source_guides)
        target_style = vote_image(nnf, lvl_source_style, patch_size)
        uniformity = None
        if config["uniformity_weight"] > 0:
            uniformity = Uniformity(nnf, (lvl_style_h, lvl_style_w), (lvl_target_h, lvl_target_w),
                                    patch_size, config["uniformity_weight"])

        for _ in range(plan["search_vote_iters_per_level"][level]):
            combined_target = torch.cat([target_style, lvl_target_guides], dim=-1)
            combined_target_padded = pad_target(combined_target, patch_size)
            cost = patch_cost(nnf, combined_source, combined_target_padded, weights, patch_size)
            for _ in range(plan["patch_match_iters_per_level"][level]):
                nnf, cost = propagate(nnf, cost, combined_source, combined_target_padded,
                                      weights, patch_size, uniformity)
                nnf, cost = random_search(nnf, cost, combined_source, combined_target_padded,
                                          weights, patch_size, uniformity)
            target_style = vote_image(nnf, lvl_source_style, patch_size)

    # Stage 4.9 (optional): extrapass3x3 refinement pass. Re-runs the same match/vote
    # loop on the finest level's converged NNF, forcing patch_size=3 and uniformity
    # off, replacing the original's re-entry into its own finest level
    # (ebsynth_cuda.cu ~1089-1095).
    if config["extra_pass_3x3"]:
        combined_source = build_combined_source(source_style, source_guides)
        target_style = vote_image(nnf, source_style, 3)
        for _ in range(plan["search_vote_iters_per_level"][-1]):
            combined_target_padded = pad_target(torch.cat([target_style, target_guides], dim=-1), 3)
            cost = patch_cost(nnf, combined_source, combined_target_padded, weights, 3)
            for _ in range(plan["patch_match_iters_per_level"][-1]):
                nnf, cost = propagate(nnf, cost, combined_source, combined_target_padded, weights, 3)
                nnf, cost = random_search(nnf, cost, combined_source, combined_target_padded, weights, 3)
            target_style = vote_image(nnf, source_style, 3)

    # Stage 5: save the output image
    assert target_style.shape == (target_h, target_w, style_c), \
        "engine output shape drifted from the planned canvas"
    save_image_from_vram(target_style, config["output_file"])
    print(f"result was written to {config['output_file']}")


if __name__ == "__main__":
    main()
