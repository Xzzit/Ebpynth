import torch
import torch.nn.functional as F

try:
    from .nnf import init_random_nnf
    from .patchmatch import run_patchmatch
except ImportError:
    from nnf import init_random_nnf
    from patchmatch import run_patchmatch


def level_size(base_h, base_w, num_levels, level):
    """
    Replicates pyramidLevelSize (ebsynth_cuda.cu ~line 742): scales the full-res
    size by 2^-(numLevels-1-level), so level=numLevels-1 (last, finest) is the
    original size and level=0 (first) is the coarsest. int() truncates exactly
    like the original's V2f -> V2i cast.
    """
    scale = 2.0 ** -(num_levels - 1 - level)
    return int(base_h * scale), int(base_w * scale)


def resize_image(img, new_h, new_w):
    """
    Bilinear resample, replacing krnlResampleBilinear (ebsynth_cuda.cu ~line 524).
    Always resamples from the ORIGINAL full-resolution tensor down to a level's
    size (not progressively level-to-level), matching the original's resampleGPU
    calls, which all source from pyramid[levelCount-1].
    """
    chw = img.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)


def upscale_nnf(nnf, new_target_h, new_target_w, new_source_h, new_source_w, patch_size):
    """
    Carries a coarse level's converged NNF up to the next, roughly-2x-finer level
    as its starting guess, replacing nnfUpscale (ebsynth_cuda.cu ~line 387): each
    new pixel inherits its coarse "parent" pixel's match, doubled, with a +0/+1
    jitter from (x%2, y%2) so a 2x2 block of children doesn't all collapse onto
    the exact same starting patch — cheap free diversity for the next level's
    search to build on, instead of every child needing to rediscover it from
    identical priors.

    Bounds are clamped to this project's [r, size-1-r] NNF invariant (r =
    patch_size // 2), same as nnf.py/propagate.py/random_search.py. Note: the
    original clamps to [patchSize, size-1-patchSize] here specifically — a
    stricter, inconsistent-with-its-own-nnfInitRandom margin. We keep one
    invariant everywhere instead of replicating that asymmetry; it's a strict
    subset of the safe range, so nothing breaks, it's simply not bug-for-bug here.
    """
    old_h, old_w = nnf.shape[0], nnf.shape[1]
    yy, xx = torch.meshgrid(
        torch.arange(new_target_h, device=nnf.device),
        torch.arange(new_target_w, device=nnf.device), indexing="ij")

    py = torch.clamp(yy // 2, 0, old_h - 1)
    px = torch.clamp(xx // 2, 0, old_w - 1)
    parent = nnf[py, px]

    child_y = parent[..., 0] * 2 + (yy % 2)
    child_x = parent[..., 1] * 2 + (xx % 2)

    r = patch_size // 2
    child_y = torch.clamp(child_y, r, new_source_h - 1 - r)
    child_x = torch.clamp(child_x, r, new_source_w - 1 - r)
    return torch.stack([child_y, child_x], dim=-1)


def run_pyramid(source_style, source_guides, target_guides,
                 style_weights, guide_weights, patch_size,
                 num_pyramid_levels, search_vote_iters_per_level, patch_match_iters_per_level,
                 uniformity_weight=0.0, extra_pass_3x3=False):
    """
    Coarse-to-fine driver, replacing the per-level loop shell in ebsynthCuda
    (ebsynth_cuda.cu ~lines 828-1064) around the single-level engine already
    built in Task G. Runs run_patchmatch once per level, level 0 (coarsest)
    first: each level resamples the ORIGINAL full-res style/guides down to that
    level's size, starts from either a random NNF (level 0) or the previous
    level's NNF carried up via upscale_nnf, refines it with run_patchmatch, and
    hands the result to the next, finer level. The last level is full resolution,
    so its returned target_style is the final synthesized image — unless
    extra_pass_3x3 is set, in which case one more run_patchmatch call refines it
    further at patch_size=3 with uniformity disabled (see the original's
    inExtraPass re-entry into its finest level, ebsynth_cuda.cu ~lines 1089-1095),
    and that becomes the final result instead.
    """
    style_h, style_w = source_style.shape[0], source_style.shape[1]
    target_h, target_w = target_guides.shape[0], target_guides.shape[1]

    nnf = None
    target_style = None
    for level in range(num_pyramid_levels):
        lvl_style_h, lvl_style_w = level_size(style_h, style_w, num_pyramid_levels, level)
        lvl_target_h, lvl_target_w = level_size(target_h, target_w, num_pyramid_levels, level)

        lvl_source_style = resize_image(source_style, lvl_style_h, lvl_style_w)
        lvl_source_guides = resize_image(source_guides, lvl_style_h, lvl_style_w)
        lvl_target_guides = resize_image(target_guides, lvl_target_h, lvl_target_w)

        if level == 0:
            nnf = init_random_nnf(lvl_target_h, lvl_target_w, lvl_style_h, lvl_style_w, patch_size)
        else:
            nnf = upscale_nnf(nnf, lvl_target_h, lvl_target_w, lvl_style_h, lvl_style_w, patch_size)

        nnf, target_style = run_patchmatch(
            nnf, lvl_source_style, lvl_source_guides, lvl_target_guides,
            style_weights, guide_weights, patch_size,
            search_vote_iters_per_level[level], patch_match_iters_per_level[level],
            uniformity_weight)

    if extra_pass_3x3:
        # Same finest-level iteration counts, sharper 3x3 patches, uniformity off
        # (3x3 patches are too small for occupancy-fighting to make sense) —
        # continues refining from the finest level's own (nnf, target_style)
        # rather than starting over, using the ORIGINAL full-res tensors (the
        # finest level's resized copies are identical in content, but this avoids
        # depending on loop-scoped variables surviving past the for-loop).
        nnf, target_style = run_patchmatch(
            nnf, source_style, source_guides, target_guides,
            style_weights, guide_weights, patch_size=3,
            num_search_vote_iters=search_vote_iters_per_level[-1],
            num_patch_match_iters=patch_match_iters_per_level[-1],
            uniformity_weight=0.0)

    return nnf, target_style


# Sandbox validation grid execution (run from the repo root: python synthesis/pyramid.py)
if __name__ == "__main__":
    import os
    import sys
    import time

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    from utils import load_image_to_vram, save_image_from_vram, merge_guides, plan_pyramid
    from arguments import parse_arguments

    # ① level_size sanity: finest level must reproduce the exact original size,
    # and sizes must strictly increase (or stay equal) going coarse -> fine
    sizes = [level_size(540, 960, 6, lvl) for lvl in range(6)]
    print("Level sizes (540x960, 6 levels):", sizes)
    assert sizes[-1] == (540, 960), "finest level must equal the original resolution"
    assert all(sizes[i][0] <= sizes[i + 1][0] and sizes[i][1] <= sizes[i + 1][1] for i in range(5)), \
        "level sizes must be non-decreasing from coarse to fine"

    # ② upscale_nnf sanity: a coarse NNF's structure should survive doubling —
    # each 2x2 child block's floor-halved value must recover its exact parent
    coarse_nnf = init_random_nnf(4, 4, 4, 4, patch_size=3)
    fine_nnf = upscale_nnf(coarse_nnf, 8, 8, 8, 8, patch_size=3)
    assert fine_nnf.shape == (8, 8, 2)
    recovered_parent_y = fine_nnf[..., 0] // 2
    recovered_parent_x = fine_nnf[..., 1] // 2
    # Allow off-by-clamp at the coarse NNF's own borders (clamping after *2 can
    # shift a value away from its unclamped parent*2+jitter); check the bulk matches
    yy, xx = torch.meshgrid(torch.arange(8, device="cuda"), torch.arange(8, device="cuda"), indexing="ij")
    py, px = torch.clamp(yy // 2, 0, 3), torch.clamp(xx // 2, 0, 3)
    expected_parent = coarse_nnf[py, px]
    match_rate = ((recovered_parent_y == expected_parent[..., 0]) &
                  (recovered_parent_x == expected_parent[..., 1])).float().mean().item()
    print(f"upscale_nnf parent-recovery rate: {match_rate * 100:.1f}%")
    assert match_rate > 0.8, "upscaled NNF drifted too far from its coarse parent"
    print("Pyramid math sanity checks passed ✓")

    # ③ Milestone: the full engine end-to-end with REAL config-derived hyperparameters
    # (not the light-weight demo settings from Task G) — first real look at what the
    # coarse-to-fine pipeline buys over Task G's single full-res-only milestone.
    config = parse_arguments([
        "stylize.py",
        "-style", "examples/video/output_frames/000.png",
        "-guide", "examples/video/video_frames/000.jpg", "examples/video/video_frames/001.jpg",
        "-output", "examples/video/temp/task_h_result.png",
    ])
    style = load_image_to_vram(config["style_file"])
    source_guides, target_guides, guide_channels = merge_guides(config["guides"], style.shape, config["style_file"])
    plan = plan_pyramid(config, style.shape, target_guides.shape, guide_channels)

    print(f"Running full pyramid: {plan['num_pyramid_levels']} levels, "
          f"{config['num_search_vote_iters']} vote iters x {config['num_patch_match_iters']} pm iters per level")
    t0 = time.time()
    final_nnf, final_target_style = run_pyramid(
        style, source_guides, target_guides,
        plan["style_weights"], plan["guide_weights"], config["patch_size"],
        plan["num_pyramid_levels"], plan["search_vote_iters_per_level"], plan["patch_match_iters_per_level"])
    elapsed = time.time() - t0

    assert final_target_style.shape == style.shape, "final level's output must match the style image's channel count"
    save_image_from_vram(final_target_style, config["output_file"])
    print(f"🏆 Task H milestone written to {config['output_file']} ({elapsed:.1f}s, full pyramid, real default hyperparameters)")

    # ④ Task J milestone: extra_pass_3x3 on vs off, real example, quantified via a
    # Laplacian-variance sharpness proxy (higher = more fine detail) rather than
    # just eyeballing it — confirms the pass genuinely sharpens, not just perturbs.
    def sharpness(img):
        gray = img.float().mean(dim=-1, keepdim=True).permute(2, 0, 1).unsqueeze(0)
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=img.device).view(1, 1, 3, 3)
        return F.conv2d(gray, kernel, padding=1).var().item()

    results = {}
    for extra in (False, True):
        # Same seed before each run so both go through an IDENTICAL coarse-to-fine
        # synthesis (same random NNF inits, same random-search draws at every level)
        # and only diverge at the optional extra pass — otherwise two independent
        # random runs' natural variance would swamp the (real but smaller) effect
        # being measured here.
        torch.manual_seed(42)
        _, out = run_pyramid(
            style, source_guides, target_guides,
            plan["style_weights"], plan["guide_weights"], config["patch_size"],
            plan["num_pyramid_levels"], plan["search_vote_iters_per_level"], plan["patch_match_iters_per_level"],
            uniformity_weight=config["uniformity_weight"], extra_pass_3x3=extra)
        suffix = "extrapass" if extra else "noextrapass"
        out_path = f"examples/video/temp/task_j_{suffix}.png"
        save_image_from_vram(out, out_path)
        results[extra] = (out, sharpness(out))
        print(f"extrapass3x3={extra}: sharpness={results[extra][1]:.2f} -> {out_path}")

    assert not torch.equal(results[False][0], results[True][0]), \
        "extrapass3x3 must actually change the output, not be a no-op"
    assert results[True][1] > results[False][1], \
        "extrapass3x3's smaller 3x3 patches should sharpen fine detail, not blur it"
    print("🏆 Task J milestone passed: extrapass3x3 measurably sharpens the finest level's result ✓")
