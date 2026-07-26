import torch
import torch.nn.functional as F


def level_size(base_h, base_w, num_levels, level):
    """
    Resolution of one pyramid level: the full-res size scaled by
    2^-(num_levels-1-level), so level 0 is the coarsest and level num_levels-1 is the
    original size.

    Float scale then truncate, not an integer right shift — the two disagree by a
    pixel at some sizes, and this must agree with plan_pyramid's level-count math.

    Args:
        base_h, base_w: full (finest) resolution.
        num_levels: total pyramid level count.
        level: 0-indexed level, 0 = coarsest, num_levels-1 = finest.

    Returns:
        (h, w) ints — this level's resolution.
    """
    scale = 2.0 ** -(num_levels - 1 - level)
    return int(base_h * scale), int(base_w * scale)


def resize_image(img, new_h, new_w):
    """
    Bilinear resample down to a pyramid level's resolution.

    Always resamples from the ORIGINAL full-resolution tensor, never progressively
    level-to-level, so repeated interpolation can't accumulate blur down the pyramid.

    Args:
        img: uint8 CUDA tensor, shape (H, W, C) — always the original full-res image.
        new_h, new_w: target resolution for this pyramid level.

    Returns:
        uint8 CUDA tensor, shape (new_h, new_w, C).
    """
    chw = img.permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)


def upscale_nnf(nnf, new_target_h, new_target_w, new_source_h, new_source_w, patch_size):
    """
    Carries a coarse level's converged NNF up to the next, roughly-2x-finer level as
    its starting guess: each new pixel inherits its coarse "parent" pixel's match,
    doubled, plus a +0/+1 jitter from (x%2, y%2).

    The jitter matters: without it a 2x2 block of children all start from the exact
    same source patch, and the finer level's search has to rediscover that diversity
    from identical priors.

    Coordinates are clamped to the project-wide [r, size-1-r] NNF invariant
    (r = patch_size // 2), same as nnf.py / propagate.py / random_search.py.

    Args:
        nnf: int64 CUDA tensor, shape (H_old_target, W_old_target, 2) — the coarse
             level's converged NNF.
        new_target_h, new_target_w: next (finer) level's target resolution.
        new_source_h, new_source_w: next (finer) level's source resolution.
        patch_size: odd int, side length of the square patch.

    Returns:
        int64 CUDA tensor, shape (new_target_h, new_target_w, 2) — starting NNF
        guess for the next level.
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


# Sandbox validation grid execution (run from the repo root: python synthesis/pyramid.py)
if __name__ == "__main__":
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    from synthesis.nnf import init_random_nnf

    # 1. level_size sanity: finest level must reproduce the exact original size,
    # and sizes must strictly increase (or stay equal) going coarse -> fine
    sizes = [level_size(540, 960, 6, lvl) for lvl in range(6)]
    print("Level sizes (540x960, 6 levels):", sizes)
    assert sizes[-1] == (540, 960), "finest level must equal the original resolution"
    assert all(sizes[i][0] <= sizes[i + 1][0] and sizes[i][1] <= sizes[i + 1][1] for i in range(5)), \
        "level sizes must be non-decreasing from coarse to fine"

    # 2. upscale_nnf sanity: a coarse NNF's structure should survive doubling —
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
