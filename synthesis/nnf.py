import torch


def init_random_nnf(target_h, target_w, source_h, source_w, patch_size):
    """
    Blind first guess: every target pixel picks a uniformly random source patch center.

    Args:
        target_h, target_w: output (target) resolution.
        source_h, source_w: style/source resolution.
        patch_size: odd int, side length of the square patch.

    Returns:
        NNF: int64 CUDA tensor, shape (target_h, target_w, 2), last dim is (y, x).
        Both coordinates are constrained to [r, size-1-r] with r = patch_size // 2, so
        a full patch_size x patch_size window around the center always fits inside the
        source image. Every later stage (propagation, random search) must preserve
        this invariant, so vote_image can gather center+offset without bounds checking.
    """
    r = patch_size // 2
    ys = torch.randint(r, source_h - r, (target_h, target_w), dtype=torch.int64, device="cuda")
    xs = torch.randint(r, source_w - r, (target_h, target_w), dtype=torch.int64, device="cuda")
    return torch.stack([ys, xs], dim=-1)
