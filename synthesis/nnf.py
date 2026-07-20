import torch


def init_random_nnf(target_h, target_w, source_h, source_w, patch_size):
    """
    Blind first guess: every target pixel picks a uniformly random source patch center.

    Returns the NNF (Nearest-Neighbor Field) — an int64 (H_target, W_target, 2) CUDA
    tensor in (y, x) order. Both coordinates are constrained so that a full
    patch_size x patch_size window around the center always fits inside the source
    image: centers live in [r, size-1-r] with r = patch_size // 2. Every later stage
    (propagation, random search) must preserve this invariant, so vote_image can
    gather center+offset without any bounds checking.
    """
    r = patch_size // 2
    ys = torch.randint(r, source_h - r, (target_h, target_w), dtype=torch.int64, device="cuda")
    xs = torch.randint(r, source_w - r, (target_h, target_w), dtype=torch.int64, device="cuda")
    return torch.stack([ys, xs], dim=-1)
