import torch
import torch.nn.functional as F


def build_cost_weights(style_weights, guide_weights):
    """
    Concatenates style_weights + guide_weights into one CUDA float tensor. The
    original patch cost is Σ styleWeights·(styleDiff)² + Σ guideWeights·(guideDiff)²
    (PatchSSD_Split, ebsynth_cuda.cu ~line 601) — since that's just a weighted sum of
    squared per-channel diffs either way, treating style and guide channels as one
    concatenated channel axis with one concatenated weight vector is mathematically
    identical and collapses the whole patch cost into a single weighted SSD.

    Args:
        style_weights: list of C_style floats (plan_pyramid's per-style-channel weights).
        guide_weights: list of ΣC_guide floats (plan_pyramid's per-guide-channel weights).

    Returns:
        float32 CUDA tensor, shape (C_style + ΣC_guide,).
    """
    return torch.tensor(list(style_weights) + list(guide_weights), dtype=torch.float32, device="cuda")


def build_combined_source(source_style, source_guides):
    """
    Concatenates the style and source-guide tensors along the channel axis — the
    fixed half of the cost function. Built once per pyramid level since neither
    input changes while the NNF is being refined within that level.

    Args:
        source_style: uint8 CUDA tensor, shape (H_style, W_style, C_style).
        source_guides: uint8 CUDA tensor, shape (H_style, W_style, ΣC_guide).

    Returns:
        uint8 CUDA tensor, shape (H_style, W_style, C_style + ΣC_guide).
    """
    return torch.cat([source_style, source_guides], dim=-1)


def pad_target(combined_target, patch_size):
    """
    Replicate-pads the running target-side feature tensor by r on every side, so
    every patch window — even one centered on the image border — can be read as a
    plain static slice with no per-pixel bounds checking. Source-side patches never
    need this: the NNF invariant (nnf.py) already keeps every source patch center
    at least r away from the source border.

    Args:
        combined_target: uint8 or float CUDA tensor, shape (H_target, W_target, C).
        patch_size: odd int, side length of the square patch.

    Returns:
        float32 CUDA tensor, shape (H_target + 2r, W_target + 2r, C), r = patch_size // 2.
    """
    r = patch_size // 2
    chw = combined_target.permute(2, 0, 1).unsqueeze(0).float()
    padded = F.pad(chw, (r, r, r, r), mode="replicate")
    return padded.squeeze(0).permute(1, 2, 0)


def patch_cost(nnf, combined_source, combined_target_padded, weights, patch_size):
    """
    Full-field weighted patch SSD, replacing PatchSSD_Split (ebsynth_cuda.cu ~line
    601): for every target pixel q, compares the patch_size x patch_size patch
    centered at q (in combined_target) against the patch centered at nnf[q] (in
    combined_source). Same gather-per-offset trick as vote_image (synthesis/vote.py):
    for a fixed (dy, dx), the source side is one flat-index gather and the target
    side is one static slice of the padded tensor — patch_size² of these, accumulated.

    Args:
        nnf: int64 CUDA tensor, shape (H_target, W_target, 2), (y, x) source coords.
        combined_source: uint8 CUDA tensor, shape (H_source, W_source, C).
        combined_target_padded: float32 CUDA tensor, shape (H_target+2r, W_target+2r, C),
                                 as returned by pad_target.
        weights: float32 CUDA tensor, shape (C,).
        patch_size: odd int, side length of the square patch.

    Returns:
        float32 CUDA tensor, shape (H_target, W_target) — per-pixel patch cost.
    """
    r = patch_size // 2
    src_h, src_w, channels = combined_source.shape
    tgt_h, tgt_w = nnf.shape[0], nnf.shape[1]
    src_flat = combined_source.reshape(-1, channels).float()

    cost = torch.zeros((tgt_h, tgt_w), dtype=torch.float32, device=nnf.device)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            flat_idx = (nnf[..., 0] + dy) * src_w + (nnf[..., 1] + dx)
            s_val = src_flat[flat_idx]
            t_val = combined_target_padded[dy + r:dy + r + tgt_h, dx + r:dx + r + tgt_w]
            diff = t_val - s_val
            cost += (weights * diff * diff).sum(-1)
    return cost
