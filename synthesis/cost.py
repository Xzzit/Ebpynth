import torch
import torch.nn.functional as F


def build_channel_scales(style_weights, guide_weights):
    """
    Per-channel √weight — the scale factor folded into both feature tensors.

    Style and guide channels are treated as one concatenated channel axis carrying one
    concatenated weight vector, which collapses the whole patch cost into a single
    weighted SSD; scoring the two terms separately and adding them is mathematically
    identical, since either way it is a weighted sum of squared per-channel diffs.

    Why sqrt: Σ w_c·(t_c - s_c)² is identically Σ (√w_c·t_c - √w_c·s_c)². Folding √w
    into BOTH feature tensors once per pyramid level (build_combined_source and
    pad_target below) turns the weighted SSD into a plain SSD, so patch_cost's hot
    loop drops the per-offset weight multiply entirely. That loop runs patch_size²
    times per call and thousands of times per synthesis, and the engine is bound by
    op-dispatch overhead rather than bandwidth, so removing ops from it is what
    actually buys time.

    Args:
        style_weights: list of C_style floats (plan_pyramid's per-style-channel weights).
        guide_weights: list of ΣC_guide floats (plan_pyramid's per-guide-channel weights).

    Returns:
        float32 CUDA tensor, shape (C_style + ΣC_guide,) — √weight per channel.
    """
    weights = torch.tensor(list(style_weights) + list(guide_weights), dtype=torch.float32, device="cuda")
    return weights.sqrt()


def build_combined_source(source_style, source_guides, scales):
    """
    Concatenates the style and source-guide tensors along the channel axis and
    pre-scales them by √weight — the fixed half of the cost function.

    Built once per pyramid level, since neither input changes while the NNF is refined
    within a level. That also hoists the uint8→float32 conversion out of patch_cost,
    which would otherwise redo it on every call.

    Args:
        source_style: uint8 CUDA tensor, shape (H_style, W_style, C_style).
        source_guides: uint8 CUDA tensor, shape (H_style, W_style, ΣC_guide).
        scales: float32 CUDA tensor, shape (C,), from build_channel_scales.

    Returns:
        float32 CUDA tensor, shape (H_style, W_style, C_style + ΣC_guide), √w-scaled.
    """
    return torch.cat([source_style, source_guides], dim=-1).float() * scales


def pad_target(combined_target, patch_size, scales):
    """
    Pre-scales the running target-side feature tensor by √weight, then replicate-pads
    it by r on every side.

    The padding is what lets every patch window — even one centered on the image
    border — be read as a plain static slice with no per-pixel bounds checking.
    Source-side patches never need it: the NNF invariant (nnf.py) already keeps every
    source patch center at least r away from the source border.

    The result is made contiguous: patch_cost slices it patch_size² times per call,
    and a permuted (non-contiguous) view makes every one of those reads strided.

    Args:
        combined_target: uint8 or float CUDA tensor, shape (H_target, W_target, C).
        patch_size: odd int, side length of the square patch.
        scales: float32 CUDA tensor, shape (C,), from build_channel_scales.

    Returns:
        float32 CUDA tensor, shape (H_target + 2r, W_target + 2r, C), r = patch_size // 2.
    """
    r = patch_size // 2
    scaled = combined_target.float() * scales
    chw = scaled.permute(2, 0, 1).unsqueeze(0)
    padded = F.pad(chw, (r, r, r, r), mode="replicate")
    return padded.squeeze(0).permute(1, 2, 0).contiguous()


def patch_cost(nnf, combined_source, combined_target_padded, patch_size):
    """
    Full-field patch SSD: for every target pixel q, compares the patch_size x
    patch_size patch centered at q (target side) against the patch centered at nnf[q]
    (source side).

    Same gather-per-offset trick as vote_image (synthesis/vote.py): for a fixed
    (dy, dx), the source side is one flat-index gather and the target side is one
    static slice of the padded tensor — patch_size² of these, accumulated.

    No weight argument: both inputs arrive pre-scaled by √weight, so a plain squared
    difference already IS the weighted SSD (see build_channel_scales).

    Batching all patch_size² offsets into one gather was tried and is a large net
    loss — it materialises a (H·W, patch_size², C) intermediate (1.5 GB at 540x960,
    8 GB on the stylit example) and the extra memory traffic costs far more than the
    saved dispatches. Keep the loop streaming.

    Args:
        nnf: int64 CUDA tensor, shape (H_target, W_target, 2), (y, x) source coords.
        combined_source: float32 CUDA tensor, shape (H_source, W_source, C), √w-scaled.
        combined_target_padded: float32 CUDA tensor, shape (H_target+2r, W_target+2r, C),
                                 as returned by pad_target.
        patch_size: odd int, side length of the square patch.

    Returns:
        float32 CUDA tensor, shape (H_target, W_target) — per-pixel patch cost.
    """
    r = patch_size // 2
    src_h, src_w, channels = combined_source.shape
    tgt_h, tgt_w = nnf.shape[0], nnf.shape[1]
    src_flat = combined_source.reshape(-1, channels)

    # Flat source index of each patch CENTER, hoisted: the per-offset index is then
    # a single add of the scalar (dy*src_w + dx) instead of rebuilding it from the
    # NNF's two coordinate planes every iteration.
    base = nnf[..., 0] * src_w + nnf[..., 1]

    cost = torch.zeros((tgt_h, tgt_w), dtype=torch.float32, device=nnf.device)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            s_val = src_flat[base + (dy * src_w + dx)]
            t_val = combined_target_padded[dy + r:dy + r + tgt_h, dx + r:dx + r + tgt_w]
            diff = t_val - s_val
            cost += diff.square().sum(-1)
    return cost
