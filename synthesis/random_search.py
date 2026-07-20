import torch

try:
    from .cost import patch_cost
except ImportError:
    from cost import patch_cost


def random_search(nnf, cost, combined_source, combined_target_padded, weights, patch_size):
    """
    Replaces the growing-radius trial loop around krnlRandomSearchPass
    (ebsynth_cuda.cu ~line 355): for r = 1, 2, 4, 8, ... (doubling until half the
    source's largest dimension), every pixel samples ONE random candidate within
    +-r of its current match and keeps it if it's cheaper. Propagation alone only
    ever spreads matches that already exist somewhere in the image; this is what
    lets PatchMatch discover genuinely new, better matches and escape local optima.
    """
    src_h, src_w, _ = combined_source.shape
    tgt_h, tgt_w = nnf.shape[0], nnf.shape[1]
    r_patch = patch_size // 2
    max_radius = max(src_h, src_w) // 2

    radius = 1
    while radius < max_radius:
        offset_y = torch.randint(-radius, radius + 1, (tgt_h, tgt_w), device=nnf.device)
        offset_x = torch.randint(-radius, radius + 1, (tgt_h, tgt_w), device=nnf.device)

        cand_y = torch.clamp(nnf[..., 0] + offset_y, r_patch, src_h - 1 - r_patch)
        cand_x = torch.clamp(nnf[..., 1] + offset_x, r_patch, src_w - 1 - r_patch)
        cand_nnf = torch.stack([cand_y, cand_x], dim=-1)

        cand_cost = patch_cost(cand_nnf, combined_source, combined_target_padded, weights, patch_size)
        improved = cand_cost < cost
        nnf = torch.where(improved.unsqueeze(-1), cand_nnf, nnf)
        cost = torch.where(improved, cand_cost, cost)

        radius *= 2

    return nnf, cost
