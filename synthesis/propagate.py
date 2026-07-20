import torch

try:
    from .cost import patch_cost
except ImportError:
    from cost import patch_cost


def propagate(nnf, cost, combined_source, combined_target_padded, weights, patch_size, uniformity=None):
    """
    Jump-flood propagation, replacing krnlPropagationPass (ebsynth_cuda.cu ~line 187).
    The original runs on a serial scanline — each pixel reads a neighbor that was
    already updated earlier in the same pass, letting a good match crawl across the
    whole image in one pass. A fully parallel rewrite can't rely on that ordering
    (every pixel updates simultaneously), so — exactly like the original CUDA
    version already does to make its own execution parallel-friendly — propagation
    happens at shrinking jump distances (4, then 2, then 1) instead of a single
    1-pixel step, so information can still cross the image in a handful of passes.

    For each jump distance r and each of 4 directions (-r,0)/(+r,0)/(0,-r)/(0,+r):
    "if my neighbor r pixels away is using source position s for itself, then
    s - offset is what my own patch would use if it shared that neighbor's alignment
    — worth trying." Candidates are tried one direction at a time (not batched),
    each immediately updating the running best — so a later direction's comparison
    (and, with uniformity enabled, its occupancy bookkeeping) always sees the
    outcome of the earlier ones, mirroring tryNeighborsOffset's sequential calls.
    """
    src_h, src_w, _ = combined_source.shape
    tgt_h, tgt_w = nnf.shape[0], nnf.shape[1]
    r_patch = patch_size // 2
    yy, xx = torch.meshgrid(
        torch.arange(tgt_h, device=nnf.device), torch.arange(tgt_w, device=nnf.device), indexing="ij")

    for jump in (4, 2, 1):
        best_nnf, best_cost = nnf, cost
        for oy, ox in ((-jump, 0), (jump, 0), (0, -jump), (0, jump)):
            # Clamped (not wrapped) neighbor lookup — a wraparound would pull in a
            # candidate from the opposite edge of the image, which is meaningless.
            ny = torch.clamp(yy + oy, 0, tgt_h - 1)
            nx = torch.clamp(xx + ox, 0, tgt_w - 1)
            neighbor_val = nnf[ny, nx]  # (H, W, 2): what my neighbor currently uses

            cand_y = neighbor_val[..., 0] - oy
            cand_x = neighbor_val[..., 1] - ox
            valid = (cand_y >= r_patch) & (cand_y <= src_h - 1 - r_patch) & \
                    (cand_x >= r_patch) & (cand_x <= src_w - 1 - r_patch)
            # Clamp so the gather below is always safe; invalid candidates are
            # rejected anyway via the +inf cost override two lines down.
            cand_nnf = torch.stack([
                torch.clamp(cand_y, r_patch, src_h - 1 - r_patch),
                torch.clamp(cand_x, r_patch, src_w - 1 - r_patch),
            ], dim=-1)

            cand_cost = patch_cost(cand_nnf, combined_source, combined_target_padded, weights, patch_size)
            cand_cost = torch.where(valid, cand_cost, torch.full_like(cand_cost, float("inf")))

            if uniformity is None:
                improved = cand_cost < best_cost
            else:
                improved = uniformity.score(cand_cost, cand_nnf) < uniformity.score(best_cost, best_nnf)
                uniformity.update(best_nnf, cand_nnf, improved)

            best_nnf = torch.where(improved.unsqueeze(-1), cand_nnf, best_nnf)
            best_cost = torch.where(improved, cand_cost, best_cost)

        nnf, cost = best_nnf, best_cost

    return nnf, cost
