import torch

try:
    from .cost import patch_cost
except ImportError:
    from cost import patch_cost


def propagate(nnf, cost, combined_source, combined_target_padded, patch_size, uniformity=None):
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

    Args:
        nnf: int64 CUDA tensor, shape (H_target, W_target, 2), (y, x) source coords.
        cost: float32 CUDA tensor, shape (H_target, W_target) — nnf's current cost.
        combined_source: float32 CUDA tensor, shape (H_source, W_source, C), √w-scaled.
        combined_target_padded: float32 CUDA tensor, shape (H_target+2r, W_target+2r, C).
        patch_size: odd int, side length of the square patch.
        uniformity: optional Uniformity instance; None disables the occupancy penalty.

    Returns:
        (nnf, cost) — same shapes/dtypes as the inputs, updated in place of a copy.
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

            cand_cost = patch_cost(cand_nnf, combined_source, combined_target_padded, patch_size)
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


# Sandbox validation grid execution (run from the repo root: python synthesis/propagate.py)
if __name__ == "__main__":
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    from synthesis.cost import build_channel_scales, build_combined_source, pad_target
    from synthesis.nnf import init_random_nnf
    from synthesis.random_search import random_search

    # Synthetic convergence check of the PatchMatch core (propagate + random_search
    # together): source_guide == target_guide exactly (same random-noise image, no
    # repeated texture to create false matches), so the true global optimum is the
    # identity NNF with zero cost. style_weights are zeroed so only the guide term
    # drives matching. (Formerly run_patchmatch's sandbox; the match/vote loop now
    # lives inline in stylize.py, so the core iteration is tested here instead.)
    torch.manual_seed(0)
    size, patch_size = 32, 5
    noise_style = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")
    noise_guide = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")

    nnf = init_random_nnf(size, size, size, size, patch_size)
    scales = build_channel_scales([0.0, 0.0, 0.0], [1.0 / 3, 1.0 / 3, 1.0 / 3])
    combined_source = build_combined_source(noise_style, noise_guide, scales)
    combined_target_padded = pad_target(torch.cat([noise_style, noise_guide], dim=-1), patch_size, scales)

    cost = patch_cost(nnf, combined_source, combined_target_padded, patch_size)
    initial_cost = cost.mean().item()
    for _ in range(20):
        nnf, cost = propagate(nnf, cost, combined_source, combined_target_padded, patch_size)
        nnf, cost = random_search(nnf, cost, combined_source, combined_target_padded, patch_size)
    final_cost = cost.mean().item()

    print(f"Synthetic identity-guide test: mean cost {initial_cost:.1f} -> {final_cost:.1f}")
    # NOTE: this floor is loose on purpose. A 2-pixel ring around the border can
    # never reach zero cost — replicate-padding fabricates edge content that has no
    # match anywhere in the valid source region — so the all-pixel mean cost has an
    # inherent floor. The real correctness bar is the interior check right below.
    assert final_cost < initial_cost * 0.3, "PatchMatch did not converge on the trivial identity case"

    r = patch_size // 2
    interior = nnf[r:size - r, r:size - r]
    yy, xx = torch.meshgrid(
        torch.arange(r, size - r, device="cuda"), torch.arange(r, size - r, device="cuda"), indexing="ij")
    correct = (interior[..., 0] == yy) & (interior[..., 1] == xx)
    print(f"Interior pixels recovering the true identity match: {correct.float().mean().item() * 100:.1f}%")
    assert correct.float().mean() > 0.95, "too few interior pixels recovered the identity NNF"
    print("Synthetic convergence test passed ✓")
