import torch

try:
    from .cost import build_cost_weights, build_combined_source, pad_target, patch_cost
    from .propagate import propagate
    from .random_search import random_search
    from .vote import vote_image
except ImportError:
    from cost import build_cost_weights, build_combined_source, pad_target, patch_cost
    from propagate import propagate
    from random_search import random_search
    from vote import vote_image


def run_patchmatch(nnf, source_style, source_guides, target_guides,
                    style_weights, guide_weights, patch_size,
                    num_search_vote_iters, num_patch_match_iters):
    """
    Single pyramid level's worth of PatchMatch, replacing the per-level body of
    ebsynthCuda's main loop (ebsynth_cuda.cu ~lines 910-1063) — minus the
    uniformity/occupancy term and the converged-pixel mask/stopthreshold skip
    (both Task I; every pixel is fully re-evaluated every iteration here) and
    minus the pyramid itself (Task H; this only ever runs at one resolution).

    Structure mirrors the original exactly:
      for vote_iter in num_search_vote_iters:
          re-derive patch cost against the current running reconstruction
          for pm_iter in num_patch_match_iters:
              propagate, then random-search
          re-vote to refresh the running reconstruction from the (now better) NNF

    Returns the final (nnf, target_style) — target_style is the running
    reconstruction, i.e. what stylize.py's output_image should become.
    """
    weights = build_cost_weights(style_weights, guide_weights)
    combined_source = build_combined_source(source_style, source_guides)
    target_style = vote_image(nnf, source_style, patch_size)

    for _ in range(num_search_vote_iters):
        combined_target = torch.cat([target_style, target_guides], dim=-1)
        combined_target_padded = pad_target(combined_target, patch_size)
        cost = patch_cost(nnf, combined_source, combined_target_padded, weights, patch_size)

        for _ in range(num_patch_match_iters):
            nnf, cost = propagate(nnf, cost, combined_source, combined_target_padded, weights, patch_size)
            nnf, cost = random_search(nnf, cost, combined_source, combined_target_padded, weights, patch_size)

        target_style = vote_image(nnf, source_style, patch_size)

    return nnf, target_style


# Sandbox validation grid execution (run from the repo root: python synthesis/patchmatch.py)
if __name__ == "__main__":
    import os
    import sys
    import time

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    from utils import load_image_to_vram, save_image_from_vram
    from synthesis.nnf import init_random_nnf

    # ① Fast correctness check on a tiny synthetic case: source_guide == target_guide
    # exactly (same random-noise image, no repeated texture to create false matches),
    # so the true global optimum is the identity NNF with zero cost. style_weights are
    # zeroed out so only the guide term drives matching — isolates PatchMatch's search
    # behavior from vote_image's own correctness (already covered in synthesis/vote.py).
    torch.manual_seed(0)
    size, patch_size = 32, 5
    noise_style = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")
    noise_guide = torch.randint(0, 256, (size, size, 3), dtype=torch.uint8, device="cuda")

    nnf = init_random_nnf(size, size, size, size, patch_size)
    style_weights = [0.0, 0.0, 0.0]
    guide_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]

    weights = build_cost_weights(style_weights, guide_weights)
    combined_source = build_combined_source(noise_style, noise_guide)
    combined_target_padded = pad_target(torch.cat([noise_style, noise_guide], dim=-1), patch_size)
    initial_cost = patch_cost(nnf, combined_source, combined_target_padded, weights, patch_size).mean().item()

    final_nnf, _ = run_patchmatch(
        nnf, noise_style, noise_guide, noise_guide,
        style_weights, guide_weights, patch_size,
        num_search_vote_iters=5, num_patch_match_iters=4)

    final_target = pad_target(torch.cat([noise_style, noise_guide], dim=-1), patch_size)
    final_cost = patch_cost(final_nnf, combined_source, final_target, weights, patch_size).mean().item()

    print(f"Synthetic identity-guide test: mean cost {initial_cost:.1f} -> {final_cost:.1f}")
    # NOTE: this floor is loose on purpose. A 2-pixel ring around the border can
    # never reach zero cost — replicate-padding fabricates edge content that has no
    # match anywhere in the valid source region — so the all-pixel mean cost has an
    # inherent floor. The real correctness bar is the interior check right below.
    assert final_cost < initial_cost * 0.3, "PatchMatch did not converge on the trivial identity case"

    r = patch_size // 2
    interior = final_nnf[r:size - r, r:size - r]
    yy, xx = torch.meshgrid(
        torch.arange(r, size - r, device="cuda"), torch.arange(r, size - r, device="cuda"), indexing="ij")
    correct = (interior[..., 0] == yy) & (interior[..., 1] == xx)
    print(f"Interior pixels recovering the true identity match: {correct.float().mean().item() * 100:.1f}%")
    assert correct.float().mean() > 0.95, "too few interior pixels recovered the identity NNF"
    print("Synthetic convergence test passed ✓")

    # ② Milestone: run on the real example pair and save the result. No pyramid yet
    # (Task H), so this works at full resolution with modest iteration counts —
    # noticeably slower and lower-quality than the final pipeline will be once
    # coarse-to-fine kicks in, but enough to show the mosaic turning into a picture.
    style = load_image_to_vram("examples/video/output_frames/000.png")
    guide_src = load_image_to_vram("examples/video/video_frames/000.jpg")
    guide_tgt = load_image_to_vram("examples/video/video_frames/001.jpg")

    h, w, style_c = style.shape
    guide_c = guide_src.shape[-1]
    patch_size = 5
    demo_style_weights = [1.0 / style_c] * style_c
    demo_guide_weights = [1.0 / guide_c] * guide_c

    nnf = init_random_nnf(h, w, h, w, patch_size)
    t0 = time.time()
    final_nnf, final_target_style = run_patchmatch(
        nnf, style, guide_src, guide_tgt,
        demo_style_weights, demo_guide_weights, patch_size,
        num_search_vote_iters=3, num_patch_match_iters=3)
    elapsed = time.time() - t0

    out_path = "examples/video/temp/task_g_result.png"
    save_image_from_vram(final_target_style, out_path)
    print(f"🏆 Task G milestone written to {out_path} ({elapsed:.1f}s, no pyramid yet — Task H speeds this up)")
