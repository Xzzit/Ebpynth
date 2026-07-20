import torch


def compute_omega(nnf, source_shape, patch_size):
    """
    Occupancy count per source pixel: how many currently-active target patches
    claim to draw from it, replacing the Omega initialization loop
    (ebsynth_cuda.cu ~lines 885-906). Mirror image of patch_cost's gather loop
    (cost.py): same nested (dy, dx) offset loop, but scattering +1 INTO the source
    instead of gathering values FROM it — "which source pixels does this NNF
    touch" genuinely needs a scatter, unlike vote_image's "who covers me" question,
    which had a fixed slice per offset because it asked about the TARGET side.
    """
    r = patch_size // 2
    src_h, src_w = source_shape
    omega = torch.zeros(src_h * src_w, dtype=torch.float32, device=nnf.device)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            flat_idx = ((nnf[..., 0] + dy) * src_w + (nnf[..., 1] + dx)).reshape(-1)
            omega.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    return omega.reshape(src_h, src_w)


def ideal_omega(target_shape, source_shape, patch_size):
    """
    The per-window occupancy a source patch would carry under perfectly uniform
    usage, replacing tryPatch's omegaBest (ebsynth_cuda.cu ~line 140): the target's
    pixels collectively make target_area * patch_size² claims; spread evenly over
    the source, patch_size² of those land on any given patch_size x patch_size
    window on average.
    """
    target_area = target_shape[0] * target_shape[1]
    source_area = source_shape[0] * source_shape[1]
    return (target_area / source_area) * (patch_size * patch_size)


class Uniformity:
    """
    Bundles the occupancy state (Omega) and the uniformity penalty weight so
    propagate/random_search can factor "don't overuse this source patch" into
    their accept/reject decisions, replacing tryPatch's lambda*occupancy term
    (ebsynth_cuda.cu ~lines 95-156). One instance lives for an entire pyramid
    level (constructed once from that level's starting NNF, updated incrementally
    as matches change across every vote/patchmatch iteration of that level) —
    pass uniformity=None anywhere it's accepted to disable the term entirely
    (Task G/H's original behavior, still the default).
    """

    def __init__(self, nnf, source_shape, target_shape, patch_size, weight):
        self.omega = compute_omega(nnf, source_shape, patch_size)
        self.ideal = ideal_omega(target_shape, source_shape, patch_size)
        self.weight = weight
        self.patch_size = patch_size

    def _patch_omega_sum(self, nnf):
        r = self.patch_size // 2
        src_w = self.omega.shape[1]
        omega_flat = self.omega.reshape(-1)
        total = torch.zeros(nnf.shape[0], nnf.shape[1], dtype=torch.float32, device=nnf.device)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                flat_idx = (nnf[..., 0] + dy) * src_w + (nnf[..., 1] + dx)
                total += omega_flat[flat_idx]
        return total

    def score(self, cost, nnf):
        occupancy_ratio = self._patch_omega_sum(nnf) / (self.patch_size * self.patch_size) / self.ideal
        return cost + self.weight * occupancy_ratio

    def update(self, old_nnf, new_nnf, changed):
        """
        Moves each changed pixel's occupancy claim from its old source patch to
        its new one: -1 across the old window, +1 across the new one, scattered
        (not sliced) since old/new positions vary per pixel. scatter_add_ correctly
        accumulates when several pixels happen to land on the same source pixel in
        the same call, unlike a plain indexed write would.
        """
        r = self.patch_size // 2
        src_w = self.omega.shape[1]
        omega_flat = self.omega.reshape(-1)
        delta = changed.reshape(-1).float()
        old_y, old_x = old_nnf[..., 0].reshape(-1), old_nnf[..., 1].reshape(-1)
        new_y, new_x = new_nnf[..., 0].reshape(-1), new_nnf[..., 1].reshape(-1)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                old_idx = (old_y + dy) * src_w + (old_x + dx)
                new_idx = (new_y + dy) * src_w + (new_x + dx)
                omega_flat.scatter_add_(0, old_idx, -delta)
                omega_flat.scatter_add_(0, new_idx, delta)


# Sandbox validation grid execution (run from the repo root: python synthesis/uniformity.py)
if __name__ == "__main__":
    import os
    import sys
    import time

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    from utils import load_image_to_vram, save_image_from_vram, merge_guides, plan_pyramid
    from arguments import parse_arguments
    from synthesis.nnf import init_random_nnf
    from synthesis.patchmatch import run_patchmatch
    from synthesis.pyramid import run_pyramid

    # ① Quantitative check: source deliberately much smaller than target (16x16 vs
    # 40x40, both unrelated random noise so no "true" correspondence exists) forces
    # heavy source-patch reuse by pigeonhole. style_weights are zeroed so only the
    # guide term drives matching, isolating the uniformity effect from vote's own
    # behavior (already covered by synthesis/vote.py and synthesis/patchmatch.py).
    torch.manual_seed(0)
    src_size, tgt_size, patch_size = 16, 40, 5
    noise_style = torch.randint(0, 256, (src_size, src_size, 3), dtype=torch.uint8, device="cuda")
    noise_source_guide = torch.randint(0, 256, (src_size, src_size, 3), dtype=torch.uint8, device="cuda")
    noise_target_guide = torch.randint(0, 256, (tgt_size, tgt_size, 3), dtype=torch.uint8, device="cuda")
    style_weights = [0.0, 0.0, 0.0]
    guide_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]

    stats = {}
    for uw in (0.0, 3500.0):
        torch.manual_seed(1)
        nnf = init_random_nnf(tgt_size, tgt_size, src_size, src_size, patch_size)
        final_nnf, _ = run_patchmatch(
            nnf, noise_style, noise_source_guide, noise_target_guide,
            style_weights, guide_weights, patch_size,
            num_search_vote_iters=6, num_patch_match_iters=4, uniformity_weight=uw)
        omega = compute_omega(final_nnf, (src_size, src_size), patch_size)
        stats[uw] = (omega.max().item(), omega.var().item(), omega.mean().item())
        print(f"uniformity_weight={uw}: max={stats[uw][0]:.1f} var={stats[uw][1]:.1f} mean={stats[uw][2]:.1f}")

    assert abs(stats[0.0][2] - stats[3500.0][2]) < 1e-2, \
        "total occupancy (mean) must be conserved regardless of uniformity — only its spread should change"
    assert stats[3500.0][1] < stats[0.0][1] * 0.9, \
        "uniformity_weight did not meaningfully flatten the occupancy distribution"
    assert stats[3500.0][0] < stats[0.0][0], "uniformity_weight did not reduce peak source-patch overuse"
    print("Uniformity flattens occupancy without changing total usage ✓")

    # ② Milestone: the full engine with the CLI's actual default uniformity_weight
    # (3500) vs. it switched off, on the real example pair — visually, uniformity
    # mainly shows up as reduced "stamping" of one favorite source patch across
    # unrelated regions of the output.
    config = parse_arguments([
        "stylize.py",
        "-style", "examples/video/output_frames/000.png",
        "-guide", "examples/video/video_frames/000.jpg", "examples/video/video_frames/001.jpg",
        "-output", "examples/video/temp/task_i_result.png",
    ])
    style = load_image_to_vram(config["style_file"])
    source_guides, target_guides, guide_channels = merge_guides(config["guides"], style.shape, config["style_file"])
    plan = plan_pyramid(config, style.shape, target_guides.shape, guide_channels)

    t0 = time.time()
    final_nnf, final_target_style = run_pyramid(
        style, source_guides, target_guides,
        plan["style_weights"], plan["guide_weights"], config["patch_size"],
        plan["num_pyramid_levels"], plan["search_vote_iters_per_level"], plan["patch_match_iters_per_level"],
        uniformity_weight=config["uniformity_weight"])
    elapsed = time.time() - t0

    save_image_from_vram(final_target_style, config["output_file"])
    omega = compute_omega(final_nnf, (style.shape[0], style.shape[1]), config["patch_size"])
    print(f"🏆 Task I milestone written to {config['output_file']} "
          f"({elapsed:.1f}s, uniformity_weight={config['uniformity_weight']}, "
          f"final source-occupancy max={omega.max().item():.1f} var={omega.var().item():.1f})")
