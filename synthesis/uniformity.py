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

    Args:
        nnf: int64 CUDA tensor, shape (H_target, W_target, 2), (y, x) source coords.
        source_shape: (H_source, W_source) tuple.
        patch_size: odd int, side length of the square patch.

    Returns:
        float32 CUDA tensor, shape (H_source, W_source) — occupancy count per pixel.
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

    Args:
        target_shape: (H_target, W_target) tuple.
        source_shape: (H_source, W_source) tuple.
        patch_size: odd int, side length of the square patch.

    Returns:
        float — the ideal (uniform-usage) occupancy scalar.
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
        """
        Args:
            nnf: int64 CUDA tensor, shape (H_target, W_target, 2) — this level's
                 starting NNF, used to seed the occupancy table.
            source_shape: (H_source, W_source) tuple.
            target_shape: (H_target, W_target) tuple.
            patch_size: odd int, side length of the square patch.
            weight: float, the uniformity penalty weight (-uniformity CLI value).
        """
        self.omega = compute_omega(nnf, source_shape, patch_size)
        self.ideal = ideal_omega(target_shape, source_shape, patch_size)
        self.weight = weight
        self.patch_size = patch_size

    def _patch_omega_sum(self, nnf):
        """
        Args:
            nnf: int64 CUDA tensor, shape (H_target, W_target, 2).

        Returns:
            float32 CUDA tensor, shape (H_target, W_target) — summed occupancy
            over each nnf-indicated patch window.
        """
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
        """
        Args:
            cost: float32 CUDA tensor, shape (H_target, W_target) — raw patch cost.
            nnf: int64 CUDA tensor, shape (H_target, W_target, 2).

        Returns:
            float32 CUDA tensor, shape (H_target, W_target) — cost plus the
            occupancy penalty, used in place of raw cost for accept/reject decisions.
        """
        occupancy_ratio = self._patch_omega_sum(nnf) / (self.patch_size * self.patch_size) / self.ideal
        return cost + self.weight * occupancy_ratio

    def update(self, old_nnf, new_nnf, changed):
        """
        Moves each changed pixel's occupancy claim from its old source patch to
        its new one: -1 across the old window, +1 across the new one, scattered
        (not sliced) since old/new positions vary per pixel. scatter_add_ correctly
        accumulates when several pixels happen to land on the same source pixel in
        the same call, unlike a plain indexed write would.

        Args:
            old_nnf, new_nnf: int64 CUDA tensors, shape (H_target, W_target, 2).
            changed: bool CUDA tensor, shape (H_target, W_target) — which pixels
                     actually moved from old_nnf to new_nnf.
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

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    from synthesis.cost import build_cost_weights, build_combined_source, pad_target, patch_cost
    from synthesis.nnf import init_random_nnf
    from synthesis.propagate import propagate
    from synthesis.random_search import random_search
    from synthesis.vote import vote_image

    # Quantitative check: source deliberately much smaller than target (16x16 vs
    # 40x40, both unrelated random noise so no "true" correspondence exists) forces
    # heavy source-patch reuse by pigeonhole. style_weights are zeroed so only the
    # guide term drives matching. The match/vote loop below mirrors stylize.py's
    # single-level structure (where the real loop now lives inline).
    torch.manual_seed(0)
    src_size, tgt_size, patch_size = 16, 40, 5
    noise_style = torch.randint(0, 256, (src_size, src_size, 3), dtype=torch.uint8, device="cuda")
    noise_source_guide = torch.randint(0, 256, (src_size, src_size, 3), dtype=torch.uint8, device="cuda")
    noise_target_guide = torch.randint(0, 256, (tgt_size, tgt_size, 3), dtype=torch.uint8, device="cuda")
    weights = build_cost_weights([0.0, 0.0, 0.0], [1.0 / 3, 1.0 / 3, 1.0 / 3])

    stats = {}
    for uw in (0.0, 3500.0):
        torch.manual_seed(1)
        nnf = init_random_nnf(tgt_size, tgt_size, src_size, src_size, patch_size)
        combined_source = build_combined_source(noise_style, noise_source_guide)
        target_style = vote_image(nnf, noise_style, patch_size)
        uniformity = None
        if uw > 0:
            uniformity = Uniformity(nnf, (src_size, src_size), (tgt_size, tgt_size), patch_size, uw)

        for _ in range(6):
            padded = pad_target(torch.cat([target_style, noise_target_guide], dim=-1), patch_size)
            cost = patch_cost(nnf, combined_source, padded, weights, patch_size)
            for _ in range(4):
                nnf, cost = propagate(nnf, cost, combined_source, padded, weights, patch_size, uniformity)
                nnf, cost = random_search(nnf, cost, combined_source, padded, weights, patch_size, uniformity)
            target_style = vote_image(nnf, noise_style, patch_size)

        omega = compute_omega(nnf, (src_size, src_size), patch_size)
        stats[uw] = (omega.max().item(), omega.var().item(), omega.mean().item())
        print(f"uniformity_weight={uw}: max={stats[uw][0]:.1f} var={stats[uw][1]:.1f} mean={stats[uw][2]:.1f}")

    assert abs(stats[0.0][2] - stats[3500.0][2]) < 1e-2, \
        "total occupancy (mean) must be conserved regardless of uniformity — only its spread should change"
    assert stats[3500.0][1] < stats[0.0][1] * 0.9, \
        "uniformity_weight did not meaningfully flatten the occupancy distribution"
    assert stats[3500.0][0] < stats[0.0][0], "uniformity_weight did not reduce peak source-patch overuse"
    print("Uniformity flattens occupancy without changing total usage ✓")
