import torch
import torch.nn.functional as F


def compute_omega(nnf, source_shape, patch_size):
    """
    Occupancy count per source pixel: how many currently-active target patches claim
    to draw from it.

    Mirror image of patch_cost's gather loop (cost.py): same nested (dy, dx) offset
    loop, but scattering +1 INTO the source instead of gathering values FROM it.
    "Which source pixels does this NNF touch" genuinely needs a scatter, unlike
    vote_image's "who covers me", which gets a fixed slice per offset because it asks
    about the TARGET side.

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
    The occupancy one source pixel would carry under perfectly uniform usage — the
    reference value score() measures actual occupancy against.

    Every target pixel scatters patch_size² claims, so the target makes
    target_area * patch_size² of them in total; spread evenly over source_area
    pixels, that is the per-pixel share below.

    Args:
        target_shape: (H_target, W_target) tuple.
        source_shape: (H_source, W_source) tuple.
        patch_size: odd int, side length of the square patch.

    Returns:
        float — the ideal (uniform-usage) per-source-pixel occupancy.
    """
    target_area = target_shape[0] * target_shape[1]
    source_area = source_shape[0] * source_shape[1]
    return (target_area / source_area) * (patch_size * patch_size)


class Uniformity:
    """
    Occupancy state (Omega) plus the penalty weight, so propagate/random_search can
    factor "don't overuse this source patch" into their accept/reject decisions:
    they compare cost + weight*(occupancy/ideal) instead of raw cost.

    One instance lives for an entire pyramid level — constructed from that level's
    starting NNF, then updated incrementally as matches change across every
    vote/patchmatch iteration of the level. Pass uniformity=None anywhere it is
    accepted to disable the penalty entirely.
    """

    def __init__(self, nnf, source_shape, target_shape, patch_size, weight):
        """
        Args:
            nnf: int64 CUDA tensor, shape (H_target, W_target, 2) — this level's
                 starting NNF, used to seed the occupancy table.
            source_shape: (H_source, W_source) tuple.
            target_shape: (H_target, W_target) tuple.
            patch_size: odd int, side length of the square patch.
            weight: float, the penalty weight (-uniformity CLI value).
        """
        self.omega = compute_omega(nnf, source_shape, patch_size)
        self.ideal = ideal_omega(target_shape, source_shape, patch_size)
        self.weight = weight
        self.patch_size = patch_size
        # weight / patch_size² / ideal collapsed into one constant, so score() is a
        # single multiply-add instead of two divides, a multiply and an add
        self._scale = weight / (patch_size * patch_size) / self.ideal
        self._box = None  # lazily built by _box_sum, invalidated by update

    def _box_sum(self):
        """
        Occupancy summed over the patch window centered on EVERY source pixel, as one
        flat float32 tensor — the key to making score() cheap.

        score() asks "what is Omega's total over the window at nnf[q]" for every target
        pixel q. Done directly that is patch_size² gathers (one per window offset), which
        is what this used to do. But that quantity depends only on the window's center,
        not on which target pixel asked for it: it is a plain box filter over Omega,
        computable for all source positions at once, after which each target pixel needs
        a single gather. patch_size² gathers collapse to one pooling op plus one gather.

        divisor_override=1 turns avg_pool2d into a plain sum. Zero padding is safe here:
        the NNF invariant keeps every center at least r from the source border, so the
        fabricated padding ring is never gathered from.

        Cached because a Uniformity instance answers many score() calls between Omega
        mutations — within one propagate direction, the candidate and the incumbent are
        both scored against the same Omega. update() drops the cache.

        Returns:
            float32 CUDA tensor, shape (H_source * W_source,) — flattened box sums.
        """
        if self._box is None:
            r = self.patch_size // 2
            padded = F.avg_pool2d(self.omega.unsqueeze(0).unsqueeze(0), self.patch_size,
                                  stride=1, padding=r, divisor_override=1)
            self._box = padded.reshape(-1)
        return self._box

    def score(self, cost, nnf):
        """
        Raw patch cost plus the occupancy penalty — the value propagate and
        random_search actually compare when uniformity is enabled.

        Args:
            cost: float32 CUDA tensor, shape (H_target, W_target) — raw patch cost.
            nnf: int64 CUDA tensor, shape (H_target, W_target, 2).

        Returns:
            float32 CUDA tensor, shape (H_target, W_target) — the penalized cost.
        """
        src_w = self.omega.shape[1]
        patch_occupancy = self._box_sum()[nnf[..., 0] * src_w + nnf[..., 1]]
        return cost + patch_occupancy * self._scale

    def update(self, old_nnf, new_nnf, changed):
        """
        Moves each changed pixel's occupancy claim from its old source patch to
        its new one: -1 across the old window, +1 across the new one, scattered
        (not sliced) since old/new positions vary per pixel. scatter_add_ correctly
        accumulates when several pixels happen to land on the same source pixel in
        the same call, unlike a plain indexed write would.

        The -old and +new scatters are fused into a single call over a concatenated
        index/delta pair, built once outside the offset loop: each offset is then one
        add and one scatter rather than two of each plus four index-arithmetic ops.

        Args:
            old_nnf, new_nnf: int64 CUDA tensors, shape (H_target, W_target, 2).
            changed: bool CUDA tensor, shape (H_target, W_target) — which pixels
                     actually moved from old_nnf to new_nnf.
        """
        r = self.patch_size // 2
        src_w = self.omega.shape[1]
        omega_flat = self.omega.reshape(-1)
        delta = changed.reshape(-1).float()

        old_base = (old_nnf[..., 0] * src_w + old_nnf[..., 1]).reshape(-1)
        new_base = (new_nnf[..., 0] * src_w + new_nnf[..., 1]).reshape(-1)
        both_base = torch.cat([old_base, new_base])
        both_delta = torch.cat([-delta, delta])

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                omega_flat.scatter_add_(0, both_base + (dy * src_w + dx), both_delta)

        self._box = None  # Omega moved; the cached box sums are stale


# Sandbox validation grid execution (run from the repo root: python synthesis/uniformity.py)
if __name__ == "__main__":
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    from synthesis.cost import build_channel_scales, build_combined_source, pad_target, patch_cost
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
    scales = build_channel_scales([0.0, 0.0, 0.0], [1.0 / 3, 1.0 / 3, 1.0 / 3])

    stats = {}
    for uw in (0.0, 3500.0):
        torch.manual_seed(1)
        nnf = init_random_nnf(tgt_size, tgt_size, src_size, src_size, patch_size)
        combined_source = build_combined_source(noise_style, noise_source_guide, scales)
        target_style = vote_image(nnf, noise_style, patch_size)
        uniformity = None
        if uw > 0:
            uniformity = Uniformity(nnf, (src_size, src_size), (tgt_size, tgt_size), patch_size, uw)

        for _ in range(6):
            padded = pad_target(torch.cat([target_style, noise_target_guide], dim=-1), patch_size, scales)
            cost = patch_cost(nnf, combined_source, padded, patch_size)
            for _ in range(4):
                nnf, cost = propagate(nnf, cost, combined_source, padded, patch_size, uniformity)
                nnf, cost = random_search(nnf, cost, combined_source, padded, patch_size, uniformity)
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
