def plan_pyramid(config, style_shape, target_shape, guide_channels):
    """
    Plans the coarse-to-fine pyramid schedule and per-channel weight vectors,
    replacing the original's tail-end scalar math (ebsynth.cpp lines ~383-426).
    Pure CPU-side arithmetic — no tensors involved: the CUDA kernel builds the
    actual pyramid itself, this only fills in its control sheet.

    Args:
        config: the dict produced by arguments.parse_arguments().
        style_shape: (H, W, C) shape of the loaded style tensor.
        target_shape: shape of the merged target guides — only [0]/[1] (H, W) are used.
        guide_channels: per-guide aligned channel counts from merge_guides().

    Returns a dict mirroring ebsynthRunCuda's control parameters:
        num_pyramid_levels: int, auto-derived when -1 and always clamped to the max
        search_vote_iters_per_level / patch_match_iters_per_level /
        stop_threshold_per_level: int lists of length num_pyramid_levels
        style_weights: float list of length C_style, summing to style_weight
        guide_weights: float list of length sum(guide_channels); each guide's
                       weight is spread evenly across its own channels
    """
    style_h, style_w, style_c = style_shape
    target_h, target_w = target_shape[0], target_shape[1]
    patch_size = config["patch_size"]

    # Auto level count: keep halving until the smallest of the four dimensions can no
    # longer fit one search window of (2*patchsize+1) pixels (ebsynth.cpp lines ~405-416).
    # Float scale + int truncation reproduces pyramidLevelSize()'s V2f->V2i rounding.
    min_dim = min(style_h, style_w, target_h, target_w)
    max_levels = 0
    for level in range(32, -1, -1):
        if int(min_dim * (2.0 ** -level)) >= (2 * patch_size + 1):
            max_levels = level + 1
            break

    num_levels = config["num_pyramid_levels"]
    if num_levels == -1:
        num_levels = max_levels
    num_levels = min(num_levels, max_levels)  # explicit user values get silently clamped too

    # The kernel API allows different effort per level; the CLI semantics never use
    # that freedom, so each array is just the same scalar replicated L times.
    search_vote_iters_per_level = [config["num_search_vote_iters"]] * num_levels
    patch_match_iters_per_level = [config["num_patch_match_iters"]] * num_levels
    stop_threshold_per_level = [config["stop_threshold"]] * num_levels

    # Style camp: total say of style_weight (default 1.0), split evenly per channel
    style_weight = config["style_weight"]
    if style_weight < 0:
        style_weight = 1.0
    style_weights = [style_weight / float(style_c)] * style_c

    # Guide camp: each guide defaults to 1/numGuides, then spreads over its channels,
    # so unweighted guides collectively always speak with a total voice of 1.0
    num_guides = len(guide_channels)
    guide_weights = []
    for i, channels in enumerate(guide_channels):
        weight = config["guides"][i]["weight"]
        if weight < 0:
            weight = 1.0 / float(num_guides)
        guide_weights.extend([weight / float(channels)] * channels)

    return {
        "num_pyramid_levels": num_levels,
        "search_vote_iters_per_level": search_vote_iters_per_level,
        "patch_match_iters_per_level": patch_match_iters_per_level,
        "stop_threshold_per_level": stop_threshold_per_level,
        "style_weights": style_weights,
        "guide_weights": guide_weights,
    }


# Sandbox validation grid execution
if __name__ == "__main__":
    mock_config = {
        "patch_size": 5,
        "num_pyramid_levels": -1,
        "num_search_vote_iters": 6,
        "num_patch_match_iters": 4,
        "stop_threshold": 5,
        "style_weight": -1.0,
        "guides": [
            {"source": "a.png", "target": "b.png", "weight": -1.0},  # defaults to 1/2
            {"source": "c.png", "target": "d.png", "weight": 2.0},
        ],
    }

    plan = plan_pyramid(mock_config, (540, 960, 3), (540, 960, 6), [3, 3])
    import pprint
    pprint.pprint(plan)

    # 540/2^5 = 16.875 -> 16 >= 11 holds, 540/2^6 = 8 < 11 fails => 6 levels
    assert plan["num_pyramid_levels"] == 6
    assert plan["search_vote_iters_per_level"] == [6] * 6
    assert plan["patch_match_iters_per_level"] == [4] * 6
    assert plan["stop_threshold_per_level"] == [5] * 6
    assert abs(sum(plan["style_weights"]) - 1.0) < 1e-6
    # guide 0: 0.5 spread over 3 channels; guide 1: 2.0 spread over 3 channels
    expected = [0.5 / 3] * 3 + [2.0 / 3] * 3
    assert all(abs(a - b) < 1e-6 for a, b in zip(plan["guide_weights"], expected))

    # Explicitly requested levels must be silently clamped to the derivable maximum
    mock_config["num_pyramid_levels"] = 99
    assert plan_pyramid(mock_config, (540, 960, 3), (540, 960, 6), [3, 3])["num_pyramid_levels"] == 6

    # A tiny image can't pyramid much: min_dim=32, patchsize=5 -> 32/2 = 16 >= 11 => 2 levels
    mock_config["num_pyramid_levels"] = -1
    assert plan_pyramid(mock_config, (32, 32, 3), (64, 64, 6), [3, 3])["num_pyramid_levels"] == 2

    print("🏆 Pyramid Planning Engine Test Passed!")
