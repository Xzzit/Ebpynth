import sys
import torch

try:
    from .image_io import load_image_to_vram
except ImportError:
    from image_io import load_image_to_vram

# Mirrored from ebpynth/include/ebsynth.h — the CUDA kernel's hard channel limits.
EBSYNTH_MAX_STYLE_CHANNELS = 8
EBSYNTH_MAX_GUIDE_CHANNELS = 24


def _expand_channels(tensor: torch.Tensor, num_channels: int) -> torch.Tensor:
    """
    Re-expands a channel-collapsed (H, W, C) guide tensor back up to num_channels,
    replicating how the original packing loops always draw from the full RGBA buffer
    (ebsynth.cpp lines ~347-381): lane order is R, then A if 2 channels, else G/B/A.
    """
    c = tensor.shape[-1]
    if c == num_channels:
        return tensor

    # Rebuild the four RGBA lanes from the collapsed layout produced by load_image_to_vram
    if c == 1:
        r = g = b = tensor[..., 0]
        a = torch.full_like(r, 255)
    elif c == 2:
        r = g = b = tensor[..., 0]
        a = tensor[..., 1]
    elif c == 3:
        r, g, b = tensor[..., 0], tensor[..., 1], tensor[..., 2]
        a = torch.full_like(r, 255)
    else:
        r, g, b, a = tensor[..., 0], tensor[..., 1], tensor[..., 2], tensor[..., 3]

    if num_channels == 1:
        lanes = [r]
    elif num_channels == 2:
        lanes = [r, a]
    elif num_channels == 3:
        lanes = [r, g, b]
    else:
        lanes = [r, g, b, a]

    return torch.stack(lanes, dim=-1)


def merge_guides(guides, style_shape, style_file):
    """
    Twists all guide channels together into two mega feature tensors via torch.cat,
    replacing the original per-guide load/validate loop and the two nested per-pixel
    packing loops (ebsynth.cpp lines ~327-381) in one vectorized shot.

    Args:
        guides: list of {"source": path, "target": path, "weight": float} dicts,
                exactly as produced by arguments.parse_arguments()["guides"].
        style_shape: the (H, W, C) shape of the already-loaded style tensor.
        style_file: style image path, used only for error messages.

    Returns:
        (source_guides, target_guides, guide_channel_counts)
        source_guides: (H_style, W_style, sum_C) uint8 CUDA contiguous tensor
        target_guides: (H_target, W_target, sum_C) uint8 CUDA contiguous tensor
        guide_channel_counts: per-guide aligned channel count list — needed later
                              to spread each guide's weight across its channels.
    """
    style_h, style_w, style_c = style_shape

    if len(guides) == 0:
        print("error: at least one -guide <source> <target> pair is required!", file=sys.stderr)
        sys.exit(1)

    source_list = []
    target_list = []
    guide_channel_counts = []
    target_h = target_w = None

    for i, guide in enumerate(guides):
        src = load_image_to_vram(guide["source"])
        tgt = load_image_to_vram(guide["target"])

        if src.shape[0] != style_h or src.shape[1] != style_w:
            print(f"error: source guide '{guide['source']}' doesn't match the resolution of '{style_file}'", file=sys.stderr)
            sys.exit(1)
        if i == 0:
            target_h, target_w = tgt.shape[0], tgt.shape[1]
        elif tgt.shape[0] != target_h or tgt.shape[1] != target_w:
            print(f"error: target guide '{guide['target']}' doesn't match the resolution of '{guides[0]['target']}'", file=sys.stderr)
            sys.exit(1)

        # A guide's source/target can collapse to different channel counts; the original
        # aligns both sides up to std::max before packing (ebsynth.cpp line ~338)
        num_channels = max(src.shape[-1], tgt.shape[-1])
        source_list.append(_expand_channels(src, num_channels))
        target_list.append(_expand_channels(tgt, num_channels))
        guide_channel_counts.append(num_channels)

    num_guide_channels_total = sum(guide_channel_counts)
    if style_c > EBSYNTH_MAX_STYLE_CHANNELS:
        print(f"error: too many style channels ({style_c}), maximum number is {EBSYNTH_MAX_STYLE_CHANNELS}", file=sys.stderr)
        sys.exit(1)
    if num_guide_channels_total > EBSYNTH_MAX_GUIDE_CHANNELS:
        print(f"error: too many guide channels ({num_guide_channels_total}), maximum number is {EBSYNTH_MAX_GUIDE_CHANNELS}", file=sys.stderr)
        sys.exit(1)

    source_guides = torch.cat(source_list, dim=-1).contiguous()
    target_guides = torch.cat(target_list, dim=-1).contiguous()
    return source_guides, target_guides, guide_channel_counts


# Sandbox validation grid execution (run from the repo root: python utils/guide_merge.py)
if __name__ == "__main__":
    style_path = "examples/video/output_frames/000.png"
    style = load_image_to_vram(style_path)
    guides = [
        {"source": "examples/video/video_frames/000.jpg", "target": "examples/video/video_frames/001.jpg", "weight": -1.0},
        {"source": "examples/video/video_frames/000.jpg", "target": "examples/video/video_frames/002.jpg", "weight": 1.5},
    ]
    src_merged, tgt_merged, channels = merge_guides(guides, style.shape, style_path)
    print("🏆 Guide Merging Engine Test Passed!")
    print(f"  source_guides: {src_merged.shape}, {src_merged.dtype}, {src_merged.device}, contiguous={src_merged.is_contiguous()}")
    print(f"  target_guides: {tgt_merged.shape}, {tgt_merged.dtype}, {tgt_merged.device}, contiguous={tgt_merged.is_contiguous()}")
    print(f"  per-guide channels: {channels}")

    # Synthetic channel-alignment check: 1-channel source vs 3-channel target must align to 3
    gray = torch.full((4, 4, 1), 7, dtype=torch.uint8, device="cuda")
    rgb = torch.arange(48, dtype=torch.uint8, device="cuda").reshape(4, 4, 3)
    expanded = _expand_channels(gray, 3)
    assert expanded.shape == (4, 4, 3) and torch.all(expanded == 7), "1→3 expansion failed"
    ga = torch.stack([gray[..., 0], torch.full((4, 4), 128, dtype=torch.uint8, device="cuda")], dim=-1)
    expanded2 = _expand_channels(ga, 4)
    assert expanded2.shape == (4, 4, 4) and torch.all(expanded2[..., 3] == 128), "2→4 expansion failed"
    print("  channel alignment (std::max) checks passed ✓")
