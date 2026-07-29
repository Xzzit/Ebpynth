import cv2
import numpy as np
import torch


def edge_guide(frame):
    """
    Sobel gradient magnitude — the structural half of the appearance match.

    The colour guide alone lets the search drift across regions that merely share a
    tint; an edge channel pins synthesis to the scene's actual contours, which is what
    keeps silhouettes from smearing as the style propagates away from a keyframe.

    Args:
        frame: uint8 CUDA tensor, shape (H, W, 3) RGB.

    Returns:
        uint8 CUDA tensor, shape (H, W, 1) — magnitude normalized to 0..255.
    """
    rgb = frame.cpu().numpy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                   cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    mag = np.clip(mag / max(mag.max(), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return torch.from_numpy(mag).unsqueeze(-1).cuda().contiguous()


def identity_ramp(height, width, device="cuda"):
    """
    The positional guide's source side: each pixel labelled with its own coordinate.

    Encoded as R = x, G = y, B = 0. Propagation advects this ramp frame by frame, so
    the target side answers "which part of the KEYFRAME does this pixel correspond to"
    — correspondence the colour and edge guides cannot express on their own, and the
    reason a moving subject keeps its style locked to its own body rather than to a
    fixed screen region.

    8-bit quantization is the accepted cost here (at 960 wide, one step is ~3.8 px);
    the coarse pyramid levels never resolve finer than that anyway.

    Args:
        height, width: frame size.
        device: torch device.

    Returns:
        uint8 CUDA tensor, shape (H, W, 3).
    """
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    r = xx / max(width - 1, 1) * 255.0
    g = yy / max(height - 1, 1) * 255.0
    b = torch.zeros_like(r)
    return torch.stack([r, g, b], dim=-1).round().clamp(0, 255).to(torch.uint8).contiguous()


# Sandbox validation grid execution (run from the repo root: python video/guides.py)
if __name__ == "__main__":
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    # 1. The ramp must be monotone in the right axis, and span the full 0..255 range —
    # a transposed ramp would still look like a plausible gradient while encoding the
    # wrong correspondence entirely.
    ramp = identity_ramp(120, 200)
    assert ramp.shape == (120, 200, 3) and ramp.dtype == torch.uint8
    assert int(ramp[0, 0, 0]) == 0 and int(ramp[0, -1, 0]) == 255, "R must sweep x"
    assert int(ramp[0, 0, 1]) == 0 and int(ramp[-1, 0, 1]) == 255, "G must sweep y"
    assert torch.all(ramp[..., 0].diff(dim=1) >= 0), "R must be non-decreasing along x"
    assert torch.all(ramp[..., 1].diff(dim=0) >= 0), "G must be non-decreasing along y"
    assert torch.all(ramp[..., 2] == 0), "B is unused and must stay zero"
    assert torch.all(ramp[..., 0] == ramp[0, :, 0]), "R must not vary with y"
    print("Identity ramp sweeps x in R and y in G, full range ✓")

    # 2. Edges should fire on real structure. A synthetic hard edge is the honest test:
    # response on the boundary must dominate the flat regions either side of it.
    block = torch.zeros((64, 64, 3), dtype=torch.uint8, device="cuda")
    block[:, 32:] = 255
    e = edge_guide(block)
    assert e.shape == (64, 64, 1) and e.dtype == torch.uint8
    boundary = e[:, 30:34].float().mean()
    flat = torch.cat([e[:, :26], e[:, 38:]], dim=1).float().mean()
    print(f"Synthetic step edge: boundary {boundary:.1f} vs flat {flat:.1f}")
    assert boundary > 10 * max(flat.item(), 1.0), "edge guide did not isolate the boundary"
    print("Edge guide responds to structure, not to flat regions ✓")
