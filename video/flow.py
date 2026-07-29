import torch
import torch.nn.functional as F


class OpticalFlow:
    """
    RAFT wrapper producing dense per-pixel motion between two frames.

    Flow is what makes video stylization temporally coherent: it is how the previous
    output and the accumulated positional guide get carried into the next frame's
    geometry. Weights are fetched once into the torch hub cache on first construction.
    """

    def __init__(self, small=False):
        """
        Args:
            small: use raft_small (~4 MB, faster) instead of raft_large (~20 MB).
                   Flow quality sets the ceiling on temporal coherence, and flow is a
                   rounding error next to synthesis time, so large is the default.
        """
        from torchvision.models.optical_flow import (
            raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights)
        if small:
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.model = self.model.eval().cuda()

    @staticmethod
    def _prepare(frame):
        """(H, W, 3) uint8 -> (1, 3, H', W') float in [-1, 1], H'/W' padded to /8."""
        chw = frame.permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        h, w = chw.shape[-2:]
        pad_h, pad_w = (-h) % 8, (-w) % 8
        if pad_h or pad_w:
            chw = F.pad(chw, (0, pad_w, 0, pad_h), mode="replicate")
        return chw, h, w

    @torch.no_grad()
    def between(self, frame_from, frame_to):
        """
        Motion of every pixel of frame_from towards where it lands in frame_to.

        Args:
            frame_from, frame_to: uint8 CUDA tensors, shape (H, W, 3) RGB.

        Returns:
            float32 CUDA tensor, shape (2, H, W) — channel 0 is dx, channel 1 is dy,
            in pixels. Feed straight to warp() to pull frame_to into frame_from's frame
            of reference.
        """
        a, h, w = self._prepare(frame_from)
        b, _, _ = self._prepare(frame_to)
        return self.model(a, b)[-1][0, :, :h, :w]


def warp(image, flow):
    """
    Resamples an image along a flow field: output(p) = image(p + flow(p)).

    Pair this with `flow = OpticalFlow.between(frame_t, frame_prev)` to drag something
    that lives in frame_prev's geometry into frame_t's geometry — which is exactly what
    the temporal and positional guides need each step.

    Border padding (rather than zeros) keeps pixels that flow in from outside the frame
    plausible instead of black, since black would read as real content downstream.

    Args:
        image: uint8 or float CUDA tensor, shape (H, W, C).
        flow: float32 CUDA tensor, shape (2, H, W), as returned by OpticalFlow.between.

    Returns:
        float32 CUDA tensor, shape (H, W, C) — caller decides how to quantize.
    """
    h, w = flow.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device, dtype=torch.float32),
        torch.arange(w, device=flow.device, dtype=torch.float32), indexing="ij")
    # grid_sample wants normalized [-1, 1] coordinates, x first
    gx = (xx + flow[0]) / max(w - 1, 1) * 2.0 - 1.0
    gy = (yy + flow[1]) / max(h - 1, 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

    chw = image.permute(2, 0, 1).unsqueeze(0).float()
    sampled = F.grid_sample(chw, grid, mode="bilinear", padding_mode="border",
                            align_corners=True)
    return sampled.squeeze(0).permute(1, 2, 0)


def warp_u8(image, flow):
    """warp() plus the round/clamp back to the project's uint8 (H, W, C) convention."""
    return warp(image, flow).round().clamp(0, 255).to(torch.uint8)


# Sandbox validation grid execution (run from the repo root: python video/flow.py)
if __name__ == "__main__":
    # warp's sign convention is the one thing here that fails SILENTLY: get it backwards
    # and every temporal guide is shifted the wrong way, which does not crash and does
    # not look obviously wrong on a still — it just quietly destroys coherence. So pin
    # it against a synthetic flow whose correct answer is known exactly.
    h, w = 32, 48
    img = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device="cuda")

    # Zero flow must be the identity
    zero = torch.zeros((2, h, w), device="cuda")
    assert torch.equal(warp_u8(img, zero), img), "zero flow is not the identity"
    print("Zero flow reproduces the image exactly ✓")

    # out(p) = img(p + flow(p)), so a constant +5 in x shifts content LEFT by 5
    shift = 5
    flow = torch.zeros((2, h, w), device="cuda")
    flow[0] = shift
    shifted = warp_u8(img, flow)
    assert torch.equal(shifted[:, :w - shift], img[:, shift:]), \
        "a positive dx must pull content leftward — sign convention is inverted"
    print(f"Constant dx=+{shift} shifts content left by {shift} ✓")

    flow = torch.zeros((2, h, w), device="cuda")
    flow[1] = shift
    shifted = warp_u8(img, flow)
    assert torch.equal(shifted[:h - shift], img[shift:]), \
        "a positive dy must pull content upward — y sign convention is inverted"
    print(f"Constant dy=+{shift} shifts content up by {shift} ✓")

    # Border padding, not zero padding: flowing in from outside must not fabricate black
    flow = torch.full((2, h, w), -float(w), device="cuda")
    assert warp_u8(img, flow).float().mean() > 0, "out-of-frame sampling produced black"
    print("Out-of-frame sampling clamps to the border instead of black ✓")
    print("Flow warp convention checks passed ✓")
