import torch


def gather_image(nnf, source_style):
    """
    Simplest possible imaging: each target pixel directly copies the single style
    pixel its NNF entry points at. One flat-index gather, no averaging — the
    training-wheels version used to verify the data flow before real voting.
    """
    src_h, src_w, channels = source_style.shape
    flat_idx = nnf[..., 0] * src_w + nnf[..., 1]
    return source_style.reshape(-1, channels)[flat_idx]


def vote_image(nnf, source_style, patch_size):
    """
    Real plain-average voting (EBSYNTH_VOTEMODE_PLAIN): a target pixel q is covered
    by patch_size² patches; the patch centered at p = q - d claims that q looks like
    style pixel nnf[p] + d. Average every claim.

    Vectorized as one sliced gather per offset d — patch_size² gathers total, no
    scatter needed, because "which pixels does patch p cover" flips into "which
    patch centers cover pixel q", and for a fixed d that is just a shifted slice.

    Relies on the NNF center invariant from init_random_nnf ([r, size-1-r]), which
    guarantees nnf[p] + d never leaves the source image.
    """
    tgt_h, tgt_w = nnf.shape[0], nnf.shape[1]
    src_h, src_w, channels = source_style.shape
    style_flat = source_style.reshape(-1, channels).float()
    r = patch_size // 2

    acc = torch.zeros((tgt_h, tgt_w, channels), dtype=torch.float32, device=nnf.device)
    cnt = torch.zeros((tgt_h, tgt_w, 1), dtype=torch.float32, device=nnf.device)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            # Target pixels q whose claiming patch center p = q - (dy, dx) is itself
            # a valid target pixel; near the border fewer patches cover q, hence cnt
            qy0, qy1 = max(0, dy), min(tgt_h, tgt_h + dy)
            qx0, qx1 = max(0, dx), min(tgt_w, tgt_w + dx)
            centers = nnf[qy0 - dy:qy1 - dy, qx0 - dx:qx1 - dx]

            flat_idx = (centers[..., 0] + dy) * src_w + (centers[..., 1] + dx)
            acc[qy0:qy1, qx0:qx1] += style_flat[flat_idx]
            cnt[qy0:qy1, qx0:qx1] += 1.0

    return (acc / cnt).round().clamp(0, 255).to(torch.uint8)


# Sandbox validation grid execution (run from anywhere: python synthesis/vote.py)
if __name__ == "__main__":
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    os.chdir(repo_root)

    from utils import load_image_to_vram, save_image_from_vram
    from synthesis.nnf import init_random_nnf

    style = load_image_to_vram("examples/video/output_frames/000.png")
    h, w, c = style.shape
    patch_size = 5
    r = patch_size // 2

    # ① Math check: the identity NNF (every pixel maps to itself, target == source)
    # must reproduce the style image byte-for-byte through BOTH imaging paths —
    # for vote, every claim on q resolves to style[q], so the average is exact
    yy, xx = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
    identity_nnf = torch.stack([yy, xx], dim=-1)
    assert torch.equal(gather_image(identity_nnf, style), style), "gather broke on identity NNF"
    assert torch.equal(vote_image(identity_nnf, style, patch_size), style), "vote broke on identity NNF"
    print("Identity NNF reproduces the style image exactly (gather + vote) ✓")

    # ② Random NNF: invariant checks + the Task F milestone mosaics
    nnf = init_random_nnf(h, w, h, w, patch_size)
    assert nnf.shape == (h, w, 2) and nnf.dtype == torch.int64
    assert int(nnf[..., 0].min()) >= r and int(nnf[..., 0].max()) <= h - 1 - r
    assert int(nnf[..., 1].min()) >= r and int(nnf[..., 1].max()) <= w - 1 - r
    print("Random NNF respects the patch-center bounds ✓")

    out_dir = "examples/video/temp"
    gather_path = os.path.join(out_dir, "task_f_gather.png")
    vote_path = os.path.join(out_dir, "task_f_vote.png")
    save_image_from_vram(gather_image(nnf, style), gather_path)
    save_image_from_vram(vote_image(nnf, style, patch_size), vote_path)
    print(f"🏆 Task F milestone mosaics written:\n  {gather_path}  (per-pixel confetti)\n  {vote_path}  (5x5-averaged mush)")
