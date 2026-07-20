import torch
import os
import sys
import torchvision.io as tv_io

def load_image_to_vram(file_path: str) -> torch.Tensor:
    """
    Loads an image from disk, pushes it straight into GPU memory (VRAM) within the first millisecond,
    and performs parallelized channel optimization using PyTorch vectorized operations.
    Returns a high-performance GPU tensor with shape format: (H, W, C).
    """
    # Defensive Gate: Validate file physical existence prior to asset ingestion
    if not os.path.exists(file_path):
        print(f"error: failed to load '{file_path}'\nReason: File not found.", file=sys.stderr)
        sys.exit(1)

    try:
        # UNCHANGED mode loads raw native layers (1-channel gray, 3-channel RGB, or 4-channel RGBA)
        img_tensor = tv_io.read_image(file_path, tv_io.ImageReadMode.UNCHANGED)
        # Permute from standard Torch layout (C, H, W) to ebsynth expected grid (H, W, C)
        img_tensor = img_tensor.permute(1, 2, 0)

        # 🚀 THE BOOTSTRAP POINT: Throw the raw byte tensor straight into the VRAM fireplane!
        # From this instruction onwards, no data transitions ever leave the GPU grid.
        gpu_tensor = img_tensor.contiguous().cuda()
        
        # Capture spatial parameters and native channel depth
        h, w, c = gpu_tensor.shape

        # Parallelized optimization mimicking ebsynth's 'evalNumChannels' logic
        if c >= 3:
            # Massive GPU execution pass: verify if R, G, B grids are mathematically identical
            is_gray = torch.all(gpu_tensor[..., 0] == gpu_tensor[..., 1]) and torch.all(gpu_tensor[..., 1] == gpu_tensor[..., 2])
            has_alpha = False
            
            if c == 4:
                # Massive GPU execution pass: verify if there is any active alpha alpha opacity transparency
                has_alpha = torch.any(gpu_tensor[..., 3] < 255)

            # Replicate custom C++ structural channel-packing contracts using vectorized tensor slicing
            if is_gray:
                if has_alpha:
                    # Case: Grayscale + Alpha. Pack channel 0 (monochrome) and channel 3 (alpha)
                    gpu_tensor = torch.stack([gpu_tensor[..., 0], gpu_tensor[..., 3]], dim=-1)
                else:
                    # Case: Opaque Grayscale. Collapse matrix down to a single pure channel depth
                    gpu_tensor = gpu_tensor[..., 0:1]
            else:
                if not has_alpha and c == 4:
                    # Case: Fully Opaque RGBA image -> Strip redundant alpha lane to fit 3-channel RGB matrix
                    gpu_tensor = gpu_tensor[..., 0:3]
        elif c == 2:
            # Native Grayscale+Alpha encoding: still need to drop the alpha lane if it carries no
            # real transparency, matching evalNumChannels' behavior on the fully-expanded RGBA buffer
            has_alpha = torch.any(gpu_tensor[..., 1] < 255)
            if not has_alpha:
                gpu_tensor = gpu_tensor[..., 0:1]

        return gpu_tensor.contiguous()

    except Exception as err:
        print(f"error: failed to load '{file_path}'\nReason: {str(err)}", file=sys.stderr)
        sys.exit(1)


def save_image_from_vram(gpu_tensor: torch.Tensor, file_path: str) -> None:
    """
    Lands a finished (H, W, C) uint8 GPU tensor back onto the disk as a PNG,
    closing the loop opened by load_image_to_vram. Mirrors the original CLI's
    stbi_write_png call (ebsynth.cpp line ~461), which always writes PNG
    regardless of the output file's extension.
    """
    try:
        # Return trip: VRAM -> RAM, and interleaved (H, W, C) back to planar (C, H, W)
        cpu_tensor = gpu_tensor.permute(2, 0, 1).cpu().contiguous()
        c = cpu_tensor.shape[0]

        if c in (1, 3):
            tv_io.write_png(cpu_tensor, file_path)
        else:
            # torchvision's PNG encoder only takes 1 or 3 channels; alpha-bearing
            # outputs (2=gray+alpha, 4=RGBA) go through PIL like stb would
            from PIL import Image
            mode = "LA" if c == 2 else "RGBA"
            Image.fromarray(gpu_tensor.cpu().numpy(), mode=mode).save(file_path, format="PNG")

    except Exception as err:
        print(f"error: failed to save '{file_path}'\nReason: {str(err)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Quick sandbox test for the image loading utility (run from the repo root)
    test_image_path = "examples/video/video_frames/000.jpg"
    tensor = load_image_to_vram(test_image_path)
    print(f"Loaded image tensor shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")

    # Round-trip test: save to a temp file, reload, and verify bytes survive intact
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        round_trip_path = os.path.join(tmp_dir, "round_trip.png")
        save_image_from_vram(tensor, round_trip_path)
        reloaded = load_image_to_vram(round_trip_path)
        assert reloaded.shape == tensor.shape, f"shape changed: {tensor.shape} -> {reloaded.shape}"
        assert torch.equal(reloaded, tensor), "pixel data corrupted in save/load round trip"
    print("Save/load round-trip test passed ✓")