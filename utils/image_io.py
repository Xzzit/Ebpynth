import torch
import os
import sys
import torchvision.io as tv_io


def load_image_to_vram(file_path: str) -> torch.Tensor:
    """
    Loads an image from disk straight onto the GPU.

    Args:
        file_path: path to a PNG/JPG (or similar) image on disk.

    Returns:
        uint8 CUDA tensor, shape (H, W, C), contiguous. C is collapsed to the image's
        real information content: opaque grayscale -> 1, grayscale+alpha -> 2, opaque
        RGB/RGBA -> 3, RGBA with real transparency -> 4. Fewer channels means fewer
        channels in the cost function's inner loop, so this is not just tidiness.
    """
    if not os.path.exists(file_path):
        print(f"error: failed to load '{file_path}'\nReason: File not found.", file=sys.stderr)
        sys.exit(1)

    try:
        # UNCHANGED mode: load raw native layers (1-channel gray, 3-channel RGB, or 4-channel RGBA)
        img_tensor = tv_io.read_image(file_path, tv_io.ImageReadMode.UNCHANGED)
        img_tensor = img_tensor.permute(1, 2, 0)  # (C, H, W) -> this project's (H, W, C) convention
        # permute only rewrites strides; .contiguous() before .cuda() so the upload is
        # one linear copy and every later gather reads a packed layout
        gpu_tensor = img_tensor.contiguous().cuda()

        h, w, c = gpu_tensor.shape

        # Channel collapsing: drop lanes that carry no information
        if c >= 3:
            is_gray = torch.all(gpu_tensor[..., 0] == gpu_tensor[..., 1]) and torch.all(gpu_tensor[..., 1] == gpu_tensor[..., 2])
            has_alpha = False
            if c == 4:
                has_alpha = torch.any(gpu_tensor[..., 3] < 255)

            if is_gray:
                if has_alpha:
                    gpu_tensor = torch.stack([gpu_tensor[..., 0], gpu_tensor[..., 3]], dim=-1)  # gray + alpha
                else:
                    gpu_tensor = gpu_tensor[..., 0:1]  # opaque gray
            else:
                if not has_alpha and c == 4:
                    gpu_tensor = gpu_tensor[..., 0:3]  # opaque RGBA -> drop redundant alpha
        elif c == 2:
            # Native gray+alpha: still drop alpha if it carries no real transparency
            has_alpha = torch.any(gpu_tensor[..., 1] < 255)
            if not has_alpha:
                gpu_tensor = gpu_tensor[..., 0:1]

        return gpu_tensor.contiguous()

    except Exception as err:
        print(f"error: failed to load '{file_path}'\nReason: {str(err)}", file=sys.stderr)
        sys.exit(1)


def save_image_from_vram(gpu_tensor: torch.Tensor, file_path: str) -> None:
    """
    Writes a GPU image tensor to disk. Always encodes PNG, whatever extension the
    output path carries.

    Args:
        gpu_tensor: uint8 CUDA tensor, shape (H, W, C), C in {1, 2, 3, 4}.
        file_path: output path.
    """
    try:
        cpu_tensor = gpu_tensor.permute(2, 0, 1).cpu().contiguous()  # (H, W, C) -> (C, H, W)
        c = cpu_tensor.shape[0]

        if c in (1, 3):
            tv_io.write_png(cpu_tensor, file_path)
        else:
            # torchvision's PNG encoder only takes 1 or 3 channels; alpha-bearing
            # outputs (2=gray+alpha, 4=RGBA) go through PIL instead
            from PIL import Image
            mode = "LA" if c == 2 else "RGBA"
            Image.fromarray(gpu_tensor.cpu().numpy(), mode=mode).save(file_path, format="PNG")

    except Exception as err:
        print(f"error: failed to save '{file_path}'\nReason: {str(err)}", file=sys.stderr)
        sys.exit(1)
