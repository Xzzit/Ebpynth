import cv2
import numpy as np
import torch


def read_frames(video_path, crop_height=None):
    """
    Decodes a whole video into this project's tensor convention.

    Videos are commonly encoded at a height padded up to a multiple of 16 (a 540-row
    clip is stored as 544 rows), and the padding sits at the bottom. crop_height keeps
    the top N rows so frames line up with keyframes authored at the real height.

    Args:
        video_path: path to any container OpenCV can open.
        crop_height: keep only the first N rows, or None for the decoded height.

    Returns:
        (frames, fps) — frames is a list of uint8 CUDA tensors, shape (H, W, 3) RGB,
        contiguous; fps is a float.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video '{video_path}'")
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if crop_height is not None:
            rgb = rgb[:crop_height]
        frames.append(torch.from_numpy(np.ascontiguousarray(rgb)).cuda())
    cap.release()

    if not frames:
        raise RuntimeError(f"decoded zero frames from '{video_path}'")
    return frames, fps


def write_video(frames, out_path, fps):
    """
    Encodes a list of frames to H.264.

    macro_block_size=1 is essential: imageio otherwise silently resizes up to a
    multiple of 16, which would undo the crop that made frames match the keyframes.

    Args:
        frames: list of uint8 CUDA or CPU tensors, shape (H, W, 3) RGB.
        out_path: output path (.mp4).
        fps: frames per second.
    """
    import imageio

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame.cpu().numpy())
    finally:
        writer.close()
