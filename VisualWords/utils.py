# utils.py

import cv2
from typing import List
from PIL import Image

def extract_frames(
    video_path: str,
    fps: int = 1,
    min_frames: int = 3
) -> List[Image.Image]:
    """
    Extract up to `fps` frames per second; if fewer than `min_frames` are found,
    evenly sample exactly `min_frames` frames across the video instead.
    """
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = vidcap.get(cv2.CAP_PROP_FPS) or fps
    interval = max(int(video_fps / fps), 1)

    # 1) Try the simple per-second extraction
    frames: List[Image.Image] = []
    success, frame = vidcap.read()
    count = 0
    while success:
        if count % interval == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img_rgb))
        success, frame = vidcap.read()
        count += 1

    # 2) If that yielded too few, do an even sampling of exactly min_frames
    if len(frames) < min_frames and total_frames >= min_frames:
        frames = []
        for i in range(min_frames):
            # Compute target frame index
            idx = int(i * (total_frames - 1) / (min_frames - 1))
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = vidcap.read()
            if not success:
                continue
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img_rgb))

    vidcap.release()
    return frames
