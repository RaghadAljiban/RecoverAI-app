import tempfile
from typing import List, Optional

import cv2
import imageio.v3 as iio
import mediapipe as mp
import numpy as np

POSE_CONNECTIONS = list(mp.solutions.pose.POSE_CONNECTIONS)

LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 2
JOINT_COLOR = (0, 120, 255)
JOINT_RADIUS = 3


def draw_skeleton(frames: List[np.ndarray], image_landmarks: np.ndarray) -> List[np.ndarray]:
    if len(frames) != len(image_landmarks):
        raise ValueError(
            f"frames ({len(frames)}) and image_landmarks ({len(image_landmarks)}) must align"
        )

    out: List[np.ndarray] = []
    for frame, pose_arr in zip(frames, image_landmarks):
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        points: List[Optional[tuple[int, int]]] = []
        for x, _y, _z, vis in pose_arr:
            if vis > 0.0:
                points.append((int(x * w), int(_y * h)))
            else:
                points.append(None)

        for a, b in POSE_CONNECTIONS:
            pa, pb = points[a], points[b]
            if pa is not None and pb is not None:
                cv2.line(canvas, pa, pb, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

        for p in points:
            if p is not None:
                cv2.circle(canvas, p, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)

        out.append(canvas)
    return out


def write_mp4(frames: List[np.ndarray], fps: float) -> Optional[str]:
    if not frames:
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name

    rgb_stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    with iio.imopen(out_path, "w", plugin="pyav") as out:
        out.write(rgb_stack, codec="h264", fps=float(fps))

    return out_path
