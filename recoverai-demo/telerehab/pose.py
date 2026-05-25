from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from telerehab.config import (
    DEFAULT_FPS,
    FPS_MAX,
    FPS_MIN,
    POSE_DETECTION_CONFIDENCE,
    POSE_MODEL_COMPLEXITY,
    POSE_TRACKING_CONFIDENCE,
)

ProgressCallback = Optional[Callable[[float], None]]


@dataclass
class PoseResult:
    raw_world: np.ndarray
    raw_image: np.ndarray
    frames: List[np.ndarray]
    fps: float
    n_frames: int
    reported_fps: Optional[float]
    detection_rate: float

    @property
    def info(self) -> dict:
        return {
            "n_frames": int(self.n_frames),
            "fps": float(self.fps),
            "reported_fps": (None if self.reported_fps is None else float(self.reported_fps)),
            "detection_rate": float(self.detection_rate),
        }


def sanitize_fps(fps: Optional[float]) -> float:
    if fps is None:
        return DEFAULT_FPS
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    if not np.isfinite(fps) or fps < FPS_MIN or fps > FPS_MAX:
        return DEFAULT_FPS
    return fps


class PoseExtractor:
    def __init__(
        self,
        model_complexity: int = POSE_MODEL_COMPLEXITY,
        min_detection_confidence: float = POSE_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = POSE_TRACKING_CONFIDENCE,
    ):
        self.model_complexity = int(model_complexity)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)

    def extract(self, video_path: str | Path, progress_cb: ProgressCallback = None) -> PoseResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = sanitize_fps(reported_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        world_coords: List[np.ndarray] = []
        image_coords: List[np.ndarray] = []
        frames: List[np.ndarray] = []
        detections = 0

        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                arr_world = np.zeros((33, 4), dtype=np.float32)
                arr_img = np.zeros((33, 4), dtype=np.float32)

                if result.pose_world_landmarks:
                    detections += 1
                    image_lms = (
                        result.pose_landmarks.landmark if result.pose_landmarks else None
                    )
                    for j, lm in enumerate(result.pose_world_landmarks.landmark):
                        vis = (
                            float(getattr(image_lms[j], "visibility", 1.0))
                            if image_lms is not None
                            else 1.0
                        )
                        arr_world[j] = [lm.x, lm.y, lm.z, vis]

                if result.pose_landmarks:
                    for j, lm in enumerate(result.pose_landmarks.landmark):
                        arr_img[j] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]

                world_coords.append(arr_world)
                image_coords.append(arr_img)
                frames.append(frame)
                frame_idx += 1

                if progress_cb and total_frames > 0:
                    progress_cb(frame_idx / total_frames)

        cap.release()

        if not world_coords:
            raise RuntimeError("No frames could be read from the video.")

        raw_world = np.stack(world_coords).astype(np.float32)
        raw_image = np.stack(image_coords).astype(np.float32)
        n_frames = len(world_coords)
        detection_rate = detections / max(n_frames, 1)

        return PoseResult(
            raw_world=raw_world,
            raw_image=raw_image,
            frames=frames,
            fps=fps,
            n_frames=n_frames,
            reported_fps=(float(reported_fps) if reported_fps is not None else None),
            detection_rate=float(detection_rate),
        )
