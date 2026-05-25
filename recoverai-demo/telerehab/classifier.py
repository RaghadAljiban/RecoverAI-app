from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from telerehab.checkpoint import (
    CheckpointBundle,
    ExerciseRecognitionBundle,
    load_checkpoint,
    load_exercise_recognition_checkpoint,
)
from telerehab.config import (
    CHECKPOINT_PATH,
    EXERCISE_RECOGNITION_CHECKPOINT_PATH,
    GLOBAL_THRESHOLD,
    SUPPORTED_EXERCISES,
    EXERCISE_NAMES,
    TARGET_FRAMES,
)
from telerehab.features import FeatureExtractor, compute_resample_indices
from telerehab.pose import PoseExtractor, PoseResult

ProgressCallback = Optional[Callable[[float], None]]


@dataclass
class ClassificationResult:
    prediction_idx: int
    prediction_label: str
    confidence: float
    prob_correct: float
    prob_incorrect: float
    logit: float
    threshold: float
    exercise_id: int
    exercise_name: str
    predicted_exercise_id: int
    predicted_exercise_name: str
    exercise_match: bool
    exercise_confidence: float
    fps: float
    overlay_frames: List[np.ndarray]
    info: Dict[str, Any]
    debug: Dict[str, Any] = field(default_factory=dict)


class RehabClassifier:
    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        exercise_checkpoint_path: Path = EXERCISE_RECOGNITION_CHECKPOINT_PATH,
        device: Optional[torch.device] = None,
        threshold: float = GLOBAL_THRESHOLD,
    ):
        self.bundle: CheckpointBundle = load_checkpoint(checkpoint_path, device=device)

        self.exercise_bundle: ExerciseRecognitionBundle = load_exercise_recognition_checkpoint(
            exercise_checkpoint_path,
            device=self.bundle.device,
        )

        self.features = FeatureExtractor(self.bundle.train_mean, self.bundle.train_std)
        self.exercise_features = FeatureExtractor(
            self.exercise_bundle.train_mean,
            self.exercise_bundle.train_std,
        )

        self.pose = PoseExtractor()
        self.threshold = float(threshold)

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @property
    def supported_exercises(self) -> Dict[str, int]:
        return dict(SUPPORTED_EXERCISES)

    def classify(
        self,
        video_path: str | Path,
        exercise_name: str,
        progress_cb: ProgressCallback = None,
    ) -> ClassificationResult:
        if exercise_name not in SUPPORTED_EXERCISES:
            raise ValueError(f"Unsupported exercise: {exercise_name!r}")

        from telerehab.overlay import draw_skeleton

        pose_result: PoseResult = self.pose.extract(video_path, progress_cb=progress_cb)

        # Features for correct / incorrect model
        x_all, world_norm = self.features(pose_result.raw_world)

        # Features for exercise-recognition model
        x_exercise_all, _ = self.exercise_features(pose_result.raw_world)

        selected_exercise_id = SUPPORTED_EXERCISES[exercise_name]

        x_tensor = torch.tensor(x_all.T[None], dtype=torch.float32, device=self.device)
        x_exercise_tensor = torch.tensor(
            x_exercise_all.T[None],
            dtype=torch.float32,
            device=self.device,
        )

        # 1) Exercise recognition model
        with torch.no_grad():
            exercise_logits = self.exercise_bundle.model(x_exercise_tensor)
            exercise_probs = torch.softmax(exercise_logits, dim=1)[0]

            predicted_exercise_idx = int(torch.argmax(exercise_probs).item())
            exercise_confidence = float(exercise_probs[predicted_exercise_idx].item())

        predicted_exercise_id = self.exercise_bundle.index_to_exercise[predicted_exercise_idx]
        predicted_exercise_name = EXERCISE_NAMES.get(
            predicted_exercise_id,
            f"Exercise {predicted_exercise_id}",
        )

        exercise_match = int(predicted_exercise_id) == int(selected_exercise_id)

        # 2) Correct / Incorrect classification model
        exercise_idx0 = self.bundle.exercise_to_index[selected_exercise_id]
        ex_tensor = torch.tensor([exercise_idx0], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logit = self.bundle.model(x_tensor, ex_tensor)
            logit_value = float(logit[0].item())
            prob_correct = float(torch.sigmoid(logit)[0].item())

        prediction_idx = int(prob_correct >= self.threshold)
        confidence = prob_correct if prediction_idx == 1 else (1.0 - prob_correct)

        overlay_frames = draw_skeleton(pose_result.frames, pose_result.raw_image)

        sample_idx = compute_resample_indices(pose_result.raw_world.shape[0], TARGET_FRAMES)
        sampled_world = world_norm[sample_idx]

        info = pose_result.info

        debug = {
            "raw_logit": logit_value,
            "clip_duration_sec": float(info["n_frames"] / max(info["fps"], 1e-6)),
            "reported_fps": info["reported_fps"],
            "used_fps": float(info["fps"]),
            "sampled_frames": TARGET_FRAMES,
            "selected_exercise_id": int(selected_exercise_id),
            "predicted_exercise_id": int(predicted_exercise_id),
            "predicted_exercise_name": predicted_exercise_name,
            "exercise_match": bool(exercise_match),
            "exercise_confidence": float(exercise_confidence),
            "sample_index_preview": (
                sample_idx[:8].astype(int).tolist()
                + (["..."] if len(sample_idx) > 16 else [])
                + sample_idx[-8:].astype(int).tolist()
            ),
            "raw_visibility_mean": float(pose_result.raw_world[..., 3].mean()),
            "normalized_xyz_min": float(world_norm[..., :3].min()),
            "normalized_xyz_max": float(world_norm[..., :3].max()),
            "sampled_xyz_min": float(sampled_world[..., :3].min()),
            "sampled_xyz_max": float(sampled_world[..., :3].max()),
        }

        return ClassificationResult(
            prediction_idx=prediction_idx,
            prediction_label=("CORRECT" if prediction_idx == 1 else "INCORRECT"),
            confidence=float(confidence),
            prob_correct=float(prob_correct),
            prob_incorrect=float(1.0 - prob_correct),
            logit=logit_value,
            threshold=self.threshold,
            exercise_id=int(selected_exercise_id),
            exercise_name=exercise_name,
            predicted_exercise_id=int(predicted_exercise_id),
            predicted_exercise_name=predicted_exercise_name,
            exercise_match=bool(exercise_match),
            exercise_confidence=float(exercise_confidence),
            fps=float(pose_result.fps),
            overlay_frames=overlay_frames,
            info=info,
            debug=debug,
        )