from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from telerehab.config import DROPOUT_DEFAULT, SUPPORTED_EXERCISES
from telerehab.model import ConditionedTCN, ExerciseTCN


@dataclass
class CheckpointBundle:
    model: ConditionedTCN
    train_mean: np.ndarray
    train_std: np.ndarray
    input_dim: int
    ex_embed_dim: int
    active_exercise_ids: List[int]
    exercise_to_index: Dict[int, int]
    global_threshold: float
    tuned_thresholds: Dict[int, float] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def load_checkpoint(
    path: Path,
    device: Optional[torch.device] = None,
    dropout: float = DROPOUT_DEFAULT,
) -> CheckpointBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    train_mean = np.asarray(ckpt["train_mean"], dtype=np.float32)
    train_std = np.asarray(ckpt["train_std"], dtype=np.float32)
    train_std = np.maximum(train_std, 1e-6)

    input_dim = int(ckpt["input_dim"])
    ex_embed_dim = int(ckpt.get("exercise_embedding_dim", 16))

    active_ids = ckpt.get("active_exercise_ids")
    if not active_ids:
        active_ids = sorted(SUPPORTED_EXERCISES.values())
    active_ids = [int(x) for x in active_ids]
    exercise_to_index = {ex_id: idx for idx, ex_id in enumerate(active_ids)}

    model = ConditionedTCN(
        input_dim=input_dim,
        num_exercises=len(active_ids),
        ex_embed_dim=ex_embed_dim,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tuned = ckpt.get("tuned_thresholds") or {}
    tuned_thresholds = {int(k): float(v) for k, v in tuned.items()}

    return CheckpointBundle(
        model=model,
        train_mean=train_mean,
        train_std=train_std,
        input_dim=input_dim,
        ex_embed_dim=ex_embed_dim,
        active_exercise_ids=active_ids,
        exercise_to_index=exercise_to_index,
        global_threshold=float(ckpt.get("global_threshold", 0.5)),
        tuned_thresholds=tuned_thresholds,
        device=device,
    )
@dataclass
class ExerciseRecognitionBundle:
    model: ExerciseTCN
    train_mean: np.ndarray
    train_std: np.ndarray
    input_dim: int
    active_exercise_ids: List[int]
    exercise_to_index: Dict[int, int]
    index_to_exercise: Dict[int, int]
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def load_exercise_recognition_checkpoint(
    path: Path,
    device: Optional[torch.device] = None,
    dropout: float = DROPOUT_DEFAULT,
) -> ExerciseRecognitionBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Exercise recognition checkpoint not found: {path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    train_mean = np.asarray(ckpt["train_mean"], dtype=np.float32)
    train_std = np.asarray(ckpt["train_std"], dtype=np.float32)
    train_std = np.maximum(train_std, 1e-6)

    input_dim = int(ckpt["input_dim"])

    active_ids = ckpt.get("active_exercise_ids")
    if not active_ids:
        active_ids = sorted(SUPPORTED_EXERCISES.values())

    active_ids = [int(x) for x in active_ids]

    exercise_to_index = {ex_id: idx for idx, ex_id in enumerate(active_ids)}
    index_to_exercise = {idx: ex_id for ex_id, idx in exercise_to_index.items()}

    model = ExerciseTCN(
        input_dim=input_dim,
        num_exercises=len(active_ids),
        dropout=dropout,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return ExerciseRecognitionBundle(
        model=model,
        train_mean=train_mean,
        train_std=train_std,
        input_dim=input_dim,
        active_exercise_ids=active_ids,
        exercise_to_index=exercise_to_index,
        index_to_exercise=index_to_exercise,
        device=device,
    )