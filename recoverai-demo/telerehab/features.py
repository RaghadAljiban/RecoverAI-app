import numpy as np

from telerehab.config import (
    BONE_CONNECTIONS,
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
    TARGET_FRAMES,
)


def compute_resample_indices(n_frames: int, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((target_frames,), dtype=np.int64)
    if n_frames == target_frames:
        return np.arange(target_frames, dtype=np.int64)
    idxs = np.linspace(0, n_frames - 1, target_frames)
    idxs = np.round(idxs).astype(np.int64)
    return np.clip(idxs, 0, n_frames - 1)


class FeatureExtractor:
    def __init__(
        self,
        train_mean: np.ndarray,
        train_std: np.ndarray,
        target_frames: int = TARGET_FRAMES,
    ):
        self.train_mean = np.asarray(train_mean, dtype=np.float32)
        self.train_std = np.maximum(np.asarray(train_std, dtype=np.float32), 1e-6)
        self.target_frames = int(target_frames)

    def __call__(self, raw_world_clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        world_norm = self._normalize(raw_world_clip)
        clip_fixed = self._resample(world_norm)
        clip_fixed = self._add_velocity(clip_fixed)
        clip_fixed = self._add_bones(clip_fixed)
        angle_feat = self._extract_angles(clip_fixed)

        pose_flat = clip_fixed.reshape(self.target_frames, -1)
        x_all = np.concatenate([pose_flat, angle_feat], axis=-1).astype(np.float32, copy=False)
        x_all = np.nan_to_num(x_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        x_all = ((x_all - self.train_mean) / self.train_std).astype(np.float32, copy=False)
        return x_all, world_norm

    @staticmethod
    def _normalize(raw_clip: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        raw_clip = np.asarray(raw_clip, dtype=np.float32)
        if raw_clip.ndim != 3 or raw_clip.shape[1:] != (33, 4):
            raise ValueError(f"Expected raw clip shape (T, 33, 4), got {raw_clip.shape}")

        xyz = raw_clip[..., :3].copy()
        vis = raw_clip[..., 3:4].copy()

        hip_center = 0.5 * (xyz[:, L_HIP] + xyz[:, R_HIP])
        shoulder_center = 0.5 * (xyz[:, L_SHOULDER] + xyz[:, R_SHOULDER])

        torso_len = np.linalg.norm(shoulder_center - hip_center, axis=-1)
        shoulder_width = np.linalg.norm(xyz[:, L_SHOULDER] - xyz[:, R_SHOULDER], axis=-1)
        hip_width = np.linalg.norm(xyz[:, L_HIP] - xyz[:, R_HIP], axis=-1)

        scale_candidates = np.concatenate([
            torso_len[torso_len > eps],
            shoulder_width[shoulder_width > eps],
            hip_width[hip_width > eps],
        ])
        scale = float(np.median(scale_candidates)) if scale_candidates.size else 1.0
        scale = max(scale, eps)

        xyz = (xyz - hip_center[:, None, :]) / scale
        return np.concatenate([xyz, vis], axis=-1).astype(np.float32, copy=False)

    def _resample(self, clip: np.ndarray) -> np.ndarray:
        clip = np.asarray(clip, dtype=np.float32)
        if clip.ndim != 3 or clip.shape[1:] != (33, 4):
            raise ValueError(f"Expected clip shape (T, 33, 4), got {clip.shape}")
        T = clip.shape[0]
        if T == 0:
            return np.zeros((self.target_frames, 33, 4), dtype=np.float32)
        if T == self.target_frames:
            return clip.astype(np.float32, copy=False)
        idxs = compute_resample_indices(T, self.target_frames)
        return clip[idxs].astype(np.float32, copy=False)

    @staticmethod
    def _add_velocity(pose_seq: np.ndarray) -> np.ndarray:
        pose_seq = np.asarray(pose_seq, dtype=np.float32)
        xyz = pose_seq[..., :3]
        vis = pose_seq[..., 3:4]
        vel = np.zeros_like(xyz, dtype=np.float32)
        vel[1:] = xyz[1:] - xyz[:-1]
        return np.concatenate([xyz, vis, vel], axis=-1).astype(np.float32, copy=False)

    @staticmethod
    def _add_bones(pose_seq: np.ndarray) -> np.ndarray:
        pose_seq = np.asarray(pose_seq, dtype=np.float32)
        xyz = pose_seq[..., :3]
        bones = np.zeros_like(xyz, dtype=np.float32)
        for parent, child in BONE_CONNECTIONS:
            bones[:, child] = xyz[:, child] - xyz[:, parent]
        return np.concatenate([pose_seq, bones], axis=-1).astype(np.float32, copy=False)

    @staticmethod
    def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        ba = a - b
        bc = c - b
        ba = ba / np.maximum(np.linalg.norm(ba, axis=-1, keepdims=True), eps)
        bc = bc / np.maximum(np.linalg.norm(bc, axis=-1, keepdims=True), eps)
        cosang = np.clip(np.sum(ba * bc, axis=-1), -1.0, 1.0)
        return np.arccos(cosang).astype(np.float32)

    @classmethod
    def _extract_angles(cls, pose_seq: np.ndarray) -> np.ndarray:
        xyz = np.asarray(pose_seq, dtype=np.float32)[..., :3]
        return np.stack([
            cls._angle_3pts(xyz[:, L_SHOULDER], xyz[:, L_ELBOW], xyz[:, L_WRIST]),
            cls._angle_3pts(xyz[:, R_SHOULDER], xyz[:, R_ELBOW], xyz[:, R_WRIST]),
            cls._angle_3pts(xyz[:, L_HIP], xyz[:, L_KNEE], xyz[:, L_ANKLE]),
            cls._angle_3pts(xyz[:, R_HIP], xyz[:, R_KNEE], xyz[:, R_ANKLE]),
            cls._angle_3pts(xyz[:, L_ELBOW], xyz[:, L_SHOULDER], xyz[:, L_HIP]),
            cls._angle_3pts(xyz[:, R_ELBOW], xyz[:, R_SHOULDER], xyz[:, R_HIP]),
            cls._angle_3pts(xyz[:, L_SHOULDER], xyz[:, L_HIP], xyz[:, L_KNEE]),
            cls._angle_3pts(xyz[:, R_SHOULDER], xyz[:, R_HIP], xyz[:, R_KNEE]),
        ], axis=-1).astype(np.float32)
