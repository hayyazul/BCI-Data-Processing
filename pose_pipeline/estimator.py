
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from pupil_apriltags import Detector
from typing import List, Optional, Dict
import warnings

from models import CameraIntrinsics, TagMount, Position, PositionEstimate


# ==============================================================================
# DETECTOR PARAMS
# ==============================================================================

@dataclass
class DetectorParams:
    """Tunable inputs to pupil_apriltags.Detector. The estimator rebuilds its
    cached detectors whenever any of these change, so they're safe to swap at
    runtime from a tuner UI."""
    quad_decimate: float = 1.0
    quad_sigma: float = 0.8
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    nthreads: int = 4
    decision_margin_threshold: float = 0.0  # post-filter, not passed to Detector

    @staticmethod
    def motion_optimized() -> "DetectorParams":
        return DetectorParams(quad_decimate=1.0, quad_sigma=0.8,
                              refine_edges=True, decode_sharpening=0.25,
                              nthreads=4, decision_margin_threshold=0.0)

    @staticmethod
    def max_reliability() -> "DetectorParams":
        return DetectorParams(quad_decimate=1.0, quad_sigma=1.0,
                              refine_edges=True, decode_sharpening=0.5,
                              nthreads=4, decision_margin_threshold=0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "DetectorParams":
        # Forward-compat: drop unknown keys, fall back to defaults for missing.
        defaults = DetectorParams()
        kw = {f: d.get(f, getattr(defaults, f)) for f in defaults.__dataclass_fields__}
        return DetectorParams(**kw)


# ==============================================================================
# POSE ESTIMATOR
# ==============================================================================

class MonocularPoseEstimator:
    def __init__(self, camera_intrinsics: CameraIntrinsics,
                 motion_optimized: bool = True,
                 params: Optional[DetectorParams] = None):
        self.camera_intrinsics = camera_intrinsics
        self.detectors: Dict[str, Detector] = {}
        if params is not None:
            self.params = params
        else:
            self.params = (DetectorParams.motion_optimized()
                           if motion_optimized else DetectorParams.max_reliability())
        # Per-Position state for single-tag (sticky) mode.
        self._sticky_tag_per_position: Dict[str, int] = {}

    def set_params(self, params: DetectorParams) -> None:
        """Swap detector params and invalidate cached Detector instances if any
        Detector-affecting field actually changed. Threshold-only changes are
        free."""
        old = self.params
        self.params = params
        rebuild_keys = ("quad_decimate", "quad_sigma", "refine_edges",
                        "decode_sharpening", "nthreads")
        if any(getattr(old, k) != getattr(params, k) for k in rebuild_keys):
            self.detectors = {}

    def _get_detector(self, tag_family: str) -> Detector:
        if tag_family not in self.detectors:
            p = self.params
            self.detectors[tag_family] = Detector(
                families=tag_family, nthreads=int(p.nthreads),
                quad_decimate=float(p.quad_decimate),
                quad_sigma=float(p.quad_sigma),
                refine_edges=bool(p.refine_edges),
                decode_sharpening=float(p.decode_sharpening),
            )
        return self.detectors[tag_family]

    def detect_tags_with_size(self, frame: np.ndarray, tag_configs: List[TagMount]) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_tags = []
        configs_by_key = {}
        for config in tag_configs:
            configs_by_key.setdefault((config.tag_family, config.tag_size), []).append(config)

        thr = float(self.params.decision_margin_threshold)
        for (family, tag_size), configs in configs_by_key.items():
            detector = self._get_detector(family)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*more than one new minima.*")
                tags = detector.detect(
                    gray, estimate_tag_pose=True,
                    camera_params=[
                        self.camera_intrinsics.camera_matrix[0, 0],
                        self.camera_intrinsics.camera_matrix[1, 1],
                        self.camera_intrinsics.camera_matrix[0, 2],
                        self.camera_intrinsics.camera_matrix[1, 2],
                    ],
                    tag_size=tag_size,
                )
            if thr > 0:
                tags = [t for t in tags if t.decision_margin >= thr]
            all_tags.extend(tags)
        return all_tags

    def estimate_position(self, position: Position, tags: list, timestamp: float) -> List[PositionEstimate]:
        estimates = []
        detected_by_id = {tag.tag_id: tag for tag in tags}
        for tag_mount in position.tags:
            if tag_mount.tag_id not in detected_by_id:
                continue
            detected = detected_by_id[tag_mount.tag_id]
            R_tag_in_cam = detected.pose_R
            t_tag_in_cam = detected.pose_t.reshape(3, 1)
            R_mount = tag_mount.mount_orientation
            t_mount = tag_mount.mount_offset
            R_pos_in_cam = R_tag_in_cam @ R_mount
            t_pos_in_cam = R_tag_in_cam @ t_mount + t_tag_in_cam
            estimates.append(PositionEstimate(
                position_name=position.name, timestamp=timestamp,
                rotation=R_pos_in_cam, translation=t_pos_in_cam,
                confidence=detected.decision_margin,
                source_tag_id=tag_mount.tag_id,
            ))
        return estimates

    def select_single_estimate(self, position_name: str,
                               estimates: List[PositionEstimate]) -> List[PositionEstimate]:
        """Reduce estimates for a Position to a single one, sticky to the last
        used tag if it's still detected. Otherwise pick the highest-confidence
        tag and remember it. Empty input → empty output (sticky preference is
        retained for when the tag returns)."""
        if not estimates:
            return []
        last_id = self._sticky_tag_per_position.get(position_name)
        if last_id is not None:
            for e in estimates:
                if e.source_tag_id == last_id:
                    return [e]
        best = max(estimates, key=lambda e: e.confidence)
        self._sticky_tag_per_position[position_name] = best.source_tag_id
        return [best]

    def reset_sticky_selection(self) -> None:
        self._sticky_tag_per_position.clear()
