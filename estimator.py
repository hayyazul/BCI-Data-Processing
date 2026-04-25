
import cv2
import numpy as np
from pupil_apriltags import Detector
from typing import List
import warnings

from models import CameraIntrinsics, TagMount, Position, PositionEstimate


# ==============================================================================
# POSE ESTIMATOR
# ==============================================================================

class MonocularPoseEstimator:
    def __init__(self, camera_intrinsics: CameraIntrinsics, motion_optimized=True):
        self.camera_intrinsics = camera_intrinsics
        self.detectors = {}
        self.motion_optimized = motion_optimized
        
    def _get_detector(self, tag_family: str) -> Detector:
        if tag_family not in self.detectors:
            if self.motion_optimized:
                self.detectors[tag_family] = Detector(
                    families=tag_family, nthreads=4,
                    quad_decimate=1.5, quad_sigma=0.0,
                    refine_edges=True, decode_sharpening=0.5,
                )
            else:
                self.detectors[tag_family] = Detector(
                    families=tag_family, nthreads=4,
                    quad_decimate=1.0, quad_sigma=0.0,
                    refine_edges=True, decode_sharpening=0.25,
                )
        return self.detectors[tag_family]
    
    def detect_tags_with_size(self, frame: np.ndarray, tag_configs: List[TagMount]) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_tags = []
        configs_by_key = {}
        for config in tag_configs:
            configs_by_key.setdefault((config.tag_family, config.tag_size), []).append(config)
        
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