import numpy as np
from dataclasses import dataclass, field
from typing import List


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    width: int
    height: int
    
    def __post_init__(self):
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).flatten()


@dataclass
class TagMount:
    tag_id: int
    tag_family: str
    mount_orientation: np.ndarray
    mount_offset: np.ndarray
    tag_size: float
    
    def __post_init__(self):
        self.mount_orientation = np.asarray(self.mount_orientation, dtype=np.float64)
        self.mount_offset = np.asarray(self.mount_offset, dtype=np.float64).reshape(3, 1)


@dataclass
class Position:
    name: str
    tags: List[TagMount] = field(default_factory=list)


@dataclass
class PositionEstimate:
    position_name: str
    timestamp: float
    rotation: np.ndarray
    translation: np.ndarray
    confidence: float
    source_tag_id: int
    
    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3, 1)


# ==============================================================================
# ROTATION HELPERS
# ==============================================================================

def rotation_matrix_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotation_matrix_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ==============================================================================
# CAMERA
# ==============================================================================

def get_camera_intrinsics(width=1280, height=720, hfov_deg=70):
    fx = width / (2 * np.tan(np.radians(hfov_deg) / 2))
    fy = fx
    cx = width / 2
    cy = height / 2
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)
    return CameraIntrinsics(camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, width=width, height=height)


# ==============================================================================
# TAG MOUNT FACTORIES
# ==============================================================================

def tag_facing_forward(tag_id, offset, tag_size, tag_family="tag36h11"):
    return TagMount(tag_id=tag_id, tag_family=tag_family,
                    mount_orientation=np.eye(3),
                    mount_offset=np.array(offset, dtype=np.float64), tag_size=tag_size)

def tag_facing_right(tag_id, offset, tag_size, tag_family="tag36h11"):
    return TagMount(tag_id=tag_id, tag_family=tag_family,
                    mount_orientation=rotation_matrix_y(np.pi/2),
                    mount_offset=np.array(offset, dtype=np.float64), tag_size=tag_size)

def tag_facing_left(tag_id, offset, tag_size, tag_family="tag36h11"):
    return TagMount(tag_id=tag_id, tag_family=tag_family,
                    mount_orientation=rotation_matrix_y(-np.pi/2),
                    mount_offset=np.array(offset, dtype=np.float64), tag_size=tag_size)

def tag_facing_back(tag_id, offset, tag_size, tag_family="tag36h11"):
    return TagMount(tag_id=tag_id, tag_family=tag_family,
                    mount_orientation=rotation_matrix_y(np.pi),
                    mount_offset=np.array(offset, dtype=np.float64), tag_size=tag_size)

def tag_facing_up(tag_id, offset, tag_size, tag_family="tag36h11"):
    return TagMount(tag_id=tag_id, tag_family=tag_family,
                    mount_orientation=rotation_matrix_x(-np.pi/2),
                    mount_offset=np.array(offset, dtype=np.float64), tag_size=tag_size)

def tag_facing_down(tag_id, offset, tag_size, tag_family="tag36h11"):
    return TagMount(tag_id=tag_id, tag_family=tag_family,
                    mount_orientation=rotation_matrix_x(np.pi/2),
                    mount_offset=np.array(offset, dtype=np.float64), tag_size=tag_size)