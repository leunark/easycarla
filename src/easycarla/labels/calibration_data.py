from dataclasses import dataclass
import numpy as np

@dataclass
class CalibrationData:
    """
    Class to store calibration data.
    
    Attributes:
        camera_intrinsics (np.ndarray): Camera intrinsic parameters matrix.
        camera_extrinsics (np.ndarray): Camera extrinsic parameters matrix.
        lidar_to_camera (np.ndarray): Transformation matrix from LiDAR to camera coordinates.
    """
    camera_intrinsics: np.ndarray
    camera_extrinsics: np.ndarray
    lidar_to_camera: np.ndarray