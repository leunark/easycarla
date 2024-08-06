from dataclasses import dataclass
import numpy as np

@dataclass
class CalibrationData:
    """
    Class to store calibration data.

    Attributes:
        extrinsics (np.ndarray): Camera extrinsic parameters matrix.
        intrinsics (np.ndarray): Camera intrinsic parameters matrix.
        lidar_to_camera (np.ndarray): Transformation matrix from LiDAR to camera coordinates.
    """
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    lidar_to_camera: np.ndarray
