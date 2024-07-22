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
    extrinsics: np.ndarray
    intrinsics: np.ndarray = None

    @property
    def K(self):
        return self.camera_intrinsics
    
    @property
    def R(self):
        return self.camera_extrinsics[:3, :3]
    
    @property
    def T(self):
        return self.camera_intrinsics[:3, 3]
