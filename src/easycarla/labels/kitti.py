import os
import numpy as np
import cv2
from pathlib import Path

from easycarla.labels.label_data import LabelData
from easycarla.labels.label_types import ObjectType
from easycarla.sensors import LidarSensor, CameraSensor, DepthCameraSensor

class KITTIDatasetGenerator:
    def __init__(self, base_dir: Path|str):
        self.base_dir = Path(base_dir)
        self.depth_camera_dir = self.base_dir / 'image_1'
        self.camera_dir = self.base_dir / 'image_2'
        self.velodyne_dir = self.base_dir / 'velodyne'
        self.calib_dir = self.base_dir / 'calib'
        self.label_dir = self.base_dir / 'label_2'
        self.frame_id = -1
        self.create_directories()

    def create_directories(self):
        self.depth_camera_dir.mkdir(parents=True, exist_ok=True)
        self.camera_dir.mkdir(parents=True, exist_ok=True)
        self.velodyne_dir.mkdir(parents=True, exist_ok=True)
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)

    def save_depth_image(self, image: np.ndarray):
        image_path = self.depth_camera_dir / f'{self.frame_id:06}.png'
        cv2.imwrite(str(image_path), image[..., ::-1])

    def save_image(self, image: np.ndarray):
        image_path = self.camera_dir / f'{self.frame_id:06}.png'
        cv2.imwrite(str(image_path), image[..., ::-1])

    def save_point_cloud(self, point_cloud: np.ndarray):
        point_cloud_path = self.velodyne_dir / f'{self.frame_id:06}.bin'
        point_cloud.tofile(point_cloud_path)

    def save_calibration(self, P2: np.ndarray, Tr_velo_to_cam: np.ndarray):
        calib_path = self.calib_dir / 'calib.txt'
        with calib_path.open('w') as f:
            f.write(self.format_calibration(P2, Tr_velo_to_cam))

    def format_calibration(self, P2: np.ndarray, Tr_velo_to_cam: np.ndarray):
        calib_text = ''
        calib_text += 'P0: ' + ' '.join(['0'] * 12) + '\n'
        calib_text += 'P1: ' + ' '.join(['0'] * 12) + '\n'
        calib_text += 'P2: ' + ' '.join([f'{v:.12e}' for v in P2.flatten()]) + '\n'
        calib_text += 'P3: ' + ' '.join(['0'] * 12) + '\n'
        calib_text += 'R0_rect: ' + ' '.join(['1' if i % 4 == 0 else '0' for i in range(9)]) + '\n'
        calib_text += 'Tr_velo_to_cam: ' + ' '.join([f'{v:.12e}' for v in Tr_velo_to_cam.flatten()]) + '\n'
        calib_text += 'Tr_imu_to_velo: ' + ' '.join(['0'] * 12) + '\n'  # Optional, fill with zeros if not used
        return calib_text

    def save_label(self, labels: LabelData):
        label_path = self.label_dir / f'{self.frame_id:06}.txt'
        with label_path.open('w') as f:
            for i in range(len(labels.id)):
                f.write(self.format_label(labels, i))

    def format_label(self, labels: LabelData, index: int):
        bbox = labels.transform[index][:3, 3]  # Assuming this is the bounding box location in the form of [x, y, z]
        dimension = labels.dimension[index]
        rotation_y = np.arctan2(labels.transform[index][1, 0], labels.transform[index][0, 0])
        
        label_str = f"{next(iter(labels.types[index])).value} "
        label_str = f"{labels.truncation[index] if labels.truncation is not None else 0} "
        label_str += f"{labels.occlusion[index] if labels.occlusion is not None else 0} " 
        label_str += f"{labels.alpha[index] if labels.alpha is not None else 0} "
        label_str += f"{bbox[0]} {bbox[1]} {bbox[2]} "
        label_str += f"{dimension[0]} {dimension[1]} {dimension[2]} "
        label_str += f"{bbox[0]} {bbox[1]} {bbox[2]} "
        label_str += f"{rotation_y}\n"
        return label_str

    def process_frame(self, pointcloud: np.ndarray, image: np.ndarray, depth_image: np.ndarray, labels: LabelData, frame_id: int = None):
        self.frame_id = frame_id if frame_id is not None else self.frame_id + 1
        self.save_point_cloud(pointcloud)
        self.save_depth_image(depth_image)
        self.save_image(image)
        self.save_label(labels)

    def set_calibration(self, P2: np.ndarray, Tr_velo_to_cam: np.ndarray):
        self.save_calibration(P2, Tr_velo_to_cam)
