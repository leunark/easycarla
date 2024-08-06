import numpy as np
import logging
import cv2
from pathlib import Path
from tqdm import tqdm
from easycarla.labels.label_data import LabelData

class KITTIDatasetGenerator:
    """
    Class to generate KITTI dataset format from CARLA simulation data.

    Attributes:
        base_dir (Path|str): Base directory to save the dataset.
        frame_interval (int): Interval of frames to process.
        train_ratio (float): Ratio of training data.
        val_ratio (float): Ratio of validation data.
        test_ratio (float): Ratio of test data.
        frame_count (int): Number of frames to process.
    """
    def __init__(self, base_dir: Path|str, frame_interval = 1, frame_count = 1000, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15):
        self.base_dir = Path(base_dir)
        self.frame_interval = frame_interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.frame_count = frame_count

        self.depth_camera_dir = self.base_dir / 'image_1'
        self.camera_dir = self.base_dir / 'image_2'
        self.velodyne_dir = self.base_dir / 'velodyne'
        self.calib_dir = self.base_dir / 'calib'
        self.label_dir = self.base_dir / 'label_2'
        self.image_sets_dir = self.base_dir / 'ImageSets'

        self.frame_id = 0
        self.create_directories()
        self.T = np.array([  # Transformation matrix from CARLA to KITTI
            [1,  0, 0, 0],
            [0, -1, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])
        self.pbar = tqdm(total=frame_count)

    def create_directories(self):
        """Create necessary directories for saving dataset files."""
        self.depth_camera_dir.mkdir(parents=True, exist_ok=True)
        self.camera_dir.mkdir(parents=True, exist_ok=True)
        self.velodyne_dir.mkdir(parents=True, exist_ok=True)
        self.calib_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.image_sets_dir.mkdir(parents=True, exist_ok=True)

    def save_depth_image(self, image: np.ndarray):
        """
        Save depth image to file.

        Args:
            image (np.ndarray): Depth image array.
        """
        image_path = self.depth_camera_dir / f'{self.frame_id:06}.png'
        cv2.imwrite(str(image_path), image[..., ::-1])

    def save_image(self, image: np.ndarray):
        """
        Save RGB image to file.

        Args:
            image (np.ndarray): RGB image array.
        """
        image_path = self.camera_dir / f'{self.frame_id:06}.png'
        cv2.imwrite(str(image_path), image[..., ::-1])

    def save_point_cloud(self, point_cloud: np.ndarray):
        """
        Save point cloud to file.

        Args:
            point_cloud (np.ndarray): Point cloud array.
        """
        point_cloud_path = self.velodyne_dir / f'{self.frame_id:06}.bin'
        point_cloud.tofile(point_cloud_path)

    def save_calibration(self, P2: np.ndarray, Tr_velo_to_cam: np.ndarray):
        """
        Save calibration data to file.

        Args:
            P2 (np.ndarray): Projection matrix.
            Tr_velo_to_cam (np.ndarray): Transformation matrix from LiDAR to camera.
        """
        calib_path = self.calib_dir / 'calib.txt'
        with calib_path.open('w') as f:
            f.write(self.format_calibration(P2, Tr_velo_to_cam))

    def save_timestamp(self, timestamp: float):
        """
        Save timestamp to file.

        Args:
            timestamp (float): Timestamp value.
        """
        times_path = self.base_dir / 'times.txt'
        with times_path.open('a') as f:
            f.write(f"{timestamp}\n")

    def format_calibration(self, P2: np.ndarray, Tr_velo_to_cam: np.ndarray):
        """
        Format calibration data into text.

        Args:
            P2 (np.ndarray): Projection matrix.
            Tr_velo_to_cam (np.ndarray): Transformation matrix from LiDAR to camera.

        Returns:
            str: Formatted calibration data.
        """
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
        """
        Save label data to file.

        Args:
            labels (LabelData): Label data object.
        """
        label_path = self.label_dir / f'{self.frame_id:06}.txt'
        with label_path.open('w') as f:
            for i in range(len(labels.id)):
                f.write(self.format_label(labels, i))

    def save_frame_id(self):
        """
        Save frame ID to appropriate set (train, val, test) based on defined ratios.
        """
        rand = np.random.random()
        if rand < self.train_ratio:
            set_name = 'train'
        elif rand < self.train_ratio + self.val_ratio:
            set_name = 'val'
        elif rand < self.train_ratio + self.val_ratio + self.test_ratio:
            set_name = 'test'
        frame_path = self.image_sets_dir / f'{set_name}.txt'
        with frame_path.open('a') as f:
            f.write(f"{self.frame_id:06}\n")

    def format_label(self, labels: LabelData, index: int):
        """
        Format label data into text.

        Args:
            labels (LabelData): Label data object.
            index (int): Index of the label.

        Returns:
            str: Formatted label data.
        """
        bbox = labels.transform[index][:3, 3]  # Assuming this is the bounding box location in the form of [x, y, z]
        dimension = labels.dimension[index]
        # It assumes a specific rotation order (z-y-x or yaw-pitch-roll).
        yaw_angle = np.arctan2(labels.transform[index][1, 0], labels.transform[index][0, 0])
        occlusion = labels.occlusion[index] if labels.occlusion is not None else 0
        # 0: Fully visible
        if occlusion <= 0.6:
            occlusion_level = 0
        # 1: Partly occluded
        elif occlusion <= 0.8:
            occlusion_level = 1
        # 2: Largely occluded
        elif occlusion <= 1.0:
            occlusion_level = 2
        # 3: Unknown level of occlusion
        else:
            occlusion_level = 3

        alpha = labels.get_alpha()
        label_str = f"{next(iter(labels.types[index])).value} "
        label_str += f"{labels.truncation[index] if labels.truncation is not None else 0} "
        label_str += f"{occlusion_level} " 
        label_str += f"{alpha[index] if alpha is not None else 0} "
        label_str += f"{-1} {-1} {-1} {-1} "
        label_str += f"{dimension[0]} {dimension[1]} {dimension[2]} "
        label_str += f"{bbox[0]} {bbox[1]} {bbox[2]} "
        label_str += f"{yaw_angle}\n"
        return label_str

    def process_frame(self, pointcloud: np.ndarray, image: np.ndarray, depth_image: np.ndarray, labels: LabelData, timestamp: float, world_frame_id: int):
        """
        Process a single frame and save relevant data.

        Args:
            pointcloud (np.ndarray): Point cloud data.
            image (np.ndarray): RGB image.
            depth_image (np.ndarray): Depth image.
            labels (LabelData): Label data.
            timestamp (float): Timestamp value.
            world_frame_id (int): World frame ID.
        """
        if world_frame_id % self.frame_interval != 0:
            return
        if self.frame_id == self.frame_count:
            self.pbar.close()
            logging.info(f"Finished dataset with {self.frame_count} samples!")
            self.frame_id = self.frame_id + 1
            return
        if self.frame_id > self.frame_count:
            return
        
        # Transform pointcloud to kitti coordinate system
        pointcloud = np.column_stack((pointcloud[:, 0], -1 * pointcloud[:, 1], pointcloud[:, 2], pointcloud[:, 3]))
        # Same for labels
        labels = labels.apply_transform(self.T)
        self.save_point_cloud(pointcloud)
        self.save_depth_image(depth_image)
        self.save_image(image)
        self.save_label(labels)
        self.save_timestamp(timestamp)
        self.save_frame_id()

        self.pbar.update()
        self.frame_id = self.frame_id + 1

    def set_calibration(self, P2: np.ndarray, Tr_velo_to_cam: np.ndarray):
        """
        Set calibration data.

        Args:
            P2 (np.ndarray): Projection matrix.
            Tr_velo_to_cam (np.ndarray): Transformation matrix from LiDAR to camera.
        """
        # Add a column with zeros to the end of P2 to make it 3x4
        P2 = np.hstack((P2, np.zeros((3, 1))))
        # Apply transformation matrix T to Tr_velo_to_cam
        Tr_velo_to_cam = self.T @ Tr_velo_to_cam
        self.save_calibration(P2, Tr_velo_to_cam)