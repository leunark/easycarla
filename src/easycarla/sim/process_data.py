import carla
from pathlib import Path
import numpy as np
import cv2


def get_camera_intrinsic(image_rgb):
    """
    Returns the camera intrinsic matrix.
    """
    image_w = image_rgb.width
    image_h = image_rgb.height
    fov = 90  # Assuming 90° FOV for simplicity, adjust if necessary
    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    return K

def save_to_disk(lidar: carla.LidarMeasurement = None, image_rgb: carla.Image = None, image_depth: carla.Image = None, image_insemseg: carla.Image = None, output_dir: Path = Path("data")):
    if lidar is not None:
        lidar_dir = output_dir / "lidar"
        lidar_dir.mkdir(parents=True, exist_ok=True)
        lidar.save_to_disk(str(lidar_dir / f'{lidar.frame}.ply'))
    if image_rgb is not None:
        image_rgb_dir = output_dir / "image_rgb"
        image_rgb_dir.mkdir(parents=True, exist_ok=True)
        image_rgb.save_to_disk(str(image_rgb_dir / f'{image_rgb.frame}.png'))
    if image_depth is not None:
        image_depth_dir = output_dir / "image_depth"
        image_depth_dir.mkdir(parents=True, exist_ok=True)
        image_depth.save_to_disk(str(image_depth_dir / f'{image_depth.frame}.png'), carla.ColorConverter.Depth)
    if image_insemseg is not None:
        image_insemseg_dir = output_dir / "image_insemseg"
        image_insemseg_dir.mkdir(parents=True, exist_ok=True)
        image_insemseg.save_to_disk(str(image_insemseg_dir / f'{image_insemseg.frame}.png'))

def process_data(world_snapshot: carla.WorldSnapshot, lidar: carla.LidarMeasurement, image_rgb: carla.Image, image_depth: carla.Image, image_insemseg: carla.Image):

    # Get transformations
    lidar_transform = lidar.transform
    camera_transform = image_rgb.transform

    # Get camera intrinsic matrix
    K = get_camera_intrinsic(image_rgb)
    
    # Extract LiDAR points
    lidar_points = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'))
    lidar_points = np.reshape(lidar_points, (int(lidar_points.shape[0] / 4), 4))
    
    # Convert RGB image to numpy array
    img_rgb = np.frombuffer(image_rgb.raw_data, dtype=np.uint8).reshape((image_rgb.height, image_rgb.width, 4))[:, :, :3]

    # Save to disk
    save_to_disk(lidar=lidar, image_rgb=image_rgb, image_depth=image_depth, image_insemseg=image_insemseg)
