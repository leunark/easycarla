import carla
from pathlib import Path
import numpy as np
import cv2

from easycarla.sim.carla_sync_mode import CarlaSyncMode
from easycarla.sim.traffic_simulation import TrafficSimulation

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

def transform_lidar_to_camera(points, lidar_transform, camera_transform):
    """
    Transform LiDAR points to the camera coordinate system.
    """
    lidar_to_world = lidar_transform.get_matrix()
    world_to_camera = np.linalg.inv(camera_transform.get_matrix())
    lidar_to_camera = np.dot(world_to_camera, lidar_to_world)
    
    # Convert points to homogeneous coordinates
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Transform points to camera coordinate system
    points_camera = np.dot(lidar_to_camera, points.T).T
    
    return points_camera

def project_points_to_image(points, K):
    """
    Project 3D points onto 2D image plane.
    """
    # Ignore points behind the camera
    points = points[points[:, 2] > 0]
    
    # Project points onto image plane
    points_2d = np.dot(K, points[:, :3].T).T
    
    # Normalize by the third (z) coordinate
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    
    return points_2d

def draw_points_on_image(image, points):
    """
    Draw the projected points on the image.
    """
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

def process_data(world_snapshot: carla.WorldSnapshot, lidar: carla.LidarMeasurement, image_rgb: carla.Image, image_depth: carla.Image, image_insemseg: carla.Image):

    # Get transformations
    lidar_transform = lidar.transform
    camera_transform = image_rgb.transform

    # Get camera intrinsic matrix
    K = get_camera_intrinsic(image_rgb)
    
    # Extract LiDAR points
    points = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    
    # Transform LiDAR points to camera coordinate system
    points_camera = transform_lidar_to_camera(points[:, :3], lidar_transform, camera_transform)
    
    # Project points to image plane
    points_2d = project_points_to_image(points_camera, K)
    
    # Convert RGB image to numpy array
    image_array = np.frombuffer(image_rgb.raw_data, dtype=np.uint8).reshape((image_rgb.height, image_rgb.width, 4))[:, :, :3]
    
    # Draw points on the image
    image_with_points = draw_points_on_image(image_array, points_2d)
    
    # Save the image
    cv2.imwrite(Path("data/image_proj") / f"img_{image_rgb.frame}.png", image_with_points)


#lidar.save_to_disk(str(Path("data/lidar") / f'lidar_output_{lidar.frame}.ply'))
#image_rgb.save_to_disk(str(Path("data/image_rgb") / f'image_rgb_output_{image_rgb.frame}.png'))
#image_insemseg.save_to_disk(str(Path("data/image_insemseg") / f'image_insemseg_output_{image_rgb.frame}.png'))