import numpy as np

class Projection:

    @staticmethod
    def project_to_camera(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate 2D projections of multiple 3D coordinates. 
        Points must be in already in camera coordinate system.
        Returns image position x,y and depth"""
        # Store original shape
        original_shape = points.shape
        
        # Reshape to (-1, 3)
        points = points.reshape(-1, 3)
        
        # Change from UE4's coordinate system to "standard"
        # (x, y, z) -> (y, -z, x)
        points = points[:, [1, 2, 0]]
        points[:, 1] *= -1
        
        # Project 3D->2D using the camera matrix
        points_img: np.ndarray = np.dot(K, points.T).T

        # Normalize with the depth value of x but use the absolute value 
        # to ensure it won't be mirrored for objects behind the camera
        points_img[:, :2] /= np.abs(points_img[:, 2:3])

        points_img = points_img.reshape(original_shape)

        return points_img[..., :2].astype(int), points_img[..., 2]

    @staticmethod
    def project_to_camera_2(points: np.ndarray, K: np.ndarray, image_width: int, image_height: int, axis=1, keep_dim=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate 2D projections of multiple 3D coordinates. 
        Points must be already in camera coordinate system.
        Returns image positions (x,y), depths, and indices of valid points.
        """
        # Change from UE4's coordinate system to "standard"
        # (x, y, z) -> (y, -z, x)
        points = points[:, [1, 2, 0]]
        points[:, 1] *= -1
        
        # Project 3D->2D using the camera matrix
        points_img: np.ndarray = np.dot(K, points.T).T
        
        # Normalize with the depth value
        depths = points_img[:, 2]
        points_img[:, :2] /= depths[:, np.newaxis]
        
        # Round to nearest integer for pixel coordinates
        points_img = np.round(points_img).astype(int)
        
        # Create a mask for valid points (inside image and positive depth)
        valid_mask = (
            (points_img[:, 0] >= 0) & 
            (points_img[:, 0] < image_width) & 
            (points_img[:, 1] >= 0) & 
            (points_img[:, 1] < image_height) & 
            (depths > 0)
        )
        
        # Get indices of valid points
        valid_indices = np.where(valid_mask)[0]
        
        # Filter points and depths
        valid_points = points_img[valid_mask, :2]
        valid_depths = depths[valid_mask]

        return valid_points, valid_depths, valid_indices
    
    @staticmethod
    def build_projection_matrix(w: int, h: int, fov: float):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
