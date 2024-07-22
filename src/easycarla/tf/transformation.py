import numpy as np

class Transformation:

    @staticmethod
    def transform_with_matrix(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Calculate 3d transformation"""
        # Store original shape
        original_shape = points.shape
        
        # Reshape to (-1, 3)
        points_flat = points.reshape(-1, 3)
        
        # Add homogeneous coordinate
        points_homogeneous = np.hstack((points_flat, np.ones((points_flat.shape[0], 1))))
        
        # Transform to camera coordinates
        points_transformed = np.dot(matrix, points_homogeneous.T).T
        
        # Reshape back to original shape
        points_transformed = points_transformed[..., :3].reshape(original_shape)
        
        return points_transformed