from dataclasses import dataclass, replace
import numpy as np
from easycarla.labels.label_types import ObjectType

@dataclass
class LabelData:
    """
    Class to store label data for objects.

    Attributes:
        id (np.ndarray): Array of object IDs.
        transform (np.ndarray): Array of 4x4 transformation matrices.
        dimension (np.ndarray): Array of object dimensions.
        types (list[set[ObjectType]]): List of object types.
        truncation (np.ndarray, optional): Array of truncation values.
        occlusion (np.ndarray, optional): Array of occlusion values.
    """
    id: np.ndarray
    transform: np.ndarray  # 4x4 transformation matrices
    dimension: np.ndarray
    types: list[set[ObjectType]]
    truncation: np.ndarray = None
    occlusion: np.ndarray = None

    # Define the 12 edges of a cube by vertex indices
    EDGE_INDICES = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ])

    # Define diagonals of the side faces
    DIAG_INDICES = np.array([
        [0, 5], [0, 7],
        [1, 6], [1, 4],
        [2, 7], [2, 5],
        [3, 4], [3, 6],
    ])

    # Define the 8 vertices of a unit cube
    UNIT_BOX = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ])

    # Add extra vertices along each edge to improve occlusion calculation
    # A subdivision count of two adds one extra vertex because it subdivides
    # the edge into two parts
    SUBDIVISION_COUNT = 3

    @property
    def position(self) -> np.ndarray:
        """Get the position of objects."""
        return self.transform[:, :3, 3]

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix of objects."""
        return self.transform[:, :3, :3]

    @property
    def vertices(self) -> np.ndarray:
        """Get the vertices of the bounding boxes."""
        unit_box = self.UNIT_BOX
        extra_verts = []
        lines = self.DIAG_INDICES
        for i, j in lines:
            for k in range(1, self.SUBDIVISION_COUNT):
                v = unit_box[i] + (unit_box[j] - unit_box[i]) / self.SUBDIVISION_COUNT * k
                extra_verts.append(v)
        extra_verts = np.array(extra_verts)
        unit_box = np.vstack((unit_box, extra_verts))

        # Scale the unit box by the dimensions
        # unit_box: Broadcasting across multiple objects
        # dimension: Broadcasting across the 8 vertices of each box
        scaled_box = unit_box[None, :, :] * self.dimension[:, None, :]

        # Rotate and translate the scaled box
        rotated_box = np.einsum('nij,nkj->nki', self.rotation_matrix, scaled_box)
        vertices = rotated_box + self.position[:, None, :]

        return vertices

    @property
    def edges(self) -> np.ndarray:
        """Get the edges of the bounding boxes."""
        vertices = self.vertices
        edges = vertices[:, self.EDGE_INDICES]
        return edges
    
    def __len__(self) -> int:
        """Get the number of labels."""
        return len(self.transform)

    def __add__(self, other: 'LabelData') -> 'LabelData':
        """
        Add two LabelData objects.

        Args:
            other (LabelData): The other LabelData object.

        Returns:
            LabelData: Combined LabelData object.
        """
        if not isinstance(other, LabelData):
            return NotImplemented

        return LabelData(
            id=np.concatenate((self.id, other.id), axis=0),
            transform=np.concatenate((self.transform, other.transform), axis=0),
            dimension=np.concatenate((self.dimension, other.dimension), axis=0),
            truncation=np.concatenate((self.truncation, other.truncation), axis=0) if self.truncation is not None and other.truncation is not None else None,
            occlusion=np.concatenate((self.occlusion, other.occlusion), axis=0) if self.occlusion is not None and other.occlusion is not None else None,
            types=np.concatenate((self.types, other.types), axis=0) if self.types is not None and other.types is not None else None,
        )

    def filter(self, mask: np.ndarray) -> 'LabelData':
        """
        Filter the labels based on a mask.

        Args:
            mask (np.ndarray): Boolean mask.

        Returns:
            LabelData: Filtered LabelData object.
        """
        return LabelData(
            id=self.id[mask],
            transform=self.transform[mask],
            dimension=self.dimension[mask],
            truncation=self.truncation[mask] if self.truncation is not None else None,
            occlusion=self.occlusion[mask] if self.occlusion is not None else None,
            types=[self.types[i] for i in np.nonzero(mask)[0]],
        )

    def filter_by_distance(self, distance: float, target: np.ndarray = None) -> 'LabelData':
        """
        Filter the labels by distance from a target point.

        Args:
            distance (float): Maximum distance.
            target (np.ndarray, optional): Target point. Defaults to None.

        Returns:
            LabelData: Filtered LabelData object.
        """
        target = target if target is not None else np.zeros(3)
        delta_pos = self.position - target
        sq_dist = np.sum(delta_pos**2, axis=1)
        mask = sq_dist <= distance**2
        return self.filter(mask)
    
    def filter_by_direction(self, dot_threshold: float = 0.0, target: np.ndarray = None, forward: np.ndarray = None) -> 'LabelData':
        """
        Filter the labels by direction relative to a target point.

        Args:
            dot_threshold (float, optional): Dot product threshold. Defaults to 0.0.
            target (np.ndarray, optional): Target point. Defaults to None.
            forward (np.ndarray, optional): Forward direction. Defaults to None.

        Returns:
            LabelData: Filtered LabelData object.
        """
        target = target if target is not None else np.zeros(3)
        forward = forward if forward is not None else np.array([1, 0, 0])
        delta_pos = self.position - target
        delta_pos_unit = delta_pos / np.linalg.norm(delta_pos, axis=1)[:, None]
        mask = np.dot(delta_pos_unit, forward) > dot_threshold
        return self.filter(mask)
    
    def filter_by_id(self, ids: np.ndarray) -> 'LabelData':
        """
        Filter the labels by IDs.

        Args:
            ids (np.ndarray): Array of IDs.

        Returns:
            LabelData: Filtered LabelData object.
        """
        mask = np.isin(self.id, ids)
        return self.filter(mask)

    def apply_transform(self, matrix: np.ndarray) -> 'LabelData':
        """
        Apply a transformation matrix to the labels.

        Args:
            matrix (np.ndarray): Transformation matrix.

        Returns:
            LabelData: Transformed LabelData object.
        """
        new_transform = np.einsum('ij,njk->nik', matrix, self.transform)
        return replace(self, transform=new_transform)

    def get_alpha(self, target: np.ndarray = None, forward: np.ndarray = None) -> np.ndarray:
        """
        Calculate alpha values for the labels.

         Args:
            target (np.ndarray, optional): Target point. Defaults to None.
            forward (np.ndarray, optional): Forward direction. Defaults to None.

        Returns:
            np.ndarray: Alpha values.
        """ 
        target = target if target is not None else np.zeros(3)
        forward = forward if forward is not None else np.array([1, 0, 0])

        # Calculate delta for each position
        delta = self.position - target

        # Project delta onto the xy-plane and normalize
        delta_xy = delta[:, :2]
        delta_xy_norm = np.linalg.norm(delta_xy, axis=1)[:, None]
        delta_xy_unit = delta_xy / delta_xy_norm

        # Project forward onto the xy-plane and normalize
        forward_xy = forward[:2]
        forward_xy_norm = np.linalg.norm(forward_xy)
        forward_xy_unit = forward_xy / forward_xy_norm

        # Calculate dot product
        dot_product = np.dot(delta_xy_unit, forward_xy_unit)

        # Calculate cross product (only z-component is needed for 2D vectors)
        cross_product = delta_xy_unit[:, 0] * forward_xy_unit[1] - delta_xy_unit[:, 1] * forward_xy_unit[0]

        # Calculate angle using arctan2
        angle = np.arctan2(cross_product, dot_product)

        return angle
