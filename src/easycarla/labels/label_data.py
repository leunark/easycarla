from dataclasses import dataclass
import numpy as np
from easycarla.labels.label_types import ObjectType


@dataclass
class LabelData:
    """All attributes of shape (N, ...) containing all labels"""
    id: np.ndarray
    transform: np.ndarray  # 4x4 transformation matrices
    dimension: np.ndarray
    types: list[set[ObjectType]]
    truncation: np.ndarray = None
    occlusion: np.ndarray = None
    alpha: np.ndarray = None

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
    # A subdivsion count of two adds one extra vertex because it subdivides
    # the edge into two parts
    SUBDIVISION_COUNT = 3

    
    @property
    def position(self) -> np.ndarray:
        return self.transform[:, :3, 3]

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.transform[:, :3, :3]

    @property
    def vertices(self) -> np.ndarray:
        # Add extra verts on diagonals
        unit_box = self.UNIT_BOX
        exra_verts = []
        lines = self.DIAG_INDICES
        for i, j in lines:
            for k in range(1, self.SUBDIVISION_COUNT):
                v = unit_box[i] + (unit_box[j] - unit_box[i]) / self.SUBDIVISION_COUNT * k
                exra_verts.append(v)
        exra_verts = np.array(exra_verts)
        unit_box = np.vstack((unit_box, exra_verts))

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
        vertices = self.vertices
        edges = vertices[:, self.EDGE_INDICES]
        return edges
    
    def __len__(self) -> int:
        return len(self.transform)

    def __add__(self, other: 'LabelData') -> 'LabelData':
        if not isinstance(other, LabelData):
            return NotImplemented

        return LabelData(
            id=np.concatenate((self.id, other.id), axis=0),
            transform=np.concatenate((self.transform, other.transform), axis=0),
            dimension=np.concatenate((self.dimension, other.dimension), axis=0),
            truncation=np.concatenate((self.truncation, other.truncation), axis=0) if self.truncation is not None and other.truncation is not None else None,
            occlusion=np.concatenate((self.occlusion, other.occlusion), axis=0) if self.occlusion is not None and other.occlusion is not None else None,
            alpha=np.concatenate((self.alpha, other.alpha), axis=0) if self.alpha is not None and other.alpha is not None else None,
            types=np.concatenate((self.types, other.types), axis=0) if self.types is not None and other.types is not None else None,
        )

    def filter(self, mask: np.ndarray) -> 'LabelData':
        self.id = self.id[mask]
        self.transform = self.transform[mask]
        self.dimension = self.dimension[mask]
        self.types = [self.types[i] for i in np.nonzero(mask)[0]]
        if self.truncation is not None: self.truncation = self.truncation[mask]
        if self.occlusion is not None: self.occlusion = self.occlusion[mask]
        if self.alpha is not None: self.alpha = self.alpha[mask]
        return self

    def filter_by_distance(self, distance: float, target: np.ndarray = None) -> 'LabelData':
        target = target if target is not None else np.zeros(3)
        delta_pos = self.position - target
        sq_dist = np.sum(delta_pos**2, axis=1)
        mask = sq_dist <= distance**2
        return self.filter(mask)
    
    def filter_by_direction(self, dot_threshold: float = 0.0, target: np.ndarray = None, forward: np.ndarray = None) -> 'LabelData':
        target = target if target is not None else np.zeros(3)
        forward = forward if forward is not None else np.array([1, 0, 0])
        delta_pos = self.position - target
        delta_pos_unit = delta_pos / np.linalg.norm(delta_pos, axis=1)[:, None]
        mask = np.dot(delta_pos_unit, forward) > dot_threshold
        return self.filter(mask)
    
    def filter_by_id(self, ids: np.ndarray):
        mask = np.isin(self.id, ids)
        return self.filter(mask)

    def apply_transform(self, matrix: np.ndarray) -> 'LabelData':
        self.transform = np.einsum('ij,njk->nik', matrix, self.transform)
        return self

    def get_alpha(self, target: np.ndarray = None, forward: np.ndarray = None):
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
