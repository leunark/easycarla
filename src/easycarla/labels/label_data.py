from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Union
from easycarla.labels.label_types import ObjectType

@dataclass
class LabelData:

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

    @property
    def position(self) -> np.ndarray:
        return self.transform[:, :3, 3]

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.transform[:, :3, :3]

    @property
    def euler_angles(self) -> np.ndarray:
        return Rotation.from_matrix(self.rotation_matrix).as_euler('xyz')

    @property
    def vertices(self) -> np.ndarray:
        # Define the 8 vertices of a unit cube
        unit_box = np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ])
        
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
            types=self.types + other.types,
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

    def filter_by_distance(self, target: np.ndarray, distance: float) -> 'LabelData':
        delta_pos = self.position - target
        sq_dist = np.sum(delta_pos**2, axis=1)
        mask = sq_dist <= distance**2
        return self.filter(mask)
    
    def filter_by_direction(self, target: np.ndarray, forward: np.ndarray, dot_threshold: float = 0.1) -> 'LabelData':
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

    @classmethod
    def from_position_and_rotation_matrix(cls, position: np.ndarray, rotation_matrix: np.ndarray, **kwargs):
        n = position.shape[0]
        transform = np.tile(np.eye(4), (n, 1, 1))
        transform[:, :3, :3] = rotation_matrix
        transform[:, :3, 3] = position
        return cls(transform=transform, **kwargs)

    @classmethod
    def from_position_and_euler_angles(cls, position: np.ndarray, euler_angles: np.ndarray, **kwargs):
        rotation_matrix = Rotation.from_euler('xyz', euler_angles).as_matrix()
        return cls.from_position_and_rotation_matrix(position, rotation_matrix, **kwargs)
