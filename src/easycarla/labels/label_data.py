import numpy as np
from typing import Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field
from easycarla.labels.label_types import ObjectType

@dataclass
class LabelData:
    """
    A class to hold label data for object detection and tracking.

    Attributes:
    -----------
    object_types : Optional[list[set[ObjectType]]]
        An array of shape (n,) representing the types of objects detected. Each entry is an integer code corresponding to a specific object type.
    bbox_2d : Optional[np.ndarray]
        A 2D bounding box array of shape (n, 4) for the detected objects. Each entry contains [x1, y1, x2, y2] coordinates.
    bbox_3d : Optional[np.ndarray]
        A 3D bounding box array of shape (n, 7) for the detected objects. Each entry contains [x, y, z, l, w, h, theta] representing the center of the base of the box (x, y, z), dimensions (length, width, height), and orientation angle (theta) in radians.
    truncation : Optional[np.ndarray]
        An array of shape (n,) representing the truncation values of the detected objects. Each entry is a float in the range [0, 1], where 0 means not truncated and 1 means fully truncated.
    occlusion : Optional[np.ndarray]
        An array of shape (n,) representing the occlusion levels of the detected objects. Each entry is an integer: 0 (fully visible), 1 (partly occluded), 2 (largely occluded), or 3 (unknown).
    alpha : Optional[np.ndarray]
        An array of shape (n,) representing the observation angles of the detected objects. Each entry is a float in radians, typically in the range [-pi, pi].
    """

    object_types: list[set[ObjectType]] = None
    bbox_2d: Optional[np.ndarray] = None
    bbox_3d: Optional[np.ndarray] = None
    truncation: Optional[np.ndarray] = None
    occlusion: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.object_types)

    def __add__(self, other: any) -> "LabelData":
        """
        Adds two LabelData objects together. Ensures that the data shapes are correct and that either both or none of the attributes are None.

        Parameters:
        -----------
        other : LabelData
            The other LabelData object to be added.

        Returns:
        --------
        LabelData
            A new LabelData object containing the combined data.
        
        Raises:
        -------
        ValueError
            If one of the attributes is None and the other is not, or if the shapes of the arrays do not match.
        """
        if not isinstance(other, LabelData):
            return NotImplemented

        def validate_and_concatenate(attr1, attr2, name):
            """
            Validates and concatenates two attributes of LabelData objects.

            Parameters:
            -----------
            attr1 : Optional[np.ndarray]
                The first attribute to be validated and concatenated.
            attr2 : Optional[np.ndarray]
                The second attribute to be validated and concatenated.
            name : str
                The name of the attribute, used for error messages.

            Returns:
            --------
            Optional[np.ndarray]
                The concatenated array, or None if both attributes are None.

            Raises:
            -------
            ValueError
                If one attribute is None and the other is not, or if the shapes do not match.
            """
            if attr1 is None and attr2 is None:
                return None
            if attr1 is None or attr2 is None:
                raise ValueError(f"Both LabelData objects must have {name} defined, or both must be None.")
            if attr1.shape[1:] != attr2.shape[1:]:
                raise ValueError(f"The shape of {name} does not match: {attr1.shape} vs {attr2.shape}.")
            return np.concatenate((attr1, attr2), axis=0)

        object_types = self.object_types + other.object_types
        bbox_2d = validate_and_concatenate(self.bbox_2d, other.bbox_2d, "bbox_2d")
        bbox_3d = validate_and_concatenate(self.bbox_3d, other.bbox_3d, "bbox_3d")
        truncation = validate_and_concatenate(self.truncation, other.truncation, "truncation")
        occlusion = validate_and_concatenate(self.occlusion, other.occlusion, "occlusion")
        alpha = validate_and_concatenate(self.alpha, other.alpha, "alpha")

        return LabelData(object_types, bbox_2d, bbox_3d, truncation, occlusion, alpha)
    
    def filter_by_distance(self, target: np.ndarray, distance: float) -> 'LabelData':
        delta_pos = (self.bbox_3d[:, 0] - target)
        sq_dist = (delta_pos**2).sum(axis=1)
        mask = sq_dist <= distance**2
        return self.filter(mask)
    
    def filter_by_direction(self, target: np.ndarray, forward: float, dot_threshold = 0.1) -> 'LabelData':
        delta_pos = (self.bbox_3d[:, 0] - target)
        delta_pos_unit: np.ndarray = delta_pos / np.linalg.norm(delta_pos, axis=1)[:, None]
        mask = delta_pos_unit.dot(forward) > dot_threshold
        return self.filter(mask)
    
    def filter(self, mask: np.ndarray) -> 'LabelData':
        self.object_types = [self.object_types[index] for index in np.nonzero(mask)[0]]
        if self.bbox_2d is not None: self.bbox_2d = self.bbox_2d[mask]
        if self.bbox_3d is not None: self.bbox_3d = self.bbox_3d[mask]
        if self.truncation is not None: self.truncation = self.truncation[mask]
        if self.occlusion is not None: self.occlusion = self.occlusion[mask]
        if self.alpha is not None: self.alpha = self.alpha[mask]
        return self