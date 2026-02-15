import os
import numpy as np
import torch
import transforms3d
import yaml
from collections import defaultdict


def plane2pose(plane_parameters):
    r3 = plane_parameters[:3]
    r2 = np.zeros_like(r3)
    r2[0], r2[1], r2[2] = (-r3[1], r3[0], 0) if r3[2] * r3[2] <= 0.5 else (-r3[2], 0, r3[0])
    r1 = np.cross(r2, r3)
    pose = np.zeros([4, 4], dtype=np.float32)
    pose[0, :3] = r1
    pose[1, :3] = r2
    pose[2, :3] = r3
    pose[2, 3] = plane_parameters[3]
    pose[3, 3] = 1
    return pose


def plane2euler(plane_parameters, axes='sxyz'):
    pose = plane2pose(plane_parameters)
    T, R, Z, S = transforms3d.affines.decompose(pose)
    euler = transforms3d.euler.mat2euler(R, axes=axes)
    return T, euler


class MergeLoader(yaml.SafeLoader):
    pass

def construct_mapping(loader, node):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=False)
        value = loader.construct_object(value_node, deep=True)
        
        if key in mapping:
            # Duplicate key found - merge values
            existing = mapping[key]
            # Both values should be lists in YAML (e.g., [1] and [4])
            # Merge them by extending
            if isinstance(existing, list) and isinstance(value, list):
                mapping[key] = existing + value
            elif isinstance(existing, list):
                mapping[key] = existing + [value]
            elif isinstance(value, list):
                mapping[key] = [existing] + value
            else:
                mapping[key] = [existing, value]
        else:
            # First occurrence of this key
            mapping[key] = value
    
    return mapping


MergeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    construct_mapping
)

