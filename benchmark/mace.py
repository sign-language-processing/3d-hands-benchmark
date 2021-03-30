import numpy as np

import math
from scipy.spatial.transform import Rotation


def rotate_to_normal(pose: np.ndarray, normal: np.ndarray, around: np.ndarray):
    # Let's rotate the points such that the normal is the new Z axis
    # Following https://stackoverflow.com/questions/1023948/rotate-normal-vector-onto-axis-plane
    old_x_axis = np.array([1, 0, 0])

    z_axis = normal
    y_axis = np.cross(old_x_axis, z_axis)
    x_axis = np.cross(z_axis, y_axis)

    axis = np.stack([x_axis, y_axis, z_axis])

    return np.dot(pose - around, axis.T)


def get_hand_normal(pose: np.ndarray):
    plane_points = [
        0,  # Wrist
        17,  # Pinky CMC
        5,  # Index CMC
    ]

    triangle = pose[plane_points]

    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]

    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    return normal, triangle[0]


def get_hand_rotation(pose: np.ndarray):
    p1 = pose[0]  # Wrist
    p2 = pose[9]  # Middle CMC
    vec = p2 - p1

    return 90 + math.degrees(math.atan2(vec[1], vec[0]))


def rotate_hand(pose: np.ndarray, angle: float):
    r = Rotation.from_euler('z', angle, degrees=True)
    return np.dot(pose, r.as_matrix())


def scale_hand(pose: np.ndarray, size=200):
    p1 = pose[0]  # Wrist
    p2 = pose[9]  # Middle CMC
    current_size = np.power(p2 - p1, 2).sum()

    pose *= size / current_size
    pose -= pose[0]  # move to Wrist
    return pose


def normalized_hand(pose: np.ndarray):
    if np.all(pose == 0):
        return pose

    # First rotate to normal
    normal, base = get_hand_normal(pose)
    pose = rotate_to_normal(pose, normal, base)

    # Then rotate on the X-Y plane such that the BASE-M_CMC is on the Y axis
    angle = get_hand_rotation(pose)
    pose = rotate_hand(pose, angle)

    # Scale pose such that BASE-M_CMC is of size 200
    pose = scale_hand(pose, 200)

    return pose


def paired_error(poses: np.ndarray):
    error = 0
    for hs in poses:
        hs_error = 0
        for h1 in hs:
            for h2 in hs:
                hs_error += np.power((h1 - h2), 2).sum()
        error += hs_error / (len(hs) ** 2)

    return error / len(poses)


def normalize_hands(poses: np.ndarray):
    return np.array([[normalized_hand(p) for p in ps] for ps in poses])


def mace(poses: np.ndarray):
    poses = normalize_hands(poses)
    return paired_error(poses)
