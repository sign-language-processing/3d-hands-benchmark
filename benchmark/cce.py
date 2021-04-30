import numpy as np


def cce(poses: np.ndarray):
    assert poses.shape[1:] == (261, 6, 21, 3)

    poses_flat = poses.reshape((len(poses), -1, 21, 3))
    poses_std = np.std(poses_flat, axis=0)
    return np.average(poses_std.sum(axis=1).sum(axis=1)).item()
