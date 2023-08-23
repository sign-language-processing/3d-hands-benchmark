import cv2
import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


def cce(poses: np.ndarray):
    assert poses.shape[1:] == (261, 6, 21, 3)

    poses_flat = poses.reshape((len(poses), -1, 21, 3))
    poses_std = np.std(poses_flat, axis=0)
    return np.average(poses_std.sum(axis=1).sum(axis=1)).item()


def visualize_cce(poses: np.ndarray, file_path: str):  # shape (num_runs, 21, 3)
    with open('example.pose', 'rb') as f:
        pose = Pose.read(f.read())

    pose.body.data = ma.array(poses) - poses[:, :, :1]
    pose.body.confidence = np.ones(poses.shape[:-1])
    pose.focus()

    vis = PoseVisualizer(pose)
    vis.save_gif(file_path + '.gif', vis.draw())

    pose.body.data = pose.body.data.transpose((1, 0, 2, 3))
    pose.body.confidence = pose.body.confidence.transpose((1, 0, 2))
    vis = PoseVisualizer(pose)
    cce_frame = next(vis.draw())
    cv2.imwrite(file_path + '.png', cce_frame)


if __name__ == "__main__":
    poses = np.load('systems/mediapipe/v0.10.3.npy')
    for i in range(10):
        file_path = f'../assets/cce/{i}'
        visualize_cce(poses[:, i*10:i*10 + 1, 0], file_path)
