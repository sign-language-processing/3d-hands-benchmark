import cv2
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.pose_header import PoseNormalizationInfo
import numpy as np
from numpy import ma
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer


def normalize_hands(poses: np.ndarray):
    plane = PoseNormalizationInfo(p1=0, p2=17, p3=5)
    line = PoseNormalizationInfo(p1=0, p2=9)
    normalizer = PoseNormalizer(plane=plane, line=line, size=200)
    masked_array = ma.array(poses)
    return normalizer(masked_array).data


def mace_single(poses: np.ndarray):
    poses = normalize_hands(poses)
    poses_std = np.std(poses, axis=0)
    return poses_std.sum().item()


def mace(poses: np.ndarray):
    assert poses.shape[1:] == (261, 6, 21, 3)

    runs = [mace_single(p) for p in poses]
    return {
        "mean": np.average(runs),
        "std": np.std(runs)
    }


def visualize_mace(poses: np.ndarray, file_path: str):  # shape (6, 21, 3)
    with open('example.pose', 'rb') as f:
        pose = Pose.read(f.read())

    # add dim at 0
    poses = np.expand_dims(poses, 0)
    poses = normalize_hands(poses)

    pose.body.data = ma.array(poses)
    pose.body.confidence = np.ones(poses.shape[:-1])
    pose.focus()

    vis = PoseVisualizer(pose)
    mace_frame = next(vis.draw())
    cv2.imwrite(file_path, mace_frame)


if __name__ == "__main__":
    poses = np.load('systems/mediapipe/v0.10.3.npy')
    for i in range(10):
        file_path = f'../assets/mace/{i}.png'
        visualize_mace(poses[0, i*10], file_path)


