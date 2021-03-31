import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderDimensions
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.openpose import OpenPose_Hand_Component

pose_components = [OpenPose_Hand_Component("hand")]

with open('systems/mediapipe/submission.npy', 'rb') as f:
    poses = np.load(f)

pose_body_data = poses[:, 0, :1, :, :]
pose_body_data -= pose_body_data[:, :1, :1] # move to wrist
pose_body_conf = np.ones(pose_body_data.shape[:-1])
pose_body = NumPyPoseBody(fps=5, data=pose_body_data, confidence=pose_body_conf)

dimensions = PoseHeaderDimensions(width=1, height=1)
pose_header = PoseHeader(version=0.1, dimensions=dimensions, components=pose_components)
pose = Pose(header=pose_header, body=pose_body)
pose.focus()

v = PoseVisualizer(pose)
v.save_video("crop_consistency.mp4", v.draw())
