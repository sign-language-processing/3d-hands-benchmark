from tqdm import tqdm
from PIL import Image as PILImage
import numpy as np
import mediapipe as mp
import cv2
import os

# For static images:
hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0)


def pose_image(filename: str, image_crop_scale=1.5):
    # MediaPipe expects an RGB tensor, with some padding
    transparent_image = PILImage.open(filename)
    im_size = int(max(transparent_image.size[0], transparent_image.size[1]) * image_crop_scale)
    image = PILImage.new("RGB", (im_size, im_size), (255, 255, 255))
    paste_loc = ((im_size - transparent_image.size[0]) // 2, (im_size - transparent_image.size[1]) // 2)
    channels = transparent_image.split()
    image.paste(transparent_image, box=paste_loc, mask=channels[3] if len(channels) > 3 else None)

    # flip it around y-axis for correct handedness output
    image = cv2.flip(np.array(image), 1)
    pose = hands.process(image)

    # If hand not found, return 0s
    if not pose.multi_hand_landmarks:
        print("failed to extract hand pose")
        return np.zeros(shape=(21, 3))

    image_height, image_width, _ = image.shape
    landmarks = [pose.multi_hand_landmarks[0].landmark[i] for i in range(21)]
    return np.array([[l.x * image_width, l.y * image_height, l.z * image_width] for l in landmarks])


if __name__ == "__main__":

    hands_dir = "../../hands"
    poses = []
    for hs_group in tqdm(sorted(os.listdir(hands_dir))):
        for hs in tqdm(sorted(os.listdir(os.path.join(hands_dir, hs_group)))):
            hs_poses = []
            hs_path = os.path.join(hands_dir, hs_group, hs)
            for h in tqdm(sorted(os.listdir(hs_path))):
                pose = pose_image(os.path.join(hs_path, h), 1.5)
                assert pose.shape == (21, 3)
                hs_poses.append(pose)

            # Hand shape 01-05-015 only includes 4 orientations
            for i in range(len(hs_poses), 6):
                hs_poses.append(np.zeros((21, 3)))
            poses.append(hs_poses)

    with open('submission.npy', 'wb') as f:
        np.save(f, np.array(poses, dtype=np.float32))