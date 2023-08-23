import os
from multiprocessing import cpu_count, Pool

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def pose_image(hands, transparent_image: PILImage, image_crop_scale=1.5):
    # MediaPipe expects an RGB tensor, with some padding
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
        return None

    image_height, image_width, _ = image.shape
    landmarks = [pose.multi_hand_landmarks[0].landmark[i] for i in range(21)]
    return np.array([[l.x * image_width, l.y * image_height, l.z * image_width] for l in landmarks])


def pose_crop(args):
    images, crop = args
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0)

    poses = []
    for hs in images:
        hs_poses = []
        for h in hs:
            pose = pose_image(hands, h, crop)
            if pose is None:
                pose = np.zeros(shape=(21, 3))
            assert pose.shape == (21, 3)
            hs_poses.append(pose)

        # Hand shape 01-05-015 only includes 4 orientations
        for i in range(len(hs_poses), 6):
            hs_poses.append(np.zeros((21, 3)))
        poses.append(hs_poses)
    return np.array(poses)


def load_images():
    print("Loading images")

    images = []
    hands_dir = "../../hands"
    for hs_group in tqdm(sorted(os.listdir(hands_dir))):
        for hs in sorted(os.listdir(os.path.join(hands_dir, hs_group))):
            group = []
            hs_path = os.path.join(hands_dir, hs_group, hs)
            for h in sorted(os.listdir(hs_path)):
                os.path.join(hs_path, h)
                with open(os.path.join(hs_path, h), "rb") as img:
                    group.append(PILImage.open(img).copy())
            images.append(group)
    return images

if __name__ == "__main__":
    images = load_images()

    n_crops = cpu_count()
    crops = [1 + 2 * i / n_crops for i in range(n_crops)]  # range [1,3)
    crops_args = [(images, c) for c in crops]

    with Pool(cpu_count()) as p:
        crops_poses = list(process_map(pose_crop, crops_args, max_workers=cpu_count()))

    crops_poses = np.array(crops_poses)
    print("Saving", crops_poses.shape)

    with open(f"v{mp.__version__}.npy", 'wb') as f:
        np.save(f, np.array(crops_poses, dtype=np.float32))
