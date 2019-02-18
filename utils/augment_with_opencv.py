import random
import cv2

import numpy as np


ORIGINAL_SIZE = 96  # original size of the images - do not change

# AUGMENTATION VARIABLES
CROP_SIZE = 90  # final size after crop
RANDOM_ROTATION = 3  # range (0-180), 180 allows all rotation variations, 0=no change
RANDOM_SHIFT = 2  # center crop shift in x and y axes, 0=no change. This cannot be more than (ORIGINAL_SIZE - CROP_SIZE)//2
RANDOM_BRIGHTNESS = 7  # range (0-100), 0=no change
RANDOM_CONTRAST = 5  # range (0-100), 0=no change
RANDOM_90_DEG_TURN = 1  # 0 or 1= random turn to left or right

def readCroppedImage(path, augmentations=True):
    # augmentations parameter is included for counting statistics from images, where we don't want augmentations

    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])

    if (not augmentations):
        return rgb_img / 255

    # random rotation
    rotation = random.randint(-RANDOM_ROTATION, RANDOM_ROTATION)
    if (RANDOM_90_DEG_TURN == 1):
        rotation += random.randint(-1, 1) * 90
    M = cv2.getRotationMatrix2D((48, 48), rotation, 1)  # the center point is the rotation anchor
    rgb_img = cv2.warpAffine(rgb_img, M, (96, 96))

    # random x,y-shift
    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)
    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT)

    # crop to center and normalize to 0-1 range
    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2
    end_crop = start_crop + CROP_SIZE
    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255

    # Random flip
    flip_hor = bool(random.getrandbits(1))
    flip_ver = bool(random.getrandbits(1))
    if (flip_hor):
        rgb_img = rgb_img[:, ::-1]
    if (flip_ver):
        rgb_img = rgb_img[::-1, :]

    # Random brightness
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
    rgb_img = rgb_img + br

    # Random contrast
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
    rgb_img = rgb_img * cr

    # clip values to 0-1 range
    rgb_img = np.clip(rgb_img, 0, 1.0)

    return rgb_img

