import argparse
import os
import random

import cv2
import numpy as np


def random_shift(shift_min, shift_max):
    range_choice = random.choice([(shift_min, shift_max), (-shift_max, -shift_min)])
    return random.randint(*range_choice)


def event_transfer_single(image_path, contrast_min=0.05, contrast_max=0.5, shift_min=1, shift_max=3):
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    gray_img_log = np.log1p(gray_img)

    contrast_threshold = random.uniform(contrast_min, contrast_max)

    xshift = random_shift(shift_min, shift_max)
    yshift = random_shift(shift_min, shift_max)

    pic_shape = [gray_img.shape[0], gray_img.shape[1], 3]
    img = np.full(pic_shape, [255, 255, 255], dtype=np.uint8)

    for i in range(abs(yshift), gray_img.shape[0] - abs(yshift)):
        for j in range(abs(xshift), gray_img.shape[1] - abs(xshift)):
            delta_L = gray_img_log[i + yshift, j + xshift] - gray_img_log[i, j]
            if delta_L > contrast_threshold:
                img[i, j] = [255, 0, 0]
            elif delta_L < -contrast_threshold:
                img[i, j] = [0, 0, 255]

    return img
