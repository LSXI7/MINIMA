# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
import torchvision.transforms.functional as TF


def get_bbox_from_mask(mask):
    h, w = mask.shape[0], mask.shape[1]
    if mask.sum() < 10:
        return 0, h, 0, w
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (y1, y2, x1, x2)


def pad_to_square(image, pad_value=255, random=False):
    H, W = image.shape[0], image.shape[1]
    if H == W:
        return image, 0, 0

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0, padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
    else:
        pad_param = ((padd_1, padd_2), (0, 0), (0, 0))

    # print(pad_param, pad_value)
    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image, padd_1, padd_2


def box_in_box(small_box, big_box):
    y1, y2, x1, x2 = small_box
    y1_b, _, x1_b, _ = big_box
    y1, y2, x1, x2 = y1 - y1_b, y2 - y1_b, x1 - x1_b, x2 - x1_b
    return (y1, y2, x1, x2)


def box2squre(image, box):
    H, W = image.shape[0], image.shape[1]
    y1, y2, x1, x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = y2 - y1, x2 - x1

    if h >= w:
        x1 = cx - h // 2
        x2 = x1 + h
    else:
        y1 = cy - w // 2
        y2 = y1 + w
    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    return (y1, y2, x1, x2)


def expand_bbox(mask,
                yyxx,
                ratio=1.0,
                min_crop=0,
                expand_type='center',
                to_square=False):
    y1, y2, x1, x2 = yyxx
    h = y2 - y1 + 1
    w = x2 - x1 + 1

    H, W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    def expand(k):
        if isinstance(ratio, tuple) or isinstance(ratio, list):
            r = np.random.uniform(*ratio)
            k = k * r
        else:
            k = ratio * k
        return k

    new_h = expand(h)
    new_w = expand(w)
    new_h = max(new_h, min_crop)
    new_w = max(new_w, min_crop)

    if to_square:
        if new_w / new_h < 0.334:
            new_w = new_w + 1.0 / 3.0 * new_h
        elif new_h / new_w < 0.334:
            new_h = new_h + 1.0 / 3.0 * new_w

    if expand_type == 'center':
        x1 = max(0, int(xc - new_w * 0.5))
        x2 = min(W, int(xc + new_w * 0.5))
        y1 = max(0, int(yc - new_h * 0.5))
        y2 = min(H, int(yc + new_h * 0.5))
    else:
        x1 = max(0, min(x1,
                        int(x2 - new_w * np.random.uniform(w / new_w, 1.0))))
        x2 = min(W, max(x2, x1 + new_w))
        y1 = max(0, min(y1,
                        int(y2 - new_h * np.random.uniform(h / new_h, 1.0))))
        y2 = min(H, max(y2, y1 + new_h))

    return (int(y1), int(y2), int(x1), int(x2))


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2, pad1, pad2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = TF.resize(pred, (H2, W2), antialias=True)

    if W1 < W2:
        # pad width
        assert H1 == H2 and (pad1 + W1) == (W2 - pad2)
        pred = pred[:, :, pad1 + 2:(W2 - pad2 - 2)]
        tar_image[:, y1:y2, x1 + 2:x2 - 2] = pred
    elif H1 < H2:
        # pad height
        assert W1 == W2 and (pad1 + H1) == (H2 - pad2)
        pred = pred[:, pad1 + 2:(H2 - pad2 - 2), :]
        tar_image[:, y1 + 2:y2 - 2, x1:x2] = pred
    else:
        tar_image[:, y1:y2, x1:x2] = pred
    return tar_image


def save_image(image, save_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)
