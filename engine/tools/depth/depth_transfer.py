import os

import cv2
import matplotlib
import numpy as np
import torch
from tools.depth.depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm


def init_depth_model(device, encoder):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'weights/depth_anything_v2/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(device).eval()
    return model


def process_image(image, model):
    depth = model.infer_image(image)  # HxW raw depth map in numpy
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    return depth_colored


def depth_transfer_single(img_path, model):
    image = cv2.imread(img_path)

    assert image is not None, f"Failed to read image: {img_path}"
    processed_image = process_image(image, model)
    return processed_image
