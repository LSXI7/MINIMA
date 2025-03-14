import argparse
import logging
import os
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tools.infrared.scepter.modules.inference.stylebooth_inference import StyleboothInference, TunerInference
from tools.infrared.scepter.modules.utils.config import Config
from tools.infrared.scepter.modules.utils.logger import get_logger
from torchvision.utils import save_image
from tqdm import tqdm


class TunerModel:
    def __init__(self, model_path):
        self.MODEL_PATH = model_path


def resize_pil_image(pil_image, target_size):
    try:
        # Newer versions of Pillow
        resized_image = pil_image.resize(target_size, resample=Image.LANCZOS)
    except TypeError:
        # Older versions of Pillow
        resized_image = pil_image.resize(target_size, Image.LANCZOS)
    return resized_image


def pad_long_to_1024(image, fill_color=(0, 0, 0)):
    origin_w, origin_h = image.size

    if origin_w > origin_h:
        new_w = 1024
        new_h = round(origin_h * new_w / origin_w)
    else:
        new_h = 1024
        new_w = round(origin_w * new_h / origin_h)
    # image = TF.resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
    image = resize_pil_image(image, (new_w, new_h))
    padding_w = 1024 - new_w
    padding_h = 1024 - new_h

    padding = (0, 0, padding_w, padding_h)
    padded_image = ImageOps.expand(image, padding, fill_color)
    return padded_image, origin_w, origin_h, new_w, new_h


def pad_to_multiple_of_8(image, fill_color=(0, 0, 0)):
    w, h = image.size
    new_w = (w + 7) // 8 * 8
    new_h = (h + 7) // 8 * 8
    padding_w = new_w - w
    padding_h = new_h - h
    padding = (0, 0, padding_w, padding_h)
    padded_image = ImageOps.expand(image, padding, fill_color)
    return padded_image, w, h, new_w, new_h


def init_infrared_model(args, device):
    logger = get_logger(name=f'scepter')
    config_file = 'tools/infrared/stylebooth_tb_pro.yaml'
    cfg = Config(cfg_file=config_file)

    diff_infer = StyleboothInference(logger=logger, device=device)
    diff_infer.init_from_cfg(cfg)
    tuner_model_list = [TunerModel(f"{args.adapter_model_path}")]
    return diff_infer, tuner_model_list


def infrared_transfer_single(args, image_path, diff_infer, tuner_model_list, big_enough_option=True):
    image = Image.open(image_path)
    W, H = image.size
    big_enough = W > 900 or H > 900
    if big_enough and big_enough_option:
        padded_image, original_width, original_height, padded_width, padded_height = pad_to_multiple_of_8(image)

        input_dict = {
            'prompt': "Convert the image to an infrared image",
            'target_size_as_tuple': [padded_height, padded_width],
        }

        other_args = {
            'style_edit_image': padded_image,
            'style_guide_scale_text': args.style_guide_scale_text,
            'style_guide_scale_image': args.style_guide_scale_image,
            'num_samples': args.num_samples,
            'tuner_model': tuner_model_list,
        }
        if args.seed != -1:
            other_args['seed'] = args.seed

        output = diff_infer(input_dict, **other_args)
        images = output['images']  # torch

        images = images[:, :, :original_height, :original_width]


    else:
        padded_image, original_width, original_height, padded_width, padded_height = pad_long_to_1024(image)

        input_dict = {
            'prompt': "Convert the image to an infrared image",
            'target_size_as_tuple': [1024, 1024],
            # 'guide_scale': {'text': args.style_guide_scale_text, 'image': args.style_guide_scale_image},
        }

        other_args = {
            'style_edit_image': padded_image,
            'style_guide_scale_text': args.style_guide_scale_text,
            'style_guide_scale_image': args.style_guide_scale_image,
            'num_samples': args.num_samples,
            'tuner_model': tuner_model_list,
        }
        if args.seed != -1:
            other_args['seed'] = args.seed

        output = diff_infer(input_dict, **other_args)
        images = output['images']  # torch

        images = images[:, :, :padded_height, :padded_width]

        images = F.interpolate(
            images,
            size=(original_height, original_width),
            mode='bicubic',
            align_corners=True
        )

    return images
