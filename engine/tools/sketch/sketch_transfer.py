import os

import torch
from PIL import Image
from kornia.enhance import equalize_clahe
from tools.sketch.anime_to_sketch.data import read_img_path, tensor_to_img, no_save_image
from tools.sketch.anime_to_sketch.model import create_model


def init_sketch_model(args, device):
    model_variant = args.model_variant
    model = create_model(model_variant)
    model = model.to(device)
    model.eval()
    return model


def sketch_transfer_single(image_path, model, device, load_size, clahe_clip):
    img, aus_resize = read_img_path(image_path, load_size)

    if clahe_clip > 0:
        img = (img + 1) / 2
        img = equalize_clahe(img, clip_limit=clahe_clip)
        img = (img - .5) / .5

    with torch.no_grad():
        aus_tensor = model(img.to(device))

    aus_img = tensor_to_img(aus_tensor)
    result = no_save_image(aus_img, aus_resize)
    return result
