import glob
import os
import sys

import numpy as np
import tools.normal.dsine.projects.dsine.config as config
import tools.normal.dsine.utils.utils as utils
import torch
import torch.nn.functional as F
from PIL import Image
from tools.normal.dsine.utils.projection import intrins_from_fov, intrins_from_txt
from torchvision import transforms
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def init_normal_model(device):
    args = config.get_args(test=True)
    print(args)
    if args.NNET_architecture == 'v00':
        from tools.normal.dsine.models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == 'v01':
        from tools.normal.dsine.models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == 'v02':
        from tools.normal.dsine.models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == 'v02_kappa':
        from tools.normal.dsine.models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception('invalid arch')

    model = DSINE(args).to(device)
    # print(f"Loaded model from {args.ckpt_path}")
    model = utils.load_checkpoint(args.ckpt_path, model)

    model.eval()
    return model


def normal_transfer_single(img_path, model, device):
    ext = os.path.splitext(img_path)[1]
    img = Image.open(img_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # pad input
    _, _, orig_H, orig_W = img.shape
    lrtb = utils.get_padding(orig_H, orig_W)
    img = F.pad(img, lrtb, mode="constant", value=0.0)
    img = normalize(img)

    # get intrinsics
    intrins_path = img_path.replace(ext, '.txt')
    if os.path.exists(intrins_path):
        # NOTE: camera intrinsics should be given as a txt file
        # it should contain the values of fx, fy, cx, cy
        intrins = intrins_from_txt(intrins_path, device=device).unsqueeze(0)
    else:
        # NOTE: if intrins is not given, we just assume that the principal point is at the center
        # and that the field-of-view is 60 degrees (feel free to modify this assumption)
        intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
    intrins[:, 0, 2] += lrtb[0]
    intrins[:, 1, 2] += lrtb[2]

    pred_norm = model(img, intrins=intrins)[-1]
    pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2] + orig_H, lrtb[0]:lrtb[0] + orig_W]

    # save to output folder
    # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
    # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files

    pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
    pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
    im = Image.fromarray(pred_norm[0, ...])
    # im.save(target_path)
    return im
