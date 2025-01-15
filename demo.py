import argparse
import cv2
import json
import logging
import matplotlib.cm as cm
import numpy as np
import os
import os.path as osp
import time
import torch
import warnings
from collections import defaultdict, OrderedDict
from kornia.geometry.transform import warp_perspective
from pathlib import Path
from tqdm import tqdm

from src.utils.load_model import load_model
from src.utils.metrics import estimate_pose, relative_pose_error, error_auc, symmetric_epipolar_distance_numpy, \
    epidist_prec
from src.utils.plotting import dynamic_alpha, error_colormap, make_matching_figure


def save_matching_figure(path, img0, img1, mkpts0, mkpts1, inlier_mask, color, ):
    """ Make and save matching figures
    """
    inlier_mask = inlier_mask.astype(bool).squeeze()
    mkpts0_inliers = mkpts0[inlier_mask]
    mkpts1_inliers = mkpts1[inlier_mask]
    color = color[inlier_mask]
    if inlier_mask is None or len(inlier_mask) == 0:
        return

    text = [f'Matches:{len(mkpts0_inliers)}']

    # make the figure
    figure = make_matching_figure(img0, img1, mkpts0_inliers, mkpts1_inliers,
                                  color, text=text, path=path, dpi=150)


def save_matching_figure2(path, img0, img1, mkpts0, mkpts1, inlier_mask, color):
    """ Make and save matching figures
    """
    mkpts0_inliers = mkpts0
    mkpts1_inliers = mkpts1
    text = [f'Matches:{len(mkpts0_inliers)}']

    # make the figure
    figure = make_matching_figure(img0, img1, mkpts0_inliers, mkpts1_inliers,
                                  color, text=text, path=path, dpi=150)


def eval_relapose(
        matcher,
        pair,
        save_figs,
        figures_dir=None,
        method=None,
):
    # Eval on pair
    im0 = pair['im0']
    im1 = pair['im1']

    match_res = matcher(im0, im1)
    img0_color = cv2.imread(im0)
    img1_color = cv2.imread(im1)
    img0_color = cv2.cvtColor(img0_color, cv2.COLOR_BGR2RGB)
    img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
    mkpts0 = match_res['mkpts0']
    mkpts1 = match_res['mkpts1']
    mconf = match_res['mconf']
    if len(mconf) > 0:
        conf_min = mconf.min()
        conf_max = mconf.max()
        mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)
    color = cm.jet(mconf)

    if len(mkpts0) >= 4:
        ret_H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
    else:
        inliers = None
        ret_H = None
    print(f"Number of inliers: {inliers.sum() if inliers is not None else 0}")
    if save_figs:
        img0_name = f"fig1_{osp.basename(pair['im0']).split('.')[0]}"
        img1_name = f"fig2_{osp.basename(pair['im1']).split('.')[0]}"
        fig_path = osp.join(figures_dir, f"{img0_name}_{img1_name}_after_ransac_{method}.jpg")
        save_matching_figure(path=fig_path,
                             img0=img0_color,
                             img1=img1_color,
                             mkpts0=mkpts0,
                             mkpts1=mkpts1,
                             inlier_mask=inliers,
                             color=color,
                             )
        fig_path = osp.join(figures_dir, f"{img0_name}_{img1_name}_before_ransac_{method}.jpg")
        save_matching_figure2(path=fig_path,
                              img0=img0_color,
                              img1=img1_color,
                              mkpts0=mkpts0,
                              mkpts1=mkpts1,
                              inlier_mask=inliers,
                              color=color,
                              )
        if ret_H is not None:
            img0_color=cv2.cvtColor(img0_color, cv2.COLOR_RGB2BGR)
            im0_tensor = torch.tensor(img0_color, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.
            ret_H = torch.tensor(ret_H, dtype=torch.float32).unsqueeze(0)
            im0_tensor = H_transform(im0_tensor, ret_H)
            im0 = im0_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
            fig_path = osp.join(figures_dir, f"{img0_name}_after_homography_{method}.jpg")
            cv2.imwrite(fig_path, im0)


def H_transform(img2_tensor, homography):
    image_shape = img2_tensor.shape[2:]
    img2_tensor = warp_perspective(img2_tensor, homography, image_shape, align_corners=True)

    return img2_tensor


def test_relative_pose_demo(
        method="xoftr",
        save_dir=None,
        save_figs=False,
        args=None

):
    # Load pairs
    scene_pairs = {'im0': args.fig1, 'im1': args.fig2}

    # Load method
    # matcher = eval(f"load_{method}")(args)
    matcher = load_model(method, args)
    # Eval
    eval_relapose(
        matcher,
        scene_pairs,
        save_figs=save_figs,
        figures_dir=save_dir,
        method=method,
    )


if __name__ == '__main__':
    def add_common_arguments(parser):
        parser.add_argument('--exp_name', type=str, default="VisSYN")
        parser.add_argument('--fig1', type=str, default="./demo/vis_test.png")
        parser.add_argument('--fig2', type=str, default="./demo/depth_test.png")
        parser.add_argument('--save_dir', type=str, default="./demo/")


    def add_method_arguments(parser, method):
        if method == "xoftr":
            parser.add_argument('--match_threshold', type=float, default=0.3)
            parser.add_argument('--fine_threshold', type=float, default=0.1)
            parser.add_argument('--ckpt', type=str, default="./weights/weights_xoftr_640.ckpt")

        elif method == "loftr":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/minima_loftr.ckpt")
            parser.add_argument('--thr', type=float, default=0.2)
        elif method == "sp_lg":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/minima_lightglue.pth")
        elif method == "roma":
            parser.add_argument('--ckpt2', type=str,
                                default="large")
            parser.add_argument('--ckpt', type=str, default='./weights/minima_roma.pth')

        else:
            raise ValueError(f"Unknown method: {method}")

        add_common_arguments(parser)


    parser = argparse.ArgumentParser(description='Benchmark Relative Pose')

    parser.add_argument('--method', type=str, required=True,
                        choices=["xoftr", 'sp_lg', 'loftr', 'roma'],
                        help="Select the method to use: xoftr, sp_lg, loftr, roma")

    args, remaining_args = parser.parse_known_args()

    add_method_arguments(parser, args.method)

    args = parser.parse_args()

    print(args)

    save_dir = args.save_dir
    save_figs = True
    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_relative_pose_demo(
            args.method,
            save_dir=args.save_dir,
            save_figs=save_figs,
            args=args
        )
    print(f"Elapsed time: {time.time() - tt}")
