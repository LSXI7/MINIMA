import argparse
import cv2
import json
import logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import time
import torch
import warnings
from collections import defaultdict, OrderedDict
from kornia.geometry.transform import warp_perspective
from math import pi
from load_model import load_model, choose_method_arguments, add_method_arguments
from src.utils.metrics import error_auc
from src.utils.plotting import dynamic_alpha, make_matching_figure2
from src.utils.sample_h import sample_homography
from tqdm import tqdm


def save_matching_figure2(path, img0, img1, mkpts0, mkpts1, mean_distance, correct_mask, svg=False):
    """ Make and save matching figures
    """
    # bool---float
    correct_mask = correct_mask.astype(float)

    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n = mkpts0.shape[0]
    # matching info
    alpha = dynamic_alpha(len(correct_mask))
    mask = correct_mask
    color = np.zeros((n, 3), dtype=np.uint8)

    color[mask == 0] = (255, 0, 0)

    color[mask == 1] = (0, 255, 0)
    n_correct = int(n_correct)

    text_precision = [
        f'Precision(3px) ({100 * precision:.1f}%): {n_correct}/{len(mkpts0)}']

    # if name is not None:
    #     text = [name]
    # else:
    text = []

    error_text = [f"Mean Distance: {mean_distance:.2f} px"]
    text += error_text

    text += text_precision
    # color=
    # make the figure
    img0 = img0 / 255
    img1 = img1 / 255
    figure = make_matching_figure2(img0, img1, mkpts0, mkpts1,
                                   color, text=text, path=path, dpi=100, svg=svg)


def load_vis_event_pairs_npz(root, txt_file):
    """Load information for scene and image pairs from npz files.
    Args:
        npz_root: Directory path for npz files
        npz_list: File containing the names of the npz files to be used
    """
    pair = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            vis_image_path = os.path.join(root, line)
            event_image_path = os.path.join(root, line.replace('images', 'events'))
            if os.path.exists(vis_image_path) and os.path.exists(event_image_path):
                pair.append((vis_image_path, event_image_path))

    scene_pairs = {}
    pairs = []

    config = {
        'perspective': True,  # Enable perspective transformation
        'scaling': True,  # Enable scaling transformation
        'rotation': True,  # Enable rotation transformation
        'translation': True,  # Enable translation
        'n_scales': 5,  # Number of scaling samples
        'n_angles': 25,  # Number of rotation samples
        'scaling_amplitude': 0.05,
        'perspective_amplitude_x': 0.05,
        'perspective_amplitude_y': 0.05,
        'patch_ratio': 0.8,
        'max_angle': 10 * (pi / 180),
        'allow_artifacts': False,
        'translation_overflow': 0.0
    }

    for vis_image_path, event_image_path in pair:
        vis_file = vis_image_path
        Event_file = event_image_path
        image0_shape = [480, 640]  # we fix the image size to 480x640
        image1_shape = [480, 640]
        H0 = sample_homography(image0_shape, vis_file, config).squeeze().numpy()
        H1 = sample_homography(image1_shape, Event_file, config).squeeze().numpy()
        pairs.append({'im0': vis_file, 'im1': Event_file, 'H': H0})
        pairs.append({'im0': Event_file, 'im1': vis_file, 'H': H1})
    # pairs.sort()
    scene_pairs['event'] = pairs

    return scene_pairs


def H_transform(img2_tensor, homography):
    image_shape = img2_tensor.shape[2:]
    img2_tensor = warp_perspective(img2_tensor, homography, image_shape, align_corners=True)

    return img2_tensor


def aggregiate_scenes(scene_pose_auc, thresholds):
    """Averages the auc results for cloudy_cloud and cloudy_sunny scenes
    """
    temp_pose_auc = {}
    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name] = [np.zeros(len(thresholds), dtype=np.float32), 0]  # [sum, total_number]
    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name][0] += scene_pose_auc[npz_name]
        temp_pose_auc[scene_name][1] += 1

    agg_pose_auc = {}
    for scene_name in temp_pose_auc.keys():
        agg_pose_auc[scene_name] = temp_pose_auc[scene_name][0] / temp_pose_auc[scene_name][1]

    return agg_pose_auc


def order_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    return rect


def draw_homography_comparison(image1, image2, real_warped_corners, warped_corners, mean_dist, file_name, save_path,
                               method):
    # Ensure the input image is in the correct format
    border_size = 100
    image2 = expand_image(image2, border_size)
    image1 = expand_image(image1, border_size)
    real_warped_corners += border_size
    warped_corners += border_size
    if image1.dtype != np.uint8:
        image1 = cv2.convertScaleAbs(image1)
    if image2.dtype != np.uint8:
        image2 = cv2.convertScaleAbs(image2)
    real_warped_corners = np.array(real_warped_corners, dtype=np.int32)
    warped_corners = np.array(warped_corners, dtype=np.int32)

    real_warped_corners = real_warped_corners.reshape((-1, 1, 2))
    warped_corners = warped_corners.reshape((-1, 1, 2))

    # combined_image = np.hstack((image1, image2))
    combined_image = np.hstack((image2, image1))

    combined_image = cv2.polylines(combined_image, [real_warped_corners], isClosed=True, color=(0, 255, 0),
                                   thickness=2)
    combined_image = cv2.polylines(combined_image, [warped_corners], isClosed=True, color=(0, 0, 255),
                                   thickness=2)

    plt.figure(figsize=(12, 6))
    combined_image = combined_image.astype('uint8')
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Homography Comparison ({method} Mean Distance: {mean_dist:.2f})')
    plt.axis('off')

    save_full_path = os.path.join(save_path, f"{file_name}_homography_comparison.png")
    plt.savefig(save_full_path)
    plt.close()
    print(f"Saved homography comparison image to: {save_full_path}")


def expand_image(image, border_size):
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])


def ensure_three_channels(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def compute_mean_distance(real_H, pred_H, H, W, visualize=False, save_path=None, file_name=None, image1=None,
                          image2=None, method=None):
    corners = np.array([[0, 0, 1],
                        [W - 1, 0, 1],
                        [0, H - 1, 1],
                        [W - 1, H - 1, 1]])
    real_H = np.linalg.inv(real_H)
    pred_H = np.linalg.inv(pred_H)
    # Compute warped corners using both estimated and real homographies
    real_warped_corners = np.dot(corners, np.transpose(real_H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(pred_H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    # Order corners to form a rectangle
    real_warped_corners = order_corners(real_warped_corners)
    warped_corners = order_corners(warped_corners)
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    if visualize:
        draw_homography_comparison(image1, image2, real_warped_corners, warped_corners, mean_dist, file_name, save_path,
                                   method)
    return mean_dist


def compute_matching_accuracy(mkpts0, mkpts1, H):
    n = mkpts0.shape[0]
    if n == 0:
        return [0, 0, 0], 0
    mkpts0_homogeneous = np.hstack([mkpts0, np.ones((mkpts0.shape[0], 1))])

    projected_pts = H @ mkpts0_homogeneous.T
    projected_pts /= projected_pts[2, :]

    projected_pts = projected_pts[:2, :].T

    distances = np.linalg.norm(projected_pts - mkpts1, axis=1)

    thresholds = [1, 3, 5]
    accuracies = []

    for threshold in thresholds:
        correct_matches = np.sum(distances <= threshold)
        accuracy = correct_matches / mkpts0.shape[0]
        accuracies.append(accuracy)

    return accuracies, n


def make_matching_plot_fast(img1, img2, mkpts0, mkpts1, mkpts0_f, mkpts1_f, color, text, show_keypoints=True,
                            mask=None):
    """
    Generates a visualization of image matches.
    """
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape
    out = np.zeros((max(H1, H2), W1 + W2, 3), dtype=np.uint8)
    out[:H1, :W1, :] = img1
    out[:H2, W1:, :] = img2

    # Draw matches
    for i, (pt1, pt2) in enumerate(zip(mkpts0, mkpts1)):
        if mask[i] == 1:
            c = (0, 255, 0)
        else:
            c = (0, 0, 255)
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + W1), int(pt2[1]))
        cv2.line(out, pt1, pt2, color=(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)), thickness=1)

    if show_keypoints:
        for pt1, pt2 in zip(mkpts0_f, mkpts1_f):
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0] + W1), int(pt2[1]))
            cv2.circle(out, pt1, 2, (0, 255, 0), 1)
            cv2.circle(out, pt2, 2, (0, 255, 0), 1)

    scale = 1
    for i, t in enumerate(text):
        cv2.putText(out, t, (5, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def make_matching_plot_fast_correct(img1, img2, mkpts0, mkpts1, mkpts0_f, mkpts1_f, color, text, show_keypoints=True):
    """
    Generates a visualization of image matches.
    """
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape
    out = np.zeros((max(H1, H2), W1 + W2, 3), dtype=np.uint8)
    out[:H1, :W1, :] = img1
    out[:H2, W1:, :] = img2

    # Draw matches
    for (pt1, pt2, c) in zip(mkpts0, mkpts1, color):
        c = (0, 255, 0)
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + W1), int(pt2[1]))
        cv2.line(out, pt1, pt2, color=(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)), thickness=1)

    if show_keypoints:
        for pt1, pt2 in zip(mkpts0_f, mkpts1_f):
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0] + W1), int(pt2[1]))
            cv2.circle(out, pt1, 2, (0, 255, 0), 1)
            cv2.circle(out, pt2, 2, (0, 255, 0), 1)

    scale = 1
    for i, t in enumerate(text):
        cv2.putText(out, t, (5, 20 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def compute_mask(real_H, mkpts02, mkpts12, threshold=3):
    mkpts02_h = np.hstack([mkpts02, np.ones((mkpts02.shape[0], 1))])
    mkpts12_h = np.hstack([mkpts12, np.ones((mkpts12.shape[0], 1))])

    projected_mkpts12_h = (real_H @ mkpts02_h.T).T
    projected_mkpts12 = projected_mkpts12_h[:, :2] / projected_mkpts12_h[:, 2, np.newaxis]

    projected_mkpts02_h = (np.linalg.inv(real_H) @ mkpts12_h.T).T
    projected_mkpts02 = projected_mkpts02_h[:, :2] / projected_mkpts02_h[:, 2, np.newaxis]

    error12 = np.linalg.norm(mkpts12 - projected_mkpts12, axis=1)
    error02 = np.linalg.norm(mkpts02 - projected_mkpts02, axis=1)
    mean_error = (error12 + error02) / 2

    mask = mean_error < threshold
    return mask


def eval_relapose(
        matcher,
        scene_pairs,
        save_figs,
        figures_dir=None,
        method=None,
        print_out=False,
        debug=False,
        read_color=False,
):
    scene_pose_auc = {}
    precs = {}
    precs_no_inlier = {}
    for scene_name in scene_pairs.keys():
        print(f"scene_name: {scene_name}")
        scene_dir = osp.join(figures_dir, scene_name.split(".")[0])
        if save_figs and not osp.exists(scene_dir):
            os.makedirs(scene_dir)

        statis = defaultdict(list)
        # continue

        groups = scene_pairs[scene_name]

        # Eval on pairs
        logging.info(f"\nStart evaluation on VisEvent \n")
        for i, pair in tqdm(enumerate(groups), smoothing=.1, total=len(groups)):
            if debug and i > 10:
                break

            im0 = pair['im0']
            im1 = pair['im1']
            real_H = pair['H']
            # real_H = np.eye(3)
            if read_color:
                im0 = cv2.imread(im0, cv2.IMREAD_COLOR)
                im1 = cv2.imread(im1, cv2.IMREAD_COLOR)

                im0 = cv2.resize(im0, (640, 480))
                im1 = cv2.resize(im1, (640, 480))
                im0_tensor = torch.tensor(im0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.
                real_H = torch.tensor(real_H, dtype=torch.float32).unsqueeze(0)
                im0_tensor = H_transform(im0_tensor, real_H)
                im0 = im0_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
                real_H = real_H.squeeze().cpu().numpy()
            else:
                im0 = cv2.imread(im0, cv2.IMREAD_GRAYSCALE)
                im1 = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
                im0 = cv2.resize(im0, (640, 480))
                im1 = cv2.resize(im1, (640, 480))
                im0_tensor = torch.tensor(im0, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
                real_H = torch.tensor(real_H, dtype=torch.float32).unsqueeze(0)
                im0_tensor = H_transform(im0_tensor, real_H)
                im0 = im0_tensor.squeeze().cpu().numpy() * 255
                real_H = real_H.squeeze().cpu().numpy()

            match_res = matcher(im0, im1)

            matches = match_res['matches']
            mkpts0 = match_res['mkpts0']
            mkpts1 = match_res['mkpts1']
            img0 = match_res['img0']
            img1 = match_res['img1']
            mconf = match_res['mconf']
            if len(mconf) > 0:
                conf_min = mconf.min()
                conf_max = mconf.max()
                mconf = (mconf - conf_min) / (conf_max - conf_min + 1e-5)
            color = cm.jet(mconf)

            if len(img0.shape) == 2:
                H, W = img0.shape
                img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            else:
                H, W, _ = img0.shape

            # Calculate pose errors
            img0_name = f"{'vis' if 'images' in pair['im0'] else 'event'}_{osp.basename(pair['im0']).split('.')[0]}"
            img1_name = f"{'vis' if 'images' in pair['im1'] else 'event'}_{osp.basename(pair['im1']).split('.')[0]}"
            file_name = f"{img0_name}_{img1_name}"

            # try:
            if len(mkpts0) >= 4:
                ret_H, inliers = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC)

                if ret_H is not None:

                    mean_dist = compute_mean_distance(real_H, ret_H, H, W, visualize=save_figs,
                                                      save_path=scene_dir,
                                                      file_name=file_name, image1=img1,
                                                      image2=img0, method=method)
                else:
                    mean_dist = np.inf
            else:
                ret_H = None
                mean_dist = np.inf

            if save_figs:
                # mkpts0 = mkpts0[inliers.ravel() == 1]
                # mkpts1 = mkpts1[inliers.ravel() == 1]
                mask0 = compute_mask(real_H, mkpts1, mkpts0, threshold=3)
                # number_inliers0 = np.sum(inliers)
                # number_mask0 = np.sum(mask0)
                # plot2(file_name, scene_dir, img1, img0, mkpts1, mkpts0, color, method, mask0, number_inliers0,
                #      number_mask0, mean_dist)
                fig_path = osp.join(scene_dir, f"{img0_name}_{img1_name}_{method}.jpg")
                save_matching_figure2(path=fig_path,
                                      img0=img0,
                                      img1=img1,
                                      mkpts0=mkpts0,
                                      mkpts1=mkpts1,
                                      mean_distance=mean_dist,
                                      correct_mask=mask0,
                                      svg=args.svg
                                      )
            if ret_H is None:
                statis['mean_dist'].append(np.inf)
                statis['failed'].append(i)
                statis['matching_accuracy'].append((0, 0, 0))
                statis['n'].append(0)
            else:
                matching_accuracy, n = compute_matching_accuracy(mkpts0, mkpts1, real_H)
                statis['mean_dist'].append(mean_dist)
                statis['matching_accuracy'].append(matching_accuracy)
                statis['n'].append(n)

                if print_out:
                    logging.info(f"#M={len(matches)} R={mean_dist:.3f}, t={matching_accuracy:.3f}")

        logging.info(f"Scene: {scene_name} Total samples: {len(groups)} Failed:{len(statis['failed'])}. \n")
        mean_dist_all = np.array(statis['mean_dist'])
        thresholds = [1, 3, 5, 7, 10, 15, 20]
        Homograpy_auc = error_auc(mean_dist_all, thresholds)
        logging.info('statis[mean_dist]: %s', statis['mean_dist'])
        logging.info('statis[matching_accuracy]: %s', statis['matching_accuracy'])
        logging.info('statis[n]: %s', statis['n'])
        total_match_nums = np.zeros(3)
        count = 0
        nums_image = 0
        all_accuracies_array = np.zeros(3)
        for accuracies, nums_match in zip(statis['matching_accuracy'], statis['n']):
            # First parameter: total number of matches / total number of matches
            accuracies = np.array(accuracies)
            match_nums = accuracies * nums_match
            total_match_nums += match_nums
            count += nums_match

            # Second parameter: total number of matches / total number of images
            nums_image += 1

            # Third parameter: sum of single image matching accuracy / total number of images
            all_accuracies_array += accuracies

            # Fourth parameter, total number of matches / total number of images

        average_matching_accuracy = total_match_nums / count
        average_matching_nums = count / nums_image
        average_accuracies_array = all_accuracies_array / nums_image
        average_matching_accuracy_nums = total_match_nums / nums_image
        filtered_mean_dist = mean_dist_all[np.isfinite(mean_dist_all)]

        average_mean_dist = np.mean(filtered_mean_dist)

        logging.info(f"\nAverage Mean Dist: {average_mean_dist}\n")
        logging.info(f"\nAverage Matching Accuracy: {average_matching_accuracy}\n")
        logging.info(f"\nAverage Matching Nums: {average_matching_nums}\n")
        logging.info(f"\nAverage Accuracies Array: {average_accuracies_array}\n")
        logging.info(f"\nAverage Matching Accuracy Nums: {average_matching_accuracy_nums}\n")
        scene_pose_auc[scene_name] = 100 * np.array([Homograpy_auc[f'auc@{t}'] for t in thresholds])
        logging.info(f"{scene_name} {Homograpy_auc}")

    agg_pose_auc = aggregiate_scenes(scene_pose_auc, thresholds)

    agg_precs, agg_precs_no_inlier = aggregate_precisions(precs, precs_no_inlier)

    return scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier


def aggregate_precisions(precs, precs_no_inlier):
    """Aggregate precision values across cloudy_cloud and cloudy_sunny scenes."""
    temp_precs = defaultdict(lambda: defaultdict(list))
    temp_precs_no_inlier = defaultdict(lambda: defaultdict(list))

    for scene_name, precision_dict in precs.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs[main_scene][threshold].append(precision)

    for scene_name, precision_dict in precs_no_inlier.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs_no_inlier[main_scene][threshold].append(precision)

    agg_precs = {scene: {threshold: np.mean(values) for threshold, values in thresholds_dict.items()}
                 for scene, thresholds_dict in temp_precs.items()}

    agg_precs_no_inlier = {scene: {threshold: np.mean(values) for threshold, values in thresholds_dict.items()}
                           for scene, thresholds_dict in temp_precs_no_inlier.items()}

    return agg_precs, agg_precs_no_inlier


def plot2(file_name, scene_dir, img0, img1, mkpts0, mkpts1, color, method, mask1, number_inliers, number_mask,
          mean_dist):
    save_path_cv = os.path.join(scene_dir, f'{file_name}_match_result_cv_{method}.png')

    output_image = make_matching_plot_fast(
        img1, img0, mkpts1, mkpts0, mkpts1, mkpts0, color,
        text=['Mean Distance: {:.2f}'.format(mean_dist),
              '# Matches: {}'.format(number_inliers),
              f'Precision(3px)({100 * number_mask / number_inliers:.1f}%): {number_mask} / {number_inliers}', ],
        show_keypoints=False, mask=mask1
    )

    cv2.imwrite(save_path_cv, output_image)


def test_relative_pose_visevent(
        method="xoftr",
        exp_name="VisEvent",
        ransac_thres=1.5,
        print_out=False,
        save_dir=None,
        save_figs=False,
        debug=False,
        args=None

):
    # save_dir = osp.join(save_dir, time)

    if method == "roma":
        if args.ckpt is None:
            save_ = "roma"
        else:
            save_ = args.ckpt.split("/")[-1].replace(".ckpt", "")

    else:
        save_ = args.ckpt.split("/")[-1].replace(".ckpt", "")
    path_ = osp.join(save_dir, method, save_)
    read_color = False

    if method == 'roma':
        read_color = True

    if args.debug:
        path_ = osp.join(save_dir, method, save_, "debug")
    if not osp.exists(path_):
        os.makedirs(path_)

    counter = 0
    if hasattr(args, 'thr'):
        path = osp.join(path_, f"{exp_name}_thresh_{args.thr}" + "_{}")
    else:
        path = osp.join(path_, f"{exp_name}" + "_{}")
    while osp.exists(path.format(counter)):
        counter += 1
    exp_dir = path.format(counter)
    os.mkdir(exp_dir)
    results_file = osp.join(exp_dir, "results.json")
    logging.basicConfig(
        filename=results_file.replace('.json', '.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    figures_dir = osp.join(exp_dir, "match_figures")
    if save_figs:
        os.mkdir(figures_dir)

    # Log args
    logging.info(f"args: {args}")

    # Init paths
    data_root = './data/DSEC/'
    random_txt_file = "./data/DSEC/event_list.txt"
    # Load pairs
    scene_pairs = load_vis_event_pairs_npz(data_root, random_txt_file)

    # Load method
    # matcher = eval(f"load_{method}")(args)
    matcher = load_model(method, args, use_path=False)
    thresholds = [5, 10, 20]
    # Eval
    scene_pose_auc, agg_pose_auc, precs, precs_no_inlier, agg_precs, agg_precs_no_inlier = eval_relapose(
        matcher,
        scene_pairs,
        save_figs=save_figs,
        figures_dir=figures_dir,
        method=method,
        print_out=print_out,
        debug=debug,
        read_color=read_color
    )

    # Create result dict
    results = OrderedDict({"method": method,
                           "exp_name": exp_name,
                           "ransac_thres": ransac_thres,
                           "auc_thresholds": thresholds})
    results.update({key: value for key, value in vars(args).items() if key not in results})
    results.update({key: value.tolist() for key, value in agg_pose_auc.items()})
    results.update({key: value.tolist() for key, value in scene_pose_auc.items()})

    results.update({f"precs_{key}": value for key, value in precs.items()})

    results.update({f"precs_no_inlier_{key}": value for key, value in precs_no_inlier.items()})

    results.update({f"agg_precs_{key}": value for key, value in agg_precs.items()})

    results.update({f"agg_precs_no_inlier_{key}": value for key, value in agg_precs_no_inlier.items()})

    logging.info(f"Results: {json.dumps(results, indent=4)}")

    # Save to json file
    with open(results_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    logging.info(f"Results saved to {results_file}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark Relative Pose')

    choose_method_arguments(parser)
    parser.add_argument('--exp_name', type=str, default="VisEvent")
    parser.add_argument('--save_dir', type=str, default="./results_relative_event_homo/")
    parser.add_argument('--e_name', type=str, default=None)
    parser.add_argument('--ransac_thres', type=float, default=1.5)
    parser.add_argument('--print_out', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_figs', action='store_true')
    parser.add_argument('--svg', action='store_true')

    args, remaining_args = parser.parse_known_args()

    add_method_arguments(parser, args.method)

    args = parser.parse_args()

    print(args)

    if args.e_name is not None:
        save_dir = osp.join(args.save_dir, args.e_name)
    else:
        save_dir = args.save_dir

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_relative_pose_visevent(
            args.method,
            args.exp_name,
            ransac_thres=args.ransac_thres,
            print_out=args.print_out,
            save_dir=save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args
        )
    print(f"Elapsed time: {time.time() - tt}")
