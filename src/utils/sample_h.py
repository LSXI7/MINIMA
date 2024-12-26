import collections
import hashlib
import random
from math import pi

import cv2
import numpy as np
import torch
from numpy.random import uniform
from scipy import stats


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def set_random_seed(image_name):
    hash_value = generate_hash(image_name)
    seed = int(hash_value, 16) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def generate_hash(image_name):
    return hashlib.sha256(image_name.encode()).hexdigest()


def sample_homography(shape, image_name, config=None, device='cpu'):
    set_random_seed(image_name)  # set random seed according to image name
    default_config = {'perspective': True, 'scaling': True, 'rotation': True, 'translation': True,
                      'n_scales': 5, 'n_angles': 25, 'scaling_amplitude': 0.2, 'perspective_amplitude_x': 0.1,
                      'perspective_amplitude_y': 0.1, 'patch_ratio': 0.5, 'max_angle': pi / 2,
                      'allow_artifacts': False, 'translation_overflow': 0.}

    if config is not None:
        config = dict_update(default_config, config)
    else:
        config = default_config

    std_trunc = 2

    # Corners of the input patch
    margin = (1 - config['patch_ratio']) / 2
    pts1 = margin + np.array([[0, 0],
                              [0, config['patch_ratio']],
                              [config['patch_ratio'], config['patch_ratio']],
                              [config['patch_ratio'], 0]])
    pts2 = pts1.copy()

    # Random perspective and affine perturbations
    if config['perspective']:
        if not config['allow_artifacts']:
            perspective_amplitude_x = min(config['perspective_amplitude_x'], margin)
            perspective_amplitude_y = min(config['perspective_amplitude_y'], margin)
        else:
            perspective_amplitude_x = config['perspective_amplitude_x']
            perspective_amplitude_y = config['perspective_amplitude_y']

        tnorm_y = stats.truncnorm(-2, 2, loc=0, scale=perspective_amplitude_y / 2)
        tnorm_x = stats.truncnorm(-2, 2, loc=0, scale=perspective_amplitude_x / 2)
        perspective_displacement = tnorm_y.rvs(1)
        h_displacement_left = tnorm_x.rvs(1)
        h_displacement_right = tnorm_x.rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if config['scaling']:
        mu, sigma = 1, config['scaling_amplitude'] / 2
        lower, upper = mu - 2 * sigma, mu + 2 * sigma
        tnorm_s = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        scales = tnorm_s.rvs(config['n_scales'])
        # scales = np.random.uniform(0.8, 2, config['n_scales'])
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if config['allow_artifacts']:
            valid = np.arange(config['n_scales'])  # all scales are valid except scale=1
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx, :, :]

    # Random translation
    if config['translation']:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if config['allow_artifacts']:
            t_min += config['translation_overflow']
            t_max += config['translation_overflow']
        pts2 += np.array([uniform(-t_min[0], t_max[0], 1), uniform(-t_min[1], t_max[1], 1)]).T

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if config['rotation']:
        angles = np.linspace(-config['max_angle'], config['max_angle'], num=config['n_angles'])
        angles = np.concatenate((np.array([0.]), angles), axis=0)  # in case no rotation is valid
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis, :, :], rot_mat) + center

        if config['allow_artifacts']:
            valid = np.arange(config['n_angles'])  # all scales are valid except scale=1
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx, :, :]

    # Rescale to actual size
    shape = np.array(shape[::-1])  # different convention [y, x]
    pts1 *= shape[np.newaxis, :]
    pts2 *= shape[np.newaxis, :]

    # this homography is the same with tf version and this line
    homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    homography = torch.tensor(homography, device=device, dtype=torch.float32).unsqueeze(dim=0)

    homography = torch.inverse(homography)  # inverse here to be consistent with tf version

    return homography  # [1,3,3]
