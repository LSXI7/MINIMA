# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url


class StdMsg():
    def __init__(self, name='msg'):
        self.name = name

    def info(self, msg):
        sys.stdout.write('[Info]: ' + msg + '\n')

    def error(self, msg):
        sys.stdout.write('[Error]: ' + msg + '\n')

    def warning(self, msg):
        sys.stdout.write('[Warning]: ' + msg + '\n')


def move_model_to_cpu(params):
    cpu_params = OrderedDict()
    for key, val in params.items():
        cpu_params[key] = val.cpu()
    return cpu_params


def load_pretrained(model: torch.nn.Module,
                    path: str,
                    map_location='cpu',
                    logger=None,
                    sub_level=None):
    if logger:
        logger.info(
            f'Load pretrained model [{model.__class__.__name__}] from {path}')
    if os.path.exists(path):
        # From local
        state_dict = torch.load(path, map_location)
    elif path.startswith('http'):
        # From url
        state_dict = load_state_dict_from_url(path,
                                              map_location=map_location,
                                              check_hash=False)
    else:
        raise Exception(f'Cannot find {path} when load pretrained')

    return load_pretrained_dict(model, state_dict, logger, sub_level=sub_level)


def _auto_drop_invalid(model: torch.nn.Module, state_dict: dict, logger=None):
    """ Strip unmatched parameters in state_dict, e.g. shape not matched, type not matched.

    Args:
        model (torch.nn.Module):
        state_dict (dict):
        logger (logging.Logger, None):

    Returns:
        A new state dict.
    """
    ret_dict = state_dict.copy()
    invalid_msgs = []
    for key, value in model.state_dict().items():
        if key in state_dict:
            # Check shape
            new_value = state_dict[key]
            if value.shape != new_value.shape:
                invalid_msgs.append(
                    f'{key}: invalid shape, dst {value.shape} vs. src {new_value.shape}'
                )
                ret_dict.pop(key)
            elif value.dtype != new_value.dtype:
                invalid_msgs.append(
                    f'{key}: invalid dtype, dst {value.dtype} vs. src {new_value.dtype}'
                )
                ret_dict.pop(key)
    if len(invalid_msgs) > 0:
        warning_msg = 'ignore keys from source: \n' + '\n'.join(invalid_msgs)
        if logger:
            logger.warning(warning_msg)
        else:
            import warnings
            warnings.warn(warning_msg)
    return ret_dict


def load_pretrained_dict(model: torch.nn.Module,
                         state_dict: dict,
                         logger=None,
                         sub_level=None):
    """ Load parameters to model with
    1. Sub name by revise_keys For DataParallelModel or DistributeParallelModel.
    2. Load 'state_dict' again if possible by key 'state_dict' or 'model_state'.
    3. Take sub level keys from source, e.g. load 'backbone' part from a classifier into a backbone model.
    4. Auto remove invalid parameters from source.
    5. Log or warning if unexpected key exists or key misses.

    Args:
        model (torch.nn.Module):
        state_dict (dict): dict of parameters
        logger (logging.Logger, None):
        sub_level (str, optional): If not None, parameters with key startswith sub_level will remove the prefix
            to fit actual model keys. This action happens if user want to load sub module parameters
            into a sub module model.
    """
    revise_keys = [(r'^module\.', '')]

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']

    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    if sub_level:
        sub_level = sub_level if sub_level.endswith('.') else (sub_level + '.')
        sub_level_len = len(sub_level)
        state_dict = {
            key[sub_level_len:]: value
            for key, value in state_dict.items() if key.startswith(sub_level)
        }

    state_dict = _auto_drop_invalid(model, state_dict, logger=logger)

    load_status = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = load_status.unexpected_keys
    missing_keys = load_status.missing_keys
    err_msgs = []
    if unexpected_keys:
        err_msgs.append('unexpected key in source '
                        f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msgs.append('missing key in source '
                        f'state_dict: {", ".join(missing_keys)}\n')
    err_msgs = '\n'.join(err_msgs)

    if len(err_msgs) > 0:
        if logger:
            logger.warning(err_msgs)
        else:
            import warnings
            warnings.warn(err_msgs)


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
