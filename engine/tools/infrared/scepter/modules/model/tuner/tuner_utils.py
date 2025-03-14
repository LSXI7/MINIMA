# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn


def choose_weight_type(weight_type, dim):
    if weight_type == 'gate':
        scaling = nn.Linear(dim, 1)
    elif weight_type == 'scale':
        scaling = nn.Parameter(torch.Tensor(1))
        scaling.data.fill_(1)
    elif weight_type == 'scale_channel':
        scaling = nn.Parameter(torch.Tensor(dim))
        scaling.data.fill_(1)
    elif weight_type and weight_type.startswith('scalar'):
        scaling = float(weight_type.split('_')[-1])
    else:
        scaling = None
    return scaling


def get_weight_value(weight_type, scaling, x):
    if weight_type in ['gate']:
        scaling = torch.mean(torch.sigmoid(scaling(x)), dim=1).view(-1, 1, 1)
    elif weight_type in ['scale', 'scale_channel'
                         ] or weight_type.startswith('scalar'):
        scaling = scaling
    else:
        scaling = None
    return scaling
