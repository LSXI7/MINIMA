# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def transfer_size(para_num):
    if para_num > 1000 * 1000 * 1000 * 1000:
        bill = para_num / (1000 * 1000 * 1000 * 1000)
        return '{:.2f}T'.format(bill)
    elif para_num > 1000 * 1000 * 1000:
        gyte = para_num / (1000 * 1000 * 1000)
        return '{:.2f}B'.format(gyte)
    elif para_num > (1000 * 1000):
        meta = para_num / (1000 * 1000)
        return '{:.2f}M'.format(meta)
    elif para_num > 1000:
        kelo = para_num / 1000
        return '{:.2f}K'.format(kelo)
    else:
        return para_num


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return transfer_size(total_params)


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x
