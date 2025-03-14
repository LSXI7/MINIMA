# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from tools.infrared.scepter.modules.utils.config import Config
from tools.infrared.scepter.modules.utils.registry import Registry, build_from_config


def build_annotator(cfg, registry, logger=None, *args, **kwargs):
    """ After build model, load pretrained model if exists key `pretrain`.

    pretrain (str, dict): Describes how to load pretrained model.
        str, treat pretrain as model path;
        dict: should contains key `path`, and other parameters token by function load_pretrained();
    """
    if not isinstance(cfg, Config):
        raise TypeError(f'Config must be type dict, got {type(cfg)}')
    if cfg.have('PRETRAINED_MODEL'):
        pretrain_cfg = cfg.PRETRAINED_MODEL
        if pretrain_cfg is not None and not isinstance(pretrain_cfg, (str)):
            raise TypeError('Pretrain parameter must be a string')
    else:
        pretrain_cfg = None

    model = build_from_config(cfg, registry, logger=logger, *args, **kwargs)
    if pretrain_cfg is not None:
        if hasattr(model, 'load_pretrained_model'):
            model.load_pretrained_model(pretrain_cfg)
    return model


ANNOTATORS = Registry('ANNOTATORS', build_func=build_annotator)
