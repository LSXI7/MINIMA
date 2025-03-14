# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod

from tools.infrared.scepter.modules.model.registry import TOKENIZERS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@TOKENIZERS.register_class()
class BaseTokenizer(metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        pass

    def tokenize(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x, **kwargs):
        self.tokenize(x, **kwargs)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TOKENIZERS',
                            __class__.__name__,
                            BaseTokenizer.para_dict,
                            set_name=True)
