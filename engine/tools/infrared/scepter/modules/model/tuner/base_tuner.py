# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import TUNERS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@TUNERS.register_class()
class BaseTuner(BaseModel, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TUNERS',
                            __class__.__name__,
                            BaseTuner.para_dict,
                            set_name=True)
