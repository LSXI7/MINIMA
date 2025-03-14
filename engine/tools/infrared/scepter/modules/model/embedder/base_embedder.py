# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import EMBEDDERS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@EMBEDDERS.register_class()
class BaseEmbedder(BaseModel, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def encode_text(self, *args, **kwargs):
        raise NotImplementedError

    def encode_image(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_config_template():
        return dict_to_yaml('EMBEDDERS',
                            __class__.__name__,
                            BaseEmbedder.para_dict,
                            set_name=True)
