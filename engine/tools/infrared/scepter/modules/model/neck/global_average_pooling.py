# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import NECKS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@NECKS.register_class()
class GlobalAveragePooling(BaseModel):
    """Global Average Pooling neck.
    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """
    para_dict = {
        'DIM': {
            'value': 2,
            'description': 'GlobalAveragePooling dim!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(GlobalAveragePooling, self).__init__(cfg, logger=logger)
        dim = cfg.get('DIM', 2)
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
                                 f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def infer(self, x):
        if x.ndim == 2:
            return x
        return self.gap(x).view(x.size(0), -1)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            return tuple([self.infer(x) for x in inputs])
        else:
            return self.infer(inputs)

    @staticmethod
    def get_config_template():
        '''
        { "ENV" :
            { "description" : "",
              "A" : {
                    "value": 1.0,
                    "description": ""
               }
            }
        }
        :return:
        '''
        return dict_to_yaml('NECKS',
                            __class__.__name__,
                            GlobalAveragePooling.para_dict,
                            set_name=True)
