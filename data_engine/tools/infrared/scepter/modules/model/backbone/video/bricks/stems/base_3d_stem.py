# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.video.bricks.visualize_3d_module import \
    Visualize3DModule
from tools.infrared.scepter.modules.model.registry import STEMS
from tools.infrared.scepter.modules.utils.config import Config, dict_to_yaml


@STEMS.register_class()
class Base3DStem(Visualize3DModule):
    para_dict = {
        'DIM_IN': {
            'value': 3,
            'description': "the stem's dim in!"
        },
        'NUM_FILTERS': {
            'value': 64,
            'description': 'the num of filter!'
        },
        'KERNEL_SIZE': {
            'value': [1, 7, 7],
            'description': 'the kernel size!'
        },
        'DOWNSAMPLING': {
            'value': True,
            'description': 'downsample spatial data or not!'
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': False,
            'description': 'downsample temporal data or not!'
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        }
    }
    para_dict.update(Visualize3DModule.para_dict)

    def __init__(self, cfg, logger=None):
        super(Base3DStem, self).__init__(cfg, logger=logger)

        self.dim_in = cfg.get('DIM_IN', 3)
        self.num_filters = cfg.get('NUM_FILTERS', 64)
        self.kernel_size = cfg.get('KERNEL_SIZE', [1, 7, 1])
        downsampling = cfg.get('DOWNSAMPLING', True)
        downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL', False)
        # bn_params or {}
        self.bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(self.bn_params, Config):
            self.bn_params = self.bn_params.__dict__
        if downsampling:
            if downsampling_temporal:
                self.stride = (2, 2, 2)
            else:
                self.stride = (1, 2, 2)
        else:
            self.stride = (1, 1, 1)

        self._construct()

    def _construct(self):
        self.a = nn.Conv3d(self.dim_in,
                           self.num_filters,
                           kernel_size=self.kernel_size,
                           stride=self.stride,
                           padding=[
                               self.kernel_size[0] // 2,
                               self.kernel_size[1] // 2,
                               self.kernel_size[2] // 2
                           ],
                           bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)
        self.a_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.a_relu(self.a_bn(self.a(x)))

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
        return dict_to_yaml('STEM',
                            __class__.__name__,
                            Base3DStem.para_dict,
                            set_name=True)
