# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.video.bricks.base_branch import BaseBranch
from tools.infrared.scepter.modules.model.registry import BRICKS
from tools.infrared.scepter.modules.utils.config import Config, dict_to_yaml


@BRICKS.register_class()
class CSNBranch(BaseBranch):
    para_dict = {
        'DIM_IN': {
            'value': 64,
            'description': "the branch's dim in!"
        },
        'NUM_FILTERS': {
            'value': 64,
            'description': 'the num of filter!'
        },
        'DOWNSAMPLING': {
            'value': True,
            'description': 'downsample spatial data or not!'
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': True,
            'description': 'downsample temporal data or not!'
        },
        'EXPANISION_RATIO': {
            'value': 2,
            'description': 'expanision ratio for this branch!'
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        }
    }
    para_dict.update(BaseBranch.para_dict)

    def __init__(self, cfg, logger=None):
        self.dim_in = cfg.DIM_IN
        self.num_filters = cfg.NUM_FILTERS
        self.kernel_size = cfg.KERNEL_SIZE
        self.downsampling = cfg.get('DOWNSAMPLING', True)
        self.downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL', True)
        self.expansion_ratio = cfg.get('EXPANISION_RATIO', 2)
        # bn_params or {}
        self.bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(self.bn_params, Config):
            self.bn_params = self.bn_params.__dict__
        if self.downsampling:
            if self.downsampling_temporal:
                self.stride = (2, 2, 2)
            else:
                self.stride = (1, 2, 2)
        else:
            self.stride = (1, 1, 1)
        super(CSNBranch, self).__init__(cfg, logger=logger)

    def _construct_simple_block(self):
        raise NotImplementedError

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(in_channels=self.dim_in,
                           out_channels=self.num_filters //
                           self.expansion_ratio,
                           kernel_size=(1, 1, 1),
                           stride=(1, 1, 1),
                           padding=0,
                           bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio,
                                   **self.bn_params)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv3d(
            in_channels=self.num_filters // self.expansion_ratio,
            out_channels=self.num_filters // self.expansion_ratio,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=[
                self.kernel_size[0] // 2, self.kernel_size[1] // 2,
                self.kernel_size[2] // 2
            ],
            bias=False,
            groups=self.num_filters // self.expansion_ratio)
        self.b_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio,
                                   **self.bn_params)
        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(in_channels=self.num_filters //
                           self.expansion_ratio,
                           out_channels=self.num_filters,
                           kernel_size=(1, 1, 1),
                           stride=(1, 1, 1),
                           padding=0,
                           bias=False)
        self.c_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)

    def forward(self, x):
        if self.branch_style == 'bottleneck':
            x = self.a(x)
            x = self.a_bn(x)
            x = self.a_relu(x)

            x = self.b(x)
            x = self.b_bn(x)
            x = self.b_relu(x)

            x = self.c(x)
            x = self.c_bn(x)
            return x

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
        return dict_to_yaml('BRANCH',
                            __class__.__name__,
                            CSNBranch.para_dict,
                            set_name=True)
