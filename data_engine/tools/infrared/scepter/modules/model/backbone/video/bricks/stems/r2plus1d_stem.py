# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.video.bricks.stems.base_3d_stem import \
    Base3DStem
from tools.infrared.scepter.modules.model.registry import STEMS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@STEMS.register_class()
class R2Plus1DStem(Base3DStem):
    para_dict = {}
    para_dict.update(Base3DStem.para_dict)

    def __init__(self, cfg, logger=None):
        super(R2Plus1DStem, self).__init__(cfg, logger=logger)

    def _construct(self):
        mid_dim = int(
            math.floor(
                (self.kernel_size[0] * self.kernel_size[1] *
                 self.kernel_size[2] * self.dim_in * self.num_filters) /
                (self.kernel_size[1] * self.kernel_size[2] * self.dim_in +
                 self.kernel_size[0] * self.num_filters)))

        self.a1 = nn.Conv3d(in_channels=self.dim_in,
                            out_channels=mid_dim,
                            kernel_size=(1, self.kernel_size[1],
                                         self.kernel_size[2]),
                            stride=(1, self.stride[1], self.stride[2]),
                            padding=(0, self.kernel_size[1] // 2,
                                     self.kernel_size[2] // 2),
                            bias=False)
        self.a1_bn = nn.BatchNorm3d(mid_dim, **self.bn_params)
        self.a1_relu = nn.ReLU(inplace=True)

        self.a2 = nn.Conv3d(in_channels=mid_dim,
                            out_channels=self.num_filters,
                            kernel_size=(self.kernel_size[0], 1, 1),
                            stride=(self.stride[0], 1, 1),
                            padding=(self.kernel_size[0] // 2, 0, 0),
                            bias=False)
        self.a2_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)
        self.a2_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a1(x)
        x = self.a1_bn(x)
        x = self.a1_relu(x)

        x = self.a2(x)
        x = self.a2_bn(x)
        x = self.a2_relu(x)
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
        return dict_to_yaml('STEM',
                            __class__.__name__,
                            R2Plus1DStem.para_dict,
                            set_name=True)
