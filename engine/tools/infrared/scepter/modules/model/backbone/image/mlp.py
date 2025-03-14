# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import BACKBONES
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


def MLP_unit(input_dim, output_dim, use_bn=False, use_relu=False):
    layers = []
    layers.append(nn.Linear(input_dim, output_dim))
    if use_bn:
        layers.append(nn.BatchNorm1d(output_dim))
    if use_relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


@BACKBONES.register_class()
class MLP(BaseModel):
    para_dict = {
        'IN_DIM': {
            'value': [10, 10],
            'description':
            "the input dim for each linear, which also the previous' out dim!"
        },
        'OUT_DIM': {
            'value':
            10,
            'description':
            'The output dim for head, often this value is the classes number!'
        },
        'USE_BN': {
            'value': [True],
            'description':
            'The MLP before proj use bn or not for each layer! len() = len(IN_DIM) - 1'
        },
        'USE_RELU': {
            'value': [False],
            'description':
            'The MLP before proj use relu or not for each layer!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(MLP, self).__init__(cfg, logger=logger)
        self.dim_list = cfg.IN_DIM
        self.use_bn = cfg.USE_BN
        self.use_relu = cfg.USE_RELU
        self.out_feature = cfg.OUT_DIM
        assert len(self.dim_list) >= 1
        layers = []
        for idx, dim in enumerate(self.dim_list):
            if idx == 0:
                in_feature = dim
            else:
                out_feature = dim
                layers.append(
                    MLP_unit(in_feature,
                             out_feature,
                             use_bn=self.use_bn[idx - 1],
                             use_relu=self.use_relu[idx - 1]))
                in_feature = dim

        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(in_feature, self.out_feature)
        self.bn = nn.BatchNorm1d(self.out_feature)

    def forward(self, x):
        x = x.type(self.fc.weight.dtype)
        x = self.mlp(x)
        x = self.fc(x)
        x = self.bn(x)
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
        return dict_to_yaml('HEADS',
                            __class__.__name__,
                            MLP.para_dict,
                            set_name=True)
