# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from tools.infrared.scepter.modules.model.backbone.image.resnet_impl import (resnet18,
                                                              resnet34,
                                                              resnet50,
                                                              resnet101,
                                                              resnet152)
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import BACKBONES
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@BACKBONES.register_class('ResNet')
class ResNet(BaseModel):
    para_dict = {
        'DEPTH': {
            'value': 18,
            'description': 'the depth of network for resnet!'
        },
        'KERNEL_SIZE': {
            'value':
            7,
            'description':
            'first conv kernel size, 7 or 3 (without stride and maxpooling)!'
        },
        'USE_RELU': {
            'value': True,
            'description': 'use relu or not!'
        },
        'USE_MAXPOOL': {
            'value': True,
            'description': 'use maxpool or not!'
        },
        'FIRST_CONV_STRIDE': {
            'value': 1,
            'description': 'first conv stride 1 or 2!'
        },
        'FIRST_MAX_POOL_STRIDE': {
            'value': 1,
            'description': 'first max pool stride 1 or 2!'
        },
        'PRETRAINED': {
            'value': False,
            'description': 'if load the official pretrained model or not.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ResNet, self).__init__(cfg, logger=logger)
        depth = cfg.get('DEPTH', 18)
        pretrained = cfg.get('PRETRAINED', False)
        kernel_size = cfg.get('KERNEL_SIZE', 7)
        use_relu = cfg.get('USE_RELU', True)
        use_maxpool = cfg.get('USE_MAXPOOL', True)
        first_conv_stride = cfg.get('FIRST_CONV_STRIDE', 1)
        first_max_pool_stride = cfg.get('FIRST_MAX_POOL_STRIDE', 1)
        depth_mapper = {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
            152: resnet152
        }
        cons_func = depth_mapper.get(depth)
        if cons_func is None:
            raise KeyError(f'Unsupported depth for resnet, {depth}')
        self.model = cons_func(pretrained=pretrained,
                               kernel_size=kernel_size,
                               use_relu=use_relu,
                               use_maxpool=use_maxpool,
                               first_conv_stride=first_conv_stride,
                               first_max_pool_stride=first_max_pool_stride)

    def forward(self, x):
        return self.model.forward(x)

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
        return dict_to_yaml('BACKBONES',
                            __class__.__name__,
                            ResNet.para_dict,
                            set_name=True)
