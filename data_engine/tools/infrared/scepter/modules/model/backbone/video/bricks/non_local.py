# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
""" NonLocal block. """

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.infrared.scepter.modules.model.backbone.video.bricks.visualize_3d_module import \
    Visualize3DModule
from tools.infrared.scepter.modules.model.registry import BRICKS
from tools.infrared.scepter.modules.utils.config import Config, dict_to_yaml


@BRICKS.register_class()
class NonLocal(Visualize3DModule):
    """
    Non-local block.

    See Xiaolong Wang et al.
    Non-local Neural Networks.
    """
    para_dict = {
        'DIM_IN': {
            'value': 64,
            'description': "the branch's dim in!"
        },
        'NUM_FILTERS': {
            'value': 64,
            'description': 'the num of filter!'
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
        super(NonLocal, self).__init__(cfg, logger=logger)
        self.dim_in = cfg.DIM_IN
        self.num_filters = cfg.NUM_FILTERS
        bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(bn_params, Config):
            bn_params = bn_params.__dict__
        bn_params['eps'] = 1e-5
        self.dim_middle = self.dim_in // 2

        self.qconv = nn.Conv3d(self.dim_in,
                               self.dim_middle,
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=0)

        self.kconv = nn.Conv3d(self.dim_in,
                               self.dim_middle,
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=0)

        self.vconv = nn.Conv3d(self.dim_in,
                               self.dim_middle,
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=0)

        self.out_conv = nn.Conv3d(
            self.dim_middle,
            self.num_filters,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=0,
        )

        self.out_bn = nn.BatchNorm3d(self.num_filters, **bn_params)

    def forward(self, x):
        n, c, t, h, w = x.shape

        query = self.qconv(x).view(n, self.dim_middle, -1)
        key = self.kconv(x).view(n, self.dim_middle, -1)
        value = self.vconv(x).view(n, self.dim_middle, -1)

        attn = torch.einsum('nct,ncp->ntp', (query, key))
        attn = attn * (self.dim_middle**-0.5)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('ntg,ncg->nct', (attn, value))
        out = out.view(n, self.dim_middle, t, h, w)
        out = self.out_conv(out)
        out = self.out_bn(out)
        return x + out

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
                            NonLocal.para_dict,
                            set_name=True)
