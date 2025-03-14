# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.video.bricks.stems.base_3d_stem import \
    Base3DStem
from tools.infrared.scepter.modules.model.registry import STEMS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@STEMS.register_class()
class DownSampleStem(Base3DStem):
    para_dict = {}
    para_dict.update(Base3DStem.para_dict)

    def __init__(self, cfg, logger=None):
        super(DownSampleStem, self).__init__(cfg, logger=logger)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))

    def forward(self, x):
        return self.maxpool(self.a_relu(self.a_bn(self.a(x))))

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
                            DownSampleStem.para_dict,
                            set_name=True)
