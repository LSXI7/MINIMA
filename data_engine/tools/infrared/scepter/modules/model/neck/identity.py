# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import NECKS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@NECKS.register_class()
class Identity(BaseModel):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super(Identity, self).__init__(cfg, logger=logger)

    def forward(self, inputs):
        return inputs

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
        return dict_to_yaml('neckname',
                            __class__.__name__,
                            Identity.para_dict,
                            set_name=True)
