# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod

import torch
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


class TrainModule(BaseModel, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super(TrainModule, self).__init__(cfg, logger=logger)
        self.logger = logger
        self.cfg = cfg

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        pass

    @abstractmethod
    def forward_train(self, *inputs, **kwargs):
        pass

    @abstractmethod
    @torch.no_grad()
    def forward_test(self, *inputs, **kwargs):
        pass

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
        return dict_to_yaml('networkname',
                            __class__.__name__,
                            TrainModule.para_dict,
                            set_name=True)
