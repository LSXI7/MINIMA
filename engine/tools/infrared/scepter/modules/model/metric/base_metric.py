# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


class BaseMetric(object):
    para_dict = [{}]

    def __init__(self, cfg, logger=None):
        self.logger = logger

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

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
        return dict_to_yaml('METRICS',
                            __class__.__name__,
                            BaseMetric.para_dict,
                            set_name=True)
