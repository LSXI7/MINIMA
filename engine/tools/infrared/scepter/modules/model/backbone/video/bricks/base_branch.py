# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABCMeta, abstractmethod

from tools.infrared.scepter.modules.model.backbone.video.bricks.visualize_3d_module import \
    Visualize3DModule
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


class BaseBranch(Visualize3DModule, metaclass=ABCMeta):
    para_dict = {
        'BRANCH_STYLE': {
            'value': 'simple_block',
            'description': 'the branch style, default: simple_block!'
        },
        'CONSTRUCT_BRANCH': {
            'value': True,
            'description': 'construct branch or not!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(BaseBranch, self).__init__(cfg, logger=logger)
        self.branch_style = cfg.get('BRANCH_STYLE', 'simple_block')
        construct_branch = cfg.get('CONSTRUCT_BRANCH', True)
        if construct_branch:
            self._construct_branch()

    def _construct_branch(self):
        if self.branch_style == 'simple_block':
            self._construct_simple_block()
        elif self.branch_style == 'bottleneck':
            self._construct_bottleneck()

    @abstractmethod
    def _construct_simple_block(self):
        return

    @abstractmethod
    def _construct_bottleneck(self):
        return

    @abstractmethod
    def forward(self, x):
        return

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
                            BaseBranch.para_dict,
                            set_name=True)
