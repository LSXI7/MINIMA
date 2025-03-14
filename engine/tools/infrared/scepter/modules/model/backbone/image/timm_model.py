# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import BACKBONES
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@BACKBONES.register_class()
class TIMM_MODEL(BaseModel):
    para_dict = {
        'MODEL_NAME': {
            'value': '',
            'description': 'The name of timm!'
        },
        'NUM_CLASSES': {
            'value': 1000,
            'description': 'The num class for your task!'
        },
        'PRETRAINED': {
            'value': True,
            'description': 'Use the pretrained model or not!'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        import timm
        num_classes = cfg.NUM_CLASSES
        model_name = cfg.MODEL_NAME
        pretrained = cfg.get('PRETRAINED', False)
        self.visual = timm.create_model(model_name,
                                        pretrained=pretrained,
                                        num_classes=num_classes)

    def forward(self, x):
        out = self.visual.forward(x)
        return out

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
                            TIMM_MODEL.para_dict,
                            set_name=True)
