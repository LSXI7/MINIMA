# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from tools.infrared.scepter.modules.model.registry import TUNERS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml

from .base_tuner import BaseTuner


@TUNERS.register_class()
class SwiftFull(BaseTuner):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        self.logger = logger

    def __call__(self, *args, **kwargs):
        return None

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TUNERS',
                            __class__.__name__,
                            SwiftFull.para_dict,
                            set_name=True)


@TUNERS.register_class()
class SwiftLoRA():
    para_dict = {
        'R': {
            'value': 64,
            'description': 'Rank of lora.'
        },
        'LORA_ALPHA': {
            'value': 64,
            'description': 'Lora alpha of lora, lora_alpha/rank=weight.'
        },
        'LORA_DROPOUT': {
            'value': 0.0,
            'description': 'Lora dropout, default is 0.0.'
        },
        'BIAS': {
            'value': None,
            'description': "Linear's bias for lora."
        },
        'TARGET_MODULES': {
            'value': '',
            'description': 'The norm expression of target modules.'
        }
    }

    def __init__(self, cfg, logger=None):
        from swift import LoRAConfig
        self.logger = logger
        self.init_config = LoRAConfig(r=cfg.R,
                                      lora_alpha=cfg.LORA_ALPHA,
                                      lora_dropout=cfg.LORA_DROPOUT,
                                      bias=cfg.BIAS,
                                      target_modules=cfg.TARGET_MODULES)

    def __call__(self, *args, **kwargs):
        return self.init_config

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TUNERS',
                            __class__.__name__,
                            SwiftLoRA.para_dict,
                            set_name=True)


@TUNERS.register_class()
class SwiftAdapter(BaseTuner):
    para_dict = {
        'DIMS': {
            'value': [],
            'description': 'DIMS.'
        },
        'TARGET_MODULES': {
            'value': '',
            'description': 'The norm expression of target modules.'
        },
        'ADAPTER_LENGTH': {
            'value': '',
            'description': 'The length of adapter.'
        }
    }

    def __init__(self, cfg, logger=None):
        from swift import AdapterConfig
        self.logger = logger
        self.init_config = AdapterConfig(dim=cfg.DIMS,
                                         hidden_pos=0,
                                         target_modules=cfg.TARGET_MODULES,
                                         adapter_length=cfg.ADAPTER_LENGTH)

    def __call__(self, *args, **kwargs):
        return self.init_config

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TUNERS',
                            __class__.__name__,
                            SwiftAdapter.para_dict,
                            set_name=True)


@TUNERS.register_class()
class SwiftSCETuning(BaseTuner):
    para_dict = {
        'DIMS': {
            'value': [],
            'description': 'DIMS.'
        },
        'TARGET_MODULES': {
            'value': '',
            'description': 'The norm expression of target modules.'
        },
        'DOWN_RATIO': {
            'value': 1.0,
            'description': 'The dim down ratio of tuner hidden state.'
        },
        'TUNER_MODE': {
            'value': 'identity',
            'description': 'Location of tuner operation'
        }
    }

    def __init__(self, cfg, logger=None):
        from swift import SCETuningConfig
        self.logger = logger
        self.init_config = SCETuningConfig(dims=cfg.DIMS,
                                           target_modules=cfg.TARGET_MODULES,
                                           down_ratio=cfg.DOWN_RATIO,
                                           tuner_mode=cfg.TUNER_MODE)

    def __call__(self, *args, **kwargs):
        return self.init_config

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TUNERS',
                            __class__.__name__,
                            SwiftSCETuning.para_dict,
                            set_name=True)
