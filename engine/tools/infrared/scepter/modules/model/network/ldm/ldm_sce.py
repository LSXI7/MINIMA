# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy

import torch
import torch.nn as nn
import torchvision.transforms as TT
from tools.infrared.scepter.modules.annotator.registry import ANNOTATORS
from tools.infrared.scepter.modules.model.registry import MODELS, TUNERS
from tools.infrared.scepter.modules.utils.config import Config, dict_to_yaml

from .ldm import LatentDiffusion
from .ldm_xl import LatentDiffusionXL


@MODELS.register_class()
class LatentDiffusionSCETuning(LatentDiffusion):
    para_dict = {}
    para_dict.update(LatentDiffusion.para_dict)

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger=logger)

    def init_params(self):
        super().init_params()
        self.tuner_model_config = self.cfg.TUNER_MODEL

    def construct_network(self):
        super().construct_network()
        input_block_channels = self.model._input_block_chans
        sc_tuner_cfg = self.tuner_model_config['SC_TUNER_CFG']
        use_layers = self.tuner_model_config.get('USE_LAYERS', None)
        lsc_tuner_blocks = nn.ModuleList([])
        for i, chan in enumerate(input_block_channels[::-1]):
            if use_layers and i not in use_layers:
                lsc_tuner_blocks.append(nn.Identity())
                continue
            tuner_cfg = copy.deepcopy(sc_tuner_cfg)
            tuner_cfg['DIM'] = chan
            tuner_cfg['TUNER_LENGTH'] = int(chan *
                                            tuner_cfg.get('DOWN_RATIO', 1.0))
            sc_tuner = TUNERS.build(tuner_cfg, logger=self.logger)
            lsc_tuner_blocks.append(sc_tuner)
        self.model.lsc_identity = lsc_tuner_blocks

    def save_pretrained(self,
                        *args,
                        destination=None,
                        prefix='',
                        keep_vars=False):
        save_state = {
            key: value
            for key, value in self.state_dict().items()
            if 'lsc_identity' in key
        }
        return save_state

    def save_pretrained_config(self):
        return copy.deepcopy(self.cfg.TUNER_MODEL.cfg_dict)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            LatentDiffusionSCETuning.para_dict,
                            set_name=True)


@MODELS.register_class()
class LatentDiffusionXLSCETuning(LatentDiffusionSCETuning, LatentDiffusionXL):
    pass


@MODELS.register_class()
class LatentDiffusionSCEControl(LatentDiffusion):
    para_dict = {
        'CONTROL_MODEL': {},
    }
    para_dict.update(LatentDiffusion.para_dict)

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger=logger)

    def init_params(self):
        super().init_params()
        self.control_model_config = self.cfg.CONTROL_MODEL
        self.control_anno_config = self.cfg.CONTROL_ANNO

    def construct_network(self):
        super().construct_network()
        # anno
        self.control_processor = ANNOTATORS.build(self.control_anno_config)
        if isinstance(self.control_model_config, (dict, Config)):
            self.control_model_config = [self.control_model_config]
        control_model = nn.ModuleList([])
        for k, sub_cfg in enumerate(self.control_model_config):
            sub_cfg['INPUT_BLOCK_CHANS'] = self.model._input_block_chans
            sub_cfg['INPUT_DOWN_FLAG'] = self.model._input_down_flag
            csc_tuners = TUNERS.build(sub_cfg, logger=self.logger)
            control_model.append(csc_tuners)
        self.model.control_blocks = control_model

    @torch.no_grad()
    def get_control_input(self, control, *args, **kwargs):
        hints = []
        for ctr in control:
            hint = self.control_processor(ctr)
            hint = TT.ToTensor()(hint)
            hints.append(hint)
        hints = torch.stack(hints).to(control.device)
        return hints

    def forward_train(self, **kwargs):
        # if ('module' not in kwargs) or ('module' in kwargs and self.control_method not in kwargs['module']):
        #     kwargs['module'] = {self.control_method: self.control_model}
        if 'hint' not in kwargs and 'image_preprocess' in kwargs:
            image_preprocess = kwargs.pop('image_preprocess')
            kwargs['hint'] = self.get_control_input(image_preprocess)
        return super().forward_train(**kwargs)

    @torch.no_grad()
    @torch.autocast('cuda', dtype=torch.float16)
    def forward_test(self, **kwargs):
        # if ('module' not in kwargs) or ('module' in kwargs and self.control_method not in kwargs['module']):
        #     kwargs['module'] = {self.control_method: self.control_model}
        if 'hint' not in kwargs and 'image_preprocess' in kwargs:
            image_preprocess = kwargs.pop('image_preprocess')
            kwargs['hint'] = self.get_control_input(image_preprocess)
            kwargs['image_size'] = kwargs['hint'].shape[-2:]
        return super().forward_test(**kwargs)

    def save_pretrained(self,
                        *args,
                        destination=None,
                        prefix='',
                        keep_vars=False):
        return self.model.control_blocks.state_dict(*args,
                                                    destination=destination,
                                                    keep_vars=keep_vars)

    def save_pretrained_config(self):
        return copy.deepcopy(self.cfg.CONTROL_MODEL.cfg_dict)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            LatentDiffusionSCEControl.para_dict,
                            set_name=True)


@MODELS.register_class()
class LatentDiffusionXLSCEControl(LatentDiffusionSCEControl,
                                  LatentDiffusionXL):
    pass
