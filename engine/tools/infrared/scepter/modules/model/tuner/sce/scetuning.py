# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

import torch
import torch.nn as nn
from tools.infrared.scepter.modules.model.registry import TUNERS
from tools.infrared.scepter.modules.model.tuner.base_tuner import BaseTuner
from tools.infrared.scepter.modules.model.tuner.tuner_component import conv_nd, zero_module
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.file_system import FS


@TUNERS.register_class()
class SCTuner(BaseTuner):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.logger = logger
        dim = cfg['DIM']
        tuner_length = cfg['TUNER_LENGTH']
        tuner_name = cfg.get('TUNER_NAME', 'SCEAdapter')
        self.tuner_name = tuner_name
        if tuner_name == 'SCEAdapter':
            from .scetuning_component import SCEAdapter
            self.tuner_op = SCEAdapter(dim=dim, adapter_length=tuner_length)
        else:
            raise Exception(f'Error tuner op {tuner_name}')

    def forward(self, x, x_shortcut=None, use_shortcut=True, **kwargs):
        if self.tuner_name == 'SCEAdapter':
            out = self.tuner_op(x, x_shortcut, use_shortcut)
        else:
            out = x
        return out

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TUNERS',
                            __class__.__name__,
                            SCTuner.para_dict,
                            set_name=True)


@TUNERS.register_class()
class CSCTuners(BaseTuner):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.logger = logger
        input_block_channels = cfg['INPUT_BLOCK_CHANS']
        input_down_flag = cfg['INPUT_DOWN_FLAG']
        assert len(input_block_channels) == len(input_down_flag)
        pre_hint_in_channels = cfg.get('PRE_HINT_IN_CHANNELS', 3)
        pre_hint_out_channels = cfg.get('PRE_HINT_OUT_CHANNELS', 256)
        pre_hint_dim_ratio = cfg.get('PRE_HINT_DIM_RATIO', 1.0)
        dense_hint_kernal = cfg.get('DENSE_HINT_KERNAL', 3)
        sc_tuner_cfg = cfg['SC_TUNER_CFG']
        use_layers = cfg.get('USE_LAYERS', None)
        self.pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        self.scale = cfg.get('SCALE', 1.0)
        self.method = 'csctuning'

        # pre_hint
        dims = 2
        ch = pre_hint_out_channels
        self.pre_hint_blocks = nn.Sequential(
            conv_nd(dims,
                    pre_hint_in_channels,
                    int(16 * pre_hint_dim_ratio),
                    3,
                    padding=1),
            nn.SiLU(),
            conv_nd(dims,
                    int(16 * pre_hint_dim_ratio),
                    int(16 * pre_hint_dim_ratio),
                    3,
                    padding=1),
            nn.SiLU(),
            conv_nd(dims,
                    int(16 * pre_hint_dim_ratio),
                    int(32 * pre_hint_dim_ratio),
                    3,
                    padding=1,
                    stride=2),
            nn.SiLU(),
            conv_nd(dims,
                    int(32 * pre_hint_dim_ratio),
                    int(32 * pre_hint_dim_ratio),
                    3,
                    padding=1),
            nn.SiLU(),
            conv_nd(dims,
                    int(32 * pre_hint_dim_ratio),
                    int(96 * pre_hint_dim_ratio),
                    3,
                    padding=1,
                    stride=2),
            nn.SiLU(),
            conv_nd(dims,
                    int(96 * pre_hint_dim_ratio),
                    int(96 * pre_hint_dim_ratio),
                    3,
                    padding=1),
            nn.SiLU(),
            conv_nd(dims,
                    int(96 * pre_hint_dim_ratio),
                    ch,
                    3,
                    padding=1,
                    stride=2),
        )
        # dense_hint
        self.dense_hint_blocks = nn.ModuleList([])
        stride_list = [2 if flag else 1 for flag in input_down_flag]
        for i, chan in enumerate(input_block_channels):
            if use_layers and i not in use_layers:
                self.dense_hint_blocks.append(nn.Identity())
                continue
            self.dense_hint_blocks.append(
                nn.Sequential(
                    nn.SiLU(),
                    zero_module(
                        conv_nd(dims,
                                ch,
                                chan,
                                dense_hint_kernal,
                                padding=1,
                                stride=stride_list[i]))
                    if dense_hint_kernal == 3 else zero_module(
                        conv_nd(dims,
                                ch,
                                chan,
                                dense_hint_kernal,
                                padding=0,
                                stride=stride_list[i]))))
            ch = chan
        # tuner
        self.lsc_tuner_blocks = nn.ModuleList([])
        for i, chan in enumerate(input_block_channels[::-1]):
            if use_layers and i not in use_layers:
                self.lsc_tuner_blocks.append(nn.Identity())
                continue
            sc_tuner_cfg['DIM'] = chan
            sc_tuner_cfg['TUNER_LENGTH'] = int(chan *
                                               cfg.get('DOWN_RATIO', 1.0))
            sc_tuner = TUNERS.build(sc_tuner_cfg, logger=self.logger)
            self.lsc_tuner_blocks.append(sc_tuner)

    def load_pretrained_model(self, pretrained_model):
        if self.pretrained_model:
            with FS.get_from(self.pretrained_model,
                             wait_finish=True) as local_path:
                self.init_from_ckpt(local_path)

    def init_from_ckpt(self, path):
        model_new = OrderedDict()
        model = torch.load(path, map_location='cpu')
        for k, v in model.items():
            if k.startswith('model.'):
                k = k[len('model.'):]
            if k.startswith('0.'):
                k = k[len('0.'):]
            model_new[k] = v
        missing, unexpected = self.load_state_dict(model_new, strict=False)
        print(
            f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
        )
        if len(missing) > 0:
            print(f'Missing Keys:\n {missing}')
        if len(unexpected) > 0:
            print(f'\nUnexpected Keys:\n {unexpected}')

    @staticmethod
    def get_config_template():
        return dict_to_yaml('TUNERS',
                            __class__.__name__,
                            CSCTuners.para_dict,
                            set_name=True)
