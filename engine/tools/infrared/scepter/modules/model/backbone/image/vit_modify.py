# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.image.utils.vit import (
    MULTI_HEAD_VIT_MODEL, VIT, VIT_MODEL, MULTI_HEAD_VIT_MODEL_Split)
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import BACKBONES
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.file_system import FS


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [
                    *[f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                    'in_proj_bias', 'bias_k', 'bias_v'
            ]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ['text_projection', 'proj']:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


@BACKBONES.register_class()
class VisualTransformer(BaseModel):
    '''
      B/16: Input 224 Patch-size 16 Layers 12 Heads 12 WIDTH 768
      B/32: Input 224 Patch-size 32 Layers 12 Heads 12 WIDTH 768
      L/16: Input 224/336 Patch-size 16 Layers 24 Heads 16 WIDTH 1024
      L/14: Input 224/336 Patch-size 14 Layers 24 Heads 16 WIDTH 1024
      L/32: Input 224 Patch-size 32 Layers 24 Heads 16 WIDTH 1024
      H/14: Input ...
        INPUT_RESOLUTION: 224
        PATCH_SIZE: 32
        WIDTH: 768
        OUTPUT_DIM: 512
        LAYERS: 12
        HEADS: 12
    '''
    para_dict = {
        'PRETRAIN_PATH': {
            'value': '',
            'description': 'The file path of pretrained model!'
        },
        'PRETRAINED': {
            'value': True,
            'description': 'Use the pretrained model or not!'
        }
    }
    para_dict.update(VIT.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.pretrain_path = cfg.PRETRAIN_PATH
        self.pretrained = cfg.PRETRAINED
        self.visual = VIT(cfg)
        use_proj = cfg.get('USE_PROJ', True)
        if self.pretrained:
            with FS.get_from(self.pretrain_path,
                             wait_finish=True) as local_file:
                logger.info(f'Loading checkpoint from {self.pretrain_path}')
                visual_pre = torch.load(local_file, map_location='cpu')
                if not use_proj:
                    visual_pre.pop('proj')
                if visual_pre['conv1.weight'].dtype == torch.float16:
                    convert_weights(self.visual)
            self.visual.load_state_dict(visual_pre, strict=True)

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
                            VisualTransformer.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class SomeFTVisualTransformer(BaseModel):
    '''
        INPUT_RESOLUTION: 224
        PATCH_SIZE: 32
        WIDTH: 768
        OUTPUT_DIM: 512
        LAYERS: 12
        HEADS: 12
    '''
    para_dict = {
        'PRETRAIN_PATH': {
            'value': '',
            'description': 'The file path of pretrained model!'
        },
        'PRETRAINED': {
            'value': True,
            'description': 'Use the pretrained model or not!'
        },
        'FROZEN_LAYERS': {
            'value': 6,
            'description': 'The frozen layers number!'
        },
        'FT_LAYERS': {
            'value': 6,
            'description': 'The finetune layers number!'
        }
    }
    para_dict.update(VIT_MODEL.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.pretrain_path = cfg.PRETRAIN_PATH
        self.pretrained = cfg.PRETRAINED
        self.visual = VIT_MODEL(cfg)
        self.frozen_layers = cfg.FROZEN_LAYERS
        self.ft_layers = cfg.FT_LAYERS
        if self.pretrained:
            with FS.get_from(self.pretrain_path,
                             wait_finish=True) as local_file:
                logger.info(f'Loading checkpoint from {self.pretrain_path}')
                visual_pre = torch.load(local_file, map_location='cpu')
            state_dict_update = self.reformat_state_dict(visual_pre)
            self.visual.load_state_dict(state_dict_update, strict=True)

    def reformat_state_dict(self, state_dict):
        state_dict_update = {}
        for k, v in state_dict.items():
            if 'transformer.resblocks.' in k:
                if int(k.split('.')[2]) < self.frozen_layers:
                    state_dict_update[k.replace(
                        'transformer.resblocks',
                        'frozen_transformer.resblocks')] = v
                else:
                    new_k = k.replace('transformer.resblocks',
                                      'ft_transformer.resblocks')
                    k_tups = new_k.split('.')
                    k_tups[2] = str(int(k_tups[2]) - self.frozen_layers)
                    new_k = '.'.join(k_tups)
                    state_dict_update[new_k] = v
            else:
                state_dict_update[k] = v
        return state_dict_update

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
                            SomeFTVisualTransformer.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class MultiHeadSomeFTVisualTransformer(BaseModel):
    '''
        INPUT_RESOLUTION: 224
        PATCH_SIZE: 32
        WIDTH: 768
        OUTPUT_DIM: 512
        LAYERS: 12
        HEADS: 12
    '''
    para_dict = {}
    para_dict.update(MULTI_HEAD_VIT_MODEL.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.visual = MULTI_HEAD_VIT_MODEL(cfg)
        self.multi_head = cfg.MULTI_HEAD
        self.frozen_layers = cfg.FROZEN_LAYERS
        self.ft_layers = cfg.FT_LAYERS

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
                            MultiHeadSomeFTVisualTransformer.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class SomeFTVisualTransformerTwoPart(BaseModel):
    '''
        INPUT_RESOLUTION: 224
        PATCH_SIZE: 32
        WIDTH: 768
        OUTPUT_DIM: 512
        LAYERS: 12
        HEADS: 12
    '''
    para_dict = {}
    para_dict.update(MULTI_HEAD_VIT_MODEL_Split.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.visual = MULTI_HEAD_VIT_MODEL_Split(cfg)
        self.frozen_layers = cfg.FROZEN_LAYERS
        self.ft_layers = cfg.FT_LAYERS

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
                            SomeFTVisualTransformerTwoPart.para_dict,
                            set_name=True)
