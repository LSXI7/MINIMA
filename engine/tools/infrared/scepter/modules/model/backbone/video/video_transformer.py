# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
'''
The implementations of vivit as https://arxiv.org/abs/2103.15691.
The following setting alined the proposed model in the paper above.
Model1: Spatio-temporal attention.
    class: VideoTransformer
    stem: PatchEmbedStem/TubeletEmbeddingStem
    branch: BaseTransformerLayer
    complexity: (n_t * n_h * n_w) ** 2
Model2: Factorised encoder
    class: FactorizedVideoTransformer
    stem: PatchEmbedStem/TubeletEmbeddingStem
    branch: BaseTransformerLayer [drop_path=0.1]
    complexity: (n_h * n_w) ** 2 + n_t ** 2 [attn_dropout = 0.0 ff_dropout=0.0]
Model3: Factorised self-attention
    class: VideoTransformer
    stem: TubeletEmbeddingStem
    branch: TimesformerLayer
    complexity: (n_h * n_w) ** 2 + O(attn_temp)
Model4: Factorised dot-product attention
    coming soon...
TimesFormer:
    class: VideoTransformer
    stem: PatchEmbedStem [drop_path=0.0]
    branch: TimesformerLayer [attn_dropout = 0.1 ff_dropout=0.1]
    complexity: (n_h * n_w) ** 2 + O(attn_temp)
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional
from einops import rearrange
from tools.infrared.scepter.modules.model.backbone.video.init_helper import (
    _init_transformer_weights, trunc_normal_)
from tools.infrared.scepter.modules.model.registry import BACKBONES, BRICKS, STEMS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@BACKBONES.register_class()
class VideoTransformer(nn.Module):
    para_dict = {
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': "the input frames's channel!"
        },
        'NUM_FRAMES': {
            'value': 30,
            'description': "the num of transformer's input frames!"
        },
        'IMAGE_SIZE': {
            'value': 224,
            'description': "the input frame's size!"
        },
        'DIM': {
            'value': 768,
            'description': 'the patch embedding size!'
        },
        'PATCH_SIZE': {
            'value': 16,
            'description': 'the patch size!'
        },
        'DEPTH': {
            'value': 12,
            'description': 'the transformer network depth!'
        },
        'STEM': {
            'NAME': {
                'value':
                'PatchEmbedStem',
                'description':
                'also use TubeletEmbeddingStem, use the shared parameters as IMAGE_SIZE, PATCH_SIZE, NUM_FRAMES,'
                'NUM_INPUT_CHANNELS, DIM'
            }
        },
        'BRANCH': {
            'NAME': {
                'value':
                'BaseTransformerLayer',
                'description':
                'also use TimesformerLayer, use the shared parameters as NUM_PATCHES, NUM_FRAMES, DIM'
            }
        }
    }

    def __init__(self, cfg, logger=None):
        super(VideoTransformer, self).__init__()
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        num_frames = cfg.get('NUM_FRAMES', 8)
        image_size = cfg.get('IMAGE_SIZE', 224)
        num_features = cfg.get('DIM', 768)
        patch_size = cfg.get('PATCH_SIZE', 16)
        depth = cfg.get('DEPTH', 12)
        drop_path = cfg.get('DROP_PATH', 0.1)
        stem = cfg.STEM
        branch = cfg.BRANCH

        assert image_size % patch_size == 0, 'Image dimensions must be divided by patch size.'

        self.num_patches_per_frame = (image_size // patch_size)**2
        if stem.NAME == 'TubeletEmbeddingStem':
            self.num_patches = num_frames * self.num_patches_per_frame // stem.TUBELET_SIZE
        else:
            self.num_patches = num_frames * self.num_patches_per_frame
        assert stem.NAME in ('PatchEmbedStem', 'TubeletEmbeddingStem')
        stem.IMAGE_SIZE = image_size
        stem.PATCH_SIZE = patch_size
        stem.NUM_FRAMES = num_frames
        stem.NUM_INPUT_CHANNELS = num_input_channels
        stem.DIM = num_features

        self.stem = STEMS.build(stem, logger=logger)

        self.pos_embd = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))

        assert branch.NAME in ('BaseTransformerLayer', 'TimesformerLayer')
        if branch.NAME == 'TimesformerLayer':
            branch.NUM_PATCHES = image_size // patch_size**2
        branch.NUM_FRAMES = num_frames
        branch.DIM = num_features

        # construct spatial transformer layers
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)
               ]  # stochastic depth decay rule

        layers = []
        for i in range(depth):
            branch.DROP_PATH_PROB = dpr[i]
            layers.append(BRICKS.build(branch, logger=logger))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, video):
        x = video
        x = self.stem(x)

        cls_token = self.cls_token.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token, x), dim=1)

        x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)

        return x[:, 0]

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
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            VideoTransformer.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class FactorizedVideoTransformer(nn.Module):
    para_dict = {
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': "the input frames's channel!"
        },
        'NUM_FRAMES': {
            'value': 30,
            'description': "the num of transformer's input frames!"
        },
        'IMAGE_SIZE': {
            'value': 224,
            'description': "the input frame's size!"
        },
        'DIM': {
            'value': 768,
            'description': 'the patch embedding size!'
        },
        'PATCH_SIZE': {
            'value': 16,
            'description': 'the patch size!'
        },
        'TUBELET_SIZE': {
            'value': 2,
            'description': 'the tubelet size, also means the temporal stride!'
        },
        'DEPTH': {
            'value': 12,
            'description': 'the transformer network depth!'
        },
        'DEPTH_TEMPORAL': {
            'value': 4,
            'description': 'the temporal network depth!'
        },
        'DROP_PATH': {
            'value': 0.1,
            'description': 'the drop path module value!'
        },
        'STEM': {
            'NAME': {
                'value':
                'PatchEmbedStem',
                'description':
                'use the shared parameters as IMAGE_SIZE, PATCH_SIZE, NUM_FRAMES,'
                'NUM_INPUT_CHANNELS, DIM'
            }
        },
        'BRANCH': {
            'NAME': {
                'value':
                'BaseTransformerLayer',
                'description':
                'use the shared parameters as NUM_PATCHES, NUM_FRAMES, DIM, '
                'TUBELET_SIZE (if use TubeletEmbeddingStem)'
            }
        }
    }

    def __init__(self, cfg, logger=None):
        super(FactorizedVideoTransformer, self).__init__()
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        num_frames = cfg.get('NUM_FRAMES', 8)
        image_size = cfg.get('IMAGE_SIZE', 224)
        num_features = cfg.get('DIM', 768)
        self.patch_size = cfg.get('PATCH_SIZE', 16)
        tubelet_size = cfg.get('TUBELET_SIZE', 2)
        depth = cfg.get('DEPTH', 12)
        depth_temp = cfg.get('DEPTH_TEMPORAL', 4)
        drop_path = cfg.get('DROP_PATH', 0.1)
        stem = cfg.STEM
        branch = cfg.BRANCH

        assert image_size % self.patch_size == 0, 'Image dimensions must be divided by patch size.'

        self.num_patches_per_frame = (image_size // self.patch_size)**2
        self.num_patches = num_frames * self.num_patches_per_frame // tubelet_size

        assert stem.NAME in ('PatchEmbedStem', 'TubeletEmbeddingStem')
        stem.IMAGE_SIZE = image_size
        stem.PATCH_SIZE = self.patch_size
        stem.NUM_FRAMES = num_frames
        stem.NUM_INPUT_CHANNELS = num_input_channels
        stem.DIM = num_features
        if stem.NAME == 'TubeletEmbeddingStem':
            stem.TUBELET_SIZE = tubelet_size

        self.stem = STEMS.build(stem, logger=logger)

        self.pos_embd = nn.Parameter(
            torch.zeros(1, self.num_patches_per_frame + 1, num_features))
        self.temp_embd = nn.Parameter(
            torch.zeros(1, num_frames // tubelet_size + 1, num_features))
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))
        self.cls_token_out = nn.Parameter(torch.randn(1, 1, num_features))

        assert branch.NAME in ('BaseTransformerLayer', 'TimesformerLayer')
        if branch.NAME == 'TimesformerLayer':
            branch.NUM_PATCHES = image_size // self.patch_size**2
        branch.NUM_FRAMES = num_frames
        branch.DIM = num_features

        # construct spatial transformer layers
        dpr = [
            x.item() for x in torch.linspace(0, drop_path, depth + depth_temp)
        ]  # stochastic depth decay rule

        layers = []
        for i in range(depth):
            branch.DROP_PATH_PROB = dpr[i]
            layers.append(BRICKS.build(branch, logger=logger))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(num_features, eps=1e-6)

        # construct temporal transformer layers
        layers_temporal = []
        for i in range(depth_temp):
            branch.DROP_PATH_PROB = dpr[i + depth]
            layers_temporal.append(BRICKS.build(branch, logger=logger))
        self.layers_temporal = nn.Sequential(*layers_temporal)

        self.norm_out = nn.LayerNorm(num_features, eps=1e-6)

        # initialization
        trunc_normal_(self.pos_embd, std=.02)
        trunc_normal_(self.temp_embd, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_token_out, std=.02)
        self.apply(_init_transformer_weights)

    def forward(self, video):
        x = video
        h, w = x.shape[-2:]
        actual_num_patches_per_frame = (h // self.patch_size) * (
            w // self.patch_size)
        x = self.stem(x)

        if actual_num_patches_per_frame != self.num_patches_per_frame:
            assert not self.training
            x = rearrange(x,
                          'b (t n) c -> (b t) n c',
                          n=actual_num_patches_per_frame)
        else:
            x = rearrange(x,
                          'b (t n) c -> (b t) n c',
                          n=self.num_patches_per_frame)

        cls_token = self.cls_token.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token, x), dim=1)

        # to make the input video size changable
        if actual_num_patches_per_frame != self.num_patches_per_frame:
            actual_num_pathces_per_side = int(
                math.sqrt(actual_num_patches_per_frame))
            if not hasattr(self,
                           'new_pos_embd') or self.new_pos_embd.shape[1] != (
                               actual_num_pathces_per_side**2 + 1):
                cls_pos_embd = self.pos_embd[:, 0, :].unsqueeze(1)
                pos_embd = self.pos_embd[:, 1:, :]
                num_patches_per_side = int(
                    math.sqrt(self.num_patches_per_frame))
                pos_embd = pos_embd.reshape(1, num_patches_per_side,
                                            num_patches_per_side,
                                            -1).permute(0, 3, 1, 2)
                pos_embd = torch.nn.functional.interpolate(
                    pos_embd,
                    size=(actual_num_pathces_per_side,
                          actual_num_pathces_per_side),
                    mode='bilinear').permute(0, 2, 3, 1).reshape(
                        1, actual_num_pathces_per_side**2, -1)
                self.new_pos_embd = torch.cat((cls_pos_embd, pos_embd), dim=1)
            x += self.new_pos_embd
        else:
            x += self.pos_embd

        x = self.layers(x)
        x = self.norm(x)[:, 0]

        x = rearrange(x,
                      '(b t) c -> b t c',
                      t=self.num_patches // self.num_patches_per_frame)

        cls_token_out = self.cls_token_out.repeat((x.shape[0], 1, 1))
        x = torch.cat((cls_token_out, x), dim=1)

        x += self.temp_embd
        x = self.layers_temporal(x)
        x = self.norm_out(x)

        return x[:, 0]

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
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            FactorizedVideoTransformer.para_dict,
                            set_name=True)
