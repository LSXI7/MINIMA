# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.video.bricks.visualize_3d_module import \
    Visualize3DModule
from tools.infrared.scepter.modules.model.registry import STEMS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@STEMS.register_class()
class PatchEmbedStem(Visualize3DModule):
    para_dict = {
        'IMAGE_SIZE': {
            'value': 224,
            'description': "the stem's input frame size!"
        },
        'PATCH_SIZE': {
            'value': 16,
            'description': "the stem's input patch size!"
        },
        'NUM_FRAMES': {
            'value': 16,
            'description': "the stem's input frame num!"
        },
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': "the stem's input channels num!"
        },
        'DIM': {
            'value': 768,
            'description': "the stem's input dim!"
        }
    }
    para_dict.update(Visualize3DModule.para_dict)

    def __init__(self, cfg, logger=None):
        super(PatchEmbedStem, self).__init__(cfg, logger=logger)
        image_size = cfg.get('IMAGE_SIZE', 224)
        patch_size = cfg.get('PATCH_SIZE', 16)
        num_frames = cfg.get('NUM_FRAMES', 16)
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        dim = cfg.get('DIM', 768)
        num_patches_per_image = (image_size // patch_size)**2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(in_channels=num_input_channels,
                               out_channels=dim,
                               kernel_size=(1, patch_size, patch_size),
                               stride=(1, patch_size, patch_size),
                               bias=False)

    def forward(self, x):
        h, w, p = x.shape[3], x.shape[4], self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x

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
        return dict_to_yaml('STEM',
                            __class__.__name__,
                            PatchEmbedStem.para_dict,
                            set_name=True)


@STEMS.register_class()
class TubeletEmbeddingStem(Visualize3DModule):
    para_dict = {
        'IMAGE_SIZE': {
            'value': 224,
            'description': "the stem's input frame size!"
        },
        'PATCH_SIZE': {
            'value': 16,
            'description': "the stem's input patch size!"
        },
        'NUM_FRAMES': {
            'value': 16,
            'description': "the stem's input frame num!"
        },
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': "the stem's input channels num!"
        },
        'TUBELET_SIZE': {
            'value': 2,
            'description': "the stem's tubelet size!"
        },
        'DIM': {
            'value': 768,
            'description': "the stem's input dim!"
        }
    }
    para_dict.update(Visualize3DModule.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        image_size = cfg.get('IMAGE_SIZE', 224)
        patch_size = cfg.get('PATCH_SIZE', 16)
        num_frames = cfg.get('NUM_FRAMES', 16)
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        tubelet_size = cfg.get('TUBELET_SIZE', 2)
        dim = cfg.get('DIM', 768)
        num_patches_per_image = (image_size // patch_size)**2
        num_patches = num_patches_per_image * num_frames

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.conv1 = nn.Conv3d(in_channels=num_input_channels,
                               out_channels=dim,
                               kernel_size=(tubelet_size, patch_size,
                                            patch_size),
                               stride=(tubelet_size, patch_size, patch_size),
                               bias=False)

    def forward(self, x):
        h, w, p = x.shape[3], x.shape[4], self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'
        x = self.conv1(x)
        # b, c, t, h, w -> b, c, p (p: num patches)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # b, c, p -> b, p, c
        x = x.permute(0, 2, 1)
        return x

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
        return dict_to_yaml('STEM',
                            __class__.__name__,
                            TubeletEmbeddingStem.para_dict,
                            set_name=True)
