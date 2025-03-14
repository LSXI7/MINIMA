# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# drop_path function & DropPath class & Attention class
# Modified from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
#
# Copyright 2019, Facebook, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tools.infrared.scepter.modules.model.backbone.video.bricks.visualize_3d_module import \
    Visualize3DModule
from tools.infrared.scepter.modules.model.registry import BRICKS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, ff_dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Self-attention module.
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention.

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?
    """
    def __init__(
        self,
        dim,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        einops_from=None,
        einops_to=None,
        **einops_dims,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.ff_dropout = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v,
                                   v_) = map(lambda t: (t[:, 0:1], t[:, 1:]),
                                             (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_attn = (cls_q @ k.transpose(1, 2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v

        # rearrange across time or space
        q_, k_, v_ = map(
            lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **
                                einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r=r),
                           (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_

        # merge back time or space
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)

        # merge back the head
        x = rearrange(x, '(b h) n d -> b n (h d)', h=h)

        # combine head out
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def extra_repr(self) -> str:
        return f'partial={self.partial}, ' + \
               '' if not self.partial \
            else f'einops_from={self.einops_from}, einops_to={self.einops_to}, einops_dims={self.einops_dims}'


@BRICKS.register_class()
class BaseTransformerLayer(Visualize3DModule):
    para_dict = {
        'DIM': {
            'value': 768,
            'description': "the num of transformer's input dim!"
        },
        'NUM_HEADS': {
            'value': 12,
            'description': "the num of transformer's head!"
        },
        'ATTN_DROPOUT': {
            'value': 0.0,
            'description': 'the attention dropout of transformer!'
        },
        'FF_DROPOUT': {
            'value': 0.0,
            'description': 'the ff dropout of transformer!'
        },
        'MLP_MULT': {
            'value': 4,
            'description': "the mlp's mult!"
        },
        'DROP_PATH_PROB': {
            'value': 0.0,
            'description': 'the drop path prob!'
        }
    }
    para_dict.update(Visualize3DModule.para_dict)

    def __init__(self, cfg, logger=None):
        super(BaseTransformerLayer, self).__init__(cfg, logger=logger)
        dim = cfg.DIM
        num_heads = cfg.get('NUM_HEADS', 12)
        attn_dropout = cfg.get('ATTN_DROPOUT', 0.0)
        ff_dropout = cfg.get('FF_DROPOUT', 0.0)
        mlp_mult = cfg.get('MLP_MULT', 4)
        drop_path_prob = cfg.get('DROP_PATH_PROB', 0.0)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              attn_dropout=attn_dropout,
                              ff_dropout=ff_dropout)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim, mult=mlp_mult, ff_dropout=ff_dropout)
        self.drop_path = DropPath(drop_prob=drop_path_prob
                                  ) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
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
        return dict_to_yaml('BRANCH',
                            __class__.__name__,
                            BaseTransformerLayer.para_dict,
                            set_name=True)


@BRICKS.register_class()
class TimesformerLayer(Visualize3DModule):
    para_dict = {
        'NUM_PATCHES': {
            'value': 49,
            'description': "the num of transformer's input patches!"
        },
        'NUM_FRAMES': {
            'value': 30,
            'description': "the num of transformer's input frames!"
        },
        'DIM': {
            'value': 768,
            'description': "the num of transformer's input dim!"
        },
        'NUM_HEADS': {
            'value': 12,
            'description': "the num of transformer's head!"
        },
        'ATTN_DROPOUT': {
            'value': 0.0,
            'description': 'the attention dropout of transformer!'
        },
        'FF_DROPOUT': {
            'value': 0.0,
            'description': 'the ff dropout of transformer!'
        },
        'DROP_PATH_PROB': {
            'value': 0.0,
            'description': 'the drop path prob!'
        }
    }
    para_dict.update(Visualize3DModule.para_dict)

    def __init__(self, cfg, logger=None):
        super(TimesformerLayer, self).__init__(cfg, logger=logger)

        num_patches = cfg.get('NUM_PATCHES', 49)
        num_frames = cfg.get('NUM_FRAMES', 30)
        dim = cfg.DIM
        num_heads = cfg.get('NUM_HEADS', 12)
        attn_dropout = cfg.get('ATTN_DROPOUT', 0.0)
        ff_dropout = cfg.get('FF_DROPOUT', 0.0)
        drop_path_prob = cfg.get('DROP_PATH_PROB', 0.0)

        self.norm_temporal = nn.LayerNorm(dim, eps=1e-6)
        self.attn_temporal = Attention(dim,
                                       num_heads=num_heads,
                                       attn_dropout=attn_dropout,
                                       ff_dropout=ff_dropout,
                                       einops_from='b (f n) d',
                                       einops_to='(b n) f d',
                                       n=num_patches)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              attn_dropout=attn_dropout,
                              ff_dropout=ff_dropout,
                              einops_from='b (f n) d',
                              einops_to='(b f) n d',
                              f=num_frames)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, ff_dropout=ff_dropout)

        self.drop_path = DropPath(
            drop_path_prob) if drop_path_prob > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn_temporal(self.norm_temporal(x)))
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
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
        return dict_to_yaml('BRANCH',
                            __class__.__name__,
                            TimesformerLayer.para_dict,
                            set_name=True)
