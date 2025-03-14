# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
from einops import repeat
from tools.infrared.scepter.modules.model.backbone.autoencoder.ae_utils import (
    XFORMERS_IS_AVAILBLE, AttnBlock, Downsample, MemoryEfficientAttention,
    Normalize, ResnetBlock, Upsample, nonlinearity)
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import BACKBONES
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from torch.utils.checkpoint import checkpoint


@BACKBONES.register_class()
class Encoder(BaseModel):
    para_dict = {
        'CH': {
            'value': 128,
            'description': ''
        },
        'NUM_RES_BLOCKS': {
            'value': 2,
            'description': ''
        },
        'IN_CHANNELS': {
            'value': 3,
            'description': ''
        },
        'ATTN_RESOLUTIONS': {
            'value': [],
            'description': ''
        },
        'CH_MULT': {
            'value': [1, 2, 4, 4],
            'description': ''
        },
        'Z_CHANNELS': {
            'value': 4,
            'description': ''
        },
        'DOUBLE_Z': {
            'value': True,
            'description': ''
        },
        'DROPOUT': {
            'value': 0.0,
            'description': ''
        },
        'RESAMP_WITH_CONV': {
            'value': True,
            'description': ''
        }
    }
    para_dict.update(BaseModel.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.use_checkpoint = cfg.get('USE_CHECKPOINT', False)
        self.ch = cfg.CH
        self.out_ch = cfg.OUT_CH
        self.num_res_blocks = cfg.NUM_RES_BLOCKS
        self.in_channels = cfg.IN_CHANNELS
        self.attn_resolutions = cfg.ATTN_RESOLUTIONS
        self.ch_mult = tuple(cfg.get('CH_MULT', [1, 2, 4, 8]))
        self.z_channels = cfg.Z_CHANNELS
        self.double_z = cfg.get('DOUBLE_Z', True)
        self.dropout = cfg.get('DROPOUT', 0.0)
        self.resamp_with_conv = cfg.get('RESAMP_WITH_CONV', True)
        self.temb_ch = 0
        self.construct_model()

    def construct_model(self):
        self.num_resolutions = len(self.ch_mult)
        self.logger.info(
            f'AE Module XFORMERS_IS_AVAILBLE : {XFORMERS_IS_AVAILBLE}')
        AttentionBuilder = MemoryEfficientAttention if XFORMERS_IS_AVAILBLE else AttnBlock

        self.conv_in = torch.nn.Conv2d(self.in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = 2**(self.num_resolutions - 1)
        in_ch_mult = (1, ) + tuple(self.ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttentionBuilder(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, self.resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)
        self.mid.attn_1 = AttentionBuilder(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * self.z_channels if self.double_z else self.z_channels,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward_ori(self, x):
        # timestep embedding
        temb = None
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self.forward_ori, x)
        else:
            return self.forward_ori(x)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            Encoder.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class Decoder(BaseModel):
    para_dict = {
        'CH': {
            'value': 128,
            'description': ''
        },
        'OUT_CH': {
            'value': 3,
            'description': ''
        },
        'NUM_RES_BLOCKS': {
            'value': 2,
            'description': ''
        },
        'IN_CHANNELS': {
            'value': 3,
            'description': ''
        },
        'ATTN_RESOLUTIONS': {
            'value': [],
            'description': ''
        },
        'CH_MULT': {
            'value': [1, 2, 4, 4],
            'description': ''
        },
        'Z_CHANNELS': {
            'value': 4,
            'description': ''
        },
        'DROPOUT': {
            'value': 0.0,
            'description': ''
        },
        'RESAMP_WITH_CONV': {
            'value': True,
            'description': ''
        },
        'GIVE_PRE_END': {
            'value': False,
            'description': ''
        },
        'TANH_OUT': {
            'value': False,
            'description': ''
        }
    }
    para_dict.update(BaseModel.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.use_checkpoint = cfg.get('USE_CHECKPOINT', False)
        self.ch = cfg.CH
        self.out_ch = cfg.OUT_CH
        self.num_res_blocks = cfg.NUM_RES_BLOCKS
        self.in_channels = cfg.IN_CHANNELS
        self.attn_resolutions = cfg.ATTN_RESOLUTIONS
        self.ch_mult = tuple(cfg.get('CH_MULT', [1, 2, 4, 8]))
        self.z_channels = cfg.Z_CHANNELS
        self.dropout = cfg.get('DROPOUT', 0.0)
        self.resamp_with_conv = cfg.get('RESAMP_WITH_CONV', True)
        self.give_pre_end = cfg.get('GIVE_PRE_END', False)
        self.tanh_out = cfg.get('TANH_OUT', False)
        self.temb_ch = 0
        self.construct_model()

    def construct_model(self):
        self.num_resolutions = len(self.ch_mult)
        AttentionBuilder = MemoryEfficientAttention if XFORMERS_IS_AVAILBLE else AttnBlock

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.ch * self.ch_mult[self.num_resolutions - 1]
        self.block_in = block_in
        curr_res = 1
        # z to block_in
        self.conv_in = torch.nn.Conv2d(self.z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)
        self.mid.attn_1 = AttentionBuilder(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch * self.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttentionBuilder(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, self.resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        self.out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def mid_upsclae_transform(self, h, temb):
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        return h

    def forward(self, z, cond=None):
        # timestep embedding
        temb = None

        h = self.conv_in(z)

        # middle
        if not self.use_checkpoint:
            h = self.mid_upsclae_transform(h, temb)
        else:
            h = checkpoint(self.mid_upsclae_transform, h, temb)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            Decoder.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class RDecoder(Decoder):
    def construct_model(self):
        super().construct_model()
        self.resize_level = nn.Sequential(
            nn.Linear(self.block_in, self.block_in),
            nn.SiLU(),
            nn.Linear(self.block_in, self.block_in),
        )

    def forward(self, z, rembed=None):
        # timestep embedding
        temb = None
        h = self.conv_in(z)
        bs, channel, hdim, wdim = h.size()
        if rembed is not None:
            rembed = self.resize_level(rembed)
            rembed = repeat(rembed, 'b e->  b e hd wd', hd=hdim, wd=wdim)
            h = h + rembed

        # middle
        if not self.use_checkpoint:
            h = self.mid_upsclae_transform(h, temb)
        else:
            h = checkpoint(self.mid_upsclae_transform, h, temb)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            Decoder.para_dict,
                            set_name=True)
