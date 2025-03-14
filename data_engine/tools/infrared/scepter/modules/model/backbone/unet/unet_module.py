# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.unet.unet_utils import (
    BasicTransformerBlock, Downsample, ResBlock, SpatialTransformer,
    SpatialTransformerV2, Timestep, TimestepEmbedSequential,
    TransformerBlockV2, Upsample, conv_nd, linear, normalization,
    timestep_embedding, zero_module)
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import BACKBONES
from tools.infrared.scepter.modules.model.utils.basic_utils import exists
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.distribute import we
from tools.infrared.scepter.modules.utils.file_system import FS


def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass


@BACKBONES.register_class()
class DiffusionUNet(BaseModel):
    para_dict = {
        'IN_CHANNELS': {
            'value':
            4,
            'description':
            "Unet channels for input, considering the input image's channels."
        },
        'OUT_CHANNELS': {
            'value':
            4,
            'description':
            "Unet channels for output, considering the input image's channels."
        },
        'NUM_RES_BLOCKS': {
            'value': 2,
            'description': "The blocks's number of res."
        },
        'MODEL_CHANNELS': {
            'value': 320,
            'description': 'base channel count for the model.'
        },
        'ATTENTION_RESOLUTIONS': {
            'value': [4, 2, 1],
            'description':
            'A collection of downsample rates at which '
            'attention will take place. May be a set, list,'
            ' or tuple. For example, if this contains 4, '
            'then at 4x downsampling, attentio will be used.'
        },
        'DROPOUT': {
            'value': 0,
            'description': 'The dropout rate.'
        },
        'CHANNEL_MULT': {
            'value': [1, 2, 4, 4],
            'description': 'channel multiplier for each level of the UNet.'
        },
        'CONV_RESAMPLE': {
            'value': True,
            'description': 'Use conv to resample when downsample.'
        },
        'DIMS': {
            'value': 2,
            'description': 'The Conv dims which 2 represent Conv2D.'
        },
        'NUM_CLASSES': {
            'value':
            None,
            'description':
            'The class num for class guided setting, also can be set as continuous.'
        },
        'USE_CHECKPOINT': {
            'value': True,
            'description': 'Use gradient checkpointing to reduce memory usage.'
        },
        'USE_FP16': {
            'value': False,
            'description':
            'Set the inference precision whether use FP16 or not.'
        },
        'NUM_HEADS': {
            'value': 8,
            'description':
            'The number of attention head in each attention layer.'
        },
        'NUM_HEADS_CHANNELS': {
            'value':
            -1,
            'description':
            'If specified, ignore num_heads and instead use '
            'a fixed channel width per attention head.'
        },
        'NUM_HEADS_UPSAMPLE': {
            'value':
            -1,
            'description':
            'Works with num_heads to set a different number '
            'of head for upsampling. Deprecated.'
        },
        'USE_SCALE_SHIFT_NORM': {
            'value':
            False,
            'description':
            'The scale and shift for the outnorm of RESBLOCK, '
            'use a FiLM-like conditioning mechanism.'
        },
        'RESBLOCK_UPDOWN': {
            'value':
            False,
            'description':
            'Use residual blocks for up/downsampling, if False use Conv.'
        },
        'USE_NEW_ATTENTION_ORDER': {
            'value':
            True,
            'description':
            'Whether use new attention(qkv before split head or not) or not.'
        },
        'USE_SPATIAL_TRANSFORMER': {
            'value':
            True,
            'description':
            'Custom transformer which support the context, '
            'if context_dim is not None, the parameter must set True'
        },
        'TRANSFORMER_DEPTH': {
            'value':
            1,
            'description':
            "Custom transformer's depth, valid when USE_SPATIAL_TRANSFORMER is True."
        },
        'CONTEXT_DIM': {
            'value':
            768,
            'description':
            'Custom context info, if set, USE_SPATIAL_TRANSFORMER also set True.'
        },
        'N_EMBED': {
            'value':
            None,
            'description':
            'Whether support predict_codebook_ids or not, which is the scale of codebook.'
        },
        'LEGACY': {
            'value':
            False,
            'description':
            'Whether auto-compute dim_heads according to USE_SPATIAL_TRANSFORMER.'
        },
        'DISABLE_SELF_ATTENTIONS': {
            'value':
            None,
            'description':
            'Whether disable the self-attentions on some level, should be a list, [False, True, ...]'
        },
        'NUM_ATTENTION_BLOCKS': {
            'value': None,
            'description':
            'The number of attention blocks for attention layer.'
        },
        'DISABLE_MIDDLE_SELF_ATTN': {
            'value': False,
            'description':
            'Whether disable the self-attentions in middle blocks.'
        },
        'USE_LINEAR_IN_TRANSFORMER': {
            'value':
            False,
            'description':
            "Custom transformer's parameter, valid when USE_SPATIAL_TRANSFORMER is True."
        },
        'ADM_IN_CHANNELS': {
            'value': 2048,
            'description': "Used when num_classes == 'sequential'."
        },
    }

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger=logger)
        self.init_params(cfg)
        self.construct_network()
        self.control_blocks = None

    def init_params(self, cfg):
        self.in_channels = cfg.IN_CHANNELS
        self.model_channels = cfg.MODEL_CHANNELS
        self.out_channels = cfg.OUT_CHANNELS
        self.num_res_blocks = cfg.NUM_RES_BLOCKS
        self.attention_resolutions = cfg.ATTENTION_RESOLUTIONS

        self.num_heads = cfg.get('NUM_HEADS', -1)
        self.num_head_channels = cfg.get('NUM_HEADS_CHANNELS', -1)
        self.context_dim = cfg.CONTEXT_DIM
        self.dropout = cfg.get('DROPOUT', 0)
        self.channel_mult = tuple(cfg.get('CHANNEL_MULT', [1, 2, 4, 4]))
        self.conv_resample = cfg.get('CONV_RESAMPLE', True)
        self.dims = cfg.get('DIMS', 2)
        self.num_classes = cfg.get('NUM_CLASSES', None)
        self.use_checkpoint = cfg.get('USE_CHECKPOINT', False)
        self.use_scale_shift_norm = cfg.get('USE_SCALE_SHIFT_NORM', False)
        self.resblock_updown = cfg.get('RESBLOCK_UPDOWN', False)
        self.use_new_attention_order = cfg.get('USE_NEW_ATTENTION_ORDER', True)
        self.use_spatial_transformer = cfg.get('USE_SPATIAL_TRANSFORMER', True)
        self.transformer_depth = cfg.get('TRANSFORMER_DEPTH', 1)
        self.use_linear_in_transformer = cfg.get('USE_LINEAR_IN_TRANSFORMER',
                                                 False)
        self.disable_self_attentions = cfg.get('DISABLE_SELF_ATTENTIONS', None)
        self.disable_middle_self_attn = cfg.get('DISABLE_MIDDLE_SELF_ATTN',
                                                False)
        self.adm_in_channels = cfg.get('ADM_IN_CHANNELS', None)
        self.pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        self.ignore_keys = cfg.get('IGNORE_KEYS', [])

        assert (self.num_heads > 0 or self.num_head_channels > 0) and \
               (self.num_heads == -1 or self.num_head_channels == -1)

        if isinstance(self.num_res_blocks, int):
            self.num_res_blocks = len(
                self.channel_mult) * [self.num_res_blocks]
        elif len(self.num_res_blocks) != len(self.channel_mult):
            raise ValueError(
                'provide num_res_blocks either as an int (globally constant) or '
                'as a list/tuple (per-level) with the same length as channel_mult'
            )

    def construct_network(self):
        in_channels = self.in_channels
        model_channels = self.model_channels
        out_channels = self.out_channels
        attention_resolutions = self.attention_resolutions
        channel_mult = self.channel_mult
        num_classes = self.num_classes
        num_heads = self.num_heads
        num_head_channels = self.num_head_channels
        dims = self.dims
        dropout = self.dropout
        use_checkpoint = self.use_checkpoint
        use_scale_shift_norm = self.use_scale_shift_norm
        disable_self_attentions = self.disable_self_attentions
        disable_middle_self_attn = self.disable_middle_self_attn
        transformer_depth = self.transformer_depth
        context_dim = self.context_dim
        use_linear_in_transformer = self.use_linear_in_transformer
        resblock_updown = self.resblock_updown
        conv_resample = self.conv_resample
        adm_in_channels = self.adm_in_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == 'continuous':
                print('setting up linear c_adm embedding layer')
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == 'sequential':
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ))
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        input_down_flag = [False]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False

                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                input_down_flag.append(False)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                input_down_flag.append(True)
                ds *= 2
                self._feature_size += ch
        self._input_block_chans = copy.deepcopy(input_block_chans)
        self._input_down_flag = input_down_flag

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(ch,
                               num_heads,
                               dim_head,
                               depth=transformer_depth,
                               context_dim=context_dim,
                               disable_self_attn=disable_middle_self_attn,
                               use_linear=use_linear_in_transformer,
                               use_checkpoint=use_checkpoint),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self._middle_block_chans = [ch]

        self._output_block_chans = []
        self.output_blocks = nn.ModuleList([])
        self.lsc_identity = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.lsc_identity.append(nn.Identity())
                self._feature_size += ch
                self._output_block_chans.append(ch)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model is not None:
            with FS.get_from(pretrained_model,
                             wait_finish=True) as local_model:
                self.init_from_ckpt(local_model, ignore_keys=self.ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        if path.endswith('safetensors'):
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(path)
        else:
            sd = torch.load(path, map_location='cpu')

        new_sd = OrderedDict()
        for k, v in sd.items():
            ignored = False
            for ik in ignore_keys:
                if ik in k:
                    if we.rank == 0:
                        self.logger.info(
                            'Ignore key {} from state_dict.'.format(k))
                    ignored = True
                    break
            if not ignored:
                new_sd[k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if we.rank == 0:
            self.logger.info(
                f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def _forward_origin(self, x, emb, context, hint=None, **kwargs):
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for m_id, module in enumerate(self.output_blocks):
            skip_h = hs.pop()
            if 'tuner_scale' in kwargs and kwargs[
                    'tuner_scale'] is not None and kwargs['tuner_scale'] < 1.0:
                tuner_scale = kwargs['tuner_scale']
                tuner_h = self.lsc_identity[m_id](skip_h) - skip_h
                h = torch.cat([h, skip_h + tuner_scale * tuner_h], dim=1)
            else:
                h = torch.cat([h, self.lsc_identity[m_id](skip_h)], dim=1)
            target_size = hs[-1].shape[-2:] if len(hs) > 0 else None
            h = module(h, emb, context, target_size)
        out = self.out(h)
        return out

    def _forward_control(self, x, emb, context, hint, **kwargs):
        control_scale = kwargs.pop('control_scale', 1.0)
        multi_csc_tuners = self.control_blocks
        # hints
        multi_hint_hs = []
        for sc_id, csc_tuners in enumerate(multi_csc_tuners):
            hint_input = hint[sc_id] if isinstance(hint, list) else hint
            hint_h = csc_tuners.pre_hint_blocks(hint_input)
            hint_hs = []
            for dsh_blk in csc_tuners.dense_hint_blocks:
                hint_h = dsh_blk(hint_h)
                hint_hs.append(hint_h)
            multi_hint_hs.append(hint_hs)
        # unet
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for m_id, module in enumerate(self.output_blocks):
            skip_h = hs.pop()
            multi_control_h = 0
            for sc_id, csc_tuners in enumerate(multi_csc_tuners):
                hint_h = multi_hint_hs[sc_id][::-1][m_id]
                control_h = csc_tuners.lsc_tuner_blocks[m_id](
                    skip_h + hint_h, x_shortcut=hint_h)
                multi_control_h += csc_tuners.scale * control_h
            tuner_h = self.lsc_identity[m_id](skip_h) - skip_h
            if torch.all(
                    torch.isclose(tuner_h,
                                  torch.zeros_like(tuner_h),
                                  atol=1e-6)):
                # csc-tuner
                skip_h_new = skip_h + control_scale * multi_control_h
            else:
                # csc-tuner + sc-tuner
                tuner_scale = kwargs['tuner_scale']
                skip_h_new = skip_h + control_scale * multi_control_h + tuner_scale * tuner_h
            h = torch.cat([h, skip_h_new], dim=1)
            target_size = hs[-1].shape[-2:] if len(hs) > 0 else None
            h = module(h, emb, context, target_size)
        out = self.out(h)
        return out

    def forward(self, x, t=None, cond=dict(), **kwargs):
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if isinstance(cond, dict):
            if 'y' in cond and cond['y'] is not None:
                assert self.num_classes is not None
                emb = emb + self.label_emb(cond['y'])
            if 'concat' in cond:
                c = cond['concat']
                x = torch.cat([x, c], dim=1)
            if 'hint' in cond:
                hint = cond['hint']
            elif 'hint' in kwargs:
                hint = kwargs.pop('hint', None)
            else:
                hint = None
            context = cond.get('crossattn', None)
        else:
            context = cond
            hint = kwargs.pop('hint', None)

        if self.control_blocks is not None and hint is not None:
            out = self._forward_control(x, emb, context, hint, **kwargs)
        else:
            out = self._forward_origin(x, emb, context, **kwargs)
        return out

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            DiffusionUNet.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class DiffusionUNetXL(DiffusionUNet):
    para_dict = {
        'TRANSFORMER_DEPTH_MIDDLE': {
            'value':
            None,
            'description':
            "Custom transformer's depth of middle block, If set None, use TRANSFORMER_DEPTH last value."
        },
    }
    para_dict.update(DiffusionUNet.para_dict)

    def init_params(self, cfg):
        super().init_params(cfg)

        if isinstance(self.transformer_depth, int):
            self.transformer_depth = len(
                self.channel_mult) * [self.transformer_depth]
        elif isinstance(self.transformer_depth, list):
            assert len(self.transformer_depth) == len(self.channel_mult)

        self.transformer_depth_middle = cfg.get('TRANSFORMER_DEPTH_MIDDLE',
                                                self.transformer_depth[-1])

    def construct_network(self):
        in_channels = self.in_channels
        model_channels = self.model_channels
        out_channels = self.out_channels
        attention_resolutions = self.attention_resolutions
        channel_mult = self.channel_mult
        num_classes = self.num_classes
        num_heads = self.num_heads
        num_head_channels = self.num_head_channels
        dims = self.dims
        dropout = self.dropout
        use_checkpoint = self.use_checkpoint
        use_scale_shift_norm = self.use_scale_shift_norm
        disable_self_attentions = self.disable_self_attentions
        disable_middle_self_attn = self.disable_middle_self_attn
        transformer_depth = self.transformer_depth
        transformer_depth_middle = self.transformer_depth_middle
        context_dim = self.context_dim
        use_linear_in_transformer = self.use_linear_in_transformer
        resblock_updown = self.resblock_updown
        conv_resample = self.conv_resample
        adm_in_channels = self.adm_in_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == 'continuous':
                print('setting up linear c_adm embedding layer')
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == 'timestep':
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == 'sequential':
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ))
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        input_down_flag = [False]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False

                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                input_down_flag.append(False)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                input_down_flag.append(True)
                ds *= 2
                self._feature_size += ch
        self._input_block_chans = copy.deepcopy(input_block_chans)
        self._input_down_flag = input_down_flag

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(ch,
                               num_heads,
                               dim_head,
                               depth=transformer_depth_middle,
                               context_dim=context_dim,
                               disable_self_attn=disable_middle_self_attn,
                               use_linear=use_linear_in_transformer,
                               use_checkpoint=use_checkpoint),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self._middle_block_chans = [ch]

        self._output_block_chans = []
        self.output_blocks = nn.ModuleList([])
        self.lsc_identity = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.lsc_identity.append(nn.Identity())
                self._feature_size += ch
                self._output_block_chans.append(ch)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def _forward_origin(self, x, emb, context, hint=None, **kwargs):
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for m_id, module in enumerate(self.output_blocks):
            skip_h = hs.pop()
            if 'tuner_scale' in kwargs and kwargs[
                    'tuner_scale'] is not None and kwargs['tuner_scale'] < 1.0:
                tuner_scale = kwargs['tuner_scale']
                tuner_h = self.lsc_identity[m_id](skip_h) - skip_h
                h = torch.cat([h, skip_h + tuner_scale * tuner_h], dim=1)
            else:
                h = torch.cat([h, self.lsc_identity[m_id](skip_h)], dim=1)
            target_size = hs[-1].shape[-2:] if len(hs) > 0 else None
            h = module(h, emb, context, target_size)
        out = self.out(h)
        return out

    def _forward_control(self, x, emb, context, hint, **kwargs):
        control_scale = kwargs.pop('control_scale', 1.0)
        multi_csc_tuners = self.control_blocks
        # hints
        multi_hint_hs = []
        for sc_id, csc_tuners in enumerate(multi_csc_tuners):
            hint_input = hint[sc_id] if isinstance(hint, list) else hint
            hint_h = csc_tuners.pre_hint_blocks(hint_input)
            hint_hs = []
            for dsh_blk in csc_tuners.dense_hint_blocks:
                hint_h = dsh_blk(hint_h)
                hint_hs.append(hint_h)
            multi_hint_hs.append(hint_hs)
        # unet
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for m_id, module in enumerate(self.output_blocks):
            skip_h = hs.pop()
            multi_control_h = 0
            for sc_id, csc_tuners in enumerate(multi_csc_tuners):
                hint_h = multi_hint_hs[sc_id][::-1][m_id]
                control_h = csc_tuners.lsc_tuner_blocks[m_id](
                    skip_h + hint_h, x_shortcut=hint_h)
                multi_control_h += csc_tuners.scale * control_h
            tuner_h = self.lsc_identity[m_id](skip_h) - skip_h
            if torch.all(
                    torch.isclose(tuner_h,
                                  torch.zeros_like(tuner_h),
                                  atol=1e-6)):
                # csc-tuner
                skip_h_new = skip_h + control_scale * multi_control_h
            else:
                # csc-tuner + sc-tuner
                tuner_scale = kwargs['tuner_scale']
                skip_h_new = skip_h + control_scale * multi_control_h + tuner_scale * tuner_h
            h = torch.cat([h, skip_h_new], dim=1)
            target_size = hs[-1].shape[-2:] if len(hs) > 0 else None
            h = module(h, emb, context, target_size)
        out = self.out(h)
        return out

    def forward(self, x, t=None, cond=dict(), **kwargs):
        t_emb = timestep_embedding(t,
                                   self.model_channels,
                                   repeat_only=False,
                                   legacy=True)
        emb = self.time_embed(t_emb)
        if isinstance(cond, dict):
            if 'y' in cond:
                assert self.num_classes is not None
                emb = emb + self.label_emb(cond['y'])
            if 'concat' in cond:
                c = cond['concat']
                x = torch.cat([x, c], dim=1)
            if 'hint' in cond:
                hint = cond['hint']
            elif 'hint' in kwargs:
                hint = kwargs.pop('hint', None)
            else:
                hint = None
            context = cond.get('crossattn', None)
        else:
            context = cond
            hint = kwargs.pop('hint', None)

        if self.control_blocks is not None and hint is not None:
            out = self._forward_control(x, emb, context, hint, **kwargs)
        else:
            out = self._forward_origin(x, emb, context, **kwargs)
        return out

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            DiffusionUNetXL.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class LargenUNetXL(DiffusionUNetXL):
    para_dict = {
        'TRANSFORMER_BLOCK_TYPE': {
            'value': 'att_v1'
        },
        'IMAGE_SCALE': {
            'value': 0.0,
        },
    }
    para_dict.update(DiffusionUNetXL.para_dict)

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger=logger)
        self.init_params(cfg)
        self.construct_network()

    def init_params(self, cfg):
        super().init_params(cfg)
        self.transformer_block_type = cfg.get('TRANSFORMER_BLOCK_TYPE',
                                              'att_v1')
        TRANSFORMER_BLOCKS = {
            'att_v1': BasicTransformerBlock,
            'att_v2': TransformerBlockV2,
        }
        assert self.transformer_block_type in list(TRANSFORMER_BLOCKS.keys())
        self.transformer_block = TRANSFORMER_BLOCKS[
            self.transformer_block_type]
        self.image_scale = cfg.get('IMAGE_SCALE', 0.0)
        self.use_refine = cfg.get('USE_REFINE', False)

    def construct_network(self):
        in_channels = self.in_channels
        model_channels = self.model_channels
        out_channels = self.out_channels
        attention_resolutions = self.attention_resolutions
        channel_mult = self.channel_mult
        num_classes = self.num_classes
        num_heads = self.num_heads
        num_head_channels = self.num_head_channels
        dims = self.dims
        dropout = self.dropout
        use_checkpoint = self.use_checkpoint
        use_scale_shift_norm = self.use_scale_shift_norm
        disable_self_attentions = self.disable_self_attentions
        disable_middle_self_attn = self.disable_middle_self_attn
        transformer_depth = self.transformer_depth
        transformer_depth_middle = self.transformer_depth_middle
        context_dim = self.context_dim
        use_linear_in_transformer = self.use_linear_in_transformer
        resblock_updown = self.resblock_updown
        conv_resample = self.conv_resample
        adm_in_channels = self.adm_in_channels
        transformer_block = self.transformer_block

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == 'continuous':
                print('setting up linear c_adm embedding layer')
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == 'timestep':
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == 'sequential':
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ))
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        input_down_flag = [False]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False

                    layers.append(
                        SpatialTransformerV2(
                            ch,
                            num_heads,
                            dim_head,
                            transformer_block=transformer_block,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                input_down_flag.append(False)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                )
                ch = out_ch
                input_block_chans.append(ch)
                input_down_flag.append(True)
                ds *= 2
                self._feature_size += ch
        self._input_block_chans = copy.deepcopy(input_block_chans)
        self._input_down_flag = input_down_flag

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformerV2(ch,
                                 num_heads,
                                 dim_head,
                                 transformer_block=transformer_block,
                                 depth=transformer_depth_middle,
                                 context_dim=context_dim,
                                 disable_self_attn=disable_middle_self_attn,
                                 use_linear=use_linear_in_transformer,
                                 use_checkpoint=use_checkpoint),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self._middle_block_chans = [ch]

        self._output_block_chans = []
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    disabled_sa = disable_self_attentions[level] if exists(
                        disable_self_attentions) else False
                    layers.append(
                        SpatialTransformerV2(
                            ch,
                            num_heads,
                            dim_head,
                            transformer_block=transformer_block,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            disable_self_attn=disabled_sa,
                            use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint))
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else Upsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

                self._feature_size += ch
                self._output_block_chans.append(ch)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if self.use_refine:
            self.ref_time_embed = copy.deepcopy(self.time_embed)
            self.ref_label_emb = copy.deepcopy(self.label_emb)
            self.ref_input_blocks = copy.deepcopy(self.input_blocks)
            self.ref_input_blocks[0] = TimestepEmbedSequential(
                conv_nd(dims, 4, model_channels, 3, padding=1))
            self.ref_middle_block = copy.deepcopy(self.middle_block)
            self.ref_output_blocks = copy.deepcopy(self.output_blocks)

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model is not None:
            with FS.get_from(pretrained_model,
                             wait_finish=True) as local_model:
                self.init_from_ckpt(local_model, ignore_keys=self.ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        if path.endswith('safetensors'):
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(path)
        else:
            sd = torch.load(path, map_location='cpu')

        new_sd = OrderedDict()
        for k, v in sd.items():
            ignored = False
            for ik in ignore_keys:
                if ik in k:
                    if we.rank == 0:
                        self.logger.info(
                            'Ignore key {} from state_dict.'.format(k))
                    ignored = True
                    break
            if not ignored:
                if k == 'input_blocks.0.0.weight':
                    if we.rank == 0:
                        self.logger.info(
                            'Partial initial key {} from state_dict.'.format(
                                k))
                    new_v = torch.empty(320, self.in_channels, 3, 3)
                    nn.init.zeros_(new_v)
                    new_v[:, :v.shape[1]] = v
                    new_sd[k] = new_v
                    if self.use_refine:
                        new_sd['ref_' + k] = v
                else:
                    new_sd[k] = v
                    if self.use_refine:
                        new_sd['ref_' + k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if we.rank == 0:
            self.logger.info(
                f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def forward(self, x, t=None, cond=dict(), **kwargs):
        t_emb = timestep_embedding(t,
                                   self.model_channels,
                                   repeat_only=False,
                                   legacy=True)
        emb = self.time_embed(t_emb)

        if isinstance(cond, dict):
            if 'y' in cond:
                assert self.num_classes is not None
                emb = emb + self.label_emb(cond['y'])
                if self.use_refine:
                    ref_emb = self.ref_time_embed(t_emb)
                    assert 'null_y' in cond
                    cond_y = cond['y'].clone()
                    cond_y[:, :cond['null_y'].shape[1]] = cond['null_y']
                    ref_emb = ref_emb + self.ref_label_emb(cond_y)

            if 'concat' in cond:
                c = cond['concat']
                x = torch.cat([x, c], dim=1)

            context = cond.get('crossattn', None)
            img_context = cond.get('img_crossattn', None)

            task = cond['task']
            image_scale = cond.get('image_scale', self.image_scale)
            if 'Subject' in task and img_context is not None:
                ip_enc_scale = image_scale
                ip_dec_scale = image_scale
                num_img_tokens = img_context.shape[1]
                context = torch.cat([context, img_context], dim=1)
            else:
                ip_enc_scale = None
                ip_dec_scale = None
                num_img_tokens = None

            ref = cond.get('ref_xt', None)
            ref_context = cond.get('ref_crossattn', None)
        else:
            raise TypeError

        hs = []
        refs = []
        h = x

        if self.use_refine:
            assert ref is not None and ref_context is not None
            for i, (ref_module, module) in enumerate(
                    zip(self.ref_input_blocks, self.input_blocks)):
                ref = ref_module(ref, ref_emb, ref_context, caching=None)
                h = module(h,
                           emb,
                           context,
                           caching=None,
                           scale=ip_enc_scale,
                           num_img_token=num_img_tokens)
                refs.append(ref)
                hs.append(h)

            ref = self.ref_middle_block(ref,
                                        ref_emb,
                                        ref_context,
                                        caching=None)
            h = self.middle_block(h,
                                  emb,
                                  context,
                                  caching=None,
                                  scale=ip_enc_scale,
                                  num_img_token=num_img_tokens)

            for i, (ref_module, module) in enumerate(
                    zip(self.ref_output_blocks, self.output_blocks)):
                cache = []
                ref = torch.cat([ref, refs.pop()], dim=1)
                ref = ref_module(ref,
                                 ref_emb,
                                 ref_context,
                                 caching='write',
                                 cache=cache)
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h,
                           emb,
                           context,
                           caching='read',
                           cache=cache,
                           scale=ip_dec_scale,
                           num_img_token=num_img_tokens)
        else:
            for module in self.input_blocks:
                h = module(h,
                           emb,
                           context,
                           caching=None,
                           scale=ip_enc_scale,
                           num_img_token=num_img_tokens)
                hs.append(h)
            h = self.middle_block(h,
                                  emb,
                                  context,
                                  caching=None,
                                  scale=ip_enc_scale,
                                  num_img_token=num_img_tokens)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h,
                           emb,
                           context,
                           caching=None,
                           scale=ip_dec_scale,
                           num_img_token=num_img_tokens)

        out = self.out(h)
        return out

    @staticmethod
    def get_config_template():
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            LargenUNetXL.para_dict,
                            set_name=True)
