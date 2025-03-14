# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

import torch
from torch import nn


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FrozenTransformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

    def train(self, mode: bool = True):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VIT(nn.Module):
    para_dict = {
        'INPUT_RESOLUTION': {
            'value': 224,
            'description': 'The input resolution of vit model!'
        },
        'PATCH_SIZE': {
            'value': 32,
            'description': 'The patch size of vit model!'
        },
        'WIDTH': {
            'value': 768,
            'description': 'The input embbeding dimention!'
        },
        'OUTPUT_DIM': {
            'value': 512,
            'description': 'The output embbeding dimention!'
        },
        'LAYERS': {
            'value': 12,
            'description': "Model's all layers num!"
        },
        'HEADS': {
            'value': 12,
            'description': 'The head number of transformer!'
        },
        'EXPORT': {
            'value': False,
            'description': 'Whether export model or not!'
        },
        'TOKEN_WISE': {
            'value': False,
            'description': 'Whether output token wise feature or not!'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__()
        input_resolution = cfg.INPUT_RESOLUTION
        width = cfg.WIDTH
        patch_size = cfg.PATCH_SIZE
        layers = cfg.LAYERS
        heads = cfg.HEADS
        output_dim = cfg.OUTPUT_DIM
        use_proj = cfg.get('USE_PROJ', True)
        self.export = cfg.get('EXPORT', False)
        self.token_wise = cfg.get('TOKEN_WISE', False)
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        if use_proj:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        else:
            self.proj = None

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, x: torch.Tensor):
        x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0],
        # 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if self.export:
            x = torch.cat([
                self.class_embedding.to(x.dtype).view(x.shape[0], 1,
                                                      x.shape[-1]), x
            ],
                          dim=1)  # shape = [*, grid ** 2 + 1, width]
        else:
            x = torch.cat([
                self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                    device=x.device), x
            ],
                          dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if self.token_wise:
            return self.ln_post(x)
        x = self.ln_post(x[:, 0, :])
        if not self.export:
            if self.proj is not None:
                x = x @ self.proj
            return x
        else:
            before_proj = x
            if self.proj is not None:
                x = before_proj @ self.proj
            return before_proj, x


class VIT_MODEL(nn.Module):
    '''
        INPUT_RESOLUTION: 224
        PATCH_SIZE: 32
        WIDTH: 768
        OUTPUT_DIM: 512
        LAYERS: 12
        HEADS: 12
    '''
    para_dict = {
        'INPUT_RESOLUTION': {
            'value': 224,
            'description': 'The input resolution of vit model!'
        },
        'PATCH_SIZE': {
            'value': 32,
            'description': 'The patch size of vit model!'
        },
        'WIDTH': {
            'value': 768,
            'description': 'The input embbeding dimention!'
        },
        'OUTPUT_DIM': {
            'value': 512,
            'description': 'The output embbeding dimention!'
        },
        'FROZEN_LAYERS': {
            'value': 6,
            'description': "Frozen model's layers num!"
        },
        'FT_LAYERS': {
            'value': 6,
            'description': "Finetune model's layers num!"
        },
        'HEADS': {
            'value': 12,
            'description': 'The head number of transformer!'
        }
    }

    def __init__(self, cfg):
        super().__init__()
        input_resolution = cfg.INPUT_RESOLUTION
        patch_size = cfg.PATCH_SIZE
        width = cfg.WIDTH
        output_dim = cfg.OUTPUT_DIM
        frozen_layers = cfg.FROZEN_LAYERS
        ft_layers = cfg.FT_LAYERS
        heads = cfg.HEADS

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale *
                                            torch.randn(width))  # [768]
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))  # [50, 768]
        self.ln_pre = LayerNorm(width)

        self.frozen_transformer = FrozenTransformer(width, frozen_layers,
                                                    heads)
        self.frozen_transformer.eval()

        self.ft_transformer = Transformer(width, ft_layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.conv1(
                x)  # shape = [*, width, grid, grid] -> [1, 768, 7, 7]
            x = x.reshape(
                x.shape[0], x.shape[1],
                -1)  # shape = [*, width, grid ** 2]  -> [1, 768, 49] 49token
            x = x.permute(0, 2,
                          1)  # shape = [*, grid ** 2, width] -> [1, 49, 768]
            x = torch.cat([
                self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                    device=x.device), x
            ],
                          dim=1
                          )  # shape = [*, grid ** 2 + 1, width]-> [1, 50, 768]
            x = x + self.positional_embedding.to(x.dtype)  # [1, 50, 768]
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND [50, 1, 768]
            x = self.frozen_transformer(x)
        x = self.ft_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class MULTI_HEAD_VIT_MODEL(nn.Module):
    para_dict = {
        'INPUT_RESOLUTION': {
            'value': 224,
            'description': 'The input resolution of vit model!'
        },
        'PATCH_SIZE': {
            'value': 32,
            'description': 'The patch size of vit model!'
        },
        'WIDTH': {
            'value': 768,
            'description': 'The input embbeding dimention!'
        },
        'OUTPUT_DIM': {
            'value': 512,
            'description': 'The output embbeding dimention!'
        },
        'LAYERS': {
            'value': 12,
            'description': "All of the vit model's layers!"
        },
        'FROZEN_LAYERS': {
            'value': 6,
            'description': 'The frozen layers number!'
        },
        'FT_LAYERS': {
            'value': 6,
            'description': 'The finetune layers number!'
        },
        'MULTI_HEAD': {
            'value': 2,
            'description': 'The head number of vit!'
        },
        'HEADS': {
            'value': 12,
            'description': 'The head number of transformer!'
        }
    }

    def __init__(self, cfg):
        super().__init__()
        input_resolution = cfg.INPUT_RESOLUTION
        patch_size = cfg.PATCH_SIZE
        width = cfg.WIDTH
        output_dim = cfg.OUTPUT_DIM
        frozen_layers = cfg.FROZEN_LAYERS
        ft_layers = cfg.FT_LAYERS
        self.multi_head = cfg.MULTI_HEAD
        heads = cfg.HEADS

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale *
                                            torch.randn(width))  # [768]
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))  # [50, 768]
        self.ln_pre = LayerNorm(width)

        self.frozen_transformer = FrozenTransformer(width, frozen_layers,
                                                    heads)
        self.frozen_transformer.eval()

        if self.multi_head == 2:
            self.ft_transformer_1 = Transformer(width, ft_layers, heads)
            self.ln_post_1 = LayerNorm(width)
            self.proj_1 = nn.Parameter(scale * torch.randn(width, output_dim))

            self.ft_transformer_2 = Transformer(width, ft_layers, heads)
            self.ln_post_2 = LayerNorm(width)
            self.proj_2 = nn.Parameter(scale * torch.randn(width, output_dim))
        elif self.multi_head == 3:
            self.ft_transformer_1 = Transformer(width, ft_layers, heads)
            self.ln_post_1 = LayerNorm(width)
            self.proj_1 = nn.Parameter(scale * torch.randn(width, output_dim))

            self.ft_transformer_2 = Transformer(width, ft_layers, heads)
            self.ln_post_2 = LayerNorm(width)
            self.proj_2 = nn.Parameter(scale * torch.randn(width, output_dim))

            self.ft_transformer_3 = Transformer(width, ft_layers, heads)
            self.ln_post_3 = LayerNorm(width)
            self.proj_3 = nn.Parameter(scale * torch.randn(width, output_dim))
        elif self.multi_head == 4:
            self.ft_transformer_1 = Transformer(width, ft_layers, heads)
            self.ln_post_1 = LayerNorm(width)
            self.proj_1 = nn.Parameter(scale * torch.randn(width, output_dim))

            self.ft_transformer_2 = Transformer(width, ft_layers, heads)
            self.ln_post_2 = LayerNorm(width)
            self.proj_2 = nn.Parameter(scale * torch.randn(width, output_dim))

            self.ft_transformer_3 = Transformer(width, ft_layers, heads)
            self.ln_post_3 = LayerNorm(width)
            self.proj_3 = nn.Parameter(scale * torch.randn(width, output_dim))

            self.ft_transformer_4 = Transformer(width, ft_layers, heads)
            self.ln_post_4 = LayerNorm(width)
            self.proj_4 = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.conv1(
                x)  # shape = [*, width, grid, grid] -> [1, 768, 7, 7]
            x = x.reshape(
                x.shape[0], x.shape[1],
                -1)  # shape = [*, width, grid ** 2]  -> [1, 768, 49] 49token
            x = x.permute(0, 2,
                          1)  # shape = [*, grid ** 2, width] -> [1, 49, 768]
            x = torch.cat([
                self.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                    device=x.device), x
            ],
                          dim=1
                          )  # shape = [*, grid ** 2 + 1, width]-> [1, 50, 768]
            x = x + self.positional_embedding.to(x.dtype)  # [1, 50, 768]
            x = self.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND [50, 1, 768]
            x = self.frozen_transformer(x)

        if self.multi_head == 2:
            sub_x1 = self.ft_transformer_1(x)
            sub_x1 = sub_x1.permute(1, 0, 2)  # LND -> NLD
            sub_x1 = self.ln_post_1(sub_x1[:, 0, :])
            sub_x1 = sub_x1 @ self.proj_1

            sub_x2 = self.ft_transformer_2(x)
            sub_x2 = sub_x2.permute(1, 0, 2)  # LND -> NLD
            sub_x2 = self.ln_post_2(sub_x2[:, 0, :])
            sub_x2 = sub_x2 @ self.proj_2

            return sub_x1, sub_x2
        elif self.multi_head == 3:
            sub_x1 = self.ft_transformer_1(x)
            sub_x1 = sub_x1.permute(1, 0, 2)  # LND -> NLD
            sub_x1 = self.ln_post_1(sub_x1[:, 0, :])
            sub_x1 = sub_x1 @ self.proj_1

            sub_x2 = self.ft_transformer_2(x)
            sub_x2 = sub_x2.permute(1, 0, 2)  # LND -> NLD
            sub_x2 = self.ln_post_2(sub_x2[:, 0, :])
            sub_x2 = sub_x2 @ self.proj_2

            sub_x3 = self.ft_transformer_3(x)
            sub_x3 = sub_x3.permute(1, 0, 2)  # LND -> NLD
            sub_x3 = self.ln_post_3(sub_x3[:, 0, :])
            sub_x3 = sub_x3 @ self.proj_3

            return sub_x1, sub_x2, sub_x3
        elif self.multi_head == 4:
            sub_x1 = self.ft_transformer_1(x)
            sub_x1 = sub_x1.permute(1, 0, 2)  # LND -> NLD
            sub_x1 = self.ln_post_1(sub_x1[:, 0, :])
            sub_x1 = sub_x1 @ self.proj_1

            sub_x2 = self.ft_transformer_2(x)
            sub_x2 = sub_x2.permute(1, 0, 2)  # LND -> NLD
            sub_x2 = self.ln_post_2(sub_x2[:, 0, :])
            sub_x2 = sub_x2 @ self.proj_2

            sub_x3 = self.ft_transformer_3(x)
            sub_x3 = sub_x3.permute(1, 0, 2)  # LND -> NLD
            sub_x3 = self.ln_post_3(sub_x3[:, 0, :])
            sub_x3 = sub_x3 @ self.proj_3

            sub_x4 = self.ft_transformer_4(x)
            sub_x4 = sub_x4.permute(1, 0, 2)  # LND -> NLD
            sub_x4 = self.ln_post_4(sub_x4[:, 0, :])
            sub_x4 = sub_x4 @ self.proj_4

            return sub_x1, sub_x2, sub_x3, sub_x4


class MULTI_HEAD_VIT_MODEL_Split(nn.Module):
    para_dict = {
        'INPUT_RESOLUTION': {
            'value': 224,
            'description': 'The input resolution of vit model!'
        },
        'PATCH_SIZE': {
            'value': 32,
            'description': 'The patch size of vit model!'
        },
        'WIDTH': {
            'value': 768,
            'description': 'The input embbeding dimention!'
        },
        'OUTPUT_DIM': {
            'value': 512,
            'description': 'The output embbeding dimention!'
        },
        'LAYERS': {
            'value': 12,
            'description': "All of the vit model's layers!"
        },
        'FROZEN_LAYERS': {
            'value': 6,
            'description': 'The frozen layers number!'
        },
        'FT_LAYERS': {
            'value': 6,
            'description': 'The finetune layers number!'
        },
        'PART': {
            'value': 'backbone',
            'description': 'The part name of vit!'
        },
        'HEADS': {
            'value': 12,
            'description': 'The head number of transformer!'
        }
    }

    def __init__(self, cfg):
        super().__init__()
        input_resolution = cfg.INPUT_RESOLUTION
        patch_size = cfg.PATCH_SIZE
        width = cfg.WIDTH
        output_dim = cfg.OUTPUT_DIM
        frozen_layers = cfg.FROZEN_LAYERS
        ft_layers = cfg.FT_LAYERS
        heads = cfg.HEADS
        self.PART = cfg.PART
        scale = width**-0.5
        if self.PART == 'backbone':
            self.input_resolution = input_resolution
            self.output_dim = output_dim
            self.conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=width,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   bias=False)
            self.class_embedding = nn.Parameter(scale *
                                                torch.randn(width))  # [768]
            self.positional_embedding = nn.Parameter(scale * torch.randn(
                (input_resolution // patch_size)**2 + 1, width))  # [50, 768]
            self.ln_pre = LayerNorm(width)
            self.frozen_transformer = FrozenTransformer(
                width, frozen_layers, heads)
            self.frozen_transformer.eval()
        else:
            self.ft_transformer = Transformer(width, ft_layers, heads)
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        if self.PART == 'backbone':
            with torch.no_grad():
                x = self.conv1(
                    x)  # shape = [*, width, grid, grid] -> [1, 768, 7, 7]
                x = x.reshape(
                    x.shape[0], x.shape[1], -1
                )  # shape = [*, width, grid ** 2]  -> [1, 768, 49] 49token
                x = x.permute(
                    0, 2, 1)  # shape = [*, grid ** 2, width] -> [1, 49, 768]
                x = torch.cat(
                    [
                        self.class_embedding.to(x.dtype) +
                        torch.zeros(x.shape[0],
                                    1,
                                    x.shape[-1],
                                    dtype=x.dtype,
                                    device=x.device), x
                    ],
                    dim=1)  # shape = [*, grid ** 2 + 1, width]-> [1, 50, 768]
                x = x + self.positional_embedding.to(x.dtype)  # [1, 50, 768]
                x = self.ln_pre(x)
                x = x.permute(1, 0, 2)  # NLD -> LND [50, 1, 768]
                x = self.frozen_transformer(x)
                return x
        else:
            sub_x = self.ft_transformer(x)
            sub_x = sub_x.permute(1, 0, 2)  # LND -> NLD
            sub_x = self.ln_post(sub_x[:, 0, :])
            sub_x = sub_x @ self.proj
            return sub_x
