# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import warnings
from abc import abstractmethod
from importlib import find_loader

from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from tools.infrared.scepter.modules.model.utils.basic_utils import default, exists
from torch.utils.checkpoint import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except Exception as e:
    XFORMERS_IS_AVAILBLE = False
    warnings.warn(f'{e}')

if find_loader('flash_attn'):
    FLASH_ATTN_IS_AVAILABLE = True
    import flash_attn
    if (not hasattr(flash_attn, '__version__')) or (version.parse(
            flash_attn.__version__) < version.parse('2.0')):
        from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    else:
        from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func
else:
    FLASH_ATTN_IS_AVAILABLE = False


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f'unsupported dimensions: {dims}')


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps,
                       dim,
                       max_period=10000,
                       repeat_only=False,
                       legacy=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=timesteps.device)
        if legacy:
            args = timesteps[:, None].float() * freqs[None]
        else:
            args = torch.mm(timesteps.float().unsqueeze(1),
                            freqs.unsqueeze(0)).view(timesteps.shape[0],
                                                     len(freqs))
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class Timestep(nn.Module):
    def __init__(self, dim, legacy=False):
        super().__init__()
        self.dim = dim
        self.legacy = legacy

    def forward(self, t):
        return timestep_embedding(t, self.dim, legacy=self.legacy)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, context=None, target_size=None, **kwargs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, SpatialTransformerV2):
                x = layer(x, context, **kwargs)
            elif isinstance(layer, Upsample):
                x = layer(x, target_size)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=padding)

    def forward(self, x, target_size=None):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x.float(),
                              (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode='nearest').type_as(x)
        else:
            if target_size is None:
                x = F.interpolate(x.float(), scale_factor=2,
                                  mode='nearest').type_as(x)
            else:
                x = F.interpolate(x.float(), target_size,
                                  mode='nearest').type_as(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self,
                 channels,
                 use_conv,
                 dims=2,
                 out_channels=None,
                 padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims,
                        self.out_channels,
                        self.out_channels,
                        3,
                        padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims,
                                           channels,
                                           self.out_channels,
                                           3,
                                           padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels,
                                           1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, \
             f'q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}'
            self.num_heads = channels // num_head_channels

        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split head
            self.attention = QKVAttention(self.num_heads)
        else:
            # split head before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput head shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            'bct,bcs->bts', q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput head shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            'bct,bcs->bts', q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum('bts,bcs->bct', weight, v)
        return a.reshape(bs, -1, length)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(
            dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout),
                                 nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 dropout=0.0,
                 flash_dtype=torch.float16):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        context_dim = context_dim or dim
        assert flash_dtype in (None, torch.float16, torch.bfloat16)
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)
        self.flash_dtype = flash_dtype

        # layers
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        """x:       [B, L, C].
           context: [B, L', C'] or None.
        """
        context = x if context is None else context
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, d)
        k = self.k(context).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        attn = torch.einsum('binc,bjnc->bnij', q * self.scale, k * self.scale)
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v.float())
        # output
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x


class FlashattnMultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 dropout=0.0,
                 flash_dtype=torch.float16):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        context_dim = context_dim or dim
        assert flash_dtype in (None, torch.float16, torch.bfloat16)
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)
        self.flash_dtype = flash_dtype

        # layers
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        """x:       [B, L, C].
           context: [B, L', C'] or None.
        """
        context = x if context is None else context
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, d)
        k = self.k(context).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        if (x.device.type != 'cpu' and find_loader('flash_attn')
                and self.head_dim % 8 == 0 and self.head_dim <= 128
                and self.flash_dtype is not None):
            # flash implementation
            dtype = q.dtype
            if dtype != self.flash_dtype:
                q = q.type(self.flash_dtype)
                k = k.type(self.flash_dtype)
                v = v.type(self.flash_dtype)
            cu_seqlens_q = torch.arange(0,
                                        b * q.size(1) + 1,
                                        q.size(1),
                                        dtype=torch.int32,
                                        device=x.device)
            cu_seqlens_k = torch.arange(0,
                                        b * k.size(1) + 1,
                                        k.size(1),
                                        dtype=torch.int32,
                                        device=x.device)
            x = flash_attn_unpadded_kvpacked_func(
                q=q.reshape(-1, n, d).contiguous(),
                kv=torch.stack([k.reshape(-1, n, d),
                                v.reshape(-1, n, d)],
                               dim=1).contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q.size(1),
                max_seqlen_k=k.size(1),
                dropout_p=self.dropout.p if self.training else 0.0,
                return_attn_probs=False).reshape(b, -1, n, d).type(dtype)
        else:
            # attn = torch.einsum('binc,bjnc->bnij', q * self.scale, k * self.scale)
            # attn = F.softmax(attn.float(), dim=-1).type_as(attn)
            # x = torch.einsum('bnij,bjnc->binc', attn, v.float())
            # torch implementation
            q = q.permute(0, 2, 1, 3) * self.scale
            q = torch.clamp(q, min=-65504, max=66504)
            k = k.permute(0, 2, 3, 1) * self.scale
            k = torch.clamp(k, min=-65504, max=66504)
            v = v.permute(0, 2, 1, 3)
            if q.shape[1] == 10 and k.shape[
                    1] == 10 and q.shape[2] >= 8192 and k.shape[3] >= 8192:
                qkv = zip(q.chunk(10, dim=1), k.chunk(10, dim=1),
                          v.chunk(10, dim=1))
                tmp = []
                for q, k, v in qkv:
                    attn = torch.matmul(q, k)
                    attn = torch.clamp(attn, min=-65504, max=65504)
                    # print(f"attn1 has no inf: {torch.all(torch.isinf(attn) == False)}, attn1 dtype: {attn.dtype}")
                    # print(f"attn1 has no nan: {torch.all(torch.isnan(attn) == False)}")
                    attn = F.softmax(attn.float(), dim=-1).type_as(attn)
                    tmp.append(torch.matmul(attn, v))
                x = torch.cat(tmp, 1).permute(0, 2, 1, 3)
            else:
                attn = torch.matmul(q, k)
                attn = torch.clamp(attn, min=-65504, max=65504)
                # print(f"attn1 has no inf: {torch.all(torch.isinf(attn) == False)}, attn1 dtype: {attn.dtype}")
                # print(f"attn1 has no nan: {torch.all(torch.isnan(attn) == False)}")
                attn = F.softmax(attn.float(), dim=-1).type_as(attn)
                x = torch.matmul(attn, v).permute(0, 2, 1, 3)

        # output
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x


class XFormerMultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 dropout=0.0):
        super().__init__()
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        context_dim = context_dim or dim
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)

        # layers
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attention_op = None

    def x_form(self, x, context=None):
        context = x if context is None else context
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # compute query, key, value
        q = self.q(x).view(b, -1, n, d)
        k = self.k(context).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = xformers.ops.memory_efficient_attention(q,
                                                    k,
                                                    v,
                                                    attn_bias=None,
                                                    op=self.attention_op)
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x

    def x_ori(self, x, context=None):
        """x:       [B, L, C].
           context: [B, L', C'] or None.
        """
        context = x if context is None else context
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, d)
        k = self.k(context).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        attn = torch.einsum('binc,bjnc->bnij', q * self.scale, k * self.scale)
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v.float())
        # output
        x = x.reshape(b, -1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x

    def forward(self, x, context=None):
        """x:       [B, L, C].
                  context: [B, L', C'] or None.
               """
        if XFORMERS_IS_AVAILBLE:
            x = self.x_form(x, context=context)
        else:
            x = self.x_ori(x, context=context)
        return x


class CrossAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
                      (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))
        self.attention_op = None

    def forward(self, x, context=None, mask=None):

        if x.shape[-1] < 8:
            with torch.autocast(enabled=False, device_type='cuda'):
                q = self.to_q(x)
                context = default(context, x)
                k = self.to_k(context)
                v = self.to_v(context)
        else:
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3).reshape(b, t.shape[
                1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                    b * self.heads, t.shape[1], self.dim_head).contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q,
                                                      k,
                                                      v,
                                                      attn_bias=None,
                                                      op=self.attention_op)

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (out.unsqueeze(0).reshape(
            b, self.heads, out.shape[1],
            self.dim_head).permute(0, 2, 1,
                                   3).reshape(b, out.shape[1],
                                              self.heads * self.dim_head))
        return self.to_out(out)


class XFormersMHA_IP(nn.Module):
    def __init__(self,
                 query_dim,
                 context_dim=None,
                 heads=8,
                 dim_head=64,
                 dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))
        self.attention_op = None

    def forward(self,
                x,
                context=None,
                mask=None,
                scale=None,
                num_img_token=None):
        q = self.to_q(x)
        context = default(context, x)

        if scale is not None and num_img_token is not None:
            eos = context.shape[1] - num_img_token
            txt_context = context[:, :eos, :]
            img_context = context[:, eos:, :]

            k = self.to_k(txt_context)
            v = self.to_v(txt_context)
            k_i = self.to_k_ip(img_context)
            v_i = self.to_v_ip(img_context)

            b, _, _ = q.shape
            q, k, v, k_i, v_i = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[
                    1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                        b * self.heads, t.shape[1], self.dim_head).contiguous(
                        ),
                (q, k, v, k_i, v_i),
            )

            # actually compute the attention, what we cannot get enough of
            txt_out = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op)
            img_out = xformers.ops.memory_efficient_attention(
                q, k_i, v_i, attn_bias=None, op=self.attention_op)
            out = txt_out + scale * img_out
        else:
            k = self.to_k(context)
            v = self.to_v(context)
            b, _, _ = q.shape
            q, k, v = map(
                lambda t: t.unsqueeze(3).reshape(b, t.shape[
                    1], self.heads, self.dim_head).permute(0, 2, 1, 3).reshape(
                        b * self.heads, t.shape[1], self.dim_head).contiguous(
                        ),
                (q, k, v),
            )
            out = xformers.ops.memory_efficient_attention(q,
                                                          k,
                                                          v,
                                                          attn_bias=None,
                                                          op=self.attention_op)

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (out.unsqueeze(0).reshape(
            b, self.heads, out.shape[1],
            self.dim_head).permute(0, 2, 1,
                                   3).reshape(b, out.shape[1],
                                              self.heads * self.dim_head))
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 use_checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        AttentionBuilder = MemoryEfficientCrossAttention if XFORMERS_IS_AVAILBLE else CrossAttention
        self.attn1 = AttentionBuilder(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim
            if self.disable_self_attn else None)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = AttentionBuilder(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None):

        if self.use_checkpoint:
            return checkpoint(self._forward, x, context)
        else:
            return self._forward(x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x),
                       context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class TransformerBlockV2(nn.Module):
    def __init__(self,
                 query_dim,
                 n_heads,
                 d_head,
                 dropout=0.,
                 context_dim=None,
                 gated_ff=True,
                 use_checkpoint=False,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = MemoryEfficientCrossAttention(query_dim=query_dim,
                                                   heads=n_heads,
                                                   dim_head=d_head,
                                                   dropout=dropout,
                                                   context_dim=None)
        self.ff = FeedForward(query_dim, dropout=dropout, glu=gated_ff)
        self.attn2 = XFormersMHA_IP(query_dim=query_dim,
                                    heads=n_heads,
                                    dim_head=d_head,
                                    context_dim=context_dim)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self,
                x,
                context,
                caching=None,
                cache=None,
                scale=None,
                num_img_token=None,
                **kwargs):
        y = self.norm1(x)
        if caching == 'write':
            assert isinstance(cache, list)
            cache.append(y)
            x = self.attn1(y, context=None) + x
        elif caching == 'read':
            assert isinstance(cache, list) and len(cache) > 0
            c = cache.pop(0)
            self_ctx = torch.cat([y, c], dim=1)
            x = self.attn1(y, context=self_ctx) + x
        elif caching is None:
            x = self.attn1(y, context=None) + x
        else:
            assert False

        x = self.attn2(self.norm2(x),
                       context=context,
                       scale=scale,
                       num_img_token=num_img_token) + x
        x = self.ff(self.norm3(x)) + x

        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]

        if exists(context_dim) and not isinstance(context_dim, (list)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f'WARNING: {self.__class__.__name__}: Found context dims {context_dim} of'
                    f" depth {len(context_dim)}, which does not match the specified 'depth' of"
                    f' {depth}. Setting context_dim to {depth * [context_dim[0]]} now.'
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), 'need homogenous context_dim to match depth automatically'
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim,
                                  n_heads,
                                  d_head,
                                  dropout=dropout,
                                  context_dim=context_dim[d],
                                  disable_self_attn=disable_self_attn,
                                  use_checkpoint=use_checkpoint)
            for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class SpatialTransformerV2(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head,
                 transformer_block,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]

        if exists(context_dim) and not isinstance(context_dim, (list)):
            context_dim = [context_dim]
        if exists(context_dim) and isinstance(context_dim, list):
            if depth != len(context_dim):
                print(
                    f'WARNING: {self.__class__.__name__}: Found context dims {context_dim} of'
                    f" depth {len(context_dim)}, which does not match the specified 'depth' of"
                    f' {depth}. Setting context_dim to {depth * [context_dim[0]]} now.'
                )
                # depth does not match context dims.
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), 'need homogenous context_dim to match depth automatically'
                context_dim = depth * [context_dim[0]]
        elif context_dim is None:
            context_dim = [None] * depth

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            transformer_block(inner_dim,
                              n_heads,
                              d_head,
                              dropout=dropout,
                              context_dim=context_dim[d],
                              disable_self_attn=disable_self_attn,
                              use_checkpoint=use_checkpoint)
            for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(
                nn.Conv2d(inner_dim,
                          in_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape

        ref_mask = kwargs.pop('ref_mask', None)
        if ref_mask is not None:
            ref_mask = TF.resize(ref_mask, (h, w), antialias=True)
            ref_mask = (ref_mask > 0.5).float()
            ref_mask = rearrange(ref_mask, 'b c h w -> b (h w) c').contiguous()

        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = block(x, context=context[i], ref_mask=ref_mask, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
