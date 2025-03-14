# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
import torch.nn as nn


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


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Prompt(nn.Module):
    """The implementation of vision prompt tuning method.

    Visual prompt tuning (VPT) is proposed to initialize tunable prompt tokens
    and prepend to the original tokens in the first layer or multiple layers.
    'Visual Prompt Tuning' by Jia et al.(2022)
    See https://arxiv.org/abs/2203.12119

    Attributes:
        dim: An integer indicating the embedding dimension.
        layer_num: An integer indicating number of layers.
        prompt_length: An integer indicating the length of vision prompt tuning.
        prompt_type: A string indicating the type of vision prompt tuning.
    """
    def __init__(self, dim, layer_num, prompt_length=None, prompt_type=None):
        super(Prompt, self).__init__()
        self.dim = dim
        self.layer_num = layer_num
        self.prompt_length = prompt_length
        self.prompt_type = prompt_type

        self.prompt_token = nn.Parameter(torch.zeros(1, prompt_length, dim))
        nn.init.xavier_uniform_(self.prompt_token)

    def forward(self, x):
        B, N, C = x.shape
        prompt_token = self.prompt_token.expand(B, -1, -1)

        if self.layer_num == 0:
            x = torch.cat((x, prompt_token), dim=1)
        else:
            x = torch.cat((x[:, :-self.prompt_length, :], prompt_token), dim=1)
        return x

    def extract(self, x):
        return x[:, :-self.prompt_length, :]


class Adapter(nn.Module):
    """The implementation of adapter tuning method.

    Adapters project input tokens by an MLP layer.
    'Parameter-Efficient Transfer Learning for NLP' by Houlsby et al.(2019)
    See http://arxiv.org/abs/1902.00751

    Attributes:
        dim: An integer indicating the embedding dimension.
        adapter_length: An integer indicating the length of adapter tuning.
        adapter_type: A string indicating the type of adapter tuning.
    """
    def __init__(
        self,
        dim,
        adapter_length=None,
        adapter_type=None,
        act_layer=nn.GELU,
    ):
        super(Adapter, self).__init__()
        self.dim = dim
        self.adapter_length = adapter_length
        self.adapter_type = adapter_type
        self.ln1 = nn.Linear(dim, adapter_length)
        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim)
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln2(self.activate(self.ln1(x)))
        if identity is None:
            identity = x
        out = identity + out
        return out


class LoRA(nn.Module):
    """The implementation of LoRA tuning method.

    LoRA constructs an additional layer with low-rank decomposition matrices of the weights in the network.
    'LoRA: Low-Rank Adaptation of Large Language Models' by Hu et al.(2021)
    See https://arxiv.org/abs/2106.09685

    Attributes:
        dim: An integer indicating the embedding dimension.
        num_heads: An integer indicating number of attention head.
        lora_length: An integer indicating the length of LoRA tuning.
        lora_type: A string indicating the type of LoRA tuning.
    """
    def __init__(
        self,
        dim,
        num_heads,
        lora_length=None,
        lora_type=None,
    ):
        super(LoRA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        if isinstance(dim, int):
            self.lora_a = nn.Linear(dim, lora_length, bias=False)
            self.lora_b = nn.Linear(lora_length, dim * 3, bias=False)
        else:
            self.lora_a = nn.Linear(dim[0], lora_length, bias=False)
            self.lora_b = nn.Linear(lora_length, dim[1] * 3, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

        self.lora_length = lora_length
        self.lora_type = lora_type

    def forward(self, x, q, k, v):
        B, N, C = x.shape
        qkv_delta = self.lora_b(self.lora_a(x))
        qkv_delta = qkv_delta.reshape(B, N, 3, self.num_heads,
                                      -1).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(0)
        # q, k, v = q + q_delta, k + k_delta, v + v_delta
        k, v = k + k_delta, v + v_delta
        return q, k, v


class Prefix(nn.Module):
    """The implementation of prefix tuning method.

    Prefix tuning optimizes the task-specific vector in the multi-head attention layer.
    'Prefix-tuning: Optimizing continuous prompts for generation' by Li & Liang(2021)
    See https://arxiv.org/abs/2101.00190

    Attributes:
        dim: An integer indicating the embedding dimension.
        num_heads: An integer indicating number of attention head.
        prefix_length: An integer indicating the length of prefix tuning.
        prefix_type: A string indicating the type of prefix tuning.
    """
    def __init__(
        self,
        dim,
        num_heads,
        prefix_length=None,
        prefix_type=None,
    ):
        super(Prefix, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.prefix_length = prefix_length
        self.prefix_type = prefix_type
        self.prefix_key = nn.Parameter(torch.zeros(1, prefix_length, dim))
        self.prefix_value = nn.Parameter(torch.zeros(1, prefix_length, dim))
        nn.init.xavier_uniform_(self.prefix_key)
        nn.init.xavier_uniform_(self.prefix_value)

    def forward(self, x, q, k, v):
        B, N, C = x.shape
        prefix_key = self.prefix_key.expand(B, -1, -1).reshape(
            B, self.prefix_length, self.num_heads,
            self.dim // self.num_heads).permute(0, 2, 1, 3)
        prefix_value = self.prefix_value.expand(B, -1, -1).reshape(
            B, self.prefix_length, self.num_heads,
            self.dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = torch.cat((k, prefix_key), dim=2), torch.cat((v, prefix_value),
                                                            dim=2)
        return q, k, v
