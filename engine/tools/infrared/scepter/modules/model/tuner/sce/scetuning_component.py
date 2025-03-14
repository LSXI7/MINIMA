# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch.nn as nn
from tools.infrared.scepter.modules.model.tuner.tuner_utils import (choose_weight_type,
                                                     get_weight_value)


class SCEAdapter(nn.Module):
    def __init__(self,
                 dim,
                 adapter_length,
                 adapter_type=None,
                 adapter_weight=None,
                 act_layer=nn.GELU,
                 zero_init_last=True,
                 use_bias=True):
        super(SCEAdapter, self).__init__()
        self.dim = dim
        self.adapter_length = adapter_length
        self.adapter_type = adapter_type
        self.adapter_weight = adapter_weight
        self.zero_init_last = zero_init_last
        self.use_bias = use_bias
        self.ln1 = nn.Linear(dim, adapter_length, bias=use_bias)
        self.activate = act_layer()
        self.ln2 = nn.Linear(adapter_length, dim, bias=use_bias)
        self.init_weights()
        self.init_scaling()

    def _zero_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if self.use_bias:
                nn.init.zeros_(m.bias)

    def _kaiming_init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def init_weights(self):
        self._kaiming_init_weights(self.ln1)
        if self.zero_init_last:
            self._zero_init_weights(self.ln2)
        else:
            self._kaiming_init_weights(self.ln2)

    def init_scaling(self):
        if self.adapter_weight:
            self.scaling = choose_weight_type(self.adapter_weight, self.dim)
        else:
            self.scaling = None

    def forward(self, x, x_shortcut=None, use_shortcut=True, **kwargs):
        if x_shortcut is None:
            x_shortcut = x
        x_shape = x.shape
        if len(x_shape) == 4:
            b, d, h, w = x_shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, d)
        out = self.ln2(self.activate(self.ln1(x)))
        if self.adapter_weight:
            scaling = get_weight_value(self.adapter_weight, self.scaling, out)
            out = out * scaling if scaling is not None else out
        if len(x_shape) == 4:
            b, d, h, w = x_shape
            out = out.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        if use_shortcut:
            out = x_shortcut + out
        return out
