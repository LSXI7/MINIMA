# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import collections
import math
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.infrared.scepter.modules.model.backbone.video.bricks.base_branch import BaseBranch
from tools.infrared.scepter.modules.model.registry import BRICKS
from tools.infrared.scepter.modules.utils.config import Config, dict_to_yaml


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class TAdaConv2d(nn.Module):
    """ Performs temporally adaptive 2D convolution.
    Currently, only application on 5D tensors is supported, which makes TAdaConv2d
    essentially a 3D convolution with temporal kernel size of 1.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        """
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (Tuple[int]): kernel size of TAdaConv2d.
            stride (int, Tuple[int]): stride for the convolution in TAdaConv2d.
            padding (int, Tuple[int]): padding for the convolution in TAdaConv2d.
            dilation (Tuple[int]): dilation of the convolution in TAdaConv2d.
            groups (int): number of groups for TAdaConv2d.
            bias (bool): whether to use bias in TAdaConv2d.
        """
        super(TAdaConv2d, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)

        assert kernel_size[0] == 1
        assert stride[0] == 1
        assert padding[0] == 0
        assert dilation[0] == 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, 1, out_channels, in_channels // groups,
                         kernel_size[1], kernel_size[2]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):
        """
        Args:
            x (tensor): feature to perform convolution on.
            alpha (tensor): calibration weight for the base weights.
                W_t = alpha_t * W_b
        """
        _, _, c_out, c_in, kh, kw = self.weight.size()
        b, c_in, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(1, -1, h, w)

        # alpha: B, C, T, H(1), W(1) -> B, T, C, H(1), W(1) -> B, T, 1, C, H(1), W(1)
        # corresponding to calibrating the input channel
        weight = (alpha.permute(0, 2, 1, 3, 4).unsqueeze(2) *
                  self.weight).reshape(-1, c_in, kh, kw)

        bias = None
        if self.bias is not None:
            raise NotImplementedError
        else:
            output = F.conv2d(x,
                              weight=weight,
                              bias=bias,
                              stride=self.stride[1:],
                              padding=self.padding[1:],
                              dilation=self.dilation[1:],
                              groups=self.groups * b * t)

        output = output.view(b, t, c_out, output.size(-2),
                             output.size(-1)).permute(0, 2, 1, 3, 4)

        return output

    def extra_repr(self) -> str:
        return f'{self.in_channels}, {self.out_channels}, ' \
               f'kernel_size: {self.kernel_size}, stride: {self.stride}, ' \
               f'padding: {self.padding}, dilation: {self.dilation}, ' \
               f'groups: {self.groups}'


class RouteFuncMLP(nn.Module):
    """ The routing function for generating the calibration weights.
    """
    def __init__(self, c_in, ratio, kernels, bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.globalpool = nn.AdaptiveAvgPool3d(1)
        self.g = nn.Conv3d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=(1, 1, 1),
            padding=0,
        )
        self.a = nn.Conv3d(
            in_channels=c_in,
            out_channels=int(c_in // ratio),
            kernel_size=(kernels[0], 1, 1),
            padding=(kernels[0] // 2, 0, 0),
        )
        self.bn = nn.BatchNorm3d(int(c_in // ratio),
                                 eps=bn_eps,
                                 momentum=bn_mmt)
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv3d(in_channels=int(c_in // ratio),
                           out_channels=c_in,
                           kernel_size=(kernels[1], 1, 1),
                           padding=(kernels[1] // 2, 0, 0),
                           bias=False)
        self.b.skip_init = True
        self.b.weight.data.zero_()  # to make sure the initial values
        # for the output is 1.

    def forward(self, x):
        g = self.globalpool(x)
        x = self.avgpool(x)
        x = self.a(x + self.g(g))
        x = self.bn(x)
        x = self.relu(x)
        x = self.b(x) + 1
        return x


@BRICKS.register_class()
class TAdaConvBlockAvgPool(BaseBranch):
    """ The TAdaConv branch with average pooling as the feature aggregation scheme.

    For details, see
    Ziyuan Huang, Shiwei Zhang, Liang Pan, Zhiwu Qing, Mingqian Tang, Ziwei Liu, and Marcelo H. Ang Jr.
    "TAda! Temporally-Adaptive Convolutions for Video Understanding."
    """
    para_dict = {
        'DIM_IN': {
            'value': 64,
            'description': "the branch's dim in!"
        },
        'NUM_FILTERS': {
            'value': 64,
            'description': 'the num of filter!'
        },
        'DOWNSAMPLING': {
            'value': True,
            'description': 'downsample spatial data or not!'
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': True,
            'description': 'downsample temporal data or not!'
        },
        'EXPANISION_RATIO': {
            'value': 2,
            'description': 'expanision ratio for this branch!'
        },
        'ROUTE_FUNC_R': {
            'value': 4,
            'description': 'the route func r!'
        },
        'ROUTE_FUNC_K': {
            'value': [3, 3],
            'description': 'the route func k!'
        },
        'POOL_K': {
            'value': [3, 1, 1],
            'description': 'the pool k!'
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        }
    }
    para_dict.update(BaseBranch.para_dict)

    def __init__(self, cfg, logger=None):
        self.dim_in = cfg.DIM_IN
        self.num_filters = cfg.NUM_FILTERS
        self.kernel_size = cfg.KERNEL_SIZE
        self.downsampling = cfg.get('DOWNSAMPLING', True)
        self.downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL', True)
        self.expansion_ratio = cfg.get('EXPANISION_RATIO', 2)
        self.route_func_r = cfg.get('ROUTE_FUNC_R', 4)
        self.route_func_k = cfg.get('ROUTE_FUNC_K', [3, 3])
        self.pool_k = cfg.get('POOL_K', [3, 1, 1])
        # bn_params or {}
        self.bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(self.bn_params, Config):
            self.bn_params = self.bn_params.__dict__
        if self.downsampling:
            if self.downsampling_temporal:
                self.stride = [2, 2, 2]
            else:
                self.stride = [1, 2, 2]
        else:
            self.stride = [1, 1, 1]

        super(TAdaConvBlockAvgPool, self).__init__(cfg, logger=logger)

    def _construct_simple_block(self):
        raise NotImplementedError

    def _construct_bottleneck(self):
        self.a = nn.Conv3d(in_channels=self.dim_in,
                           out_channels=self.num_filters //
                           self.expansion_ratio,
                           kernel_size=(1, 1, 1),
                           stride=(1, 1, 1),
                           padding=0,
                           bias=False)
        self.a_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio,
                                   **self.bn_params)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = TAdaConv2d(
            in_channels=self.num_filters // self.expansion_ratio,
            out_channels=self.num_filters // self.expansion_ratio,
            kernel_size=(1, self.kernel_size[1], self.kernel_size[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.kernel_size[1] // 2, self.kernel_size[2] // 2),
            bias=False)
        self.b_rf = RouteFuncMLP(c_in=self.num_filters // self.expansion_ratio,
                                 ratio=self.route_func_r,
                                 kernels=self.route_func_k)
        self.b_bn = nn.BatchNorm3d(self.num_filters // self.expansion_ratio,
                                   **self.bn_params)
        self.b_avgpool = nn.AvgPool3d(kernel_size=self.pool_k,
                                      stride=1,
                                      padding=(self.pool_k[0] // 2,
                                               self.pool_k[1] // 2,
                                               self.pool_k[2] // 2))
        self.b_avgpool_bn = nn.BatchNorm3d(
            self.num_filters // self.expansion_ratio, **self.bn_params)
        self.b_avgpool_bn.skip_init = True
        self.b_avgpool_bn.weight.data.zero_()
        self.b_avgpool_bn.bias.data.zero_()

        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv3d(in_channels=self.num_filters //
                           self.expansion_ratio,
                           out_channels=self.num_filters,
                           kernel_size=(1, 1, 1),
                           stride=(1, 1, 1),
                           padding=0,
                           bias=False)
        self.c_bn = nn.BatchNorm3d(self.num_filters, **self.bn_params)

    def forward(self, x):
        if self.branch_style == 'simple_block':
            raise NotImplementedError

        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x, self.b_rf(x))
        x = self.b_bn(x) + self.b_avgpool_bn(self.b_avgpool(x))
        x = self.b_relu(x)

        x = self.c(x)
        x = self.c_bn(x)
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
                            TAdaConvBlockAvgPool.para_dict,
                            set_name=True)
