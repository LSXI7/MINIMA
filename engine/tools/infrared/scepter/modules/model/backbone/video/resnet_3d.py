# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch.nn as nn
from tools.infrared.scepter.modules.model.backbone.video.bricks.non_local import NonLocal
from tools.infrared.scepter.modules.model.backbone.video.init_helper import \
    _init_convnet_weights
from tools.infrared.scepter.modules.model.registry import BACKBONES, BRICKS, STEMS
from tools.infrared.scepter.modules.utils.config import Config, dict_to_yaml

_n_conv_resnet = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}


@BRICKS.register_class()
class Base3DBlock(nn.Module):
    para_dict = {
        'DIM_IN': {
            'value': 64,
            'description': 'this block input dim!'
        },
        'NUM_FILTERS': {
            'value': 64,
            'description': 'this block input num filter!'
        },
        'KERNEL_SIZE': {
            'value': [1, 7, 7],
            'description': 'the kernel size!'
        },
        'DOWNSAMPLING': {
            'value': True,
            'description': "this block's downsampling!"
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': False,
            'description': "this block's downsampling temporal!"
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        },
        'BRANCH': {
            'NAME': {
                'value':
                '',
                'description':
                "the branch's params, which shared the parameters DIM_IN, "
                'NUM_FILTERS, DOWNSAMPLING, DOWNSAMPLING_TEMPORAL, BN_PARAMS!'
            }
        }
    }

    def __init__(self, cfg, logger=None):
        super(Base3DBlock, self).__init__()
        dim_in = cfg.DIM_IN
        num_filters = cfg.NUM_FILTERS
        kernel_size = cfg.KERNEL_SIZE
        downsampling = cfg.get('DOWNSAMPLING', True)
        downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL', True)
        # bn_params or {}
        bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(bn_params, Config):
            bn_params = bn_params.__dict__
        if dim_in != num_filters or downsampling:
            if downsampling:
                if downsampling_temporal:
                    _stride = (2, 2, 2)
                else:
                    _stride = (1, 2, 2)
            else:
                _stride = (1, 1, 1)
            self.short_cut = nn.Conv3d(dim_in,
                                       num_filters,
                                       kernel_size=(1, 1, 1),
                                       stride=_stride,
                                       padding=0,
                                       bias=False)
            self.short_cut_bn = nn.BatchNorm3d(num_filters, **(bn_params
                                                               or {}))
        branch_cfg = cfg.get('BRANCH', None)
        assert branch_cfg is not None
        branch_cfg.BN_PARAMS = bn_params
        branch_cfg.DOWNSAMPLING = downsampling
        branch_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal
        branch_cfg.DIM_IN = dim_in
        branch_cfg.NUM_FILTERS = num_filters
        branch_cfg.KERNEL_SIZE = kernel_size
        self.conv_branch = BRICKS.build(branch_cfg, logger=logger)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        short_cut = x
        if hasattr(self, 'short_cut'):
            short_cut = self.short_cut_bn(self.short_cut(short_cut))
        x = self.relu(short_cut + self.conv_branch(x))
        return x

    def set_stage_block(self, stage_id, block_id):
        if hasattr(self.conv_branch, 'set_stage_block_id'):
            self.conv_branch.set_stage_block_id(stage_id, block_id)

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
                            Base3DBlock.para_dict,
                            set_name=True)


@BRICKS.register_class()
class Base3DResStage(nn.Module):
    """
    ResNet Stage containing several blocks.
    """
    para_dict = {
        'NUM_BLOCKS': {
            'value': 5,
            'description': 'this stage contains num of block!'
        },
        'USE_NON_LOCAL': {
            'value': False,
            'description': 'use non local or not!'
        },
        'NUM_FILTERS': {
            'value': 64,
            'description': 'this block input num filter!'
        },
        'NON_LOCAL': {
            'NAME': {
                'value': 'NonLocal',
                'description': 'the non local config!'
            }
        }
    }
    para_dict.update(Base3DBlock.para_dict)

    def __init__(self, cfg, logger=None):
        super(Base3DResStage, self).__init__()
        self.num_blocks = cfg.NUM_BLOCKS
        use_non_local = cfg.get('USE_NON_LOCAL', False)
        non_local_cfg = cfg.get('NON_LOCAL', None)
        res_block = Base3DBlock(cfg, logger=logger)
        self.add_module('res_{}'.format(1), res_block)
        for i in range(self.num_blocks - 1):
            dim_in = cfg.NUM_FILTERS
            downsampling = False
            cfg.DIM_IN = dim_in
            cfg.DOWNSAMPLING = downsampling
            res_block = Base3DBlock(cfg, logger=logger)
            self.add_module('res_{}'.format(i + 2), res_block)
        if use_non_local:
            non_local = NonLocal(non_local_cfg, logger=logger)
            self.add_module('nonlocal', non_local)

    def forward(self, x):
        # performs computation on the convolutions
        for i in range(self.num_blocks):
            res_block = getattr(self, 'res_{}'.format(i + 1))
            x = res_block(x)

        # performs non-local operations if specified.
        if hasattr(self, 'nonlocal'):
            non_local = getattr(self, 'nonlocal')
            x = non_local(x)
        return x

    def set_stage_id(self, stage_id):
        for i in range(self.num_blocks):
            res_block = getattr(self, 'res_{}'.format(i + 1))
            res_block.set_stage_block_id(stage_id, i)
        if hasattr(self, 'nonlocal'):
            non_local = getattr(self, 'nonlocal')
            non_local.set_stage_block_id(stage_id, self.num_blocks)

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
                            Base3DResStage.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class ResNet3D(nn.Module):
    para_dict = {
        'DEPTH': {
            'value': 18,
            'description': "resnet model's depth!"
        },
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': 'the input channels num for the model!'
        },
        'NUM_FILTERS': {
            'value': [64, 64, 128, 256, 256],
            'description': 'this num filter for each layer!'
        },
        'KERNEL_SIZE': {
            'value': [[1, 7, 7], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3]],
            'description': 'the kernel size for each layer!'
        },
        'DOWNSAMPLING': {
            'value': [True, False, True, True, True],
            'description': 'the downsample status for each layer!'
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': [False, False, False, True, True],
            'description': 'the temporal downsample status for each layer!'
        },
        'USE_NON_LOCAL': {
            'value': [False, False, False, False, False],
            'description': 'use non local or not for each layer!'
        },
        'NON_LOCAL': {
            'value': None,
            'description': 'the non local paramters!'
        },
        'STEM': {
            'NAME': {
                'value':
                'DownSampleStem',
                'description':
                'use the shared parameters as DIM_IN, NUM_FILTERS, VISUAL_CFG,'
                'KERNEL_SIZE, DOWNSAMPLING, DOWNSAMPLING_TEMPORAL, BN_PARAMS'
            }
        },
        'BRANCH': {
            'NAME': {
                'value': 'R2D3DBranch',
                'description': 'use the shared parameters as BRANCH_STYLE'
            },
            'EXPANSION_RATIO': {
                'value': 2,
                'description': 'the expansion ratio value'
            }
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        },
        'INIT_CFG': {
            'value':
            None,
            'description':
            'the parameters init config, including name key and default as kaiming!'
        },
        'VISUAL_CFG': {
            'value': None,
            'description': 'the visualize config'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ResNet3D, self).__init__()
        depth = cfg.get('DEPTH', 18)
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        num_filters = cfg.get('NUM_FILTERS', [64, 64, 128, 256, 256])
        kernel_size = cfg.get(
            'KERNEL_SIZE',
            [[1, 7, 7], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3]])
        downsampling = cfg.get('DOWNSAMPLING', [True, False, True, True, True])
        downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL',
                                        [False, False, False, True, True])
        use_non_local = cfg.get('USE_NON_LOCAL',
                                [False, False, False, False, False])
        non_local_cfg = cfg.get('NON_LOCAL', None)
        stem_cfg = cfg.get('STEM', None)
        branch_cfg = cfg.get('BRANCH', None)

        init_cfg = cfg.get('INIT_CFG', None) or dict()
        visual_cfg = cfg.get('VISUAL_CFG', None) or dict()
        # bn_params or {}
        bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(bn_params, Config):
            bn_params = bn_params.__dict__
        if len(bn_params) < 1:
            bn_params = dict(eps=1e-3, momentum=0.1)
        # Build stem cfg
        stem_cfg.DIM_IN = num_input_channels
        stem_cfg.NUM_FILTERS = num_filters[0]
        stem_cfg.KERNEL_SIZE = tuple(kernel_size[0])
        stem_cfg.DOWNSAMPLING = downsampling[0]
        stem_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[0]
        stem_cfg.BN_PARAMS = bn_params
        stem_cfg.VISUAL_CFG = visual_cfg
        self.conv1 = STEMS.build(stem_cfg, logger=logger)
        self.conv1.set_stage_block_id(0, 0)
        # ------------------- Main arch -------------------
        branch_style = 'simple_block' if depth <= 34 else 'bottleneck'
        blocks_list = _n_conv_resnet[depth]
        for stage_id, num_blocks in enumerate(blocks_list):
            stage_id = stage_id + 1
            # Build branch cfg
            block_cfg = Config(cfg_dict={}, load=False, logger=logger)
            block_cfg.NUM_BLOCKS = num_blocks
            block_cfg.USE_NON_LOCAL = use_non_local[stage_id]
            block_cfg.NUM_FILTERS = num_filters[stage_id]
            block_cfg.NON_LOCAL = non_local_cfg
            block_cfg.DIM_IN = num_filters[stage_id - 1]
            block_cfg.DOWNSAMPLING = downsampling[stage_id]
            block_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[stage_id]
            block_cfg.BRANCH = branch_cfg
            block_cfg.BRANCH.BRANCH_STYLE = branch_style
            block_cfg.KERNEL_SIZE = tuple(kernel_size[stage_id])
            conv = Base3DResStage(block_cfg, logger=None)
            setattr(self, f'conv{stage_id + 1}', conv)
        # perform initialization
        if isinstance(init_cfg, Config):
            init_cfg = init_cfg.__dict__
        if init_cfg.get('name') == 'kaiming':
            _init_convnet_weights(self)

    def forward(self, video):
        x = self.conv1(video)
        for i in range(2, 6):
            x = getattr(self, f'conv{i}')(x)
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
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            ResNet3D.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class ResNet3D_2plus1d(nn.Module):
    para_dict = {
        'DEPTH': {
            'value': 18,
            'description': "resnet model's depth!"
        },
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': 'the input channels num for the model!'
        },
        'NUM_FILTERS': {
            'value': [64, 64, 128, 256, 512],
            'description': 'this num filter for each layer!'
        },
        'KERNEL_SIZE': {
            'value': [[3, 7, 7], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'description': 'the kernel size for each layer!'
        },
        'DOWNSAMPLING': {
            'value': [True, False, True, True, True],
            'description': 'the downsample status for each layer!'
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': [False, False, True, True, True],
            'description': 'the temporal downsample status for each layer!'
        },
        'USE_NON_LOCAL': {
            'value': [False, False, False, False, False],
            'description': 'use non local or not for each layer!'
        },
        'NON_LOCAL': {
            'value': None,
            'description': 'the non local paramters!'
        },
        'STEM': {
            'NAME': {
                'value':
                'R2Plus1DStem',
                'description':
                'use the shared parameters as DIM_IN, NUM_FILTERS, VISUAL_CFG,'
                'KERNEL_SIZE, DOWNSAMPLING, DOWNSAMPLING_TEMPORAL, BN_PARAMS'
            }
        },
        'BRANCH': {
            'NAME': {
                'value': 'R2Plus1DBranch',
                'description': 'use the shared parameters as BRANCH_STYLE'
            },
            'EXPANSION_RATIO': {
                'value': 2,
                'description': 'the expansion ratio value'
            }
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        },
        'INIT_CFG': {
            'value':
            None,
            'description':
            'the parameters init config, including name key and default as kaiming!'
        },
        'VISUAL_CFG': {
            'value': None,
            'description': 'the visualize config'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ResNet3D_2plus1d, self).__init__()
        depth = cfg.get('DEPTH', 18)
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        num_filters = cfg.get('NUM_FILTERS', [64, 64, 128, 256, 512])
        kernel_size = cfg.get(
            'KERNEL_SIZE',
            [[3, 7, 7], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]])
        downsampling = cfg.get('DOWNSAMPLING', [True, False, True, True, True])
        downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL',
                                        [False, False, False, True, True])
        use_non_local = cfg.get('USE_NON_LOCAL',
                                [False, False, False, False, False])
        non_local_cfg = cfg.get('NON_LOCAL', None)
        stem_cfg = cfg.get('STEM', None)
        branch_cfg = cfg.get('BRANCH', None)

        init_cfg = cfg.get('INIT_CFG', None) or dict()
        visual_cfg = cfg.get('VISUAL_CFG', None) or dict()
        # bn_params or {}
        bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(bn_params, Config):
            bn_params = bn_params.__dict__
        if len(bn_params) < 1:
            bn_params = dict(eps=1e-3, momentum=0.1)
        # Build stem cfg
        stem_cfg.DIM_IN = num_input_channels
        stem_cfg.NUM_FILTERS = num_filters[0]
        stem_cfg.KERNEL_SIZE = tuple(kernel_size[0])
        stem_cfg.DOWNSAMPLING = downsampling[0]
        stem_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[0]
        stem_cfg.BN_PARAMS = bn_params
        stem_cfg.VISUAL_CFG = visual_cfg
        self.conv1 = STEMS.build(stem_cfg, logger=logger)
        self.conv1.set_stage_block_id(0, 0)
        # ------------------- Main arch -------------------
        branch_style = 'simple_block' if depth <= 34 else 'bottleneck'
        blocks_list = _n_conv_resnet[depth]
        for stage_id, num_blocks in enumerate(blocks_list):
            stage_id = stage_id + 1
            # Build branch cfg
            block_cfg = Config(cfg_dict={}, load=False, logger=logger)
            block_cfg.NUM_BLOCKS = num_blocks
            block_cfg.USE_NON_LOCAL = use_non_local[stage_id]
            block_cfg.NUM_FILTERS = num_filters[stage_id]
            block_cfg.NON_LOCAL = non_local_cfg
            block_cfg.DIM_IN = num_filters[stage_id - 1]
            block_cfg.DOWNSAMPLING = downsampling[stage_id]
            block_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[stage_id]
            block_cfg.BRANCH = branch_cfg
            block_cfg.BRANCH.BRANCH_STYLE = branch_style
            block_cfg.KERNEL_SIZE = tuple(kernel_size[stage_id])
            conv = Base3DResStage(block_cfg, logger=None)
            setattr(self, f'conv{stage_id + 1}', conv)
        # perform initialization
        if isinstance(init_cfg, Config):
            init_cfg = init_cfg.__dict__
        if init_cfg.get('name') == 'kaiming':
            _init_convnet_weights(self)

    def forward(self, video):
        x = self.conv1(video)
        for i in range(2, 6):
            x = getattr(self, f'conv{i}')(x)
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
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            ResNet3D_2plus1d.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class ResNet3D_CSN(nn.Module):
    para_dict = {
        'DEPTH': {
            'value': 18,
            'description': "resnet model's depth!"
        },
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': 'the input channels num for the model!'
        },
        'NUM_FILTERS': {
            'value': [64, 256, 512, 1024, 2048],
            'description': 'this num filter for each layer!'
        },
        'KERNEL_SIZE': {
            'value': [[3, 7, 7], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'description': 'the kernel size for each layer!'
        },
        'DOWNSAMPLING': {
            'value': [True, False, True, True, True],
            'description': 'the downsample status for each layer!'
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': [False, False, True, True, True],
            'description': 'the temporal downsample status for each layer!'
        },
        'USE_NON_LOCAL': {
            'value': [False, False, False, False, False],
            'description': 'use non local or not for each layer!'
        },
        'NON_LOCAL': {
            'value': None,
            'description': 'the non local paramters!'
        },
        'STEM': {
            'NAME': {
                'value':
                'DownSampleStem',
                'description':
                'use the shared parameters as DIM_IN, NUM_FILTERS, VISUAL_CFG,'
                'KERNEL_SIZE, DOWNSAMPLING, DOWNSAMPLING_TEMPORAL, BN_PARAMS'
            }
        },
        'BRANCH': {
            'NAME': {
                'value': 'CSNBranch',
                'description': 'use the shared parameters as BRANCH_STYLE'
            },
            'EXPANSION_RATIO': {
                'value': 4,
                'description': 'the expansion ratio value'
            }
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        },
        'INIT_CFG': {
            'value':
            None,
            'description':
            'the parameters init config, including name key and default as kaiming!'
        },
        'VISUAL_CFG': {
            'value': None,
            'description': 'the visualize config'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ResNet3D_CSN, self).__init__()
        depth = cfg.get('DEPTH', 18)
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        num_filters = cfg.get('NUM_FILTERS', [64, 64, 128, 256, 256])
        kernel_size = cfg.get(
            'KERNEL_SIZE',
            [[1, 7, 7], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3]])
        downsampling = cfg.get('DOWNSAMPLING', [True, False, True, True, True])
        downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL',
                                        [False, False, False, True, True])
        use_non_local = cfg.get('USE_NON_LOCAL',
                                [False, False, False, False, False])
        non_local_cfg = cfg.get('NON_LOCAL', None)
        stem_cfg = cfg.get('STEM', None)
        branch_cfg = cfg.get('BRANCH', None)

        init_cfg = cfg.get('INIT_CFG', None) or dict()
        visual_cfg = cfg.get('VISUAL_CFG', None) or dict()
        # bn_params or {}
        bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(bn_params, Config):
            bn_params = bn_params.__dict__
        if len(bn_params) < 1:
            bn_params = dict(eps=1e-3, momentum=0.1)
        # Build stem cfg
        stem_cfg.DIM_IN = num_input_channels
        stem_cfg.NUM_FILTERS = num_filters[0]
        stem_cfg.KERNEL_SIZE = tuple(kernel_size[0])
        stem_cfg.DOWNSAMPLING = downsampling[0]
        stem_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[0]
        stem_cfg.BN_PARAMS = bn_params
        stem_cfg.VISUAL_CFG = visual_cfg
        self.conv1 = STEMS.build(stem_cfg, logger=logger)
        self.conv1.set_stage_block_id(0, 0)
        # ------------------- Main arch -------------------
        branch_style = 'simple_block' if depth <= 34 else 'bottleneck'
        blocks_list = _n_conv_resnet[depth]
        for stage_id, num_blocks in enumerate(blocks_list):
            stage_id = stage_id + 1
            # Build branch cfg
            block_cfg = Config(cfg_dict={}, load=False, logger=logger)
            block_cfg.NUM_BLOCKS = num_blocks
            block_cfg.USE_NON_LOCAL = use_non_local[stage_id]
            block_cfg.NUM_FILTERS = num_filters[stage_id]
            block_cfg.NON_LOCAL = non_local_cfg
            block_cfg.DIM_IN = num_filters[stage_id - 1]
            block_cfg.DOWNSAMPLING = downsampling[stage_id]
            block_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[stage_id]
            block_cfg.BRANCH = branch_cfg
            block_cfg.BRANCH.BRANCH_STYLE = branch_style
            block_cfg.KERNEL_SIZE = tuple(kernel_size[stage_id])
            conv = Base3DResStage(block_cfg, logger=None)
            setattr(self, f'conv{stage_id + 1}', conv)
        # perform initialization
        if isinstance(init_cfg, Config):
            init_cfg = init_cfg.__dict__
        if init_cfg.get('name') == 'kaiming':
            _init_convnet_weights(self)

    def forward(self, video):
        x = self.conv1(video)
        for i in range(2, 6):
            x = getattr(self, f'conv{i}')(x)
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
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            ResNet3D_CSN.para_dict,
                            set_name=True)


@BACKBONES.register_class()
class ResNet3D_TAda(nn.Module):
    para_dict = {
        'DEPTH': {
            'value': 18,
            'description': "resnet model's depth!"
        },
        'NUM_INPUT_CHANNELS': {
            'value': 3,
            'description': 'the input channels num for the model!'
        },
        'NUM_FILTERS': {
            'value': [64, 256, 512, 1024, 2048],
            'description': 'this num filter for each layer!'
        },
        'KERNEL_SIZE': {
            'value': [[1, 7, 7], [1, 3, 3], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
            'description': 'the kernel size for each layer!'
        },
        'DOWNSAMPLING': {
            'value': [True, True, True, True, True],
            'description': 'the downsample status for each layer!'
        },
        'DOWNSAMPLING_TEMPORAL': {
            'value': [False, False, False, False, False],
            'description': 'the temporal downsample status for each layer!'
        },
        'USE_NON_LOCAL': {
            'value': [False, False, False, False, False],
            'description': 'use non local or not for each layer!'
        },
        'NON_LOCAL': {
            'value': None,
            'description': 'the non local paramters!'
        },
        'STEM': {
            'NAME': {
                'value':
                'Base2DStem',
                'description':
                'use the shared parameters as DIM_IN, NUM_FILTERS, VISUAL_CFG,'
                'KERNEL_SIZE, DOWNSAMPLING, DOWNSAMPLING_TEMPORAL, BN_PARAMS'
            }
        },
        'BRANCH': {
            'NAME': {
                'value': 'TAdaConvBlockAvgPool',
                'description': 'use the shared parameters as BRANCH_STYLE'
            },
            'EXPANSION_RATIO': {
                'value': 4,
                'description': 'the expansion ratio value'
            }
        },
        'BN_PARAMS': {
            'value':
            None,
            'description':
            'bn params data, key/value align with torch.BatchNorm3d/2d/1d!'
        },
        'INIT_CFG': {
            'name': {
                'value': 'kaiming',
                'description': 'init fn!'
            }
        },
        'VISUAL_CFG': {
            'value': None,
            'description': 'the visualize config'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ResNet3D_TAda, self).__init__()
        depth = cfg.get('DEPTH', 18)
        num_input_channels = cfg.get('NUM_INPUT_CHANNELS', 3)
        num_filters = cfg.get('NUM_FILTERS', [64, 64, 128, 256, 256])
        kernel_size = cfg.get(
            'KERNEL_SIZE',
            [[1, 7, 7], [1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3]])
        downsampling = cfg.get('DOWNSAMPLING', [True, False, True, True, True])
        downsampling_temporal = cfg.get('DOWNSAMPLING_TEMPORAL',
                                        [False, False, False, True, True])
        use_non_local = cfg.get('USE_NON_LOCAL',
                                [False, False, False, False, False])
        non_local_cfg = cfg.get('NON_LOCAL', None)
        stem_cfg = cfg.get('STEM', None)
        branch_cfg = cfg.get('BRANCH', None)

        init_cfg = cfg.get('INIT_CFG', None) or dict()
        visual_cfg = cfg.get('VISUAL_CFG', None) or dict()
        # bn_params or {}
        bn_params = cfg.get('BN_PARAMS', None) or dict()
        if isinstance(bn_params, Config):
            bn_params = bn_params.__dict__
        if len(bn_params) < 1:
            bn_params = dict(eps=1e-3, momentum=0.1)
        # Build stem cfg
        stem_cfg.DIM_IN = num_input_channels
        stem_cfg.NUM_FILTERS = num_filters[0]
        stem_cfg.KERNEL_SIZE = tuple(kernel_size[0])
        stem_cfg.DOWNSAMPLING = downsampling[0]
        stem_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[0]
        stem_cfg.BN_PARAMS = bn_params
        stem_cfg.VISUAL_CFG = visual_cfg
        self.conv1 = STEMS.build(stem_cfg, logger=logger)
        self.conv1.set_stage_block_id(0, 0)
        # ------------------- Main arch -------------------
        branch_style = 'simple_block' if depth <= 34 else 'bottleneck'
        blocks_list = _n_conv_resnet[depth]
        for stage_id, num_blocks in enumerate(blocks_list):
            stage_id = stage_id + 1
            # Build branch cfg
            block_cfg = Config(cfg_dict={}, load=False, logger=logger)
            block_cfg.NUM_BLOCKS = num_blocks
            block_cfg.USE_NON_LOCAL = use_non_local[stage_id]
            block_cfg.NUM_FILTERS = num_filters[stage_id]
            block_cfg.NON_LOCAL = non_local_cfg
            block_cfg.DIM_IN = num_filters[stage_id - 1]
            block_cfg.DOWNSAMPLING = downsampling[stage_id]
            block_cfg.DOWNSAMPLING_TEMPORAL = downsampling_temporal[stage_id]
            block_cfg.BRANCH = branch_cfg
            block_cfg.KERNEL_SIZE = tuple(kernel_size[stage_id])
            block_cfg.BRANCH.BRANCH_STYLE = branch_style
            conv = Base3DResStage(block_cfg, logger=None)
            setattr(self, f'conv{stage_id + 1}', conv)
        # perform initialization
        if isinstance(init_cfg, Config):
            init_cfg = init_cfg.__dict__
        if init_cfg.get('name') == 'kaiming':
            _init_convnet_weights(self)

    def forward(self, video):
        x = self.conv1(video)
        for i in range(2, 6):
            x = getattr(self, f'conv{i}')(x)
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
        return dict_to_yaml('BACKBONE',
                            __class__.__name__,
                            ResNet3D_2plus1d.para_dict,
                            set_name=True)
