# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.infrared.scepter.modules.model.base_model import BaseModel
from tools.infrared.scepter.modules.model.registry import HEADS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from torch.nn.parameter import Parameter


@HEADS.register_class()
class ClassifierHead(BaseModel):
    para_dict = {
        'DIM': {
            'value': 512,
            'description': 'representation dim!'
        },
        'NUM_CLASSES': {
            'value': 10,
            'description': 'number of classes.'
        },
        'DROPOUT_RATE': {
            'value': 0.0,
            'description': 'dropout rate, default 0.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ClassifierHead, self).__init__(cfg, logger=logger)
        self.dim = cfg.DIM
        self.num_classes = cfg.NUM_CLASSES
        self.dropout_rate = cfg.DROPOUT_RATE
        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.fc = nn.Linear(self.dim, self.num_classes)

    def forward(self, x, label=None):
        x = x.type(self.fc.weight.dtype)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.fc(x)

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
        return dict_to_yaml('HEADS',
                            __class__.__name__,
                            ClassifierHead.para_dict,
                            set_name=True)


class CosineLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 sigma: bool = True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, x, label=None):
        out = F.linear(F.normalize(x, p=2, dim=1),
                       F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


@HEADS.register_class()
class CosineLinearHead(BaseModel):
    para_dict = {
        'IN_DIM': {
            'value': 64,
            'description': 'the input dim for head!'
        },
        'NUM_CLASSES': {
            'value':
            10,
            'description':
            'The output dim for head, often this value is the classes number!'
        },
        'SIGMA': {
            'value': True,
            'description': 'The cosine scale which is learned by the model!'
        }
    }

    def __init__(self, cfg, logger=None):
        super(CosineLinearHead, self).__init__(cfg, logger=logger)
        self.in_features = cfg.IN_DIM
        self.out_features = cfg.NUM_CLASSES
        sigma = cfg.get('SIGMA', True)
        self.fc = CosineLinear(self.in_features, self.out_features, sigma)

    def forward(self, x, label=None):
        x = x.type(self.fc.weight.dtype)
        x = self.fc(x)
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
        return dict_to_yaml('HEADS',
                            __class__.__name__,
                            CosineLinearHead.para_dict,
                            set_name=True)


@HEADS.register_class()
class VideoClassifierHead(BaseModel):
    para_dict = {
        'DIM': {
            'value': 512,
            'description': 'representation dim!'
        },
        'NUM_CLASSES': {
            'value': 10,
            'description': 'number of classes.'
        },
        'DROPOUT_RATE': {
            'value': 0.0,
            'description': 'dropout rate, default 0.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(VideoClassifierHead, self).__init__(cfg, logger=logger)
        self.dim = cfg.DIM
        self.num_classes = cfg.NUM_CLASSES
        self.dropout_rate = cfg.get('DROPOUT_RATE', 0.5)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.out = nn.Linear(self.dim, self.num_classes, bias=True)

    def forward(self, x, label=None):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        out = self.out(x)

        return out

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
        return dict_to_yaml('HEADS',
                            __class__.__name__,
                            VideoClassifierHead.para_dict,
                            set_name=True)


@HEADS.register_class()
class VideoClassifierHeadx2(BaseModel):
    para_dict = {
        'DIM': {
            'value': 512,
            'description': 'representation dim!'
        },
        'NUM_CLASSES': {
            'value': [10, 12],
            'description': 'number of classes for two head.'
        },
        'DROPOUT_RATE': {
            'value': 0.0,
            'description': 'dropout rate, default 0.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(VideoClassifierHeadx2, self).__init__(cfg, logger=logger)
        self.dim = cfg.DIM
        self.num_classes = cfg.NUM_CLASSES
        self.dropout_rate = cfg.get('DROPOUT_RATE', 0.5)
        assert type(self.num_classes) is list
        assert len(self.num_classes) == 2

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.linear1 = nn.Linear(self.dim, self.num_classes[0], bias=True)
        self.linear2 = nn.Linear(self.dim, self.num_classes[1], bias=True)

    def forward(self, x, label=None):
        if hasattr(self, 'dropout'):
            out = self.dropout(x)
        else:
            out = x

        out1 = self.linear1(out)
        out2 = self.linear2(out)

        return out1, out2

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
        return dict_to_yaml('HEADS',
                            __class__.__name__,
                            VideoClassifierHeadx2.para_dict,
                            set_name=True)


@HEADS.register_class()
class TransformerHead(BaseModel):
    para_dict = {
        'DIM': {
            'value': 512,
            'description': 'representation dim!'
        },
        'NUM_CLASSES': {
            'value': 10,
            'description': 'number of classes.'
        },
        'DROPOUT_RATE': {
            'value': 0.0,
            'description': 'dropout rate, default 0.'
        },
        'PRE_LOGITS': {
            'value': False,
            'description': 'pre logits default False.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(TransformerHead, self).__init__(cfg, logger=logger)
        self.dim = cfg.DIM
        self.num_classes = cfg.NUM_CLASSES
        self.dropout_rate = cfg.get('DROPOUT_RATE', 0.5)
        self.pre_logits = cfg.get('PRE_LOGITS', False)
        if self.pre_logits:
            self.pre_logits = nn.Sequential(
                OrderedDict([('fc', nn.Linear(self.dim, self.dim)),
                             ('act', nn.Tanh())]))

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.linear = nn.Linear(self.dim, self.num_classes, bias=True)

    def forward(self, x, label=None):
        if hasattr(self, 'dropout'):
            out = self.dropout(x)
        else:
            out = x
        if hasattr(self, 'pre_logits'):
            out = self.pre_logits(out)
        out = self.linear(out)

        return out

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
        return dict_to_yaml('HEADS',
                            __class__.__name__,
                            TransformerHead.para_dict,
                            set_name=True)


@HEADS.register_class()
class TransformerHeadx2(BaseModel):
    para_dict = {
        'DIM': {
            'value': 512,
            'description': 'representation dim!'
        },
        'NUM_CLASSES': {
            'value': [10, 12],
            'description': 'number of classes for two head.'
        },
        'DROPOUT_RATE': {
            'value': 0.0,
            'description': 'dropout rate, default 0.'
        },
        'PRE_LOGITS': {
            'value': False,
            'description': 'pre logits default False.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(TransformerHeadx2, self).__init__(cfg, logger=logger)
        self.dim = cfg.DIM
        self.num_classes = cfg.NUM_CLASSES
        self.dropout_rate = cfg.get('DROPOUT_RATE', 0.5)
        self.pre_logits = cfg.get('PRE_LOGITS', False)
        assert type(self.num_classes) is list
        assert len(self.num_classes) == 2
        if self.pre_logits:
            self.pre_logits1 = nn.Sequential(
                OrderedDict([('fc', nn.Linear(self.dim, self.dim)),
                             ('act', nn.Tanh())]))
            self.pre_logits2 = nn.Sequential(
                OrderedDict([('fc', nn.Linear(self.dim, self.dim)),
                             ('act', nn.Tanh())]))

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)

        self.linear1 = nn.Linear(self.dim, self.num_classes[0], bias=True)
        self.linear2 = nn.Linear(self.dim, self.num_classes[1], bias=True)

    def forward(self, x, label=None):
        if hasattr(self, 'dropout'):
            out = self.dropout(x)
        else:
            out = x

        if hasattr(self, 'pre_logits1'):
            out1 = self.pre_logits1(out)
            out2 = self.pre_logits2(out)
        else:
            out1, out2 = out, out

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)

        return out1, out2

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
        return dict_to_yaml('HEADS',
                            __class__.__name__,
                            TransformerHeadx2.para_dict,
                            set_name=True)
