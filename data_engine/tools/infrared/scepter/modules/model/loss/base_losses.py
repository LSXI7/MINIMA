# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from packaging import version

import torch.nn as nn
from tools.infrared.scepter.modules.model.registry import LOSSES
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from torch.version import __version__ as torch_version


@LOSSES.register_class()
class CrossEntropy(nn.Module):
    para_dict = {
        'REDUCE': {
            'value':
            None,
            'description':
            'reduce is False, returns a loss per batch element instead '
            'and ignores :attr: size_average. Default: True'
        },
        'SIZE_AVERAGE': {
            'value':
            None,
            'description':
            'Deprecated (see :attr: reduction). By default,'
            'the losses are averaged over each loss element in the batch. Note that for'
            'some losses, there are multiple elements per sample. If the field :attr: size_average'
            'is set to False, the losses are instead summed for each minibatch. Ignored'
            'when :attr: reduce is False. Default: True'
        },
        'IGNORE_INDEX': {
            'value':
            -100,
            'description':
            'Specifies a target value that is ignored'
            'and does not contribute to the input gradient. When :attr: size_average is'
            'True, the loss is averaged over non-ignored targets. Note that'
            ':attr: ignore_index is only applicable when the target contains class indices.'
        },
        'REDUCTION': {
            'value':
            'mean',
            'description':
            'Specifies the reduction to apply to the output:'
            "'none' | 'mean' | 'sum'. 'none': no reduction will"
            "be applied, 'mean': the weighted mean of the output is taken,"
            "'sum': the output will be summed. Note: :attr: size_average"
            'and :attr:`reduce` are in the process of being deprecated, and in'
            'the meantime, specifying either of those two args will override'
            ":attr:`reduction`. Default: 'mean'"
        },
        'LABEL_SMOOTHING': {
            'value':
            0.0,
            'description':
            'A float in [0.0, 1.0]. Specifies the amount'
            'of smoothing when computing the loss, where 0.0 means no smoothing. '
        }
    }

    def __init__(self, cfg, logger=None):
        super(CrossEntropy, self).__init__()
        self.logger = logger
        size_average = cfg.get('SIZE_AVERAGE', None)
        ignore_index = cfg.get('IGNORE_INDEX', -100)
        reduce = cfg.get('REDUCE', None)
        reduction = cfg.get('REDUCTION', 'mean')

        if version.parse(torch_version) >= version.parse('1.10.0'):
            label_smoothing = cfg.get('LABEL_SMOOTHING', 0.0)
            self.loss_obj = nn.CrossEntropyLoss(
                size_average=size_average,
                ignore_index=ignore_index,
                reduce=reduce,
                reduction=reduction,
                label_smoothing=label_smoothing)
        else:
            self.loss_obj = nn.CrossEntropyLoss(size_average=size_average,
                                                ignore_index=ignore_index,
                                                reduce=reduce,
                                                reduction=reduction)

    def forward(self, input, target):
        return self.loss_obj(input, target)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

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
        return dict_to_yaml('LOSS',
                            __class__.__name__,
                            CrossEntropy.para_dict,
                            set_name=True)
