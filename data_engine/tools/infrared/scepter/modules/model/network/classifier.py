# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import OrderedDict
from functools import partial

import torch.nn as nn
from tools.infrared.scepter.modules.model.metric.registry import METRICS
from tools.infrared.scepter.modules.model.network.train_module import TrainModule
from tools.infrared.scepter.modules.model.registry import (BACKBONES, HEADS, LOSSES, MODELS,
                                            NECKS)
from tools.infrared.scepter.modules.utils.config import Config, dict_to_yaml
from torch.nn.functional import sigmoid, softmax

_ACTIVATE_MAPPER = {'softmax': partial(softmax, dim=1), 'sigmoid': sigmoid}


@MODELS.register_class()
class Classifier(TrainModule):
    """ Base classifier implementation.

    Args:
        backbones (dict): Defines backbones.
        neck (dict, optional): Defines neck. Use Identity if none.
        head (dict): Defines head.
        act_name (str): Defines activate function, 'softmax' or 'sigmoid'.
        topk (Sequence[int]): Defines how to calculate accuracy metrics.
        freeze_bn (bool): If True, freeze all BatchNorm layers including LayerNorm.
    """
    para_dict = {
        'ACT_NAME': {
            'value':
            'softmax',
            'description':
            'the activation function for logits, select from [softmax, sigmoid]!'
        },
        'FREEZE_BN': {
            'value': False,
            'description': 'if freeze bn of not'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        # Construct model
        self.backbone = BACKBONES.build(cfg.BACKBONE, logger=logger)
        necks_cfg = cfg.get('NECK',
                            Config(cfg_dict={'NAME': 'Identity'}, load=False))
        self.neck = NECKS.build(necks_cfg, logger=logger)
        self.head = HEADS.build(cfg.HEAD, logger=logger)
        freeze_bn = cfg.get('FREEZE_BN', False)
        # Construct loss
        loss = cfg.get('LOSS',
                       Config(cfg_dict={'NAME': 'CrossEntropy'}, load=False))
        self.loss = LOSSES.build(loss, logger=logger)
        act_name = cfg.get('ACT_NAME', 'softmax')
        # Construct activate function
        self.act_fn = _ACTIVATE_MAPPER[act_name]
        self.metric = METRICS.build(cfg.METRIC, logger=logger)
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        self.training = mode
        super(Classifier, self).train(mode=mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module,
                              (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                    module.train(False)
        return self

    def forward(self, img, label=None, **kwargs):
        return self.forward_train(
            img, label=label) if self.training else self.forward_test(
                img, label=label)  # noqa

    def forward_train(self, img, label=None):
        probs = self.head(self.neck(self.backbone(img)))
        if label is None:
            return probs

        ret = OrderedDict()
        loss = self.loss(probs, label)
        ret['loss'] = loss
        ret['batch_size'] = img.size(0)
        ret.update(self.metric(probs, label))
        return ret

    def forward_test(self, img, label=None):
        logits = self.act_fn(self.head(self.neck(self.backbone(img))))
        if label is not None:
            ret = OrderedDict()
            ret['logits'] = logits
            ret['batch_size'] = img.size(0)
            ret.update(self.metric(logits, label))
            return ret
        return logits

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
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            Classifier.para_dict,
                            set_name=True)


@MODELS.register_class()
class VideoClassifier(Classifier):
    """ Classifier for video.
    Default input tensor is video.

    """
    def forward(self, video, label=None, **kwargs):
        return self.forward_train(video, label=label) \
            if self.training else self.forward_test(video, label=label)

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
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            VideoClassifier.para_dict,
                            set_name=True)


@MODELS.register_class()
class VideoClassifier2x(VideoClassifier):
    """ A 2-way classifier for video.

    """
    def forward_train(self, video, label=None):
        probs0, probs1 = self.head(self.neck(self.backbone(video)))
        if label is not None:
            ret = OrderedDict()
            loss = self.loss(probs0, label[:, 0]) + self.loss(
                probs1, label[:, 1])
            ret['loss'] = loss
            ret['batch_size'] = video.size(0)
            acc_0 = self.metric(probs0, label[:, 0])
            acc_0 = {
                key.relace('@', '_0@'): value
                for key, value in acc_0.items()
            }
            acc_1 = self.metric(probs1, label[:, 1])
            acc_1 = {
                key.relace('@', '_1@'): value
                for key, value in acc_1.items()
            }
            ret.update(acc_0)
            ret.update(acc_1)
            return ret
        return {'logits0': self.act_fn(probs0), 'logits1': self.act_fn(probs1)}

    def forward_test(self, video, label=None):
        probs0, probs1 = self.head(self.neck(self.backbone(video)))
        logits0, logits1 = self.act_fn(probs0), self.act_fn(probs1)
        if label is None:
            return {'logits0': logits0, 'logits1': logits1}
        ret = OrderedDict()
        ret['logits0'] = logits0
        ret['logits1'] = logits1
        acc_0 = self.metric(probs0, label[:, 0])
        acc_0 = {key.relace('@', '_0@'): value for key, value in acc_0.items()}
        acc_1 = self.metric(probs1, label[:, 1])
        acc_1 = {key.relace('@', '_1@'): value for key, value in acc_1.items()}
        ret.update(acc_0)
        ret.update(acc_1)
        return ret

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
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            VideoClassifier2x.para_dict,
                            set_name=True)
