# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import OrderedDict

import numpy as np
import torch
from tools.infrared.scepter.modules.model.metric.base_metric import BaseMetric
from tools.infrared.scepter.modules.model.metric.registry import METRICS
from tools.infrared.scepter.modules.utils.config import dict_to_yaml


@METRICS.register_class('AccuracyMetric')
class AccuracyMetric(BaseMetric):
    para_dict = [{'TOPK': {'value': 1, 'description': 'topk accuracy!'}}]

    def __init__(self, cfg, logger=None):
        super(AccuracyMetric, self).__init__(cfg, logger=logger)
        topk = cfg.get('TOPK', 1)
        if isinstance(topk, int):
            topk = (topk, )
        self.topk = topk
        self.maxk = max(self.topk)

    @torch.no_grad()
    def __call__(self, logits, labels, label_map=None, prefix='acc'):
        """ Compute Accuracy
        Args:
            logits (torch.Tensor or numpy.ndarray):
            labels (torch.Tensor or numpy.ndarray):
            prefix (str): Prefix string of ret key, default is acc.

        Returns:
            A OrderedDict, contains accuracy tensors according to topk.

        """
        assert self.maxk <= logits.shape[-1]

        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        batch_size = logits.size(0)

        _, pred = logits.topk(self.maxk, 1, True, True)
        if label_map is not None:
            pred = torch.gather(label_map, 1, pred)
            # print(labels)
            # print(pred)

        pred = pred.t()
        corrects = pred.eq(labels.view(1, -1).expand_as(pred))

        res = OrderedDict()
        for k in self.topk:
            correct_k = corrects[:k].contiguous().view(-1).float().sum(0)
            res[f'{prefix}@{k}'] = correct_k.mul_(1.0 / batch_size)
        return res

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
        return dict_to_yaml('METRICS',
                            __class__.__name__,
                            AccuracyMetric.para_dict,
                            set_name=True)


@METRICS.register_class('EnsembleAccuracyMetric')
class EnsembleAccuracyMetric(object):
    para_dict = [{
        'TOPK': {
            'value': 1,
            'description': 'topk accuracy!'
        },
        'ENSEMBLE_METHOD': {
            'value': 'avg',
            'description': 'ensemble method from (avg, max)!'
        }
    }]

    def __init__(self, cfg, logger=None):
        topk = cfg.get('TOPK', 1)
        ensemble_method = cfg.get('ENSEMBLE_METHOD', 'avg')
        if isinstance(topk, int):
            topk = (topk, )
        self.topk = topk
        self.maxk = max(self.topk)
        assert ensemble_method in (
            'avg', 'max'
        ), f"Expected ensemble_method in ('avg', 'max'), got {ensemble_method}"
        self.ensemble_method = ensemble_method

    @torch.no_grad()
    def __call__(self, logits, labels, keys, prefix='acc'):
        """ Compute Accuracy
        Args:
            logits (torch.Tensor or numpy.ndarray):
            labels (torch.Tensor or numpy.ndarray):
            keys (List[str]): Keys to accumulate logits.
            prefix (str): Prefix string of ret key, default is acc.

        Returns:
            A OrderedDict, contains accuracy tensors according to topk.

        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        agg_keys = list(set(keys))
        keys = np.asarray(keys)

        agg_logits = []  # N * Tensor([C])
        agg_labels = []  # N * Tensor(scalar)

        for key in agg_keys:
            key_index = np.where(keys == key)[0]
            key_index = torch.from_numpy(key_index)
            key_logits = logits[key_index]

            if self.ensemble_method == 'avg':
                key_logit = torch.mean(key_logits, dim=0)
            else:
                key_logit, _ = torch.max(key_logit, dim=0)
            key_label = labels[key_index[0]]

            agg_logits.append(key_logit)
            agg_labels.append(key_label)

        agg_logits = torch.vstack(agg_logits)
        agg_labels = torch.hstack(agg_labels)

        return AccuracyMetric(self.topk)(agg_logits, agg_labels, prefix)

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
        return dict_to_yaml('METRICS',
                            __class__.__name__,
                            EnsembleAccuracyMetric.para_dict,
                            set_name=True)
