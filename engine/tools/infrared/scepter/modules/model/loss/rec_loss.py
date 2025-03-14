# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from tools.infrared.scepter.modules.model.registry import LOSSES
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from torch import nn


@LOSSES.register_class()
class ReconstructLoss(nn.Module):
    para_dict = {
        'LOSS_TYPE': {
            'value': 'l1',
            'description': 'Used loss type l1 or l2.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(ReconstructLoss, self).__init__()
        self.loss_type = cfg.get('LOSS_TYPE', 'l2')

    def forward(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target,
                                                    pred,
                                                    reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    @staticmethod
    def get_config_template():
        return dict_to_yaml('LOSS',
                            __class__.__name__,
                            ReconstructLoss.para_dict,
                            set_name=True)


@LOSSES.register_class()
class MinSNRLoss(ReconstructLoss):
    """Only used when parameterization=='eps'"""
    para_dict = {'GAMMA': {'value': 5, 'description': 'max value of snr.'}}
    para_dict.update(ReconstructLoss.para_dict)

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.gamma = cfg.get('GAMMA', 5)

    def forward(self, pred, target, alphas_cumprod, timesteps, mean=False):
        loss = super().forward(pred, target, mean=mean)

        alpha = torch.sqrt(alphas_cumprod)
        sigma = torch.sqrt(1.0 - alphas_cumprod)
        all_snr = ((alpha / sigma)**2).to(pred.device)
        snr_weight = (self.gamma / all_snr[timesteps]).clip(max=1.).float()
        return loss * snr_weight.view(-1, 1, 1, 1)
