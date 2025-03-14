# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy

import torch.nn as nn
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.distribute import gather_data, we
from tools.infrared.scepter.modules.utils.probe import (ProbeData, merge_gathered_probe,
                                         register_data)


class BaseModel(nn.Module):
    para_dict = {
        'PRETRAINED_MODEL': {
            'value': None,
            'description': 'Pretrained model path.'
        }
    }

    def __init__(self, cfg, logger=None):
        super(BaseModel, self).__init__()
        self.logger = logger
        self.cfg = cfg
        self._probe_data = {}
        self._dist_data = {}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' + ' ' + super().__repr__()

    def load_pretrained_model(self, pretrained_model):
        pass

    def register_probe(self, probe_data: dict):
        probe_da, dist_da = register_data(probe_data,
                                          key_prefix=__class__.__name__)
        self._probe_data.update(probe_da)
        for key in dist_da:
            if key not in self._dist_data:
                self._dist_data[key] = dist_da[key]
            else:
                for k, v in dist_da[key].items():
                    if k in self._dist_data[key]:
                        self._dist_data[key][k] += v
                    else:
                        self._dist_data[key][k] = v

    def probe_data(self):
        gather_probe_data = gather_data(self._probe_data)
        _dist_data_list = gather_data([self._dist_data])
        if not we.rank == 0:
            self._probe_data = {}
            self._dist_data = {}
        # Iterate recurse the sub class's probe data for time-aware data.
        for k, v in self._modules.items():
            if isinstance(getattr(self, k), BaseModel):
                for kk, vv in getattr(self, k).probe_data().items():
                    self._probe_data[f'{k}/{kk}'] = vv

        if gather_probe_data is not None:
            # Before processing, just merge the data.
            self._probe_data = merge_gathered_probe(gather_probe_data)
        reduce_dist_data = {}
        if _dist_data_list is not None:
            reduce_dist_data = {}
            for one_data in _dist_data_list:
                for k, v in one_data.items():
                    if k in reduce_dist_data:
                        for kk, vv in v.items():
                            if kk in reduce_dist_data[k]:
                                reduce_dist_data[k][kk] += vv
                            else:
                                reduce_dist_data[k][kk] = vv
                    else:
                        reduce_dist_data[k] = v
        self._dist_data = reduce_dist_data
        # Iterate recurse the sub class's probe data for reduce data.
        self._probe_data[f'{__class__.__name__}_distribute'] = ProbeData(
            self._dist_data)
        norm_dist_data = {}
        for key, value in self._dist_data.items():
            total = 0
            for k, v in value.items():
                total += v
            norm_v = {}
            for k, v in value.items():
                norm_v[k] = v / total
            norm_dist_data[key] = norm_v
        self._probe_data[f'{__class__.__name__}_norm_distribute'] = ProbeData(
            norm_dist_data)
        ret_data = copy.deepcopy(self._probe_data)
        self._probe_data = {}
        return ret_data

    def clear_probe(self):
        self._probe_data.clear()

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
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            BaseModel.para_dict,
                            set_name=True)
