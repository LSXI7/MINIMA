# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import copy
import json
import numbers
import os
import re
import sys

import yaml

from tools.infrared.scepter.modules.utils.model import StdMsg

_SECURE_KEYWORDS = [
    'ENDPOINT', 'BUCKET', 'OSS_AK', 'OSS_SK', 'OSS', 'TOKEN', 'APPKEY'
    'SECRET', 'ACCESS_ID', 'ACCESS_KEY', 'PASSWORD', 'TEMP_DIR'
]  # -> "*****"

_SECURE_VALUEWORDS = ['oss://', 'oss-']  # -> "#####"


def dict_to_yaml(module_name, name, json_config, set_name=False):
    '''
    { "ENV" :
        { "description" : "",
          "A" : {
                "value": 1.0,
                "description": ""
           }
        }
    }
    convert std dict to yaml
    :param module_name:
    :param json_config:
    :return:
    '''
    def convert_yaml_style(level=1,
                           name='ENV',
                           description='ENV PARA',
                           default='',
                           type_name='',
                           is_sys=False):
        new_line = ''
        new_line += '{}# {} DESCRIPTION: {} TYPE: {} default: {}\n'.format(
            '\t' * (level - 1), name.upper(), description, type_name,
            f'\'{default}\'' if isinstance(default, str) else default)
        if is_sys:
            if name == '-':
                new_line += '{}{}\n'.format('\t' * (level - 1), name.upper())
            else:
                new_line += '{}{}:\n'.format('\t' * (level - 1), name.upper())
        else:
            # if isinstance(default, str):
            #     default = f'\'{default}\''
            if default is None:
                new_line += '{}# {}: {}\n'.format('\t' * (level - 1),
                                                  name.upper(), default)
            else:
                new_line += '{}{}: {}\n'.format('\t' * (level - 1),
                                                name.upper(), default)
        return new_line

    def parse_dict(json_config,
                   level_num,
                   parent_key,
                   set_name=False,
                   name='',
                   parent_type='dict'):
        yaml_str = ''
        # print(level_num, json_config)
        if isinstance(json_config, dict):
            if 'value' in json_config:
                value = json_config['value']
                if isinstance(value, dict):
                    assert len(value) < 1
                    value = None
                description = json_config.get('description', '')
                yaml_str += convert_yaml_style(level=level_num - 1,
                                               name=parent_key,
                                               description=description,
                                               default=value,
                                               type_name=type(value).__name__)
                return True, yaml_str
            else:
                if len(json_config) < 1:
                    yaml_str += convert_yaml_style(level=level_num,
                                                   name='NAME',
                                                   description='',
                                                   default='',
                                                   type_name='')
                level_num += 1
                for k, v in json_config.items():
                    if k == 'description':
                        continue
                    if isinstance(v, dict):
                        is_final, new_yaml_str = parse_dict(v,
                                                            level_num,
                                                            k,
                                                            parent_type='dict')
                        if not is_final and parent_type == 'dict':
                            description = v.get('description', '')
                            yaml_str += convert_yaml_style(
                                level=level_num - 1,
                                name=k,
                                description=description,
                                default='',
                                type_name='',
                                is_sys=True)
                        if not is_final and parent_type == 'list':
                            yaml_str += convert_yaml_style(level=level_num,
                                                           name='NAME',
                                                           description='',
                                                           default=k,
                                                           type_name='')
                        yaml_str += new_yaml_str
                    elif isinstance(v, list):
                        base_yaml_str = convert_yaml_style(level=level_num - 1,
                                                           name=k,
                                                           description='',
                                                           default='',
                                                           type_name='',
                                                           is_sys=True)
                        yaml_str += base_yaml_str
                        for tup in v:
                            is_final, new_yaml_str = parse_dict(
                                tup, level_num, '-', parent_type='list')
                            if not is_final:
                                yaml_str += convert_yaml_style(level=level_num,
                                                               name='-',
                                                               description='',
                                                               default='',
                                                               type_name='',
                                                               is_sys=True)
                            yaml_str += new_yaml_str
                    else:
                        raise KeyError(
                            f'json config {json_config} must be a dict of list'
                        )

        elif isinstance(json_config, list):
            level_num += 1
            for tup in json_config:
                is_final, new_yaml_str = parse_dict(tup, level_num, '-')
                if not is_final:

                    yaml_str += convert_yaml_style(level=level_num - 1,
                                                   name='-',
                                                   description='',
                                                   default='',
                                                   type_name='',
                                                   is_sys=True)
                    if set_name:
                        yaml_str += convert_yaml_style(level=level_num,
                                                       name='NAME',
                                                       description='',
                                                       default=name,
                                                       type_name='')
                yaml_str += new_yaml_str
        else:
            raise KeyError(f'json config {json_config} must be a dict')
        return False, yaml_str

    if isinstance(json_config, dict):
        first_dict, sec_dict, third_dict = {}, {}, {}
        for key, value in json_config.items():
            if isinstance(value, dict) and len(value) > 0:
                first_dict[key] = value
            elif isinstance(value, dict) and len(value) == 0:
                sec_dict[key] = value
            elif isinstance(value, list):
                third_dict[key] = value
            else:
                raise f'Config {json_config} is illegal'
        json_config = {}
        json_config.update(first_dict)
        json_config.update(sec_dict)
        json_config.update(third_dict)

    yaml_str = f'[{module_name}] module yaml examples:\n'
    level_num = 1
    base_yaml_str = convert_yaml_style(level=level_num,
                                       name=module_name,
                                       description='',
                                       default='',
                                       type_name='',
                                       is_sys=True)
    level_num += 1

    is_final, new_yaml_str = parse_dict(json_config,
                                        level_num,
                                        module_name,
                                        set_name=isinstance(json_config, list)
                                        and set_name,
                                        name=name)
    if not is_final:
        yaml_str += base_yaml_str
        if set_name and not isinstance(json_config, list):
            yaml_str += convert_yaml_style(level=level_num,
                                           name='NAME',
                                           description='',
                                           default=name,
                                           type_name='')
        yaml_str += new_yaml_str
    else:
        yaml_str += new_yaml_str[1:]

    return yaml_str


pattern = re.compile('.*?(\${\w+}).*?')  # noqa


def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    for item in pattern.findall(value):
        var_name = item.strip('${}')
        value = value.replace(item, os.getenv(var_name, item))
    return value


yaml.SafeLoader.add_constructor('$', env_var_constructor)
yaml.SafeLoader.add_implicit_resolver('$', pattern, None)


def _parse_args(parser):
    if parser is None:
        parser = argparse.ArgumentParser(
            description='Argparser for My codebase:\n')
    else:
        assert isinstance(parser, argparse.ArgumentParser)
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Path to the configuration file',
                        required=False,
                        default=None)
    parser.add_argument('--local_rank',
                        dest='local_rank',
                        help='torch distributed launch args!',
                        default=0)
    parser.add_argument(
        '-l',
        '--launcher',
        dest='launcher',
        help='spawn launcher is using python scripts, torchrun launcher is '
        'using torchrun module, default is spawn!',
        default='spawn')

    parser.add_argument('-o',
                        '--data_online',
                        dest='data_online',
                        action='store_false',
                        help='Read data from online or save local as cache. '
                        'Default is from online.')

    parser.add_argument('-s',
                        '--share_storage',
                        dest='share_storage',
                        action='store_true',
                        help='If use nas as the common cache folder, '
                        'set True to avoid download conflict.')
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='Swich debug mode.')

    return parser.parse_args()


class Config(object):
    def __init__(self,
                 cfg_dict={},
                 load=True,
                 cfg_file=None,
                 logger=None,
                 parser_ins=None):
        '''
        support to parse json/dict/yaml_file of parameters.
        :param load: whether load parameters or not.
        :param cfg_dict: default None.
        :param cfg_level: default None, means the current cfg-level for recurrent cfg presentation.
        :param logger: logger instance for print the cfg log.
        one examples:
            import argparse
            parser = argparse.ArgumentParser(
                description="Argparser for Cate process:\n"
            )
            parser.add_argument(
                "--stage",
                dest="stage",
                help="Running stage!",
                default="train",
                choices=["train"]
            )

            cfg = Config(load=True, parser_ins=parser)
        '''
        # checking that the logger exists or not
        if logger is None:
            self.logger = StdMsg(name='Config')
        else:
            self.logger = logger
        self.cfg_dict = cfg_dict
        if load:
            if cfg_file is None:
                assert parser_ins is not None
                self.args = _parse_args(parser_ins)
                self.load_from_file(self.args.cfg_file)
                # os.environ["LAUNCHER"] = self.args.launcher
                os.environ['DATA_ONLINE'] = str(self.args.data_online).lower()
                os.environ['SHARE_STORAGE'] = str(
                    self.args.share_storage).lower()
                os.environ['ES_DEBUG'] = str(self.args.debug).lower()
            else:
                self.load_from_file(cfg_file)
            if 'ENV' not in self.cfg_dict:
                self.cfg_dict['ENV'] = {
                    'SEED': 2023,
                    'USE_PL': False,
                    'BACKEND': 'nccl',
                    'SYNC_BN': False,
                    'CUDNN_DETERMINISTIC': True,
                    'CUDNN_BENCHMARK': False
                }
                self.logger.info(
                    f"ENV is not set and will use default ENV as {self.cfg_dict['ENV']}; "
                    f'If want to change this value, please set them in your config.'
                )
            else:
                if 'SEED' not in self.cfg_dict['ENV']:
                    self.cfg_dict['ENV']['SEED'] = 2023
                    self.logger.info(
                        f"SEED is not set and will use default SEED as {self.cfg_dict['ENV']['SEED']}; "
                        f'If want to change this value, please set it in your config.'
                    )
            os.environ['ES_SEED'] = str(self.cfg_dict['ENV']['SEED'])
        self._update_dict(self.cfg_dict)
        if load:
            self.logger.info(f'Parse cfg file as \n {self.dump()}')

    def load_from_file(self, file_name):
        self.logger.info(f'Loading config from {file_name}')
        if file_name is None or not os.path.exists(file_name):
            self.logger.info(f'File {file_name} does not exist!')
            self.logger.warning(
                f"Cfg file is None or doesn't exist, Skip loading config from {file_name}."
            )
            return
        if file_name.endswith('.json'):
            self.cfg_dict = self._load_json(file_name)
            self.logger.info(
                f'System take {file_name} as json, because we find json in this file'
            )
        elif file_name.endswith('.yaml'):
            self.cfg_dict = self._load_yaml(file_name)
            self.logger.info(
                f'System take {file_name} as yaml, because we find yaml in this file'
            )
        else:
            self.logger.info(
                f'No config file found! Because we do not find json or yaml in --cfg {file_name}'
            )

    def _update_dict(self, cfg_dict):
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(load=False,
                                   cfg_dict=elem,
                                   logger=self.logger)
            elif type(elem) is list:
                config_list = []
                for idx, ele in enumerate(elem):
                    if type(ele) is str and ele[1:3] == 'e-':
                        ele = float(ele)
                        config_list.append(ele)
                    elif type(ele) is str:
                        config_list.append(ele)
                    elif type(ele) is dict:
                        config_list.append(
                            Config(load=False,
                                   cfg_dict=ele,
                                   logger=self.logger))
                    elif type(ele) is list:
                        config_list.append(ele)
                    else:
                        config_list.append(ele)
                return key, config_list
            else:
                if type(elem) is str and elem[1:3] == 'e-':
                    elem = float(elem)
                return key, elem

        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)

    def _load_json(self, cfg_file):
        '''
        :param cfg_file:
        :return:
        '''
        if cfg_file is None:
            self.logger.warning(
                f'Cfg file is None, Skip loading config from {cfg_file}.')
            return {}
        file_name = cfg_file
        try:
            cfg = json.load(open(file_name, 'r'))
        except Exception as e:
            self.logger.error(f'Load json from {cfg_file} error. Message: {e}')
            sys.exit()
        return cfg

    def _load_yaml(self, cfg_file):
        '''
        if replace some parameters from Base, You can reference the base parameters use Base.

        :param cfg_file:
        :return:
        '''
        if cfg_file is None:
            self.logger.warning(
                f'Cfg file is None, Skip loading config from {cfg_file}.')
            return {}
        file_name = cfg_file
        try:
            with open(cfg_file, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        except Exception as e:
            self.logger.error(f'Load yaml from {cfg_file} error. Message: {e}')
            sys.exit()
        if '_BASE_RUN' not in cfg.keys() and '_BASE_MODEL' not in cfg.keys(
        ) and '_BASE' not in cfg.keys():
            return cfg

        if '_BASE' in cfg.keys():
            if cfg['_BASE'][1] == '.':
                prev_count = cfg['_BASE'].count('..')
                cfg_base_file = self._path_join(
                    file_name.split('/')[:(-1 - cfg['_BASE'].count('..'))] +
                    cfg['_BASE'].split('/')[prev_count:])
            else:
                cfg_base_file = cfg['_BASE'].replace(
                    './', file_name.replace(file_name.split('/')[-1], ''))
            cfg_base = self._load_yaml(cfg_base_file)
            cfg = self._merge_cfg_from_base(cfg_base, cfg)
        else:
            if '_BASE_RUN' in cfg.keys():
                if cfg['_BASE_RUN'][1] == '.':
                    prev_count = cfg['_BASE_RUN'].count('..')
                    cfg_base_file = self._path_join(
                        file_name.split('/')[:(-1 - prev_count)] +
                        cfg['_BASE_RUN'].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg['_BASE_RUN'].replace(
                        './', file_name.replace(file_name.split('/')[-1], ''))
                cfg_base = self._load_yaml(cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base,
                                                cfg,
                                                preserve_base=True)
            if '_BASE_MODEL' in cfg.keys():
                if cfg['_BASE_MODEL'][1] == '.':
                    prev_count = cfg['_BASE_MODEL'].count('..')
                    cfg_base_file = self._path_join(
                        file_name.split('/')[:(
                            -1 - cfg['_BASE_MODEL'].count('..'))] +
                        cfg['_BASE_MODEL'].split('/')[prev_count:])
                else:
                    cfg_base_file = cfg['_BASE_MODEL'].replace(
                        './', file_name.replace(file_name.split('/')[-1], ''))
                cfg_base = self._load_yaml(cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg)
        return cfg

    def _path_join(self, path_list):
        path = ''
        for p in path_list:
            path += p + '/'
        return path[:-1]

    def items(self):
        return self.cfg_dict.items()

    def _merge_cfg_from_base(self, cfg_base, cfg, preserve_base=False):
        for k, v in cfg.items():
            if k in cfg_base.keys():
                if isinstance(v, dict):
                    self._merge_cfg_from_base(cfg_base[k], v)
                else:
                    cfg_base[k] = v
            else:
                if 'BASE' not in k or preserve_base:
                    cfg_base[k] = v
        return cfg_base

    def _merge_cfg_from_command(self, args, cfg):
        assert len(
            args.opts
        ) % 2 == 0, f'Override list {args.opts} has odd length: {len(args.opts)}'

        keys = args.opts[0::2]
        vals = args.opts[1::2]

        # maximum supported depth 3
        for idx, key in enumerate(keys):
            key_split = key.split('.')
            assert len(
                key_split
            ) <= 4, 'Key depth error. \n Maximum depth: 3\n Get depth: {}'.format(
                len(key_split))
            assert key_split[0] in cfg.keys(), 'Non-existant key: {}.'.format(
                key_split[0])
            if len(key_split) == 2:
                assert key_split[1] in cfg[
                    key_split[0]].keys(), 'Non-existant key: {}'.format(key)
            elif len(key_split) == 3:
                assert key_split[1] in cfg[
                    key_split[0]].keys(), 'Non-existant key: {}'.format(key)
                assert key_split[2] in cfg[key_split[0]][
                    key_split[1]].keys(), 'Non-existant key: {}'.format(key)
            elif len(key_split) == 4:
                assert key_split[1] in cfg[
                    key_split[0]].keys(), 'Non-existant key: {}'.format(key)
                assert key_split[2] in cfg[key_split[0]][
                    key_split[1]].keys(), 'Non-existant key: {}'.format(key)
                assert key_split[3] in cfg[key_split[0]][key_split[1]][
                    key_split[2]].keys(), 'Non-existant key: {}'.format(key)

            if len(key_split) == 1:
                cfg[key_split[0]] = vals[idx]
            elif len(key_split) == 2:
                cfg[key_split[0]][key_split[1]] = vals[idx]
            elif len(key_split) == 3:
                cfg[key_split[0]][key_split[1]][key_split[2]] = vals[idx]
            elif len(key_split) == 4:
                cfg[key_split[0]][key_split[1]][key_split[2]][
                    key_split[3]] = vals[idx]

        return cfg

    def __repr__(self):
        return '{}\n'.format(self.dump())

    def dump(self, is_secure=False):
        if not is_secure:
            return json.dumps(self.cfg_dict, indent=2)
        else:

            def make_secure(cfg):
                if isinstance(cfg, dict):
                    for key, val in cfg.items():
                        if key in _SECURE_KEYWORDS and type(val) is str:
                            cfg[key] = '*****'
                        else:
                            cfg[key] = make_secure(cfg[key])
                elif isinstance(cfg, list):
                    cfg = [make_secure(t) for t in cfg]
                elif isinstance(cfg, str):
                    for sval in _SECURE_VALUEWORDS:
                        if sval in cfg:
                            cfg = '#####'
                return cfg

            cfg_dict_copy = copy.deepcopy(self.cfg_dict)
            cfg_dict_copy = make_secure(cfg_dict_copy)
            return json.dumps(cfg_dict_copy, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)

    def have(self, name):
        if name in self.__dict__:
            return True
        return False

    def get(self, name, default=None):
        if name in self.__dict__:
            return self.__dict__[name]
        return default

    def __getitem__(self, key):
        return self.__dict__.__getitem__(key)

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if hasattr(self, 'cfg_dict') and key in self.cfg_dict:
            if isinstance(value, Config):
                value = value.cfg_dict
            self.cfg_dict[key] = value

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        self.__setattr__(key, value)

    def __iter__(self):
        return iter(self.__dict__)

    def set(self, name, value):
        new_dict = {name: value}
        self.__dict__.update(new_dict)
        self.__setattr__(name, value)

    def get_dict(self):
        return self.cfg_dict

    def get_lowercase_dict(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = self.get_dict()
        config_new = {}
        for key, val in cfg_dict.items():
            if isinstance(key, str):
                if isinstance(val, dict):
                    config_new[key.lower()] = self.get_lowercase_dict(val)
                else:
                    config_new[key.lower()] = val
            else:
                config_new[key] = val
        return config_new

    @staticmethod
    def get_plain_cfg(cfg=None):
        if isinstance(cfg, Config):
            cfg_new = {}
            cfg_dict = cfg.get_dict()
            for key, val in cfg_dict.items():
                if isinstance(val, (Config, dict, list)):
                    cfg_new[key] = Config.get_plain_cfg(val)
                elif isinstance(val, (str, numbers.Number)):
                    cfg_new[key] = val
            return cfg_new
        elif isinstance(cfg, dict):
            cfg_new = {}
            cfg_dict = cfg
            for key, val in cfg_dict.items():
                if isinstance(val, (Config, dict, list)):
                    cfg_new[key] = Config.get_plain_cfg(val)
                elif isinstance(val, (str, numbers.Number)):
                    cfg_new[key] = val
            return cfg_new
        elif isinstance(cfg, list):
            cfg_new = []
            cfg_list = cfg
            for val in cfg_list:
                if isinstance(val, (Config, dict, list)):
                    cfg_new.append(Config.get_plain_cfg(val))
                elif isinstance(val, (str, numbers.Number)):
                    cfg_new.append(val)
            return cfg_new
        else:
            return cfg

    def __len__(self):
        return len(self.cfg_dict)

    def pop(self, name):
        self.cfg_dict.pop(name)
