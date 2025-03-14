# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import hashlib
import json
import os
import warnings

import torch
from tools.infrared.scepter.modules.utils.file_system import FS

try:
    from peft.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
except Exception as e:
    warnings.warn(f'Import peft error, please deal with this problem: {e}')
try:
    from swift import Swift, SwiftModel
except Exception as e:
    warnings.warn(f'Import swift error, please deal with this problem: {e}')


class TunerInference():
    def __init__(self, logger=None):
        self.logger = logger
        self.is_register = False

    # @classmethod
    def unregister_tuner(self, tuner_model_list, diffusion_model,
                         cond_stage_model):
        self.logger.info('Unloading tuner model')
        if isinstance(diffusion_model['model'], SwiftModel):
            for adapter_name in diffusion_model['model'].adapters:
                diffusion_model['model'].deactivate_adapter(adapter_name,
                                                            offload='cpu')
        if isinstance(cond_stage_model['model'], SwiftModel):
            for adapter_name in cond_stage_model['model'].adapters:
                cond_stage_model['model'].deactivate_adapter(adapter_name,
                                                             offload='cpu')
        return

    # @classmethod
    def register_tuner(self, tuner_model_list, diffusion_model,
                       cond_stage_model):
        self.logger.info('Loading tuner model')
        if len(tuner_model_list) < 1:
            self.unregister_tuner(tuner_model_list, diffusion_model,
                                  cond_stage_model)
            return
        all_diffusion_tuner = {}
        all_cond_tuner = {}
        save_root_dir = '.cache_tuner'
        for tuner_model in tuner_model_list:
            tunner_model_folder = tuner_model.MODEL_PATH
            local_tuner_model = FS.get_dir_to_local_dir(tunner_model_folder)
            print('tunner_model_folder',tunner_model_folder)
            all_tuner_datas = os.listdir(local_tuner_model)
            cur_tuner_md5 = hashlib.md5(
                tunner_model_folder.encode('utf-8')).hexdigest()

            local_diffusion_cache = os.path.join(
                save_root_dir, cur_tuner_md5 + '_' + 'diffusion')
            local_cond_cache = os.path.join(save_root_dir,
                                            cur_tuner_md5 + '_' + 'cond')

            meta_file = os.path.join(save_root_dir,
                                     cur_tuner_md5 + '_meta.json')
            if not os.path.exists(meta_file):
                diffusion_tuner = {}
                cond_tuner = {}
                for sub in all_tuner_datas:
                    sub_file = os.path.join(local_tuner_model, sub)
                    config_file = os.path.join(sub_file, CONFIG_NAME)
                    safe_file = os.path.join(sub_file,
                                             SAFETENSORS_WEIGHTS_NAME)
                    bin_file = os.path.join(sub_file, WEIGHTS_NAME)
                    if os.path.isdir(sub_file) and os.path.isfile(config_file):
                        # diffusion or cond
                        cfg = json.load(open(config_file, 'r'))
                        if 'cond_stage_model.' in cfg['target_modules']:
                            cond_cfg = copy.deepcopy(cfg)
                            if 'cond_stage_model.*' in cond_cfg[
                                    'target_modules']:
                                cond_cfg['target_modules'] = cond_cfg[
                                    'target_modules'].replace(
                                        'cond_stage_model.*', '.*')
                            else:
                                cond_cfg['target_modules'] = cond_cfg[
                                    'target_modules'].replace(
                                        'cond_stage_model.', '')
                            if cond_cfg['target_modules'].startswith('*'):
                                cond_cfg['target_modules'] = '.' + cond_cfg[
                                    'target_modules']
                            os.makedirs(local_cond_cache + '_' + sub,
                                        exist_ok=True)
                            cond_tuner[os.path.basename(local_cond_cache) +
                                       '_' + sub] = hashlib.md5(
                                           (local_cond_cache + '_' +
                                            sub).encode('utf-8')).hexdigest()
                            os.makedirs(local_cond_cache + '_' + sub,
                                        exist_ok=True)

                            json.dump(
                                cond_cfg,
                                open(
                                    os.path.join(local_cond_cache + '_' + sub,
                                                 CONFIG_NAME), 'w'))
                        if 'model.' in cfg['target_modules'].replace(
                                'cond_stage_model.', ''):
                            diffusion_cfg = copy.deepcopy(cfg)
                            if 'model.*' in diffusion_cfg['target_modules']:
                                diffusion_cfg[
                                    'target_modules'] = diffusion_cfg[
                                        'target_modules'].replace(
                                            'model.*', '.*')
                            else:
                                diffusion_cfg[
                                    'target_modules'] = diffusion_cfg[
                                        'target_modules'].replace(
                                            'model.', '')
                            if diffusion_cfg['target_modules'].startswith('*'):
                                diffusion_cfg[
                                    'target_modules'] = '.' + diffusion_cfg[
                                        'target_modules']
                            os.makedirs(local_diffusion_cache + '_' + sub,
                                        exist_ok=True)
                            diffusion_tuner[
                                os.path.basename(local_diffusion_cache) + '_' +
                                sub] = hashlib.md5(
                                    (local_diffusion_cache + '_' +
                                     sub).encode('utf-8')).hexdigest()
                            json.dump(
                                diffusion_cfg,
                                open(
                                    os.path.join(
                                        local_diffusion_cache + '_' + sub,
                                        CONFIG_NAME), 'w'))

                        state_dict = {}
                        is_bin_file = True
                        if os.path.isfile(bin_file):
                            state_dict = torch.load(bin_file)
                        elif os.path.isfile(safe_file):
                            is_bin_file = False
                            from safetensors.torch import \
                                load_file as safe_load_file
                            state_dict = safe_load_file(
                                safe_file,
                                device='cuda'
                                if torch.cuda.is_available() else 'cpu')
                        save_diffusion_state_dict = {}
                        save_cond_state_dict = {}
                        for key, value in state_dict.items():
                            if key.startswith('model.'):
                                save_diffusion_state_dict[
                                    key[len('model.'):].replace(
                                        sub,
                                        os.path.basename(local_diffusion_cache)
                                        + '_' + sub)] = value
                            elif key.startswith('cond_stage_model.'):
                                save_cond_state_dict[
                                    key[len('cond_stage_model.'):].replace(
                                        sub,
                                        os.path.basename(local_cond_cache) +
                                        '_' + sub)] = value

                        if is_bin_file:
                            if len(save_diffusion_state_dict) > 0:
                                torch.save(
                                    save_diffusion_state_dict,
                                    os.path.join(
                                        local_diffusion_cache + '_' + sub,
                                        WEIGHTS_NAME))
                            if len(save_cond_state_dict) > 0:
                                torch.save(
                                    save_cond_state_dict,
                                    os.path.join(local_cond_cache + '_' + sub,
                                                 WEIGHTS_NAME))
                        else:
                            from safetensors.torch import \
                                save_file as safe_save_file
                            if len(save_diffusion_state_dict) > 0:
                                safe_save_file(
                                    save_diffusion_state_dict,
                                    os.path.join(
                                        local_diffusion_cache + '_' + sub,
                                        SAFETENSORS_WEIGHTS_NAME),
                                    metadata={'format': 'pt'})
                            if len(save_cond_state_dict) > 0:
                                safe_save_file(
                                    save_cond_state_dict,
                                    os.path.join(local_cond_cache + '_' + sub,
                                                 SAFETENSORS_WEIGHTS_NAME),
                                    metadata={'format': 'pt'})
                json.dump(
                    {
                        'diffusion_tuner': diffusion_tuner,
                        'cond_tuner': cond_tuner
                    }, open(meta_file, 'w'))
            else:
                meta_conf = json.load(open(meta_file, 'r'))
                diffusion_tuner = meta_conf['diffusion_tuner']
                cond_tuner = meta_conf['cond_tuner']
            all_diffusion_tuner.update(diffusion_tuner)
            all_cond_tuner.update(cond_tuner)
        if len(all_diffusion_tuner) > 0:

            diffusion_model['model'] = Swift.from_pretrained(
                diffusion_model['model'],
                save_root_dir,
                adapter_name=all_diffusion_tuner)
            diffusion_model['model'].set_active_adapters(
                list(all_diffusion_tuner.values()))
        if len(all_cond_tuner) > 0:
            cond_stage_model['model'] = Swift.from_pretrained(
                cond_stage_model['model'],
                save_root_dir,
                adapter_name=all_cond_tuner)
            cond_stage_model['model'].set_active_adapters(
                list(all_cond_tuner.values()))
        self.is_register = True
