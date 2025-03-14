# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os
import warnings

import torch
import torch.nn as nn
import torchvision.transforms as TT
from PIL.Image import Image
from tools.infrared.scepter.modules.model.registry import TUNERS
from tools.infrared.scepter.modules.utils.config import Config
from tools.infrared.scepter.modules.utils.distribute import we
from tools.infrared.scepter.modules.utils.file_system import FS

try:
    from swift import SwiftModel
except Exception:
    warnings.warn('Import swift failed, please check it.')


class ControlInference():
    def __init__(self, logger=None):
        self.logger = logger
        self.is_register = False

    # @classmethod
    def unregister_controllers(self, control_model_ins, diffusion_model):
        self.logger.info('Unloading control model')
        if isinstance(diffusion_model['model'], SwiftModel):
            if (hasattr(diffusion_model['model'].base_model, 'control_blocks')
                    and diffusion_model['model'].base_model.control_blocks
                ):  # noqa
                del diffusion_model['model'].base_model.control_blocks
                diffusion_model['model'].base_model.control_blocks = None
                diffusion_model['model'].base_model.control_name = []
        else:
            del diffusion_model['model'].control_blocks
            diffusion_model['model'].control_blocks = None
            diffusion_model['model'].control_name = []
        self.is_register = False

    # @classmethod
    def register_controllers(self, control_model_ins, diffusion_model):
        self.logger.info('Loading control model')
        if control_model_ins is None or control_model_ins == '':
            self.unregister_controllers(control_model_ins, diffusion_model)
            return
        if not isinstance(control_model_ins, list):
            control_model_ins = [control_model_ins]
        control_model = nn.ModuleList([])
        control_model_folder = []
        for one_control in control_model_ins:
            one_control_model_folder = one_control.MODEL_PATH
            control_model_folder.append(one_control_model_folder)
            have_list = getattr(diffusion_model['model'], 'control_name', [])
            if one_control_model_folder in have_list:
                ind = have_list.index(one_control_model_folder)
                csc_tuners = copy.deepcopy(
                    diffusion_model['model'].control_blocks[ind])
            else:
                one_local_control_model = FS.get_dir_to_local_dir(
                    one_control_model_folder)
                control_cfg = Config(cfg_file=os.path.join(
                    one_local_control_model, '0_SwiftSCETuning',
                    'configuration.json'))
                assert hasattr(control_cfg, 'CONTROL_MODEL')
                control_cfg.CONTROL_MODEL[
                    'INPUT_BLOCK_CHANS'] = diffusion_model[
                        'model']._input_block_chans
                control_cfg.CONTROL_MODEL['INPUT_DOWN_FLAG'] = diffusion_model[
                    'model']._input_down_flag
                control_cfg.CONTROL_MODEL.PRETRAINED_MODEL = os.path.join(
                    one_local_control_model, '0_SwiftSCETuning',
                    'pytorch_model.bin')
                csc_tuners = TUNERS.build(control_cfg.CONTROL_MODEL,
                                          logger=self.logger)
            control_model.append(csc_tuners)

        control_model.to(diffusion_model['device'])
        if isinstance(diffusion_model['model'], SwiftModel):
            del diffusion_model['model'].base_model.control_blocks
            diffusion_model['model'].base_model.control_blocks = control_model
            diffusion_model[
                'model'].base_model.control_name = control_model_folder
        else:
            del diffusion_model['model'].control_blocks
            diffusion_model['model'].control_blocks = control_model
            diffusion_model['model'].control_name = control_model_folder
        self.is_register = True

    @classmethod
    def get_control_input(self, control_model, control_cond_image, height,
                          width):
        hints = []
        if control_cond_image is not None and control_model is not None:
            if not isinstance(control_model, list):
                control_model = [control_model]
            if not isinstance(control_cond_image, list):
                control_cond_image = [control_cond_image]
            assert len(control_cond_image) == len(control_model)
            for img in control_cond_image:
                if isinstance(img, Image):
                    w, h = img.size
                    if not h == height or not w == width:
                        img = TT.Resize(min(height, width))(img)
                        img = TT.CenterCrop((height, width))(img)
                    hint = TT.ToTensor()(img)
                    hints.append(hint)
                else:
                    raise NotImplementedError
        if len(hints) > 0:
            hints = torch.stack(hints).to(we.device_id)
        else:
            hints = None
        return hints
