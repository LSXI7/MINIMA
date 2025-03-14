# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os.path
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL.Image import Image
from tools.infrared.scepter.modules.model.network.diffusion.diffusion import GaussianDiffusion
from tools.infrared.scepter.modules.model.network.diffusion.schedules import noise_schedule
from tools.infrared.scepter.modules.model.registry import (BACKBONES, EMBEDDERS, MODELS,
                                            TOKENIZERS)
from tools.infrared.scepter.modules.utils.distribute import we
from tools.infrared.scepter.modules.utils.file_system import FS
from tools.infrared.scepter.studio.utils.env import get_available_memory

from .control_inference import ControlInference
from .tuner_inference import TunerInference


def get_model(model_tuple):
    assert 'model' in model_tuple
    return model_tuple['model']


class DiffusionInference():
    '''
        define vae, unet, text-encoder, tuner, refiner components
        support to load the components dynamicly.
        create and load model when run this model at the first time.
    '''
    def __init__(self, logger=None,device='cpu'):
        self.logger = logger
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model'
        ]
        self.tuner_infer = TunerInference(self.logger)
        self.control_infer = ControlInference(self.logger)
        self.device = device

    def init_from_cfg(self, cfg):
        self.name = cfg.NAME
        self.is_default = cfg.get('IS_DEFAULT', False)
        module_paras = self.load_default(cfg.get('DEFAULT_PARAS', None))
        assert cfg.have('MODEL')
        cfg.MODEL = self.redefine_paras(cfg.MODEL)
        self.diffusion = self.load_schedule(cfg.MODEL.SCHEDULE)
        self.diffusion_model = self.infer_model(
            cfg.MODEL.DIFFUSION_MODEL, module_paras.get(
                'DIFFUSION_MODEL',
                None)) if cfg.MODEL.have('DIFFUSION_MODEL') else None
        self.first_stage_model = self.infer_model(
            cfg.MODEL.FIRST_STAGE_MODEL,
            module_paras.get(
                'FIRST_STAGE_MODEL',
                None)) if cfg.MODEL.have('FIRST_STAGE_MODEL') else None
        self.cond_stage_model = self.infer_model(
            cfg.MODEL.COND_STAGE_MODEL,
            module_paras.get(
                'COND_STAGE_MODEL',
                None)) if cfg.MODEL.have('COND_STAGE_MODEL') else None
        self.refiner_cond_model = self.infer_model(
            cfg.MODEL.REFINER_COND_MODEL,
            module_paras.get(
                'REFINER_COND_MODEL',
                None)) if cfg.MODEL.have('REFINER_COND_MODEL') else None
        self.refiner_diffusion_model = self.infer_model(
            cfg.MODEL.REFINER_MODEL, module_paras.get(
                'REFINER_MODEL',
                None)) if cfg.MODEL.have('REFINER_MODEL') else None
        self.tokenizer = TOKENIZERS.build(
            cfg.MODEL.TOKENIZER,
            logger=self.logger) if cfg.MODEL.have('TOKENIZER') else None

        if self.tokenizer is not None:
            self.cond_stage_model['cfg'].KWARGS = {
                'vocab_size': self.tokenizer.vocab_size
            }

    def redefine_paras(self, cfg):
        if cfg.get('PRETRAINED_MODEL', None):
            assert FS.isfile(cfg.PRETRAINED_MODEL)
            with FS.get_from(cfg.PRETRAINED_MODEL,
                             wait_finish=True) as local_path:
                if local_path.endswith('safetensors'):
                    from safetensors.torch import load_file as load_safetensors
                    sd = load_safetensors(local_path)
                else:
                    sd = torch.load(local_path, map_location='cpu')
                first_stage_model_path = os.path.join(
                    os.path.dirname(local_path), 'first_stage_model.pth')
                cond_stage_model_path = os.path.join(
                    os.path.dirname(local_path), 'cond_stage_model.pth')
                diffusion_model_path = os.path.join(
                    os.path.dirname(local_path), 'diffusion_model.pth')
                if (not os.path.exists(first_stage_model_path)
                        or not os.path.exists(cond_stage_model_path)
                        or not os.path.exists(diffusion_model_path)):
                    self.logger.info(
                        'Now read the whole model and rearrange the modules, it may take several mins.'
                    )
                    first_stage_model = OrderedDict()
                    cond_stage_model = OrderedDict()
                    diffusion_model = OrderedDict()
                    for k, v in sd.items():
                        if k.startswith('first_stage_model.'):
                            first_stage_model[k.replace(
                                'first_stage_model.', '')] = v
                        elif k.startswith('conditioner.'):
                            cond_stage_model[k.replace('conditioner.', '')] = v
                        elif k.startswith('cond_stage_model.'):
                            if k.startswith('cond_stage_model.model.'):
                                cond_stage_model[k.replace(
                                    'cond_stage_model.model.', '')] = v
                            else:
                                cond_stage_model[k.replace(
                                    'cond_stage_model.', '')] = v
                        elif k.startswith('model.diffusion_model.'):
                            diffusion_model[k.replace('model.diffusion_model.',
                                                      '')] = v
                        else:
                            continue
                    if cfg.have('FIRST_STAGE_MODEL'):
                        with open(first_stage_model_path + 'cache', 'wb') as f:
                            torch.save(first_stage_model, f)
                        os.rename(first_stage_model_path + 'cache',
                                  first_stage_model_path)
                        self.logger.info(
                            'First stage model has been processed.')
                    if cfg.have('COND_STAGE_MODEL'):
                        with open(cond_stage_model_path + 'cache', 'wb') as f:
                            torch.save(cond_stage_model, f)
                        os.rename(cond_stage_model_path + 'cache',
                                  cond_stage_model_path)
                        self.logger.info(
                            'Cond stage model has been processed.')
                    if cfg.have('DIFFUSION_MODEL'):
                        with open(diffusion_model_path + 'cache', 'wb') as f:
                            torch.save(diffusion_model, f)
                        os.rename(diffusion_model_path + 'cache',
                                  diffusion_model_path)
                        self.logger.info('Diffusion model has been processed.')
                if not cfg.FIRST_STAGE_MODEL.get('PRETRAINED_MODEL', None):
                    cfg.FIRST_STAGE_MODEL.PRETRAINED_MODEL = first_stage_model_path
                else:
                    cfg.FIRST_STAGE_MODEL.RELOAD_MODEL = first_stage_model_path
                if not cfg.COND_STAGE_MODEL.get('PRETRAINED_MODEL', None):
                    cfg.COND_STAGE_MODEL.PRETRAINED_MODEL = cond_stage_model_path
                else:
                    cfg.COND_STAGE_MODEL.RELOAD_MODEL = cond_stage_model_path
                if not cfg.DIFFUSION_MODEL.get('PRETRAINED_MODEL', None):
                    cfg.DIFFUSION_MODEL.PRETRAINED_MODEL = diffusion_model_path
                else:
                    cfg.DIFFUSION_MODEL.RELOAD_MODEL = diffusion_model_path
        return cfg

    def init_from_modules(self, modules):
        for k, v in modules.items():
            self.__setattr__(k, v)

    def infer_model(self, cfg, module_paras=None):
        module = {
            'model': None,
            'cfg': cfg,
            'device': 'offline',
            'name': cfg.NAME,
            'function_info': {},
            'paras': {}
        }
        if module_paras is None:
            return module
        function_info = {}
        paras = {
            k.lower(): v
            for k, v in module_paras.get('PARAS', {}).items()
        }
        for function in module_paras.get('FUNCTION', []):
            input_dict = {}
            for inp in function.get('INPUT', []):
                if inp.lower() in self.input:
                    input_dict[inp.lower()] = self.input[inp.lower()]
            function_info[function.NAME] = {
                'dtype': function.get('DTYPE', 'float32'),
                'input': input_dict
            }
        module['paras'] = paras
        module['function_info'] = function_info
        return module

    def init_from_ckpt(self, path, model, ignore_keys=list()):
        if path.endswith('safetensors'):
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(path)
        else:
            sd = torch.load(path, map_location='cpu')

        new_sd = OrderedDict()
        for k, v in sd.items():
            ignored = False
            for ik in ignore_keys:
                if ik in k:
                    if we.rank == 0:
                        self.logger.info(
                            'Ignore key {} from state_dict.'.format(k))
                    ignored = True
                    break
            if not ignored:
                new_sd[k] = v

        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if we.rank == 0:
            self.logger.info(
                f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def load(self, module):
        if module['device'] == 'offline':
            if module['cfg'].NAME in MODELS.class_map:
                model = MODELS.build(module['cfg'], logger=self.logger).eval()
            elif module['cfg'].NAME in BACKBONES.class_map:
                model = BACKBONES.build(module['cfg'],
                                        logger=self.logger).eval()
            elif module['cfg'].NAME in EMBEDDERS.class_map:
                model = EMBEDDERS.build(module['cfg'],
                                        logger=self.logger).eval()
            else:
                raise NotImplementedError
            if module['cfg'].get('RELOAD_MODEL', None):
                self.init_from_ckpt(module['cfg'].RELOAD_MODEL, model)
            module['model'] = model
            module['device'] = 'cpu'
        if module['device'] == 'cpu':
            module['device'] = self.device
            module['model'] = module['model'].to(self.device)
        return module

    def unload(self, module):
        if module is None:
            return module
        mem = get_available_memory()
        free_mem = int(mem['available'] / (1024**2))
        total_mem = int(mem['total'] / (1024**2))
        if free_mem < 0.5 * total_mem:
            if module['model'] is not None:
                module['model'] = module['model'].to('cpu')
                del module['model']
            module['model'] = None
            module['device'] = 'offline'
            print('delete module')
        else:
            if module['model'] is not None:
                module['model'] = module['model'].to('cpu')
                module['device'] = 'cpu'
            else:
                module['device'] = 'offline'
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return module

    def dynamic_load(self, module=None, name=''):
        self.logger.info('Loading {} model'.format(name))
        if name == 'all':
            for subname in self.loaded_model_name:
                self.loaded_model[subname] = self.dynamic_load(
                    getattr(self, subname), subname)
        elif name in self.loaded_model_name:
            if name in self.loaded_model:
                if module['cfg'] != self.loaded_model[name]['cfg']:
                    self.unload(self.loaded_model[name])
                    module = self.load(module)
                    self.loaded_model[name] = module
                    return module
                elif module['device'] == 'cpu' or module['device'] == 'offline':
                    module = self.load(module)
                    return module
                else:
                    return module
            else:
                module = self.load(module)
                self.loaded_model[name] = module
                return module
        else:
            return self.load(module)

    def dynamic_unload(self, module=None, name='', skip_loaded=False):
        self.logger.info('Unloading {} model'.format(name))
        if name == 'all':
            for name, module in self.loaded_model.items():
                module = self.unload(self.loaded_model[name])
                self.loaded_model[name] = module
        elif name in self.loaded_model_name:
            if name in self.loaded_model:
                if not skip_loaded:
                    module = self.unload(self.loaded_model[name])
                    self.loaded_model[name] = module
            else:
                self.unload(module)
        else:
            self.unload(module)

    def load_default(self, cfg):
        module_paras = {}
        if cfg is not None:
            self.paras = cfg.PARAS
            self.input = {k.lower(): v for k, v in cfg.INPUT.items()}
            self.output = {k.lower(): v for k, v in cfg.OUTPUT.items()}
            module_paras = cfg.MODULES_PARAS
        return module_paras

    def load_schedule(self, cfg):
        parameterization = cfg.get('PARAMETERIZATION', 'eps')
        assert parameterization in [
            'eps', 'x0', 'v'
        ], 'currently only supporting "eps" and "x0" and "v"'
        num_timesteps = cfg.get('TIMESTEPS', 1000)

        schedule_args = {
            k.lower(): v
            for k, v in cfg.get('SCHEDULE_ARGS', {
                'NAME': 'logsnr_cosine_interp',
                'SCALE_MIN': 2.0,
                'SCALE_MAX': 4.0
            }).items()
        }

        zero_terminal_snr = cfg.get('ZERO_TERMINAL_SNR', False)
        if zero_terminal_snr:
            assert parameterization == 'v', 'Now zero_terminal_snr only support v-prediction mode.'
        sigmas = noise_schedule(schedule=schedule_args.pop('name'),
                                n=num_timesteps,
                                zero_terminal_snr=zero_terminal_snr,
                                **schedule_args)
        diffusion = GaussianDiffusion(sigmas=sigmas,
                                      prediction_type=parameterization)
        return diffusion

    def get_batch(self, value_dict, num_samples=1):
        batch = {}
        batch_uc = {}
        N = num_samples
        device = self.device
        for key in value_dict:
            if key == 'prompt':
                if not self.tokenizer:
                    batch['prompt'] = value_dict['prompt']
                    batch_uc['prompt'] = value_dict['negative_prompt']
                else:
                    batch['tokens'] = self.tokenizer(value_dict['prompt']).to(
                        self.device)
                    batch_uc['tokens'] = self.tokenizer(
                        value_dict['negative_prompt']).to(self.device)
            elif key == 'original_size_as_tuple':
                batch['original_size_as_tuple'] = (torch.tensor(
                    value_dict['original_size_as_tuple']).to(device).repeat(
                        N, 1))
            elif key == 'crop_coords_top_left':
                batch['crop_coords_top_left'] = (torch.tensor(
                    value_dict['crop_coords_top_left']).to(device).repeat(
                        N, 1))
            elif key == 'aesthetic_score':
                batch['aesthetic_score'] = (torch.tensor(
                    [value_dict['aesthetic_score']]).to(device).repeat(N, 1))
                batch_uc['aesthetic_score'] = (torch.tensor([
                    value_dict['negative_aesthetic_score']
                ]).to(device).repeat(N, 1))

            elif key == 'target_size_as_tuple':
                batch['target_size_as_tuple'] = (torch.tensor(
                    value_dict['target_size_as_tuple']).to(device).repeat(
                        N, 1))
            elif key == 'image':
                batch[key] = self.load_image(value_dict[key], num_samples=N)
            else:
                batch[key] = value_dict[key]

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc

    def load_image(self, image, num_samples=1):
        if isinstance(image, torch.Tensor):
            pass
        elif isinstance(image, Image):
            pass
        elif isinstance(image, Image):
            pass

    def get_function_info(self, module, function_name=None):
        all_function = module['function_info']
        if function_name in all_function:
            return function_name, all_function[function_name]['dtype']
        if function_name is None and len(all_function) == 1:
            for k, v in all_function.items():
                return k, v['dtype']

    def encode_first_stage(self, x, **kwargs):
        _, dtype = self.get_function_info(self.first_stage_model, 'encode')
        with torch.autocast('cuda',
                            enabled=dtype == 'float16',
                            dtype=getattr(torch, dtype)):
            z = get_model(self.first_stage_model).encode(x)
            return self.first_stage_model['paras']['scale_factor'] * z

    def decode_first_stage(self, z):
        _, dtype = self.get_function_info(self.first_stage_model, 'decode')
        with torch.autocast('cuda',
                            enabled=dtype == 'float16',
                            dtype=getattr(torch, dtype)):
            z = 1. / self.first_stage_model['paras']['scale_factor'] * z
            return get_model(self.first_stage_model).decode(z)

    @torch.no_grad()
    def __call__(self,
                 input,
                 num_samples=1,
                 intermediate_callback=None,
                 refine_strength=0,
                 img_to_img_strength=0,
                 cat_uc=True,
                 tuner_model=None,
                 control_model=None,
                 **kwargs):

        value_input = copy.deepcopy(self.input)
        value_input.update(input)
        print(value_input)
        height, width = value_input['target_size_as_tuple']
        value_output = copy.deepcopy(self.output)
        batch, batch_uc = self.get_batch(value_input, num_samples=1)

        # register tuner
        if tuner_model is not None and tuner_model != '' and len(
                tuner_model) > 0:
            if not isinstance(tuner_model, list):
                tuner_model = [tuner_model]
            self.dynamic_load(self.diffusion_model, 'diffusion_model')
            self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
            self.tuner_infer.register_tuner(tuner_model, self.diffusion_model,
                                            self.cond_stage_model)
            self.dynamic_unload(self.diffusion_model,
                                'diffusion_model',
                                skip_loaded=True)
            self.dynamic_unload(self.cond_stage_model,
                                'cond_stage_model',
                                skip_loaded=True)

        # register control
        if control_model is not None and control_model != '':
            self.dynamic_load(self.diffusion_model, 'diffusion_model')
            hints = ControlInference.get_control_input(
                control_model, kwargs.pop('control_cond_image', None), height,
                width)
            self.control_infer.register_controllers(control_model,
                                                    self.diffusion_model)
            self.dynamic_unload(self.diffusion_model,
                                'diffusion_model',
                                skip_loaded=True)
        else:
            hints = None

        # first stage encode
        image = input.pop('image', None)
        if image is not None and img_to_img_strength > 0:
            # run image2image
            b, c, ori_width, ori_height = image.shape
            if not (ori_width == width and ori_height == height):
                image = F.interpolate(image, (width, height), mode='bicubic')
            self.dynamic_load(self.first_stage_model, 'first_stage_model')
            input_latent = self.encode_first_stage(image)
            self.dynamic_unload(self.first_stage_model,
                                'first_stage_model',
                                skip_loaded=True)
        else:
            input_latent = None
        if 'input_latent' in value_output and input_latent is not None:
            value_output['input_latent'] = input_latent
        # cond stage
        self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
        function_name, dtype = self.get_function_info(self.cond_stage_model)
        with torch.autocast('cuda',
                            enabled=dtype == 'float16',
                            dtype=getattr(torch, dtype)):
            if self.tokenizer:
                if not hasattr(get_model(self.cond_stage_model), 'tokenizer'):
                    setattr(get_model(self.cond_stage_model), 'tokenizer',
                            self.tokenizer)
                context = getattr(get_model(self.cond_stage_model),
                                  function_name)(batch['tokens'])
                null_context = getattr(get_model(self.cond_stage_model),
                                       function_name)(batch_uc['tokens'])

            else:
                context = getattr(get_model(self.cond_stage_model),
                                  function_name)(batch)
                null_context = getattr(get_model(self.cond_stage_model),
                                       function_name)(batch_uc)
        self.dynamic_unload(self.cond_stage_model,
                            'cond_stage_model',
                            skip_loaded=True)

        if refine_strength > 0 and self.refiner_diffusion_model is not None:
            assert self.refiner_cond_model is not None
            self.refiner_cond_model = self.load(self.refiner_cond_model)
            function_name, dtype = self.get_function_info(
                self.refiner_cond_model)
            with torch.autocast('cuda',
                                enabled=dtype == 'float16',
                                dtype=getattr(torch, dtype)):
                if self.tokenizer:
                    refine_context = getattr(
                        get_model(self.refiner_cond_model),
                        function_name)(batch['tokens'])
                    refine_null_context = getattr(
                        get_model(self.refiner_cond_model),
                        function_name)(batch_uc['tokens'])
                else:
                    refine_context = getattr(
                        get_model(self.refiner_cond_model),
                        function_name)(batch)
                    refine_null_context = getattr(
                        get_model(self.refiner_cond_model),
                        function_name)(batch_uc)
            self.refiner_cond_model = self.unload(self.refiner_cond_model)

        # get noise
        seed = kwargs.pop('seed', -1)
        g = torch.Generator(device=self.device)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        if 'seed' in value_output:
            value_output['seed'] = seed
        for sample_id in range(num_samples):
            if self.diffusion_model is not None:
                noise = torch.empty(
                    1,
                    4,
                    height // self.first_stage_model['paras']['size_factor'],
                    width // self.first_stage_model['paras']['size_factor'],
                    device=self.device).normal_(generator=g)

                self.dynamic_load(self.diffusion_model, 'diffusion_model')
                # UNet use input n_prompt
                function_name, dtype = self.get_function_info(
                    self.diffusion_model)
                with torch.autocast('cuda',
                                    enabled=dtype == 'float16',
                                    dtype=getattr(torch, dtype)):
                    latent = self.diffusion.sample(
                        noise=noise,
                        x=input_latent,
                        denoising_strength=img_to_img_strength
                        if input_latent is not None else 1.0,
                        refine_strength=refine_strength,
                        solver=value_input.get('sample', 'ddim'),
                        model=get_model(self.diffusion_model),
                        model_kwargs=[{
                            'cond': context,
                            'hint': hints
                        }, {
                            'cond': null_context,
                            'hint': hints
                        }],
                        steps=value_input.get('sample_steps', 50),
                        guide_scale=value_input.get('guide_scale', 7.5),
                        guide_rescale=value_input.get('guide_rescale', 0.5),
                        discretization=value_input.get('discretization',
                                                       'trailing'),
                        show_progress=True,
                        seed=seed,
                        condition_fn=None,
                        clamp=None,
                        sharpness=value_input.get('sharpness', 0.0),
                        percentile=None,
                        t_max=None,
                        t_min=None,
                        discard_penultimate_step=None,
                        intermediate_callback=intermediate_callback,
                        cat_uc=value_input.get('cat_uc', cat_uc),
                        **kwargs)

                self.dynamic_unload(self.diffusion_model,
                                    'diffusion_model',
                                    skip_loaded=True)

            # apply refiner
            if refine_strength > 0 and self.refiner_diffusion_model is not None:
                assert self.refiner_diffusion_model is not None
                # decode intermidiet latent before refine
                self.first_stage_model = self.load(self.first_stage_model)
                before_refiner_samples = self.decode_first_stage(
                    latent).float()
                self.first_stage_model = self.unload(self.first_stage_model)

                before_refiner_samples = torch.clamp(
                    (before_refiner_samples + 1.0) / 2.0, min=0.0, max=1.0)
                if 'before_refine_images' in value_output:
                    if value_output['before_refine_images'] is None or (
                            isinstance(value_output['before_refine_images'],
                                       list)
                            and len(value_output['before_refine_images']) < 1):
                        value_output['before_refine_images'] = []
                    value_output['before_refine_images'].append(
                        before_refiner_samples)
                self.refiner_model = self.load(self.refiner_diffusion_model)
                function_name, dtype = self.get_function_info(
                    self.refiner_model)
                with torch.autocast('cuda',
                                    enabled=dtype == 'float16',
                                    dtype=getattr(torch, dtype)):
                    latent = self.diffusion.sample(
                        noise=noise,
                        x=latent,
                        denoising_strength=img_to_img_strength
                        if input_latent is not None else 1.0,
                        refine_strength=refine_strength,
                        refine_stage=True,
                        solver=value_input.get('refine_sample', 'ddim'),
                        model=get_model(self.refiner_model),
                        model_kwargs=[{
                            'cond': refine_context
                        }, {
                            'cond': refine_null_context
                        }],
                        steps=value_input.get('refine_sample_steps', 50),
                        guide_scale=value_input.get('refine_guide_scale', 7.5),
                        guide_rescale=value_input.get('refine_guide_rescale',
                                                      0.5),
                        discretization=value_input.get('refine_discretization',
                                                       'trailing'),
                        show_progress=True,
                        seed=seed,
                        condition_fn=None,
                        clamp=None,
                        percentile=None,
                        t_max=None,
                        t_min=None,
                        discard_penultimate_step=None,
                        return_intermediate=None,
                        intermediate_callback=intermediate_callback,
                        cat_uc=cat_uc,
                        **kwargs)
                self.refiner_model = self.unload(self.refiner_model)

            if 'latent' in value_output:
                if value_output['latent'] is None or (
                        isinstance(value_output['latent'], list)
                        and len(value_output['latent']) < 1):
                    value_output['latent'] = []
                value_output['latent'].append(latent)

            self.dynamic_load(self.first_stage_model, 'first_stage_model')
            x_samples = self.decode_first_stage(latent).float()
            self.dynamic_unload(self.first_stage_model,
                                'first_stage_model',
                                skip_loaded=True)
            images = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            if 'images' in value_output:
                if value_output['images'] is None or (
                        isinstance(value_output['images'], list)
                        and len(value_output['images']) < 1):
                    value_output['images'] = []
                value_output['images'].append(images)

        for k, v in value_output.items():
            if isinstance(v, list):
                value_output[k] = torch.cat(v, dim=0)
            if isinstance(v, torch.Tensor):
                value_output[k] = v.cpu()

        # unregister tuner
        if tuner_model is not None and tuner_model != '' and len(
                tuner_model) > 0:
            self.tuner_infer.unregister_tuner(tuner_model,
                                              self.diffusion_model,
                                              self.cond_stage_model)

        # unregister control
        if control_model is not None and control_model != '':
            self.control_infer.unregister_controllers(control_model,
                                                      self.diffusion_model)

        return value_output
