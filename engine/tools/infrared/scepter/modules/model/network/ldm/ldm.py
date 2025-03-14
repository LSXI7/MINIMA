# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers
import random
from collections import OrderedDict

import torch
from tools.infrared.scepter.modules.model.network.diffusion.diffusion import GaussianDiffusion
from tools.infrared.scepter.modules.model.network.diffusion.schedules import noise_schedule
from tools.infrared.scepter.modules.model.network.train_module import TrainModule
from tools.infrared.scepter.modules.model.registry import (BACKBONES, EMBEDDERS, LOSSES,
                                            MODELS, TOKENIZERS)
from tools.infrared.scepter.modules.model.utils.basic_utils import count_params, default
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.distribute import we
from tools.infrared.scepter.modules.utils.file_system import FS


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@MODELS.register_class()
class LatentDiffusion(TrainModule):
    para_dict = {
        'PARAMETERIZATION': {
            'value':
            'v',
            'description':
            "The prediction type, you can choose from 'eps' and 'x0' and 'v'",
        },
        'TIMESTEPS': {
            'value': 1000,
            'description': 'The schedule steps for diffusion.',
        },
        'SCHEDULE_ARGS': {},
        'MIN_SNR_GAMMA': {
            'value': None,
            'description': 'The minimum snr gamma, default is None.',
        },
        'ZERO_TERMINAL_SNR': {
            'value': False,
            'description': 'Whether zero terminal snr, default is False.',
        },
        'PRETRAINED_MODEL': {
            'value': None,
            'description': "Whole model's pretrained model path.",
        },
        'IGNORE_KEYS': {
            'value': [],
            'description': 'The ignore keys for pretrain model loaded.',
        },
        'SCALE_FACTOR': {
            'value': 0.18215,
            'description': 'The vae embeding scale.',
        },
        'SIZE_FACTOR': {
            'value': 8,
            'description': 'The vae size factor.',
        },
        'DEFAULT_N_PROMPT': {
            'value': '',
            'description': 'The default negtive prompt.',
        },
        'TRAIN_N_PROMPT': {
            'value': '',
            'description': 'The negtive prompt used in train phase.',
        },
        'P_ZERO': {
            'value': 0.0,
            'description': 'The prob for zero or negtive prompt.',
        },
        'USE_EMA': {
            'value': True,
            'description': 'Use Ema or not. Default True',
        },
        'DIFFUSION_MODEL': {},
        'DIFFUSION_MODEL_EMA': {},
        'FIRST_STAGE_MODEL': {},
        'COND_STAGE_MODEL': {},
        'TOKENIZER': {}
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.init_params()
        self.construct_network()

    def init_params(self):
        self.parameterization = self.cfg.get('PARAMETERIZATION', 'eps')
        assert self.parameterization in [
            'eps', 'x0', 'v'
        ], 'currently only supporting "eps" and "x0" and "v"'
        self.num_timesteps = self.cfg.get('TIMESTEPS', 1000)

        self.schedule_args = {
            k.lower(): v
            for k, v in self.cfg.get('SCHEDULE_ARGS', {
                'NAME': 'logsnr_cosine_interp',
                'SCALE_MIN': 2.0,
                'SCALE_MAX': 4.0
            }).items()
        }

        self.min_snr_gamma = self.cfg.get('MIN_SNR_GAMMA', None)

        self.zero_terminal_snr = self.cfg.get('ZERO_TERMINAL_SNR', False)
        if self.zero_terminal_snr:
            assert self.parameterization == 'v', 'Now zero_terminal_snr only support v-prediction mode.'

        self.sigmas = noise_schedule(schedule=self.schedule_args.pop('name'),
                                     n=self.num_timesteps,
                                     zero_terminal_snr=self.zero_terminal_snr,
                                     **self.schedule_args)

        self.diffusion = GaussianDiffusion(
            sigmas=self.sigmas, prediction_type=self.parameterization)

        self.pretrained_model = self.cfg.get('PRETRAINED_MODEL', None)
        self.ignore_keys = self.cfg.get('IGNORE_KEYS', [])

        self.model_config = self.cfg.DIFFUSION_MODEL
        self.first_stage_config = self.cfg.FIRST_STAGE_MODEL
        self.cond_stage_config = self.cfg.COND_STAGE_MODEL
        self.tokenizer_config = self.cfg.get('TOKENIZER', None)
        self.loss_config = self.cfg.get('LOSS', None)

        self.scale_factor = self.cfg.get('SCALE_FACTOR', 0.18215)
        self.size_factor = self.cfg.get('SIZE_FACTOR', 8)
        self.default_n_prompt = self.cfg.get('DEFAULT_N_PROMPT', '')
        self.default_n_prompt = '' if self.default_n_prompt is None else self.default_n_prompt
        self.p_zero = self.cfg.get('P_ZERO', 0.0)
        self.train_n_prompt = self.cfg.get('TRAIN_N_PROMPT', '')
        if self.default_n_prompt is None:
            self.default_n_prompt = ''
        if self.train_n_prompt is None:
            self.train_n_prompt = ''
        self.use_ema = self.cfg.get('USE_EMA', True)
        self.model_ema_config = self.cfg.get('DIFFUSION_MODEL_EMA', None)

    def construct_network(self):
        self.model = BACKBONES.build(self.model_config, logger=self.logger)
        self.logger.info('all parameters:{}'.format(count_params(self.model)))
        if self.use_ema and self.model_ema_config:
            self.model_ema = BACKBONES.build(self.model_ema_config,
                                             logger=self.logger)
            self.model_ema = self.model_ema.eval()
            for param in self.model_ema.parameters():
                param.requires_grad = False
        if self.loss_config:
            self.loss = LOSSES.build(self.loss_config, logger=self.logger)
        if self.tokenizer_config is not None:
            self.tokenizer = TOKENIZERS.build(self.tokenizer_config,
                                              logger=self.logger)

        self.first_stage_model = MODELS.build(self.first_stage_config,
                                              logger=self.logger)
        self.first_stage_model = self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        if self.tokenizer_config is not None:
            self.cond_stage_config.KWARGS = {
                'vocab_size': self.tokenizer.vocab_size
            }
        if self.cond_stage_config == '__is_unconditional__':
            print(
                f'Training {self.__class__.__name__} as an unconditional model.'
            )
            self.cond_stage_model = None
        else:
            model = EMBEDDERS.build(self.cond_stage_config, logger=self.logger)
            self.cond_stage_model = model.eval().requires_grad_(False)
            self.cond_stage_model.train = disabled_train

    def load_pretrained_model(self, pretrained_model):
        if pretrained_model is not None:
            with FS.get_from(pretrained_model,
                             wait_finish=True) as local_model:
                self.init_from_ckpt(local_model, ignore_keys=self.ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
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
                if k.startswith('model.diffusion_model.'):
                    k = k.replace('model.diffusion_model.', 'model.')
                k = k.replace('post_quant_conv',
                              'conv2') if 'post_quant_conv' in k else k
                k = k.replace('quant_conv',
                              'conv1') if 'quant_conv' in k else k
                new_sd[k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if we.rank == 0:
            self.logger.info(
                f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def encode_condition(self, input, method='encode_text'):
        if hasattr(self.cond_stage_model, method):
            return getattr(self.cond_stage_model,
                           method)(input, tokenizer=self.tokenizer)
        else:
            return self.cond_stage_model(input)

    def forward_train(self, image=None, noise=None, prompt=None, **kwargs):

        x_start = self.encode_first_stage(image, **kwargs)
        t = torch.randint(0,
                          self.num_timesteps, (x_start.shape[0], ),
                          device=x_start.device).long()
        context = {}
        if prompt and self.cond_stage_model:
            zeros = (torch.rand(len(prompt)) < self.p_zero).numpy().tolist()
            prompt = [
                self.train_n_prompt if zeros[idx] else p
                for idx, p in enumerate(prompt)
            ]
            self.register_probe({'after_prompt': prompt})
            with torch.autocast(device_type='cuda', enabled=False):
                context = self.encode_condition(
                    self.tokenizer(prompt).to(we.device_id))
        if 'hint' in kwargs and kwargs['hint'] is not None:
            hint = kwargs.pop('hint')
            if isinstance(context, dict):
                context['hint'] = hint
            else:
                context = {'crossattn': context, 'hint': hint}
        else:
            hint = None
        if self.min_snr_gamma is not None:
            alphas = self.diffusion.alphas.to(we.device_id)[t]
            sigmas = self.diffusion.sigmas.pow(2).to(we.device_id)[t]
            snrs = (alphas / sigmas).clamp(min=1e-20)
            min_snrs = snrs.clamp(max=self.min_snr_gamma)
            weights = min_snrs / snrs
        else:
            weights = 1
        self.register_probe({'snrs_weights': weights})

        loss = self.diffusion.loss(x0=x_start,
                                   t=t,
                                   model=self.model,
                                   model_kwargs={'cond': context},
                                   noise=noise,
                                   **kwargs)
        loss = loss * weights
        loss = loss.mean()
        ret = {'loss': loss, 'probe_data': {'prompt': prompt}}
        return ret

    def noise_sample(self, batch_size, h, w, g):
        noise = torch.empty(batch_size, 4, h, w,
                            device=we.device_id).normal_(generator=g)
        return noise

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @torch.no_grad()
    @torch.autocast('cuda', dtype=torch.float16)
    def forward_test(self,
                     prompt=None,
                     n_prompt=None,
                     sampler='ddim',
                     sample_steps=50,
                     seed=2023,
                     guide_scale=7.5,
                     guide_rescale=0.5,
                     discretization='trailing',
                     run_train_n=True,
                     **kwargs):
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        num_samples = len(prompt)
        if 'dynamic_encode_text' in kwargs and kwargs.pop(
                'dynamic_encode_text'):
            method = 'dynamic_encode_text'
        else:
            method = 'encode_text'

        n_prompt = default(n_prompt, [self.default_n_prompt] * len(prompt))
        assert isinstance(prompt, list) and \
               isinstance(n_prompt, list) and \
               len(prompt) == len(n_prompt)
        # with torch.autocast(device_type="cuda", enabled=False):
        context = self.encode_condition(self.tokenizer(prompt).to(
            we.device_id),
                                        method=method)
        null_context = self.encode_condition(self.tokenizer(n_prompt).to(
            we.device_id),
                                             method=method)
        if 'hint' in kwargs and kwargs['hint'] is not None:
            hint = kwargs.pop('hint')
            if isinstance(context, dict):
                context['hint'] = hint
            else:
                context = {'crossattn': context, 'hint': hint}
            if isinstance(null_context, dict):
                null_context['hint'] = hint
            else:
                null_context = {'crossattn': null_context, 'hint': hint}
        else:
            hint = None
        if 'index' in kwargs:
            kwargs.pop('index')
        image_size = None
        if 'meta' in kwargs:
            meta = kwargs.pop('meta')
            if 'image_size' in meta:
                h = int(meta['image_size'][0][0])
                w = int(meta['image_size'][1][0])
                image_size = [h, w]
        if 'image_size' in kwargs and kwargs['image_size'] is not None:
            image_size = kwargs.pop('image_size')
        if isinstance(image_size, numbers.Number):
            image_size = [image_size, image_size]
        if image_size is None:
            image_size = [1024, 1024]
        height, width = image_size
        noise = self.noise_sample(num_samples, height // self.size_factor,
                                  width // self.size_factor, g)
        # UNet use input n_prompt
        samples = self.diffusion.sample(solver=sampler,
                                        noise=noise,
                                        model=self.model,
                                        model_kwargs=[{
                                            'cond': context
                                        }, {
                                            'cond': null_context
                                        }],
                                        steps=sample_steps,
                                        guide_scale=guide_scale,
                                        guide_rescale=guide_rescale,
                                        discretization=discretization,
                                        show_progress=True,
                                        seed=seed,
                                        condition_fn=None,
                                        clamp=None,
                                        percentile=None,
                                        t_max=None,
                                        t_min=None,
                                        discard_penultimate_step=None,
                                        return_intermediate=None,
                                        **kwargs)
        x_samples = self.decode_first_stage(samples).float()
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        # UNet use train n_prompt
        if not self.default_n_prompt == self.train_n_prompt and run_train_n:
            train_n_prompt = [self.train_n_prompt] * len(prompt)
            null_train_context = self.encode_condition(
                self.tokenizer(train_n_prompt).to(we.device_id), method=method)

            tn_samples = self.diffusion.sample(solver=sampler,
                                               noise=noise,
                                               model=self.model,
                                               model_kwargs=[{
                                                   'cond': context
                                               }, {
                                                   'cond':
                                                   null_train_context
                                               }],
                                               steps=sample_steps,
                                               guide_scale=guide_scale,
                                               guide_rescale=guide_rescale,
                                               discretization=discretization,
                                               show_progress=we.rank == 0,
                                               seed=seed,
                                               condition_fn=None,
                                               clamp=None,
                                               percentile=None,
                                               t_max=None,
                                               t_min=None,
                                               discard_penultimate_step=None,
                                               return_intermediate=None,
                                               **kwargs)

            t_x_samples = self.decode_first_stage(tn_samples).float()

            t_x_samples = torch.clamp((t_x_samples + 1.0) / 2.0,
                                      min=0.0,
                                      max=1.0)
        else:
            train_n_prompt = ['' for _ in prompt]
            t_x_samples = [None for _ in prompt]

        outputs = list()
        for i, (p, np, tnp, img, t_img) in enumerate(
                zip(prompt, n_prompt, train_n_prompt, x_samples, t_x_samples)):
            one_tup = {'prompt': p, 'n_prompt': np, 'image': img}
            if hint is not None:
                one_tup.update({'hint': hint[i]})
            if t_img is not None:
                one_tup['train_n_prompt'] = tnp
                one_tup['train_n_image'] = t_img
            outputs.append(one_tup)

        return outputs

    @torch.no_grad()
    def log_images(self, image=None, prompt=None, n_prompt=None, **kwargs):
        results = self.forward_test(prompt=prompt, n_prompt=n_prompt, **kwargs)
        outputs = list()
        for img, res in zip(image, results):
            one_tup = {
                'orig': torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0),
                'recon': res['image'],
                'prompt': res['prompt'],
                'n_prompt': res['n_prompt']
            }
            if 'hint' in res:
                one_tup.update({'hint': res['hint']})
            if 'train_n_prompt' in res:
                one_tup['train_n_prompt'] = res['train_n_prompt']
                one_tup['train_n_image'] = res['train_n_image']
            outputs.append(one_tup)
        return outputs

    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        z = self.first_stage_model.encode(x)
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            LatentDiffusion.para_dict,
                            set_name=True)
