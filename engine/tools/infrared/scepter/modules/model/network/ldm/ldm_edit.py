# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import numbers
import random

import numpy as np
import torch
from tools.infrared.scepter.modules.model.network.ldm.ldm import LatentDiffusion
from tools.infrared.scepter.modules.model.registry import MODELS
from tools.infrared.scepter.modules.model.utils.basic_utils import default
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.distribute import we


@MODELS.register_class()
class LatentDiffusionEdit(LatentDiffusion):
    para_dict = {
        'CONCAT_NO_SCALE_FACTOR': {
            'value': False,
            'description': 'Whether concat input scaled after VAE.'
        },
        'I_ZERO': {
            'value': 0.0,
            'description': 'P-zero of concat image.'
        },
    }
    para_dict.update(LatentDiffusion.para_dict)

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger=logger)
        self.concat_no_scale_factor = self.cfg.get('CONCAT_NO_SCALE_FACTOR',
                                                   False)
        self.i_zero = self.cfg.get('I_ZERO', 0.0)
        # # overwrite the original diffusion
        # self.diffusion = GaussianDiffusion(
        #     sigmas=self.sigmas, prediction_type=self.parameterization)

    def forward_train(self, image=None, noise=None, prompt=None, **kwargs):
        condition_attn = None
        if 'condition_attn' in kwargs and kwargs['condition_attn'] is not None:
            condition_attn = kwargs.pop('condition_attn')
            if np.random.uniform() < self.i_zero:
                condition_attn = torch.zeros_like(condition_attn)

        condition_cat = None
        if 'condition_cat' in kwargs and kwargs['condition_cat'] is not None:
            condition_cat = kwargs.pop('condition_cat')
            if np.random.uniform() < self.i_zero:
                condition_cat = torch.zeros_like(condition_cat)

        ###############################
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
                if condition_attn is None:
                    context['crossattn'] = self.encode_condition(prompt)
                else:
                    image_feature = self.encode_condition(condition_attn,
                                                          type='image')
                    context['crossattn'] = self.encode_condition(prompt,
                                                                 image_feature,
                                                                 type='hybrid')
            if isinstance(context['crossattn'], dict):
                attn = context.pop('crossattn')
                context.update(attn)

        if condition_cat is not None:
            cat = self.encode_first_stage(condition_cat, **kwargs)
            if self.concat_no_scale_factor:
                cat /= self.scale_factor
            context['concat'] = cat

        if 'hint' in kwargs and kwargs['hint'] is not None:
            hint = kwargs.pop('hint')
            context['hint'] = hint

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
        # ret = {'loss': loss, 'probe_data': {'prompt': prompt, 'concat': condition_cat, 'attn': condition_attn}}
        return ret

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
        condition_attn = None
        if 'condition_attn' in kwargs and kwargs['condition_attn'] is not None:
            condition_attn = kwargs.pop('condition_attn')

        condition_cat = None
        if 'condition_cat' in kwargs and kwargs['condition_cat'] is not None:
            condition_cat = kwargs.pop('condition_cat')

        image = None
        if 'image' in kwargs and kwargs['image'] is not None:
            image = kwargs.pop('image')

        ###############################
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        # torch.manual_seed(seed)
        num_samples = len(prompt)

        # if 'dynamic_encode_text' in kwargs and kwargs.pop(
        #         'dynamic_encode_text'):
        #     method = 'dynamic_encode_text'
        # else:
        #     method = 'encode_text'

        n_prompt = default(n_prompt, [self.default_n_prompt] * len(prompt))
        assert isinstance(prompt, list) and \
               isinstance(n_prompt, list) and \
               len(prompt) == len(n_prompt)

        context = {}
        with torch.autocast(device_type='cuda', enabled=False):
            if condition_attn is None:
                context['crossattn'] = self.encode_condition(prompt)
            else:
                image_feature = self.encode_condition(condition_attn,
                                                      type='image')
                context['crossattn'] = self.encode_condition(prompt,
                                                             image_feature,
                                                             type='hybrid')
            if isinstance(context['crossattn'], dict):
                attn = context.pop('crossattn')
                context.update(attn)

        null_context = {}
        null_context['crossattn'] = self.encode_condition(n_prompt)
        if isinstance(null_context['crossattn'], dict):
            attn = null_context.pop('crossattn')
            null_context.update(attn)

        if 'hint' in kwargs and kwargs['hint'] is not None:
            hint = kwargs.pop('hint')
            context['hint'] = hint
            null_context['hint'] = hint
        else:
            hint = None

        model_kwargs = [{'cond': context}]
        if condition_cat is not None:
            cat = self.encode_first_stage(condition_cat, **kwargs)
            if self.concat_no_scale_factor:
                cat /= self.scale_factor
            context['concat'] = cat
            null_context['concat'] = torch.zeros_like(cat)

            mid_context = {}
            mid_context.update(null_context)
            mid_context.update({'concat': cat})
            model_kwargs.append({'cond': mid_context})
            model_kwargs.append({'cond': null_context})

        if 'index' in kwargs:
            kwargs.pop('index')
        image_size = None

        if 'meta' in kwargs:
            meta = kwargs.pop('meta')
            if 'image_size' in meta:
                h = int(meta['image_size'][0][0])
                w = int(meta['image_size'][1][0])
                image_size = [h, w]
        if 'image_size' in kwargs:
            image_size = kwargs.pop('image_size')
        if condition_cat is not None:
            image_size = condition_cat.shape[2:4]
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
                                        model_kwargs=model_kwargs,
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
        if image is not None:
            x_samples = torch.cat([image, x_samples], dim=-1)
        if condition_attn is not None:
            condition_attn = torch.clamp((condition_attn + 1.8) / 4,
                                         min=0.0,
                                         max=1.0)
            x_samples = torch.cat([condition_attn, x_samples], dim=-1)
        if condition_cat is not None:
            condition_cat = torch.clamp((condition_cat + 1.0) / 2.0,
                                        min=0.0,
                                        max=1.0)
            x_samples = torch.cat([condition_cat, x_samples], dim=-1)
        # UNet use train n_prompt
        # deleted!
        train_n_prompt = ['' for _ in prompt]
        t_x_samples = [None for _ in prompt]

        outputs = list()
        for i, (p, n_p, tnp, img, t_img) in enumerate(
                zip(prompt, n_prompt, train_n_prompt, x_samples, t_x_samples)):
            one_tup = {'prompt': p, 'n_prompt': n_p, 'image': img}

            if hint is not None:
                one_tup.update({'hint': hint[i]})
            if t_img is not None:
                one_tup['train_n_prompt'] = tnp
                one_tup['train_n_image'] = t_img
            outputs.append(one_tup)

        return outputs

    def encode_condition(self, data, data2=None, type='text'):
        assert hasattr(self, 'tokenizer')
        if type == 'image' and (
                hasattr(self.cond_stage_model, 'build_new_tokens')
                and not hasattr(self.cond_stage_model, 'new_tokens_to_ids')):
            self.cond_stage_model.build_new_tokens(self.tokenizer)

        if type == 'text':
            text = self.tokenizer(data).to(we.device_id)
            return self.cond_stage_model.encode_text(text)
        elif type == 'image':
            return self.cond_stage_model.encode_image(data)
        elif type == 'hybrid':
            text = self.tokenizer(data).to(we.device_id)
            return self.cond_stage_model.encode_text(text, data2)

    def save_pretrained(self,
                        *args,
                        destination=None,
                        prefix='',
                        keep_vars=False):
        return self.model.state_dict(*args,
                                     destination=destination,
                                     keep_vars=keep_vars)

    def save_pretrained_config(self):
        return copy.deepcopy(self.cfg.COND_STAGE_MODEL.cfg_dict)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            LatentDiffusionEdit.para_dict,
                            set_name=True)
