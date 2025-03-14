# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tools.infrared.scepter.modules.model.network.ldm import LatentDiffusion
from tools.infrared.scepter.modules.model.registry import BACKBONES, MODELS
from tools.infrared.scepter.modules.model.utils.basic_utils import default
from tools.infrared.scepter.modules.utils.config import dict_to_yaml
from tools.infrared.scepter.modules.utils.distribute import we


@MODELS.register_class()
class LatentDiffusionXL(LatentDiffusion):
    para_dict = {
        'LOAD_REFINER': {
            'value': False,
            'description': 'Whether load REFINER or Not.'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.load_refiner = cfg.get('LOAD_REFINER', False)
        self.latent_cache_data = {}
        self.SURPPORT_RATIOS = {
            '0.5': (704, 1408),
            '0.52': (704, 1344),
            '0.57': (768, 1344),
            '0.6': (768, 1280),
            '0.68': (832, 1216),
            '0.72': (832, 1152),
            '0.78': (896, 1152),
            '0.82': (896, 1088),
            '0.88': (960, 1088),
            '0.94': (960, 1024),
            '1.0': (1024, 1024),
            '1.07': (1024, 960),
            '1.13': (1088, 960),
            '1.21': (1088, 896),
            '1.29': (1152, 896),
            '1.38': (1152, 832),
            '1.46': (1216, 832),
            '1.67': (1280, 768),
            '1.75': (1344, 768),
            '1.91': (1344, 704),
            '2.0': (1408, 704),
            '2.09': (1472, 704),
            '2.4': (1536, 640),
            '2.5': (1600, 640),
            '2.89': (1664, 576),
            '3.0': (1728, 576),
        }

    def construct_network(self):
        super().construct_network()
        self.refiner_cfg = self.cfg.get('REFINER_MODEL', None)
        self.refiner_cond_cfg = self.cfg.get('REFINER_COND_MODEL', None)
        if self.refiner_cfg and self.load_refiner:
            self.refiner_model = BACKBONES.build(self.refiner_cfg,
                                                 logger=self.logger)
            self.refiner_cond_model = BACKBONES.build(self.refiner_cond_cfg,
                                                      logger=self.logger)
        else:
            self.refiner_model = None
            self.refiner_cond_model = None
        self.input_keys = self.get_unique_embedder_keys_from_conditioner(
            self.cond_stage_model)
        if self.refiner_cond_model:
            self.input_refiner_keys = self.get_unique_embedder_keys_from_conditioner(
                self.refiner_cond_model)
        else:
            self.input_refiner_keys = []

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
                if k.startswith('conditioner.'):
                    k = k.replace('conditioner.', 'cond_stage_model.')
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

    def get_unique_embedder_keys_from_conditioner(self, conditioner):
        input_keys = []
        for x in conditioner.embedders:
            input_keys.extend(x.input_keys)
        return list(set(input_keys))

    def get_batch(self, keys, value_dict, num_samples=1):
        batch = {}
        batch_uc = {}
        N = num_samples
        device = we.device_id
        for key in keys:
            if key == 'prompt':
                batch['prompt'] = value_dict['prompt']
                batch_uc['prompt'] = value_dict['negative_prompt']
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
            else:
                batch[key] = value_dict[key]

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc

    def forward_train(self, image=None, noise=None, prompt=None, **kwargs):
        with torch.autocast('cuda', enabled=False):
            x_start = self.encode_first_stage(image, **kwargs)

        t = torch.randint(0,
                          self.num_timesteps, (x_start.shape[0], ),
                          device=x_start.device).long()

        if prompt and self.cond_stage_model:
            zeros = (torch.rand(len(prompt)) < self.p_zero).numpy().tolist()
            prompt = [
                self.train_n_prompt if zeros[idx] else p
                for idx, p in enumerate(prompt)
            ]
            self.register_probe({'after_prompt': prompt})
            batch = {'prompt': prompt}
            for key in self.input_keys:
                if key not in kwargs:
                    continue
                batch[key] = kwargs[key].to(we.device_id)
            context = getattr(self.cond_stage_model, 'encode')(batch)
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

    def check_valid_inputs(self, kwargs):
        batch_data = {}
        all_keys = set(self.input_keys + self.input_refiner_keys)
        for key in all_keys:
            if key in kwargs:
                batch_data[key] = kwargs.pop(key)
        return batch_data

    @torch.no_grad()
    def forward_test(self,
                     prompt=None,
                     n_prompt=None,
                     image=None,
                     sampler='ddim',
                     sample_steps=50,
                     seed=2023,
                     guide_scale=7.5,
                     guide_rescale=0.5,
                     discretization='trailing',
                     img_to_img_strength=0.0,
                     run_train_n=True,
                     refine_strength=0.0,
                     refine_sampler='ddim',
                     **kwargs):
        g = torch.Generator(device=we.device_id)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        g.manual_seed(seed)
        num_samples = len(prompt)
        n_prompt = default(n_prompt, [self.default_n_prompt] * len(prompt))
        assert isinstance(prompt, list) and \
               isinstance(n_prompt, list) and \
               len(prompt) == len(n_prompt)
        image_size = None
        if 'meta' in kwargs:
            meta = kwargs.pop('meta')
            if 'image_size' in meta:
                h = int(meta['image_size'][0][0])
                w = int(meta['image_size'][1][0])
                image_size = [h, w]
        if 'image_size' in kwargs:
            image_size = kwargs.pop('image_size')
        if image_size is None or isinstance(image_size, numbers.Number):
            image_size = [1024, 1024]
        pre_batch = self.check_valid_inputs(kwargs)
        if len(pre_batch) > 0:
            batch = {'prompt': prompt}
            batch.update(pre_batch)
            batch_uc = {'prompt': n_prompt}
            batch_uc.update(pre_batch)
        else:
            height, width = image_size
            if image is None:
                ori_width = width
                ori_height = height
            else:
                ori_height, ori_width = image.shape[-2:]

            value_dict = {
                'original_size_as_tuple': [ori_height, ori_width],
                'target_size_as_tuple': [height, width],
                'prompt': prompt,
                'negative_prompt': n_prompt,
                'crop_coords_top_left': [0, 0]
            }
            if refine_strength > 0:
                assert 'aesthetic_score' in kwargs and 'negative_aesthetic_score' in kwargs
                value_dict['aesthetic_score'] = kwargs.pop('aesthetic_score')
                value_dict['negative_aesthetic_score'] = kwargs.pop(
                    'negative_aesthetic_score')

            batch, batch_uc = self.get_batch(self.input_keys,
                                             value_dict,
                                             num_samples=num_samples)

        context = getattr(self.cond_stage_model, 'encode')(batch)
        null_context = getattr(self.cond_stage_model, 'encode')(batch_uc)

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
        height, width = batch['target_size_as_tuple'][0].cpu().numpy().tolist()
        noise = self.noise_sample(num_samples, height // self.size_factor,
                                  width // self.size_factor, g)
        if image is not None and img_to_img_strength > 0:
            # run image2image
            if not (ori_width == width and ori_height == height):
                image = F.interpolate(image, (height, width), mode='bicubic')
            with torch.autocast('cuda', enabled=False):
                z = self.encode_first_stage(image, **kwargs)
        else:
            z = None

        # UNet use input n_prompt
        samples = self.diffusion.sample(
            noise=noise,
            x=z,
            denoising_strength=img_to_img_strength if z is not None else 1.0,
            refine_strength=refine_strength,
            solver=sampler,
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

        # apply refiner
        if refine_strength > 0:
            assert self.refiner_model is not None
            assert self.refiner_cond_model is not None
            with torch.autocast('cuda', enabled=False):
                before_refiner_samples = self.decode_first_stage(
                    samples).float()
            before_refiner_samples = torch.clamp(
                (before_refiner_samples + 1.0) / 2.0, min=0.0, max=1.0)

            if len(pre_batch) > 0:
                batch = {'prompt': prompt}
                batch.update(pre_batch)
                batch_uc = {'prompt': n_prompt}
                batch_uc.update(pre_batch)
            else:
                batch, batch_uc = self.get_batch(self.input_refiner_keys,
                                                 value_dict,
                                                 num_samples=num_samples)

            context = getattr(self.refiner_cond_model, 'encode')(batch)
            null_context = getattr(self.refiner_cond_model, 'encode')(batch_uc)

            samples = self.diffusion.sample(
                noise=noise,
                x=samples,
                denoising_strength=img_to_img_strength
                if z is not None else 1.0,
                refine_strength=refine_strength,
                refine_stage=True,
                solver=sampler,
                model=self.refiner_model,
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
        else:
            before_refiner_samples = [None for _ in prompt]

        with torch.autocast('cuda', enabled=False):
            x_samples = self.decode_first_stage(samples).float()
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        # UNet use train n_prompt
        if not self.default_n_prompt == self.train_n_prompt and run_train_n:
            train_n_prompt = [self.train_n_prompt] * len(prompt)
            if len(pre_batch) > 0:
                pre_batch = {'prompt': prompt}
                batch.update(pre_batch)
                batch_uc = {'prompt': train_n_prompt}
                batch_uc.update(pre_batch)
            else:
                value_dict['negative_prompt'] = train_n_prompt
                batch, batch_uc = self.get_batch(self.input_keys,
                                                 value_dict,
                                                 num_samples=num_samples)

            context = getattr(self.cond_stage_model, 'encode')(batch)
            null_context = getattr(self.cond_stage_model, 'encode')(batch_uc)

            tn_samples = self.diffusion.sample(
                noise=noise,
                x=z,
                denoising_strength=img_to_img_strength
                if z is not None else 1.0,
                refine_strength=refine_strength,
                solver=sampler,
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

            if refine_strength > 0:
                assert self.refiner_model is not None
                assert self.refiner_cond_model is not None
                with torch.autocast('cuda', enabled=False):
                    before_refiner_t_samples = self.decode_first_stage(
                        samples).float()
                before_refiner_t_samples = torch.clamp(
                    (before_refiner_t_samples + 1.0) / 2.0, min=0.0, max=1.0)

                if len(pre_batch) > 0:
                    pre_batch = {'prompt': prompt}
                    batch.update(pre_batch)
                    batch_uc = {'prompt': train_n_prompt}
                    batch_uc.update(pre_batch)
                else:
                    batch, batch_uc = self.get_batch(self.input_refiner_keys,
                                                     value_dict,
                                                     num_samples=num_samples)

                context = getattr(self.refiner_cond_model, 'encode')(batch)
                null_context = getattr(self.refiner_cond_model,
                                       'encode')(batch_uc)
                tn_samples = self.diffusion.sample(
                    noise=noise,
                    x=tn_samples,
                    denoising_strength=img_to_img_strength
                    if z is not None else 1.0,
                    refine_strength=refine_strength,
                    refine_stage=True,
                    solver=sampler,
                    model=self.refiner_model,
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
            else:
                before_refiner_t_samples = [None for _ in prompt]

            t_x_samples = self.decode_first_stage(tn_samples).float()

            t_x_samples = torch.clamp((t_x_samples + 1.0) / 2.0,
                                      min=0.0,
                                      max=1.0)
        else:
            train_n_prompt = ['' for _ in prompt]
            t_x_samples = [None for _ in prompt]
            before_refiner_t_samples = [None for _ in prompt]

        outputs = list()
        for i, (p, np, tnp, img, r_img, t_img, r_t_img) in enumerate(
                zip(prompt, n_prompt, train_n_prompt, x_samples,
                    before_refiner_samples, t_x_samples,
                    before_refiner_t_samples)):
            one_tup = {
                'prompt': p,
                'n_prompt': np,
                'image': img,
                'before_refiner_image': r_img
            }
            if hint is not None:
                one_tup.update({'hint': hint[i]})
            if t_img is not None:
                one_tup['train_n_prompt'] = tnp
                one_tup['train_n_image'] = t_img
                one_tup['train_n_before_refiner_image'] = r_t_img
            outputs.append(one_tup)

        return outputs

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            LatentDiffusionXL.para_dict,
                            set_name=True)
