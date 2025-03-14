# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import random


import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tools.infrared.scepter.modules.utils.distribute import we

from .control_inference import ControlInference
from .diffusion_inference import DiffusionInference
from .tuner_inference import TunerInference


def get_model(model_tuple):
    assert 'model' in model_tuple
    return model_tuple['model']


class StyleboothInference(DiffusionInference):
    '''
        define vae, unet, text-encoder, tuner, refiner components
        support to load the components dynamicly.
        create and load model when run this model at the first time.
    '''
    def __init__(self, logger=None,device='cuda'):
        self.logger = logger
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model'
        ]
        self.tuner_infer = TunerInference(self.logger)
        self.control_infer = ControlInference(self.logger)
        self.device=device

    def get_batch(self, value_dict, num_samples=1):
        batch = {}
        batch_uc = {}
        N = num_samples
        print(self.device)
        # device = we.device_id
        device = self.device
        for key in value_dict:
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
            elif key == 'image':
                batch[key] = self.load_image(value_dict[key], num_samples=N)
            else:
                batch[key] = value_dict[key]

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc

    def encode_condition(self, data, data2=None, type='text'):
        cond_stage_model = get_model(self.cond_stage_model)
        assert hasattr(self, 'tokenizer')
        with torch.autocast(device_type='cuda', enabled=False):
            if type == 'image' and (
                    hasattr(cond_stage_model, 'build_new_tokens')
                    and not hasattr(cond_stage_model, 'new_tokens_to_ids')):
                cond_stage_model.build_new_tokens(self.tokenizer)

            if type == 'text':
                # text = self.tokenizer(data).to(we.device_id)
                text = self.tokenizer(data).to(self.device)
                return cond_stage_model.encode_text(text)
            elif type == 'image':
                return cond_stage_model.encode_image(data)
            elif type == 'hybrid':
                # text = self.tokenizer(data).to(we.device_id)
                text = self.tokenizer(data).to(self.device)
                return cond_stage_model.encode_text(text, data2)

    def process_edit_image(self, images, height, width):
        if not isinstance(images, list):
            images = [images]
        tensors = []
        for img in images:
            w, h = img.size
            if not h == height or not w == width:
                scale = max(width / w, height / h)
                new_size = (int(h * scale), int(w * scale))
                img = TF.resize(img,
                                new_size,
                                interpolation=TF.InterpolationMode.BICUBIC)
                img = TF.center_crop(img, (height, width))
            # tensor = TF.to_tensor(img).to(we.device_id)
            tensor = TF.to_tensor(img).to(self.device)
            tensors.append(tensor)
        tensors = TF.normalize(torch.stack(tensors),
                               mean=[0.5, 0.5, 0.5],
                               std=[0.5, 0.5, 0.5])
        return tensors

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
                 stylebooth_state=True,
                 style_edit_image=None,
                 style_exemplar_image=None,
                 style_guide_scale_text=None,
                 style_guide_scale_image=None,
                 **kwargs):

        if not stylebooth_state:
            raise gr.Error('EDIT model must be used with StyleBooth settings')

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
            self.control_infer.register_controllers(control_model,
                                                    self.diffusion_model)
            self.dynamic_unload(self.diffusion_model,
                                'diffusion_model',
                                skip_loaded=True)

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
        context = {}
        if style_exemplar_image is not None:
            if not isinstance(style_exemplar_image, list):
                style_exemplar_image = [style_exemplar_image]
            style_exemplar_image = [
                TF.resize(x, (224, 224),
                          interpolation=TF.InterpolationMode.BICUBIC)
                for x in style_exemplar_image
            ]
            style_exemplar_image = [
                # TF.to_tensor(x).to(we.device_id) for x in style_exemplar_image
                TF.to_tensor(x).to(self.device) for x in style_exemplar_image
            ]
            style_exemplar_image = TF.normalize(
                torch.stack(style_exemplar_image),
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
            image_feature = self.encode_condition(style_exemplar_image,
                                                  type='image')
            context['crossattn'] = self.encode_condition(batch['prompt'],
                                                         image_feature,
                                                         type='hybrid')
        else:
            context['crossattn'] = self.encode_condition(batch['prompt'])
        null_context = {}
        null_context['crossattn'] = self.encode_condition(batch_uc['prompt'])

        self.dynamic_unload(self.cond_stage_model,
                            'cond_stage_model',
                            skip_loaded=True)

        model_kwargs = [{'cond': context}]

        # style first stage encode
        if style_edit_image is not None:
            style_edit_image = self.process_edit_image(style_edit_image,
                                                       height, width)
            self.dynamic_load(self.first_stage_model, 'first_stage_model')
            cond_concat = self.encode_first_stage(style_edit_image)
            cond_concat /= self.first_stage_model['paras']['scale_factor']
            self.dynamic_unload(self.first_stage_model,
                                'first_stage_model',
                                skip_loaded=True)

            context['concat'] = cond_concat
            null_context['concat'] = torch.zeros_like(cond_concat)
            mid_context = {}
            mid_context.update(null_context)
            mid_context.update({'concat': cond_concat})
            model_kwargs.append({'cond': mid_context})
            model_kwargs.append({'cond': null_context})

        # get noise
        seed = kwargs.pop('seed', -1)
        # g = torch.Generator(device=we.device_id)
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
                    # device=we.device_id).normal_(generator=g)
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
                        model_kwargs=model_kwargs,
                        steps=value_input.get('sample_steps', 50),
                        guide_scale={
                            'text': style_guide_scale_text,
                            'image': style_guide_scale_image
                        },
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
