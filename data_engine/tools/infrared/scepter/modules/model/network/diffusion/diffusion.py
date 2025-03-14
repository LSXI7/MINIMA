# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
GaussianDiffusion wraps operators for denoising diffusion models, including the
diffusion and denoising processes, as well as the loss evaluation.
"""
import copy
import random

import torch

from .schedules import karras_schedule
from .solvers import (
    sample_ddim, sample_dpm_2, sample_dpm_2_ancestral, sample_dpmpp_2m,
    sample_dpmpp_2m_sde, sample_dpmpp_2m_sde_lcm, sample_dpmpp_2s_ancestral,
    sample_dpmpp_sde, sample_euler, sample_euler_ancestral, sample_heun,
    sample_onestep, stochastic_iterative_sampler,
    stochastic_iterative_sampler2, stochastic_iterative_sampler3)

__all__ = ['GaussianDiffusion']


def _i(tensor, t, x):
    """
    Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).to(x.device)


def _unpack_2d_ks(kernel_size):
    if isinstance(kernel_size, int):
        ky = kx = kernel_size
    else:
        assert len(
            kernel_size) == 2, '2D Kernel size should have a length of 2.'
        ky, kx = kernel_size

    ky = int(ky)
    kx = int(kx)
    return ky, kx


def _compute_zero_padding(kernel_size):
    ky, kx = _unpack_2d_ks(kernel_size)
    return (ky - 1) // 2, (kx - 1) // 2


def _bilateral_blur(
    input,
    guidance,
    kernel_size,
    sigma_color,
    sigma_space,
    border_type='reflect',
    color_distance_type='l1',
):

    if isinstance(sigma_color, torch.Tensor):
        sigma_color = sigma_color.to(device=input.device,
                                     dtype=input.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = torch.nn.functional.pad(input, (pad_x, pad_x, pad_y, pad_y),
                                           mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(
        -2)  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = torch.nn.functional.pad(guidance,
                                                  (pad_x, pad_x, pad_y, pad_y),
                                                  mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(
            3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == 'l1':
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == 'l2':
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError('color_distance_type only acceps l1 or l2')
    color_kernel = (-0.5 / sigma_color**2 *
                    color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size,
                                         sigma_space,
                                         device=input.device,
                                         dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    return out


def get_gaussian_kernel1d(
    kernel_size,
    sigma,
    force_even,
    *,
    device=None,
    dtype=None,
):

    return gaussian(kernel_size, sigma, device=device, dtype=dtype)


def gaussian(window_size, sigma, *, device=None, dtype=None):

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) -
         window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def get_gaussian_kernel2d(
    kernel_size,
    sigma,
    force_even=False,
    *,
    device=None,
    dtype=None,
):

    sigma = torch.Tensor([[sigma, sigma]]).to(device=device, dtype=dtype)

    ksize_y, ksize_x = _unpack_2d_ks(kernel_size)
    sigma_y, sigma_x = sigma[:, 0, None], sigma[:, 1, None]

    kernel_y = get_gaussian_kernel1d(ksize_y,
                                     sigma_y,
                                     force_even,
                                     device=device,
                                     dtype=dtype)[..., None]
    kernel_x = get_gaussian_kernel1d(ksize_x,
                                     sigma_x,
                                     force_even,
                                     device=device,
                                     dtype=dtype)[..., None]

    return kernel_y * kernel_x.view(-1, 1, ksize_x)


def adaptive_anisotropic_filter(x, g=None):
    if g is None:
        g = x
    s, m = torch.std_mean(g, dim=(1, 2, 3), keepdim=True)
    s = s + 1e-5
    guidance = (g - m) / s
    y = _bilateral_blur(x,
                        guidance,
                        kernel_size=(13, 13),
                        sigma_color=3.0,
                        sigma_space=3.0,
                        border_type='reflect',
                        color_distance_type='l1')
    return y


class GaussianDiffusion(object):
    def __init__(self, sigmas, prediction_type='eps'):
        assert prediction_type in {'x0', 'eps', 'v'}
        self.sigmas = sigmas  # noise coefficients
        self.alphas = torch.sqrt(1 - sigmas**2)  # signal coefficients
        self.num_timesteps = len(sigmas)
        self.prediction_type = prediction_type

    def diffuse(self, x0, t, noise=None):
        """
        Add Gaussian noise to signal x0 according to:
        q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I).
        """
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt

    def denoise(self,
                xt,
                t,
                s,
                model,
                model_kwargs={},
                guide_scale=None,
                guide_rescale=None,
                clamp=None,
                sharpness=0.0,
                percentile=None,
                cat_uc=False,
                **kwargs):
        """
        Apply one step of denoising from the posterior distribution q(x_s | x_t, x0).
        Since x0 is not available, estimate the denoising results using the learned
        distribution p(x_s | x_t, \hat{x}_0 == f(x_t)). # noqa
        """
        s = t - 1 if s is None else s

        # hyperparams
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.
        sigmas_s = torch.sqrt(1 - alphas_s**2)

        # precompute variables
        betas = 1 - (alphas / alphas_s)**2
        coef1 = betas * alphas_s / sigmas**2
        coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var = betas * (sigmas_s / sigmas)**2
        log_var = torch.log(var).clamp_(-20, 20)

        # prediction
        if guide_scale is None:
            if isinstance(model_kwargs, dict):
                out = model(xt, t=t, **model_kwargs, **kwargs)
            elif isinstance(model_kwargs, list) and len(model_kwargs) > 0:
                out = model(xt, t=t, **model_kwargs[0], **kwargs)
            else:
                raise Exception('Error')
        else:
            # classifier-free guidance (arXiv:2207.12598)
            # model_kwargs[0]: conditional kwargs
            # model_kwargs[1]: non-conditional kwargs
            assert isinstance(model_kwargs, list) and len(model_kwargs) >= 2
            if isinstance(guide_scale, float) or isinstance(guide_scale, int):
                assert len(model_kwargs) == 2
                if guide_scale == 1.:
                    out = model(xt, t=t, **model_kwargs[0], **kwargs)
                else:
                    if cat_uc:

                        def parse_model_kwargs(prev_value, value):
                            if isinstance(value, torch.Tensor):
                                prev_value = torch.cat([prev_value, value],
                                                       dim=0)
                            elif isinstance(value, dict):
                                for k, v in value.items():
                                    prev_value[k] = parse_model_kwargs(
                                        prev_value[k], v)
                            elif isinstance(value, list):
                                for idx, v in enumerate(value):
                                    prev_value[idx] = parse_model_kwargs(
                                        prev_value[idx], v)
                            return prev_value

                        all_model_kwargs = copy.deepcopy(model_kwargs[0])
                        for model_kwarg in model_kwargs[1:]:
                            for key, value in model_kwarg.items():
                                all_model_kwargs[key] = parse_model_kwargs(
                                    all_model_kwargs[key], value)
                        all_out = model(xt.repeat(2, 1, 1, 1),
                                        t=t.repeat(2),
                                        **all_model_kwargs,
                                        **kwargs)
                        y_out, u_out = all_out.chunk(2)
                    else:
                        y_out = model(xt, t=t, **model_kwargs[0], **kwargs)
                        u_out = model(xt, t=t, **model_kwargs[1], **kwargs)
                    # todo sharpness
                    # sharpness sampling
                    if sharpness is not None and sharpness > 0:
                        positive_x0 = alphas * xt - sigmas * y_out
                        negative_x0 = alphas * xt - sigmas * u_out

                        positive_eps = xt - positive_x0
                        negative_eps = xt - negative_x0

                        global_diffusion_progress = (
                            1 - t / 999.0).detach().cpu().numpy().tolist()[0]
                        alpha = 0.001 * sharpness * global_diffusion_progress
                        positive_eps_degraded = adaptive_anisotropic_filter(
                            x=positive_eps, g=positive_x0)
                        positive_eps_degraded_weighted = positive_eps_degraded * alpha + positive_eps * (
                            1.0 - alpha)

                        final_eps = negative_eps + guide_scale * (
                            positive_eps_degraded_weighted - negative_eps)
                        final_x0 = xt - final_eps
                        out = (alphas * xt - final_x0) / sigmas
                    else:
                        out = u_out + guide_scale * (y_out - u_out)
            elif isinstance(guide_scale, dict):
                assert len(model_kwargs) == 3
                y_out = model(xt, t=t, **model_kwargs[0], **kwargs)
                m_out = model(xt, t=t, **model_kwargs[1], **kwargs)
                u_out = model(xt, t=t, **model_kwargs[2], **kwargs)
                out = u_out + guide_scale['image'] * (
                    m_out - u_out) + guide_scale['text'] * (y_out - m_out)
            elif isinstance(guide_scale, list):
                assert len(guide_scale) == len(model_kwargs) - 1
                y_out = model(xt, t=t, **model_kwargs[0], **kwargs)
                outs = [y_out]
                for i in range(1, len(model_kwargs)):
                    outs.append(model(xt, t=t, **model_kwargs[i], **kwargs))
                out = outs[-1]
                for i in range(len(guide_scale)):
                    out += guide_scale[i] * (outs[-i - 2] - outs[-i - 1])

            # rescale the output according to arXiv:2305.08891
            if guide_rescale is not None and guide_rescale > 0.0:
                assert guide_rescale >= 0 and guide_rescale <= 1
                ratio = (
                    y_out.flatten(1).std(dim=1) /
                    (out.flatten(1).std(dim=1) + 1e-12)).view((-1, ) + (1, ) *
                                                              (y_out.ndim - 1))
                out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0
        # compute x0
        if self.prediction_type == 'x0':
            x0 = out
        elif self.prediction_type == 'eps':
            x0 = (xt - sigmas * out) / alphas
        elif self.prediction_type == 'v':
            x0 = alphas * xt - sigmas * out
        else:
            raise NotImplementedError(
                f'prediction_type {self.prediction_type} not implemented')

        # restrict the range of x0
        if percentile is not None:
            # NOTE: percentile should only be used when data is within range [-1, 1]
            assert percentile > 0 and percentile <= 1
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1, ) + (1, ) * (xt.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        # recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps

    def loss(self,
             x0,
             t,
             model,
             model_kwargs={},
             reduction='mean',
             noise=None,
             **kwargs):
        # hyperparams
        sigmas = _i(self.sigmas, t, x0)
        alphas = _i(self.alphas, t, x0)

        # diffuse and denoise
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.diffuse(x0, t, noise)
        out = model(xt, t=t, **model_kwargs, **kwargs)

        # mse loss
        target = {
            'eps': noise,
            'x0': x0,
            'v': alphas * noise - sigmas * x0
        }[self.prediction_type]
        loss = (out - target).pow(2)
        if reduction == 'mean':
            loss = loss.flatten(1).mean(dim=1)
        return loss

    @torch.no_grad()
    def sample(self,
               noise,
               model,
               x=None,
               denoising_strength=1.0,
               refine_stage=False,
               refine_strength=0.0,
               model_kwargs={},
               condition_fn=None,
               guide_scale=None,
               guide_rescale=None,
               clamp=None,
               sharpness=0.0,
               percentile=None,
               solver='euler_a',
               steps=20,
               t_max=None,
               t_min=None,
               discretization=None,
               discard_penultimate_step=None,
               return_intermediate=None,
               show_progress=False,
               seed=-1,
               intermediate_callback=None,
               cat_uc=False,
               add_noise=False,
               free_steps=None,
               step_offset=None,
               **kwargs):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, 'leading', 'linspace', 'trailing',
                                  'free')
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, 'x0', 'xt')

        # function of diffusion solver
        solver_fn = {
            'ddim': sample_ddim,
            'euler_ancestral': sample_euler_ancestral,
            'euler': sample_euler,
            'heun': sample_heun,
            'dpm2': sample_dpm_2,
            'dpm2_ancestral': sample_dpm_2_ancestral,
            'dpmpp_2s_ancestral': sample_dpmpp_2s_ancestral,
            'dpmpp_2m': sample_dpmpp_2m,
            'dpmpp_sde': sample_dpmpp_sde,
            'dpmpp_2m_sde': sample_dpmpp_2m_sde,
            'dpm2_karras': sample_dpm_2,
            'dpm2_ancestral_karras': sample_dpm_2_ancestral,
            'dpmpp_2s_ancestral_karras': sample_dpmpp_2s_ancestral,
            'dpmpp_2m_karras': sample_dpmpp_2m,
            'dpmpp_sde_karras': sample_dpmpp_sde,
            'dpmpp_2m_sde_karras': sample_dpmpp_2m_sde,
            'onestep': sample_onestep,
            'multistep': stochastic_iterative_sampler,
            'multistep2': stochastic_iterative_sampler2,
            'multistep3': stochastic_iterative_sampler3,
            'dpmpp_2m_sde_lcm': sample_dpmpp_2m_sde_lcm,
        }[solver]

        # options
        schedule = 'karras' if 'karras' in solver else None
        discretization = discretization or 'linspace'
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = True if solver in (
                'dpm2', 'dpm2_ancestral', 'dpmpp_2m_sde', 'dpm2_karras',
                'dpm2_ancestral_karras', 'dpmpp_2m_sde_karras') else False

        # function for denoising xt to get x0
        intermediates = []

        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()

            if isinstance(model_kwargs, list) and len(model_kwargs) == 2:
                if isinstance(
                        model_kwargs[0]['cond'], dict) and \
                        'tar_x0' in model_kwargs[0]['cond'] and \
                        'tar_mask_latent' in model_kwargs[0]['cond']:
                    tar_x0 = model_kwargs[0]['cond']['tar_x0']
                    tar_mask = model_kwargs[0]['cond']['tar_mask_latent']

                    tar_xt = self.diffuse(x0=tar_x0, t=t)
                    xt = tar_xt * (1.0 - tar_mask) + xt * tar_mask

                if isinstance(model_kwargs[0]['cond'],
                              dict) and 'ref_x0' in model_kwargs[0]['cond']:
                    model_kwargs[0]['cond']['ref_xt'] = self.diffuse(
                        x0=model_kwargs[0]['cond']['ref_x0'], t=t)
                    model_kwargs[1]['cond']['ref_xt'] = self.diffuse(
                        x0=model_kwargs[1]['cond']['ref_x0'], t=t)

            if solver in ('onestep', 'multistep', 'multistep2', 'multistep3'):
                x0 = self.denoise(xt,
                                  t,
                                  None,
                                  model,
                                  model_kwargs,
                                  guide_scale,
                                  guide_rescale,
                                  clamp,
                                  sharpness,
                                  percentile,
                                  cat_uc=cat_uc,
                                  **kwargs)[-3]
            else:
                x0 = self.denoise(xt,
                                  t,
                                  None,
                                  model,
                                  model_kwargs,
                                  guide_scale,
                                  guide_rescale,
                                  clamp,
                                  sharpness,
                                  percentile,
                                  cat_uc=cat_uc,
                                  **kwargs)[-2]

            # collect intermediate outputs
            if return_intermediate == 'xt':
                intermediates.append(xt)
            elif return_intermediate == 'x0':
                intermediates.append(x0)
            if intermediate_callback is not None:
                intermediate_callback(intermediates[-1])
            return x0

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == 'leading':
                steps = torch.arange(t_min, t_max + 1,
                                     (t_max - t_min + 1) / steps).flip(0)
            elif discretization == 'linspace':
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == 'trailing':
                steps = torch.arange(t_max, t_min - 1,
                                     -((t_max - t_min + 1) / steps))
            elif discretization == 'free':
                steps = torch.tensor(free_steps)
            else:
                raise NotImplementedError(
                    f'{discretization} discretization not implemented')
            steps = steps.clamp_(t_min, t_max)
        elif isinstance(steps, list):
            steps = torch.tensor(steps)
        steps = torch.as_tensor(steps,
                                dtype=torch.float32,
                                device=noise.device)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        t_enc = int(min(denoising_strength, 0.999) * len(steps))
        sigmas = sigmas[len(steps) - t_enc - 1:]
        if refine_strength > 0:
            t_refine = int(min(refine_strength, 0.999) * len(steps))
            if refine_stage:
                sigmas = sigmas[-t_refine:]
            else:
                sigmas = sigmas[:-t_refine + 1]
        # print(sigmas)
        if x is not None:
            noise = (x + noise * sigmas[0]) / torch.sqrt(1.0 + sigmas[0]**2.0)

        if schedule == 'karras':
            if sigmas[0] == float('inf'):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float('inf')].max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([
                    sigmas.new_tensor([float('inf')]), sigmas,
                    sigmas.new_zeros([1])
                ])
            else:
                sigmas = karras_schedule(
                    n=len(steps),
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas.max().item(),
                    rho=7.).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        kwargs['seed'] = seed
        # add noise to x0
        if add_noise:
            if 'dm_steps' in kwargs:
                if step_offset:
                    add_noise_step = -kwargs['dm_steps'] + step_offset
                    if add_noise_step < 0:
                        noise = self.diffuse(
                            noise,
                            torch.full((noise.shape[0], 1),
                                       steps[add_noise_step],
                                       dtype=torch.int))
                else:
                    noise = self.diffuse(
                        noise,
                        torch.full((noise.shape[0], 1),
                                   steps[-kwargs['dm_steps'] - 1],
                                   dtype=torch.int))
        # sampling
        x0 = solver_fn(noise,
                       model_fn,
                       sigmas,
                       show_progress=show_progress,
                       **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0

    def _sigma_to_t(self, sigma):
        if sigma == float('inf'):
            t = torch.full_like(sigma, len(self.sigmas) - 1)
        else:
            log_sigmas = torch.sqrt(self.sigmas**2 /
                                    (1 - self.sigmas**2)).log().to(sigma)
            log_sigma = sigma.log()
            dists = log_sigma - log_sigmas[:, None]
            low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(
                max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
            low, high = log_sigmas[low_idx], log_sigmas[high_idx]
            w = (low - log_sigma) / (low - high)
            w = w.clamp(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            t = t.view(sigma.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t

    def _t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = torch.sqrt(self.sigmas**2 /
                                (1 - self.sigmas**2)).log().to(t)
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        log_sigma[torch.isnan(log_sigma)
                  | torch.isinf(log_sigma)] = float('inf')
        return log_sigma.exp()

    @torch.no_grad()
    def stochastic_encode(self, x0, t, steps):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas

        t_max = None
        t_min = None

        # discretization method
        discretization = 'trailing' if self.prediction_type == 'v' else 'leading'

        # timesteps
        if isinstance(steps, int):
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min
            steps = discretize_timesteps(t_max, t_min, steps, discretization)
        steps = torch.as_tensor(steps).round().long().flip(0).to(x0.device)
        # steps = torch.as_tensor(steps).round().long().to(x0.device)

        # self.alphas_bar = torch.cumprod(1 - self.sigmas ** 2, dim=0)
        # print('sigma: ', self.sigmas, len(self.sigmas))
        # print('alpha_bar: ', self.alphas_bar, len(self.alphas_bar))
        # print('steps: ', steps, len(steps))
        # sqrt_alphas_cumprod = torch.sqrt(self.alphas_bar).to(x0.device)[steps]
        # sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_bar).to(x0.device)[steps]

        sqrt_alphas_cumprod = self.alphas.to(x0.device)[steps]
        sqrt_one_minus_alphas_cumprod = self.sigmas.to(x0.device)[steps]
        # print('sigma: ', self.sigmas, len(self.sigmas))
        # print('alpha: ', self.alphas, len(self.alphas))
        # print('steps: ', steps, len(steps))

        noise = torch.randn_like(x0)
        return (
            extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) *
            noise)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def discretize_timesteps(t_max, t_min, steps, discretization):
    """
    Implementation of timestep discretization methods.
    """
    if discretization == 'leading':
        steps = torch.arange(t_min, t_max + 1,
                             (t_max - t_min + 1) / steps).flip(0)
    elif discretization == 'linspace':
        steps = torch.linspace(t_max, t_min, steps)
    elif discretization == 'trailing':
        steps = torch.arange(t_max, t_min - 1, -((t_max - t_min + 1) / steps))
    else:
        raise NotImplementedError(
            f'{discretization} discretization not implemented')
    return steps.clamp_(t_min, t_max)


def get_scalings_for_boundary_condition(sigma):
    sigma_data = 0.5
    c_skip = (1 -
              sigma**2)**0.5 * sigma_data**2 / (sigma**2 +
                                                (1 - sigma**2) * sigma_data**2)
    c_out = (sigma * sigma_data / (sigma**2 +
                                   (1 - sigma**2) * sigma_data**2)**0.5)
    return c_skip, c_out


def v_to_x0(v, t, x_t, diffusion):
    sigmas = _i(diffusion.sigmas, t, v)
    alphas = _i(diffusion.alphas, t, v)
    return alphas * x_t - sigmas * v
