# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
ODE/SDE solver for denoising diffusion models under either variation preserving (VP) or
variation exploding (VE) settings. Under the VE setting, the diffusion process is:

q(x_t | x_0) = N(x_t | x_0, sigma_t^2 I),

where 0 <= sigma_t <= inf; while under the VP setting, the diffusion process is:

q(x_t | x_0) = N(x_t | alpha_t x_0, sigma_t^2 I),

where 0 <= sigma_t <= 1 and alpha_t^2 = 1 - sigma_t^2.
"""
from tqdm.auto import trange

import torch

__all__ = [
    'sample_euler', 'sample_euler_ancestral', 'sample_heun', 'sample_dpm_2',
    'sample_dpm_2_ancestral', 'sample_dpmpp_2s_ancestral', 'sample_dpmpp_sde',
    'sample_dpmpp_2m', 'sample_dpmpp_2m_sde', 'sample_ddim',
    'sample_dpmpp_2m_sde_lcm', 'sample_onestep',
    'stochastic_iterative_sampler', 'stochastic_iterative_sampler2',
    'stochastic_iterative_sampler3'
]

# -------------------- variation exploding (VE) solver --------------------#


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """
    Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step.
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 *
               (sigma_from**2 - sigma_to**2) / sigma_from**2)**0.5)
    sigma_down = (sigma_to**2 - sigma_up**2)**0.5
    return sigma_down, sigma_up


def get_scalings(sigma):
    c_out = -sigma
    c_in = 1 / (sigma**2 + 1.**2)**0.5
    return c_out, c_in


@torch.no_grad()
def sample_euler(noise,
                 model,
                 sigmas,
                 s_churn=0.,
                 s_tmin=0.,
                 s_tmax=float('inf'),
                 s_noise=1.,
                 seed=None,
                 show_progress=True,
                 **kwargs):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022).
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        # Euler method
        if sigmas[i] == float('inf'):
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            _, c_in = get_scalings(sigma_hat)
            denoised = model(x * c_in, sigma_hat)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_ancestral(noise,
                           model,
                           sigmas,
                           eta=1.,
                           s_noise=1.,
                           seed=None,
                           show_progress=True,
                           **kwargs):
    """
    Ancestral sampling with Euler method steps.
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma_down, sigma_up = get_ancestral_step(sigmas[i],
                                                  sigmas[i + 1],
                                                  eta=eta)
        # Euler method
        if sigmas[i] == float('inf'):
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            d = (x - denoised) / sigmas[i]
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_heun(noise,
                model,
                sigmas,
                s_churn=0.,
                s_tmin=0.,
                s_tmax=float('inf'),
                s_noise=1.,
                seed=None,
                show_progress=True,
                **kwargs):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            _, c_in = get_scalings(sigma_hat)
            denoised = model(x * c_in, sigma_hat)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                _, c_in = get_scalings(sigmas[i + 1])
                denoised_2 = model(x_2 * c_in, sigmas[i + 1])
                d_2 = (x_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_dpm_2(noise,
                 model,
                 sigmas,
                 s_churn=0.,
                 s_tmin=0.,
                 s_tmax=float('inf'),
                 s_noise=1.,
                 seed=None,
                 show_progress=True,
                 **kwargs):
    """
    A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022).
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            _, c_in = get_scalings(sigma_hat)
            denoised = model(x * c_in, sigma_hat)
            d = (x - denoised) / sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                dt = sigmas[i + 1] - sigma_hat
                x = x + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(),
                                                 0.5).exp()
                dt_1 = sigma_mid - sigma_hat
                dt_2 = sigmas[i + 1] - sigma_hat
                x_2 = x + d * dt_1
                _, c_in = get_scalings(sigma_mid)
                denoised_2 = model(x_2 * c_in, sigma_mid)
                d_2 = (x_2 - denoised_2) / sigma_mid
                x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(noise,
                           model,
                           sigmas,
                           eta=1.,
                           s_noise=1.,
                           seed=None,
                           show_progress=True,
                           **kwargs):
    """
    Ancestral sampling with DPM-Solver second-order steps.
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma_down, sigma_up = get_ancestral_step(sigmas[i],
                                                  sigmas[i + 1],
                                                  eta=eta)
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            d = (x - denoised) / sigmas[i]
            if sigma_down == 0:
                # Euler method
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
                dt_1 = sigma_mid - sigmas[i]
                dt_2 = sigma_down - sigmas[i]
                x_2 = x + d * dt_1
                _, c_in = get_scalings(sigma_mid)
                denoised_2 = model(x_2 * c_in, sigma_mid)
                d_2 = (x_2 - denoised_2) / sigma_mid
                x = x + d_2 * dt_2
                x = x + torch.randn_like(x) * s_noise * sigma_up
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(noise,
                              model,
                              sigmas,
                              eta=1.,
                              s_noise=1.,
                              seed=None,
                              show_progress=True,
                              **kwargs):
    """
    Ancestral sampling with DPM-Solver++ (2S) second-order steps.
    """
    def t_to_sigma(t):
        return t.neg().exp()

    def sigma_to_t(sigma):
        return sigma.log().neg()

    # x = noise * sigmas[0]
    x = noise * torch.sqrt(1.0 + sigmas[0]**2.0)
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma_down, sigma_up = get_ancestral_step(sigmas[i],
                                                  sigmas[i + 1],
                                                  eta=eta)
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigma_down * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            if sigma_down == 0:
                # Euler method
                d = (x - denoised) / sigmas[i]
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver++(2S)
                t, t_next = sigma_to_t(sigmas[i]), sigma_to_t(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (t_to_sigma(s) /
                       t_to_sigma(t)) * x - (-h * r).expm1() * denoised
                _, c_in = get_scalings(t_to_sigma(s))
                denoised_2 = model(x_2 * c_in, t_to_sigma(s))
                x = (t_to_sigma(t_next) /
                     t_to_sigma(t)) * x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + torch.randn_like(x) * s_noise * sigma_up
    return x


class BatchedBrownianTree:
    """
    A wrapper around torchsde.BrownianTree that enables batches of entropy.
    """
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        import torchsde
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [
            torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs)
            for s in seed
        ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1)
                         for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """
    A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """
    def __init__(self,
                 x,
                 sigma_min,
                 sigma_max,
                 seed=None,
                 transform=lambda x: x):
        self.transform = transform
        t0 = self.transform(torch.as_tensor(sigma_min))
        t1 = self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0 = self.transform(torch.as_tensor(sigma))
        t1 = self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_dpmpp_sde(noise,
                     model,
                     sigmas,
                     eta=1.,
                     s_noise=1.,
                     r=1 / 2,
                     seed=None,
                     show_progress=True,
                     **kwargs):
    """
    DPM-Solver++ (stochastic).
    """
    def t_to_sigma(t):
        return t.neg().exp()

    def sigma_to_t(sigma):
        return sigma.log().neg()

    x = noise * sigmas[0]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[
        sigmas < float('inf')].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed)
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            if sigmas[i + 1] == 0:
                # Euler method
                d = (x - denoised) / sigmas[i]
                dt = sigmas[i + 1] - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver++
                t, t_next = sigma_to_t(sigmas[i]), sigma_to_t(sigmas[i + 1])
                h = t_next - t
                s = t + h * r
                fac = 1 / (2 * r)

                # Step 1
                sd, su = get_ancestral_step(t_to_sigma(t), t_to_sigma(s), eta)
                s_ = sigma_to_t(sd)
                x_2 = (t_to_sigma(s_) /
                       t_to_sigma(t)) * x - (t - s_).expm1() * denoised
                x_2 = x_2 + noise_sampler(t_to_sigma(t),
                                          t_to_sigma(s)) * s_noise * su
                _, c_in = get_scalings(t_to_sigma(s))
                denoised_2 = model(x_2 * c_in, t_to_sigma(s))

                # Step 2
                sd, su = get_ancestral_step(t_to_sigma(t), t_to_sigma(t_next),
                                            eta)
                t_next_ = sigma_to_t(sd)
                denoised_d = (1 - fac) * denoised + fac * denoised_2
                x = (t_to_sigma(t_next_) / t_to_sigma(t)) * x - \
                    (t - t_next_).expm1() * denoised_d
                x = x + noise_sampler(t_to_sigma(t),
                                      t_to_sigma(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m(noise,
                    model,
                    sigmas,
                    seed=None,
                    show_progress=True,
                    **kwargs):
    """
    DPM-Solver++ (2M).
    """
    def t_to_sigma(t):
        return t.neg().exp()

    def sigma_to_t(sigma):
        return sigma.log().neg()

    x = noise * sigmas[0]
    old_denoised = None
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            t, t_next = sigma_to_t(sigmas[i]), sigma_to_t(sigmas[i + 1])
            h = t_next - t
            if (old_denoised is None or sigmas[i - 1] == float('inf')
                    or sigmas[i + 1] == 0):
                x = (t_to_sigma(t_next) /
                     t_to_sigma(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - sigma_to_t(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 /
                              (2 * r)) * denoised - (1 /
                                                     (2 * r)) * old_denoised
                x = (t_to_sigma(t_next) /
                     t_to_sigma(t)) * x - (-h).expm1() * denoised_d
            old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(noise,
                        model,
                        sigmas,
                        eta=1.,
                        s_noise=1.,
                        solver_type='midpoint',
                        seed=None,
                        show_progress=True,
                        **kwargs):
    """
    DPM-Solver++ (2M) SDE.
    """
    assert solver_type in {'heun', 'midpoint'}

    x = noise * sigmas[0]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[
        sigmas < float('inf')].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed)
    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + \
                    (-h - eta_h).expm1().neg() * denoised

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * \
                            (1 / r) * (denoised - old_denoised)
                    elif solver_type == 'midpoint':
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * \
                            (1 / r) * (denoised - old_denoised)

                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[
                    i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            old_denoised = denoised
            h_last = h
    return x


# -------------------- variation preserving (VP) solver --------------------#
@torch.no_grad()
def sample_ddim(noise,
                model,
                sigmas,
                eta=0.,
                seed=None,
                show_progress=True,
                **kwargs):
    """
    DDIM solver steps.
    """
    x = noise
    sigmas_vp = (sigmas**2 / (1 + sigmas**2))**0.5
    sigmas_vp[sigmas == float('inf')] = 1.
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])
        noise_factor = eta * (sigmas_vp[i + 1]**2 / sigmas_vp[i]**2 *
                              (1 - (1 - sigmas_vp[i]**2) /
                               (1 - sigmas_vp[i + 1]**2)))
        d = (x - (1 - sigmas_vp[i]**2)**0.5 * denoised) / sigmas_vp[i]
        x = (1 - sigmas_vp[i + 1] ** 2) ** 0.5 * denoised + \
            (sigmas_vp[i + 1] ** 2 - noise_factor ** 2) ** 0.5 * d
        if sigmas_vp[i + 1] > 0:
            x += noise_factor * torch.randn_like(x)
    return x


@torch.no_grad()
def sample_img2img_euler(noise,
                         model,
                         sigmas,
                         s_churn=0.,
                         s_tmin=0.,
                         s_tmax=float('inf'),
                         s_noise=1.,
                         seed=None,
                         show_progress=True,
                         **kwargs):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022).
    """
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        # Euler method
        if sigmas[i] == float('inf'):
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            denoised = model(x, sigma_hat)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
    return x


@torch.no_grad()
def sample_img2img_euler_ancestral(noise,
                                   model,
                                   sigmas,
                                   eta=1.,
                                   s_noise=1.,
                                   seed=None,
                                   show_progress=True,
                                   **kwargs):
    """
    Ancestral sampling with Euler method steps.
    """
    x = noise
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        sigma_down, sigma_up = get_ancestral_step(sigmas[i],
                                                  sigmas[i + 1],
                                                  eta=eta)
        # Euler method
        if sigmas[i] == float('inf'):
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            denoised = model(x, sigmas[i])
            d = (x - denoised) / sigmas[i]
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            if sigmas[i + 1] > 0:
                x = x + torch.randn_like(x) * s_noise * sigma_up
    return x


# --------- LCM ------------


@torch.no_grad()
def sample_dpmpp_2m_sde_lcm(noise,
                            model,
                            sigmas,
                            eta=1.,
                            s_noise=1.,
                            solver_type='midpoint',
                            show_progress=True,
                            total_sample_steps=50,
                            sample_steps=5,
                            **kwargs):
    """
    DPM-Solver++ (2M) SDE.
    """
    assert solver_type in {'heun', 'midpoint'}

    dm_steps = sample_steps
    x = noise * (sigmas[-dm_steps - 1]**2 + 1)**0.5
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[
        sigmas < float('inf')].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - dm_steps - 1,
                    len(sigmas) - 1,
                    disable=not show_progress):
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + \
                    (-h - eta_h).expm1().neg() * denoised

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * \
                            (1 / r) * (denoised - old_denoised)
                    elif solver_type == 'midpoint':
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * \
                            (1 / r) * (denoised - old_denoised)

                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[
                    i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

                old_denoised = denoised
                h_last = h
    return x


@torch.no_grad()
def sample_onestep(noise, model, sigmas, show_progress=True, **kwargs):
    """
    CM one step solver.
    """
    x = noise
    denoised = model(x, sigmas[0])
    return denoised


@torch.no_grad()
def stochastic_iterative_sampler(noise,
                                 model,
                                 sigmas,
                                 show_progress=True,
                                 **kwargs):
    """
    CM multiple steps solver.
    """
    x = noise
    sigmas_vp = (sigmas**2 / (1 + sigmas**2))**0.5
    sigmas_vp[sigmas == float('inf')] = 1.
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        denoised = model(x, sigmas[i])

        sigma = sigmas_vp[i + 1]
        alpha = torch.sqrt(1 - sigma**2)
        noises = torch.randn_like(x)
        x = alpha * denoised + sigma * noises
    return x


@torch.no_grad()
def stochastic_iterative_sampler2(noise,
                                  model,
                                  sigmas,
                                  show_progress=True,
                                  repeat=1,
                                  ema=0.05,
                                  **kwargs):
    """
    CM multiple steps solver.
    """
    # denoised_s = []
    # for _ in range(repeat):
    #     noise = torch.randn_like(noise)
    #     denoised_s.append(model(noise, sigmas[0]))

    # denoised_s = torch.cat(denoised_s, dim = 0)
    # denoised = torch.mean(denoised_s, dim=0, keepdim=True)
    denoised = model(noise, sigmas[0])
    denoised_ema = denoised

    sigmas_vp = (sigmas**2 / (1 + sigmas**2))**0.5
    sigmas_vp[sigmas == float('inf')] = 1.
    for i in trange(1, len(sigmas) - 1, disable=not show_progress):
        sigma = sigmas_vp[i]
        alpha = torch.sqrt(1 - sigma**2)
        noise = torch.randn_like(noise)
        x = alpha * denoised + sigma * noise
        denoised = model(x, sigmas[i])
        denoised_ema = ema * denoised_ema + (1.0 - ema) * denoised
    return denoised_ema


@torch.no_grad()
def stochastic_iterative_sampler3(noise,
                                  model,
                                  sigmas,
                                  show_progress=True,
                                  repeat=1,
                                  solver_eta=0.2,
                                  solver_ema=0.05,
                                  **kwargs):
    """
    CM multiple steps solver.
    """
    eta = solver_eta
    ema = solver_ema
    # denoised_s = []
    # for _ in range(repeat):
    #     noise = torch.randn_like(noise)
    #     denoised_s.append(model(noise, sigmas[0]))

    # denoised_s = torch.cat(denoised_s, dim = 0)
    # denoised = torch.mean(denoised_s, dim=0, keepdim=True)
    denoised = model(noise, sigmas[0])
    denoised_ema = denoised
    eta = float(eta)

    sigmas_vp = (sigmas**2 / (1 + sigmas**2))**0.5
    sigmas_vp[sigmas == float('inf')] = 1.
    noises = [
        noise.clone(),
    ]
    for i in trange(1, len(sigmas) - 1, disable=not show_progress):
        sigma = sigmas_vp[i]
        alpha = torch.sqrt(1 - sigma**2)
        # x = alpha * denoised + sigma * torch.randn_like(noise)
        # x = alpha * denoised + sigma * (eta * noise + (1 - eta ** 2) ** 0.5 * torch.randn_like(noise))
        # eta = 0.2*sigma / (sigma + 1.0)
        # noise = eta * noise + (1 - eta ** 2) ** 0.5 * torch.randn_like(noise)
        noise2 = 0
        for n in noises:
            noise2 += eta * n
        noise2 += (1 - (eta * len(noises))**2)**0.5 * torch.randn_like(noise)
        noises.append(noise2.clone())
        # noise = torch.randn_like(noise)
        x = alpha * denoised + sigma * noise2
        denoised = model(x, sigmas[i])
        denoised_ema = ema * denoised_ema + (1.0 - ema) * denoised
    return denoised_ema
