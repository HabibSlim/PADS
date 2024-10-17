"""
Diffusion sampling functions for the latent diffusion model.
"""

import numpy as np
import torch


def get_timesteps(num_steps, sigma_min, sigma_max, rho, device=None):
    """
    Compute the time steps for the noising schedule.
    """
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0
    return t_steps


@torch.inference_mode()
def edm_sampler(
    net,
    latents,
    cond,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    start_step=0,
    verbose=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    t_steps = get_timesteps(num_steps, sigma_min, sigma_max, rho, device=latents.device)

    # Initialize base latents depending on the starting step
    latents = latents + randn_like(latents) * t_steps[start_step]
    x_next = latents.to(torch.float64)

    gamma_base = min(S_churn / num_steps, np.sqrt(2) - 1)

    # Main sampling loop, starting from the specified step
    step_count = 0
    for k, (t_cur, t_next) in enumerate(
        zip(t_steps[start_step:-1], t_steps[start_step + 1 :])
    ):
        i = k + start_step
        x_cur = x_next

        # Increase noise temporarily.
        gamma = gamma_base if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net.edm_forward(x=x_hat, sigma=t_hat, cond=cond).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net.edm_forward(x=x_next, sigma=t_next, cond=cond).to(
                torch.float64
            )
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        step_count += 1

    if verbose:
        print(
            "Denoised for {} steps.".format(step_count),
            " starting from step ",
            start_step,
            " sigma_0=",
            t_steps[start_step],
            " sigma_N=",
            t_steps[-1],
        )

    return x_next
