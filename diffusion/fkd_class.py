"""
Feynman-Kac Diffusion (FKD) steering mechanism implementation.
"""

import torch
from enum import Enum
import numpy as np
from typing import Callable, Optional, Tuple
import logging
import os
from PIL import Image as PILImage

class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"
    RT = "rt"


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        device: torch.device = torch.device('cuda'),
        pipeline=None,
        data_type=torch.float32,
        **kwargs,
    ) -> None:
        # Initialize hyperparameters and functions

        # if kwargs:
            # logging.warning(f"FKD Steering - Unused arguments: {kwargs}")

        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps

        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn

        # Initialize device and population reward state
        self.device = device

        # initial rewards
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * reward_min_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)
        
        self.pipeline = pipeline
        self.data_type = data_type
        
        # Store initial values for reset
        self.initial_reward_value = reward_min_value
        self.resampling_interval = np.arange(
            self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency
        )
        print(f"FKD initialized with resampling timesteps: {self.resampling_interval}, potential type: {self.potential_type}, lambda: {self.lmbda}")
        self.resampling_interval = np.append(self.resampling_interval, self.time_steps - 1)

    def reset_state(self):
        """
        Reset the FKD state variables to their initial values.
        This should be called before processing a new independent set of particles
        (e.g., new prompt or switching between best/worst particles).
        """
        self.population_rs = (
            torch.ones(self.num_particles, device=self.device) * self.initial_reward_value
        )
        self.product_of_potentials = torch.ones(self.num_particles).to(self.device)

    def resample(
        self, *, sampling_idx: int, latents: torch.Tensor, x0_preds: torch.Tensor,
        ground, img_dir, save_dir, config, 
        log_probs, get_best_indices=True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        # Check if resampling is within the allowed range and conditions

        resampling_interval = self.resampling_interval

        if sampling_idx not in resampling_interval:
            return latents, None, log_probs, None # no resampling, return None for indices

        # Decode latents to population images and compute rewards
        population_images = self.latent_to_decode_fn(self.pipeline, x0_preds, self.device, self.data_type)
        
        # Save images to img_dir for reward computation
        
        os.makedirs(img_dir, exist_ok=True)
        for idx, img in enumerate(population_images):
            img_array = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8')
            pil_img = PILImage.fromarray(img_array)
            pil_img.save(os.path.join(img_dir, f"{idx:05d}.png"))
        
        eval_scores, score_metrics, raw_clip_scores = self.reward_fn(ground, img_dir, save_dir, config) # TODO: it could be more efficient
        
        # Clean up temporary images
        for filename in os.listdir(img_dir):
            file_path = os.path.join(img_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        rs_candidates = raw_clip_scores
        if not get_best_indices:
            rs_candidates = torch.tensor([-score for score in raw_clip_scores], device=self.device)
        else:
            # Ensure rs_candidates is on the correct device
            if not isinstance(rs_candidates, torch.Tensor):
                rs_candidates = torch.tensor(rs_candidates, device=self.device)
            else:
                rs_candidates = rs_candidates.to(self.device)
        
        # Compute importance weights
        if self.potential_type == PotentialType.MAX:
            rs_candidates = torch.max(rs_candidates, self.population_rs)
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        elif self.potential_type == PotentialType.RT:
            w = torch.exp(self.lmbda * rs_candidates)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        if sampling_idx == self.time_steps - 1:
            if (
                self.potential_type == PotentialType.MAX
                or self.potential_type == PotentialType.ADD
                or self.potential_type == PotentialType.RT
            ):
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            # compute effective sample size
            normalized_w = w / w.sum()
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}")
                # Resample indices based on weights
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )
                resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]

                # Resample population images
                resampled_images = population_images[indices]

                # Update product of potentials; used for max and add potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )
            else:
                # No resampling
                resampled_images = population_images
                resampled_latents = latents
                self.population_rs = rs_candidates
                # Create tensor of identity indices for consistency
                indices = torch.arange(self.num_particles, device=self.device)

        else:
            # Resample indices based on weights
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )
            resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]

            # Resample population images
            resampled_images = population_images[indices]

            # Update product of potentials; used for max and add potentials
            self.product_of_potentials = (
                self.product_of_potentials[indices] * w[indices]
            )
        log_probs = log_probs[indices]
        return resampled_latents, resampled_images, log_probs, indices


if __name__ == "__main__":

    # Demonstration of FKD resampling step
    import matplotlib.pyplot as plt
    import random

    # set seed
    random.seed(0)

    # 1x1 pixel images
    num_particles = 8
    x0s = torch.rand(num_particles, 1, 1)

    # reward darker images
    reward_function = lambda x: -0.5 * x.sum(dim=(1, 2))

    # Define the FKD steering mechanism
    fkds = FKD(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=-1,
        resampling_t_end=100,
        time_steps=100,
        reward_fn=lambda x: reward_function(x),
        device=torch.device('cpu'),
    )

    # Define the sampling index
    sampling_idx = 0

    # Perform resampling
    resampled_latents, resampled_images = fkds.resample(
        sampling_idx=sampling_idx,
        latents=x0s,
        x0_preds=x0s,
    )

    plt.rc('text', usetex=True)
    fig, axs = plt.subplots(2, num_particles)

    axs[0, 0].set_title('Initial')
    axs[1, 0].set_title('Resampled')

    for i in range(num_particles):
        axs[0, i].imshow(x0s[i].detach().numpy(), cmap='gray', vmin=0, vmax=1)
        axs[1, i].imshow(
            resampled_images[i].detach().numpy(), cmap='gray', vmin=0, vmax=1
        )

        axs[1, i].axis('off')
        axs[0, i].axis('off')

    out_path = 'resampled_examples.png'
    plt.savefig(out_path)
    print('Saved resampled examples to:', out_path)