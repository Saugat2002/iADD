import os
import sys
import torch
import contextlib
import pickle
import json
import random
import numpy as np
from functools import partial
from tqdm import tqdm as tqdm_lib
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import post_processing, seed_everything

tqdm = partial(tqdm_lib, dynamic_ncols=True)


def run_sampling(config, stage_idx=None, logger=None, wandb_run=None, pipeline=None, trainable_layers=None, resume_from_ckpt=False):
    """
    Run the sampling phase for a given stage.
    
    Args:
        config: Configuration object (can be OmegaConf or dict)
        stage_idx: Current stage index (optional, for logging)
        logger: Logger instance (optional)
        wandb_run: Existing wandb run to log to (optional)
        pipeline: Pre-loaded StableDiffusionPipeline (avoids reloading)
        trainable_layers: Pre-initialized trainable layers
        
    Returns:
        save_dir: Directory where samples were saved
    """

    if logger:
        logger.info(f"Starting sampling for stage {stage_idx}")
    else:
        print(f"Starting sampling for stage {stage_idx}")
    
    print(f'========== seed: {config.seed} ==========') 
    torch.cuda.set_device(config.dev_id)
    # Setup directories
    unique_id = config.exp_name
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)
    
    # Use stage index for directory, not run_name
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Use pre-loaded pipeline (no loading!)
    if pipeline is None:
        raise ValueError("Pipeline must be provided - should not reload model each stage!")
    
    # Setup accelerator
    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )
    
    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        # print(models)
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    

    
    # Prepare trainable layers (if not already prepared)
    if trainable_layers is not None and not hasattr(trainable_layers, '_hf_hook'):
        trainable_layers = accelerator.prepare(trainable_layers)
    
    # Load checkpoint if resume_from_ckpt is True
    if resume_from_ckpt:
        print("loading model. Please Wait.")
        # Build checkpoint path from previous stage
        prev_stage = stage_idx - 1
        checkpoint_num = config.train.num_epochs // config.train.save_interval
        checkpoint_path = os.path.join(
            config.save_path,
            config.exp_name,
            f"stage{prev_stage}",
            "checkpoints",
            f"checkpoint_{checkpoint_num}"
        )
        
        checkpoint_path = os.path.normpath(os.path.expanduser(checkpoint_path))
        if "checkpoint_" not in os.path.basename(checkpoint_path):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(checkpoint_path)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {checkpoint_path}")
            checkpoint_path = os.path.join(
                checkpoint_path,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print(f"Resuming from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        print("load successfully!")
    
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Set seed
    seed_everything(config.seed)
    
    pipeline.unet.eval()
    samples = []
    if config.sample.always_branch_at > 0:
        split_steps = [config.sample.always_branch_at]
    else:
        split_steps = [config.split_step]

    split_times = [config.split_time]
    
    total_prompts = []
    total_samples = None
    
    # Load existing prompts and samples if available
    if os.path.exists(os.path.join(save_dir, 'prompt.json')):
        with open(os.path.join(save_dir, 'prompt.json'), 'r') as f:
            total_prompts = json.load(f)
    
    if os.path.exists(os.path.join(save_dir, 'sample.pkl')):
        with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
            total_samples = pickle.load(f)
    
    global_idx = len(total_prompts)
    local_idx = 0
    
    # Load prompts
    prompt_list = []
    if len(config.prompt) == 0: #prompt = "" by default
        # Convert relative path to absolute path from project root
        prompt_file_path = config.prompt_file
        if not os.path.isabs(prompt_file_path):
            # Get project root (3 levels up from core/sampling.py -> core/ -> root/)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            prompt_file_path = os.path.join(project_root, prompt_file_path)
        
        with open(prompt_file_path) as f:
            prompt_list = json.load(f)
    prompt_idx = 0
    prompt_cnt = len(prompt_list)
    
    # Main sampling loop
    for idx in tqdm(
        range(config.sample.num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
        desc="Sampling batches"
    ):
        # generate prompts
        if len(config.prompt) != 0:
            prompts1 = [config.prompt for _ in range(config.sample.batch_size)]
        elif config.prompt_random_choose:
            prompts1 = [random.choice(prompt_list) for _ in range(config.sample.batch_size)]
        else:
            prompts1 = [prompt_list[(prompt_idx+i)%prompt_cnt] for i in range(config.sample.batch_size)]
            prompt_idx += config.sample.batch_size
        
        # Encode prompts
        prompt_ids1 = pipeline.tokenizer(
            prompts1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        
        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
        
        # Combine prompt and negative prompt
        prompt_embeds1_combine = pipeline._encode_prompt(
            None,
            accelerator.device,
            1,
            config.sample.cfg,
            None,
            prompt_embeds=prompt_embeds1,
            negative_prompt_embeds=sample_neg_prompt_embeds
        )
        
        # Prepare latents
        noise_latents1 = pipeline.prepare_latents(
            config.sample.batch_size, 
            pipeline.unet.config.in_channels, ## channels
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
            prompt_embeds1.dtype, 
            accelerator.device, 
            None ## generator
        )
        
        pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
        ts = pipeline.scheduler.timesteps
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, config.sample.eta)
        
        # For no_branching mode, prepare multiple different initial noises
        if config.sample.no_branching:
            branch_num = split_times[0]  # Get branching factor
            noise_latents_list = [noise_latents1]  # Keep first noise
            for _ in range(branch_num - 1):
                additional_noise = pipeline.prepare_latents(
                    config.sample.batch_size, 
                    pipeline.unet.config.in_channels,
                    pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                    pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                    prompt_embeds1.dtype, 
                    accelerator.device, 
                    None
                )
                noise_latents_list.append(additional_noise)
            latents = [[noise] for noise in noise_latents_list]
            log_probs = [[] for _ in range(branch_num)]  # Initialize log_probs for each branch
        else:
            latents = [[noise_latents1]]
            log_probs = [[]]

        for i, t in tqdm(
            enumerate(ts),
            desc="Timestep",
            position=3,
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):  
            # sample
            with autocast():
                with torch.no_grad():
                    if not config.sample.no_branching and ((config.sample.num_steps-i) in split_steps):
                        branch_num = split_steps.index(config.sample.num_steps-i)
                        branch_num = split_times[branch_num]
                        cur_sample_num = len(latents)
                        # split the sample
                        latents = [
                            [latent for latent in latents[k//branch_num]] 
                            for k in range(cur_sample_num*branch_num)
                            ]
                        
                        # Not creating random values - it's duplicating the accumulated log_probs from previous timesteps
                        log_probs = [
                            [log_prob for log_prob in log_probs[k//branch_num]] 
                            for k in range(cur_sample_num*branch_num)
                            ]
                        
                    for k in range(len(latents)): 
                        latents_t = latents[k][i]
                        latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                        latents_input = pipeline.scheduler.scale_model_input(latents_input, t)

                        noise_pred = pipeline.unet(
                            latents_input,
                            t,
                            encoder_hidden_states=prompt_embeds1_combine,
                            return_dict=False,
                        )[0]

                        if config.sample.cfg:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        latents_t_1, log_prob, latents_0 = ddim_step_with_logprob(pipeline.scheduler, noise_pred, t, latents_t, **extra_step_kwargs)

                        latents[k].append(latents_t_1)
                        log_probs[k].append(log_prob)

        sample_num = len(latents)
        total_prompts.extend(prompts1*sample_num)

        for k in range(sample_num): 
            images = latents_decode(pipeline, latents[k][config.sample.num_steps], accelerator.device, prompt_embeds1.dtype)
            store_latents = torch.stack(latents[k], dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            store_log_probs = torch.stack(log_probs[k], dim=1)  # (batch_size, num_steps)
            prompt_embeds = prompt_embeds1
            current_latents = store_latents[:, :-1]
            next_latents = store_latents[:, 1:]
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            samples.append(
                {
                    "prompt_embeds": prompt_embeds.cpu().detach(),
                    "timesteps": timesteps.cpu().detach(),
                    "log_probs": store_log_probs.cpu().detach(),
                    "latents": current_latents.cpu().detach(),  # each entry is the latent before timestep t
                    "next_latents": next_latents.cpu().detach(),  # each entry is the latent after timestep t
                    "images":images.cpu().detach()
                }
            )

        # if idx==0:
        #     for k,v in samples[0].items():
        #         print(k, v.shape)

        if (idx+1)%config.sample.save_interval ==0 or idx==(config.sample.num_batches_per_epoch-1):
            os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
            print(f'-----------{accelerator.process_index} save image start-----------')
            # print(samples)
            new_samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
            images = new_samples['images'][local_idx:]
            for j, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil.save(os.path.join(save_dir, f"images/{(j+global_idx):05}.png"))

            global_idx += len(images)
            local_idx += len(images)
            
            with open(os.path.join(save_dir, f'prompt.json'),'w') as f:
                json.dump(total_prompts, f)
            with open(os.path.join(save_dir, f'sample.pkl'), 'wb') as f:
                if total_samples is None: 
                    pickle.dump({
                        "prompt_embeds": new_samples["prompt_embeds"], 
                        "timesteps": new_samples["timesteps"], 
                        "log_probs": new_samples["log_probs"], 
                        "latents": new_samples["latents"], 
                        "next_latents": new_samples["next_latents"]
                        }, f)
                else: 
                    pickle.dump({
                        "prompt_embeds": torch.cat([total_samples["prompt_embeds"], new_samples["prompt_embeds"]]), 
                        "timesteps": torch.cat([total_samples["timesteps"], new_samples["timesteps"]]), 
                        "log_probs": torch.cat([total_samples["log_probs"], new_samples["log_probs"]]), 
                        "latents": torch.cat([total_samples["latents"], new_samples["latents"]]), 
                        "next_latents": torch.cat([total_samples["next_latents"], new_samples["next_latents"]])
                        }, f)
    
    # Log sampling completion to parent wandb run
    if wandb_run:
        wandb_run.log({
            f"sampling/stage_{stage_idx}/total_samples": len(total_prompts),
            f"sampling/stage_{stage_idx}/batches": config.sample.num_batches_per_epoch,
        })
    
    
    if logger:
        logger.info(f"Sampling completed for stage {stage_idx}")
    
    return save_dir
