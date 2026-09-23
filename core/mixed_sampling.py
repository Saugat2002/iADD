import os
import torch
import contextlib
import copy
import json
import pickle
import random
import numpy as np
import open_clip
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
from diffusion.fkd_class import FKD, PotentialType
from core.selection import score_fn1

tqdm = partial(tqdm_lib, dynamic_ncols=True)


def run_mixed_sampling(config, stage_idx=None, logger=None, wandb_run=None, pipeline=None, trainable_layers=None, resume_from_ckpt=False):
    """
    Mixed sampling: FK-steered + vanilla branched samples.
    
    Args:
        config: Configuration with sample.fk_mix_ratio (0.0 to 1.0)
                0.0 = all vanilla, 1.0 = all FK, 0.5 = 50/50 mix
    """
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config
        
    if logger:
        logger.info(f"Starting mixed sampling for stage {stage_idx}")
    else:
        print(f"Starting mixed sampling for stage {stage_idx}")
    
    # Get FK mix ratio from config (default 0.5)
    fk_mix_ratio = getattr(config.sample, 'fk_mix_ratio', 0.5)
    if logger:
        logger.info(f"FK mix ratio: {fk_mix_ratio:.2f} (FK: {fk_mix_ratio*100:.0f}%, Vanilla: {(1-fk_mix_ratio)*100:.0f}%)")
    
    print(f'========== seed: {config.seed} ==========') 
    torch.cuda.set_device(config.dev_id)
    
    unique_id = config.exp_name
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)
    
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'tmp_images'), exist_ok=True)
    
    if pipeline is None:
        raise ValueError("Pipeline must be provided - should not reload model each stage!")
    
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
        weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 1
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
        models.pop()
    
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    if trainable_layers is not None and not hasattr(trainable_layers, '_hf_hook'):
        trainable_layers = accelerator.prepare(trainable_layers)
    
    if resume_from_ckpt:
        print("loading model. Please Wait.")
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
    
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    seed_everything(config.seed)
    
    pipeline.unet.eval()
    
    num_particles = 4
    samples = []
    split_steps = [config.split_step]
    split_times = [config.split_time]
    
    total_prompts = []
    total_samples = None
    
    particle_multiplier = 1 if getattr(config.sample, 'only_best_fk', False) else 2
    
    if os.path.exists(os.path.join(save_dir, 'prompt.json')):
        with open(os.path.join(save_dir, 'prompt.json'), 'r') as f:
            total_prompts = json.load(f)
    
    if os.path.exists(os.path.join(save_dir, 'sample.pkl')):
        with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
            total_samples = pickle.load(f)
    
    global_idx = len(total_prompts)
    local_idx = 0
    
    prompt_list = []
    if len(config.prompt) == 0:
        prompt_file_path = config.prompt_file
        if not os.path.isabs(prompt_file_path):
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            prompt_file_path = os.path.join(project_root, prompt_file_path)
        with open(prompt_file_path) as f:
            prompt_list = json.load(f)
    prompt_idx = 0
    prompt_cnt = len(prompt_list)
    
    # Load CLIP for FK sampling
    print("Loading CLIP model for FK sampling...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', 
        pretrained='laion2B-s32B-b79K'
    )
    clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
    clip_model = clip_model.to(accelerator.device)
    print("CLIP model loaded successfully!")
    
    reward_fn_with_clip = partial(
        score_fn1,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer
    )
    
    fkd = FKD(
        potential_type="diff",
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=20,
        resampling_t_start=5,
        resampling_t_end=15,
        time_steps=20,
        reward_fn=reward_fn_with_clip,
        device=accelerator.device,
        latent_to_decode_fn=latents_decode,
        pipeline=pipeline,
        data_type=neg_prompt_embed.dtype
    )
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, config.sample.eta)
    
    # Main sampling loop
    for idx in tqdm(
        range(config.sample.num_batches_per_epoch),
        disable=not accelerator.is_local_main_process,
        position=0,
        desc="Sampling batches (mixed)"
    ):
        # Generate prompts
        if len(config.prompt) != 0:
            base_prompts = [config.prompt for _ in range(config.sample.batch_size)]
        elif config.prompt_random_choose:
            base_prompts = [random.choice(prompt_list) for _ in range(config.sample.batch_size)]
        else:
            base_prompts = [prompt_list[(prompt_idx+i)%prompt_cnt] for i in range(config.sample.batch_size)]
            prompt_idx += config.sample.batch_size
        
        # Split batch into FK and vanilla groups
        fk_batch_size = int(config.sample.batch_size * fk_mix_ratio)
        vanilla_batch_size = config.sample.batch_size - fk_batch_size
        
        fk_prompts = base_prompts[:fk_batch_size]
        vanilla_prompts = base_prompts[fk_batch_size:]
        
        all_latents = []
        all_log_probs = []
        all_prompt_embeds = []
        
        # Initialize to avoid undefined variable errors
        latents_fk = []
        latents_vanilla = []
        fk_prompts_expanded = []
        
        # ========== FK SAMPLING PART ==========
        if fk_batch_size > 0:
            fk_prompts_expanded = [prompt for prompt in fk_prompts for _ in range(num_particles * particle_multiplier)]
            
            sample_neg_prompt_embeds_fk = neg_prompt_embed.repeat(fk_batch_size * num_particles * particle_multiplier, 1, 1)
            
            prompt_ids_fk = pipeline.tokenizer(
                fk_prompts_expanded,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            
            prompt_embeds_fk = pipeline.text_encoder(prompt_ids_fk)[0]
            
            prompt_embeds_fk_combine = pipeline._encode_prompt(
                None,
                accelerator.device,
                1,
                config.sample.cfg,
                None,
                prompt_embeds=prompt_embeds_fk,
                negative_prompt_embeds=sample_neg_prompt_embeds_fk
            )
            
            noise_latents_fk = pipeline.prepare_latents(
                fk_batch_size * num_particles * particle_multiplier, 
                pipeline.unet.config.in_channels,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                prompt_embeds_fk.dtype, 
                accelerator.device, 
                None
            )
            
            pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
            ts = pipeline.scheduler.timesteps
            
            latents_fk = [[noise_latents_fk]]
            log_probs_fk = [[]]
            
            for i, t in enumerate(ts):
                with autocast():
                    with torch.no_grad():
                        for k in range(len(latents_fk)):
                            latents_t = latents_fk[k][i]
                            latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                            latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                            
                            noise_pred = pipeline.unet(
                                latents_input,
                                t,
                                encoder_hidden_states=prompt_embeds_fk_combine,
                                return_dict=False,
                            )[0]
                            
                            if config.sample.cfg:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            
                            latents_t_1, log_prob, latents_0 = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                t,
                                latents_t,
                                **extra_step_kwargs
                            )
                            
                            # FK resampling
                            if i >= fkd.resampling_t_start and i <= fkd.resampling_t_end:
                                all_resampled_latents = []
                                all_selected_log_probs = []
                                
                                only_best = getattr(config.sample, 'only_best_fk', False)
                                particles_per_prompt = num_particles * particle_multiplier
                                
                                for b in range(fk_batch_size):
                                    start_idx = b * particles_per_prompt
                                    end_idx = start_idx + particles_per_prompt
                                    
                                    if only_best:
                                        best_latents = latents_t_1[start_idx:end_idx]
                                        best_log_probs = log_prob[start_idx:end_idx]
                                        best_prompts = fk_prompts_expanded[start_idx:end_idx]
                                        latents_0_best = latents_0[start_idx:end_idx]
                                        
                                        resampled_best, _, selected_best_log_probs = fkd.resample(
                                            sampling_idx=i, 
                                            latents=best_latents, 
                                            x0_preds=latents_0_best,
                                            ground=best_prompts,
                                            img_dir=os.path.join(save_dir, 'tmp_images'),
                                            save_dir=save_dir,
                                            config=config,
                                            log_probs=best_log_probs,
                                            get_best_indices=True
                                        )
                                        
                                        all_resampled_latents.append(resampled_best)
                                        all_selected_log_probs.append(selected_best_log_probs)
                                    else:
                                        mid_idx = start_idx + num_particles
                                        
                                        # Process best particles
                                        best_latents = latents_t_1[start_idx:mid_idx]
                                        best_log_probs = log_prob[start_idx:mid_idx]
                                        best_prompts = fk_prompts_expanded[start_idx:mid_idx]
                                        latents_0_best = latents_0[start_idx:mid_idx]
                                        
                                        resampled_best, _, selected_best_log_probs = fkd.resample(
                                            sampling_idx=i, 
                                            latents=best_latents, 
                                            x0_preds=latents_0_best,
                                            ground=best_prompts,
                                            img_dir=os.path.join(save_dir, 'tmp_images'),
                                            save_dir=save_dir,
                                            config=config,
                                            log_probs=best_log_probs,
                                            get_best_indices=True
                                        )
                                        
                                        # Process worst particles
                                        worst_latents = latents_t_1[mid_idx:end_idx]
                                        worst_log_probs = log_prob[mid_idx:end_idx]
                                        worst_prompts = fk_prompts_expanded[mid_idx:end_idx]
                                        latents_0_worst = latents_0[mid_idx:end_idx]
                                        
                                        resampled_worst, _, selected_worst_log_probs = fkd.resample(
                                            sampling_idx=i, 
                                            latents=worst_latents, 
                                            x0_preds=latents_0_worst,
                                            ground=worst_prompts,
                                            img_dir=os.path.join(save_dir, 'tmp_images'),
                                            save_dir=save_dir,
                                            config=config,
                                            log_probs=worst_log_probs,
                                            get_best_indices=False
                                        )
                                        
                                        combined_latents = torch.cat([resampled_best, resampled_worst], dim=0)
                                        combined_log_probs = torch.cat([selected_best_log_probs, selected_worst_log_probs], dim=0)
                                        
                                        all_resampled_latents.append(combined_latents)
                                        all_selected_log_probs.append(combined_log_probs)
                                
                                latents_t_1 = torch.cat(all_resampled_latents, dim=0)
                                log_prob = torch.cat(all_selected_log_probs, dim=0)
                            
                            latents_fk[k].append(latents_t_1)
                            log_probs_fk[k].append(log_prob)
            
            all_latents.extend(latents_fk)
            all_log_probs.extend(log_probs_fk)
            all_prompt_embeds.extend([prompt_embeds_fk] * len(latents_fk))
        
        # ========== VANILLA BRANCHING PART ==========
        if vanilla_batch_size > 0:
            sample_neg_prompt_embeds_vanilla = neg_prompt_embed.repeat(vanilla_batch_size, 1, 1)
            
            prompt_ids_vanilla = pipeline.tokenizer(
                vanilla_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            
            prompt_embeds_vanilla = pipeline.text_encoder(prompt_ids_vanilla)[0]
            
            prompt_embeds_vanilla_combine = pipeline._encode_prompt(
                None,
                accelerator.device,
                1,
                config.sample.cfg,
                None,
                prompt_embeds=prompt_embeds_vanilla,
                negative_prompt_embeds=sample_neg_prompt_embeds_vanilla
            )
            
            noise_latents_vanilla = pipeline.prepare_latents(
                vanilla_batch_size, 
                pipeline.unet.config.in_channels,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                prompt_embeds_vanilla.dtype, 
                accelerator.device, 
                None
            )
            
            pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
            ts = pipeline.scheduler.timesteps
            
            if config.sample.no_branching:
                branch_num = split_times[0]
                noise_latents_list = [noise_latents_vanilla]
                for _ in range(branch_num - 1):
                    additional_noise = pipeline.prepare_latents(
                        vanilla_batch_size, 
                        pipeline.unet.config.in_channels,
                        pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                        pipeline.unet.config.sample_size * pipeline.vae_scale_factor,
                        prompt_embeds_vanilla.dtype, 
                        accelerator.device, 
                        None
                    )
                    noise_latents_list.append(additional_noise)
                latents_vanilla = [[noise] for noise in noise_latents_list]
                log_probs_vanilla = [[] for _ in range(branch_num)]
            else:
                latents_vanilla = [[noise_latents_vanilla]]
                log_probs_vanilla = [[]]
            
            for i, t in enumerate(ts):
                with autocast():
                    with torch.no_grad():
                        if not config.sample.no_branching and ((config.sample.num_steps-i) in split_steps):
                            split_time = split_times[split_steps.index(config.sample.num_steps-i)]
                            new_latents = []
                            new_log_probs = []
                            for latent, log_prob in zip(latents_vanilla, log_probs_vanilla):
                                for _ in range(split_time):
                                    new_latents.append(copy.deepcopy(latent))
                                    new_log_probs.append(copy.deepcopy(log_prob))
                            latents_vanilla = new_latents
                            log_probs_vanilla = new_log_probs
                        
                        for k in range(len(latents_vanilla)):
                            latents_t = latents_vanilla[k][i]
                            latents_input = torch.cat([latents_t] * 2) if config.sample.cfg else latents_t
                            latents_input = pipeline.scheduler.scale_model_input(latents_input, t)
                            
                            noise_pred = pipeline.unet(
                                latents_input,
                                t,
                                encoder_hidden_states=prompt_embeds_vanilla_combine,
                                return_dict=False,
                            )[0]
                            
                            if config.sample.cfg:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            
                            latents_t_1, log_prob, _ = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                t,
                                latents_t,
                                **extra_step_kwargs,
                            )
                            
                            latents_vanilla[k].append(latents_t_1)
                            log_probs_vanilla[k].append(log_prob)
            
            all_latents.extend(latents_vanilla)
            all_log_probs.extend(log_probs_vanilla)
            all_prompt_embeds.extend([prompt_embeds_vanilla] * len(latents_vanilla))
        
        # ========== COMBINE AND SAVE ==========
        sample_num = len(all_latents)
        prompts_repeated = (fk_prompts_expanded * len(latents_fk) if fk_batch_size > 0 else []) + \
                          (vanilla_prompts * len(latents_vanilla) if vanilla_batch_size > 0 else [])
        total_prompts.extend(prompts_repeated)
        
        for k in range(sample_num):
            # Use pre-built prompt_embeds from all_prompt_embeds
            prompt_embeds_k = all_prompt_embeds[k]
            
            images = latents_decode(pipeline, all_latents[k][config.sample.num_steps], accelerator.device, prompt_embeds_k.dtype)
            store_latents = torch.stack(all_latents[k], dim=1)
            store_log_probs = torch.stack(all_log_probs[k], dim=1)
            current_latents = store_latents[:, :-1]
            next_latents = store_latents[:, 1:]
            actual_batch_size = store_latents.shape[0]
            timesteps = pipeline.scheduler.timesteps.repeat(actual_batch_size, 1)
            
            samples.append({
                "prompt_embeds": prompt_embeds_k.cpu().detach(),
                "timesteps": timesteps.cpu().detach(),
                "log_probs": store_log_probs.cpu().detach(),
                "latents": current_latents.cpu().detach(),
                "next_latents": next_latents.cpu().detach(),
                "images": images.cpu().detach()
            })
        
        if (idx+1) % config.sample.save_interval == 0 or idx == (config.sample.num_batches_per_epoch-1):
            os.makedirs(os.path.join(save_dir, "images/"), exist_ok=True)
            print(f'-----------{accelerator.process_index} save image start-----------')
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
    
    if wandb_run:
        wandb_run.log({
            f"sampling/stage_{stage_idx}/total_samples": len(total_prompts),
            f"sampling/stage_{stage_idx}/fk_ratio": fk_mix_ratio,
            f"sampling/stage_{stage_idx}/fk_samples": int(config.sample.num_batches_per_epoch * fk_batch_size),
            f"sampling/stage_{stage_idx}/vanilla_samples": int(config.sample.num_batches_per_epoch * vanilla_batch_size),
        })
    
    if logger:
        logger.info(f"Mixed sampling completed for stage {stage_idx}")
    
    return save_dir
