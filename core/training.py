import os
import torch
import contextlib
import copy
import json
import tree
from functools import partial
from tqdm import tqdm as tqdm_lib
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusion.ddim_with_logprob import ddim_step_with_logprob
from utils.utils import load_sample_stage, seed_everything

tqdm = partial(tqdm_lib, dynamic_ncols=True)
logger = get_logger(__name__)


def run_training(config, stage_idx=None, external_logger=None, wandb_run=None, pipeline=None, trainable_layers=None, training_timesteps=None, unet_copy=None):
    """
    Run the training phase for a given stage.
    
    Args:
        config: Configuration object (can be OmegaConf or dict)
        stage_idx: Current stage index (optional, for logging)
        external_logger: External logger instance (optional)
        wandb_run: Existing wandb run to log to (optional)
        pipeline: Pre-loaded StableDiffusionPipeline (avoids reloading)
        trainable_layers: Pre-initialized trainable layers
        unet_copy: Frozen pretrained unet for KL regularization (optional)
        
    Returns:
        save_dir: Directory where checkpoints were saved
    """     
    if external_logger:
        external_logger.info(f"Starting training for stage {stage_idx}")
    else:
        print(f"Starting training for stage {stage_idx}")
    
    print("Starting training")
    torch.cuda.set_device(config.dev_id)
    
    # Setup directories
    unique_id = config.exp_name
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)
    
    stage_id = f"stage{stage_idx}"
    save_dir = os.path.join(config.save_path, unique_id, stage_id)
    
    if pipeline is None or trainable_layers is None:
        raise ValueError("Pipeline and trainable_layers must be provided - should not reload model each stage!")
    
    # Determine effective number of training timesteps for gradient accumulation
    if getattr(config.train, 'incremental_training', False) and training_timesteps is not None:
        num_train_timesteps_2 = int(len(training_timesteps))
    else:
        num_train_timesteps_2 = int(config.split_step * config.train.timestep_fraction)
    
    # Setup accelerator (per-stage, with correct gradient_accumulation_steps)
    accelerator_config = ProjectConfiguration(
        project_dir=save_dir,
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )
    
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps_2,
    )
        
    logger.info(f"\n{config}")
    seed_everything(config.seed)
    

    # Setup save/load hooks for checkpointing (using original pattern)
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
    

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    
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
    
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    
    # Prepare everything with accelerator
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)
    
    # Log training info
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )
    
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.train.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    
    if not config.sample.fk:
        assert config.sample.batch_size >= config.train.batch_size
        
        assert config.sample.batch_size % config.train.batch_size == 0
    else:
        assert (config.sample.batch_size * 4 * 2) >= config.train.batch_size
        
        assert (config.sample.batch_size * 4 * 2) % config.train.batch_size == 0

    # Load sample data
    samples = load_sample_stage(save_dir)
    accelerator.save_state()
    
    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
    init_samples = copy.deepcopy(samples)
    LossRecord = []
    GradRecord = []
    
    # Filter trajectories to only include specified timesteps for uniformly_sample_timesteps
    if config.train.uniformly_sample_timesteps and training_timesteps is not None:
        if external_logger and accelerator.is_local_main_process:
            external_logger.info(f"Filtering trajectories to timesteps: {training_timesteps}")
        
        # Convert training_timesteps to tensor indices if needed
        if isinstance(training_timesteps, list):
            timestep_indices = torch.tensor(training_timesteps, dtype=torch.long)
        else:
            timestep_indices = training_timesteps
        
        # Filter trajectory data to only keep selected timesteps
        for key in ["latents", "next_latents", "log_probs", "timesteps"]:
            if key in samples:
                # samples[key] has shape [batch_size, num_timesteps, ...]
                # We want to keep only the timesteps at the specified indices
                samples[key] = samples[key][:, timestep_indices]

    # Training loop
    for epoch in range(config.train.num_epochs):
        # Shuffle samples along batch dimension
        samples = {}
        LossRecord.append([])
        GradRecord.append([])
        
        total_batch_size = init_samples["eval_scores"].shape[0]
        perm = torch.randperm(total_batch_size)
        for k, v in init_samples.items():
            print(f"{k}: {v.shape}")
        samples = {k: v[perm] for k, v in init_samples.items()}
        
        
        
        # Shuffle timesteps (always shuffle, even for progressive training)
        current_num_timesteps = samples["timesteps"].shape[1]
        perms = torch.stack(
            [torch.randperm(current_num_timesteps) for _ in range(total_batch_size)]
        )
        for key in ["latents", "next_latents", "log_probs", "timesteps"]:
            samples[key] = samples[key][torch.arange(total_batch_size)[:, None], perms]
        
        # Training
        pipeline.unet.train()
        for idx in tqdm(
            range(0, total_batch_size // 2 * 2, config.train.batch_size),
            desc="Update",
            position=2,
            leave=False,
        ):
            LossRecord[epoch].append([])
            GradRecord[epoch].append([])
            
            sample = tree.map_structure(lambda value: value[idx:idx + config.train.batch_size].to(accelerator.device), samples)
            
            sample_batch_size = sample["prompt_embeds"].shape[0]
            train_neg_prompt_embeds = neg_prompt_embed.repeat(sample_batch_size, 1, 1)
            
            # cfg, classifier-free-guidance
            if config.train.cfg:
                embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]
            
            if getattr(config.train, 'incremental_training', False) and training_timesteps is not None:
                timestep_indices = training_timesteps
                if external_logger and accelerator.is_local_main_process and idx == 0:
                    external_logger.info(f"Training on {len(timestep_indices)}/{sample['timesteps'].shape[1]} timesteps: {timestep_indices}")
            else:
                timestep_indices = range(sample["timesteps"].shape[1])
            
            for t in tqdm(
                timestep_indices,
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                evaluation_score = sample["eval_scores"][:]
                
                with accelerator.accumulate(pipeline.unet):
                    with autocast():
                        if config.train.cfg:
                            noise_pred = pipeline.unet(
                                torch.cat([sample["latents"][:, t]] * 2),
                                torch.cat([sample["timesteps"][:, t]] * 2),
                                embeds,
                            ).sample
                            
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        else:
                            noise_pred = pipeline.unet(
                                sample["latents"][:, t], sample["timesteps"][:, t], embeds
                            ).sample
                        
                        _, total_prob, _ = ddim_step_with_logprob(
                            pipeline.scheduler,
                            noise_pred,
                            sample["timesteps"][:, t],
                            sample["latents"][:, t],
                            eta=config.sample.eta,
                            prev_sample=sample["next_latents"][:, t],
                        )
                        total_ref_prob = sample["log_probs"][:, t]
                        
                        ratio = torch.exp(total_prob - total_ref_prob)
                        temp_beta1 = torch.ones_like(evaluation_score) * config.train.beta1
                        temp_beta2 = torch.ones_like(evaluation_score) * config.train.beta2
                        sample_weight = torch.where(evaluation_score > 0, temp_beta1, temp_beta2)
                        advantages = torch.clamp(
                            evaluation_score,
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        ) * sample_weight
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.eps,
                            1.0 + config.train.eps,
                        )
                        rl_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        # KL regularizer against the frozen pretrained unet
                        kl_regularizer_loss = None
                        if getattr(config.train, 'use_kl_div_loss', False):
                            assert unet_copy is not None, "unet_copy must be provided for KL regularization"
                            with torch.no_grad():
                                if config.train.cfg:
                                    old_noise_pred = unet_copy(
                                        torch.cat([sample["latents"][:, t]] * 2).detach(),
                                        torch.cat([sample["timesteps"][:, t]] * 2).detach(),
                                        embeds.detach(),
                                    ).sample
                                    old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
                                    old_noise_pred = old_noise_pred_uncond + config.sample.guidance_scale * (
                                        old_noise_pred_text - old_noise_pred_uncond
                                    )
                                else:
                                    old_noise_pred = unet_copy(
                                        sample["latents"][:, t].detach(),
                                        sample["timesteps"][:, t].detach(),
                                        embeds.detach(),
                                    ).sample
                            kl_regularizer_loss = (noise_pred - old_noise_pred) ** 2
                            loss = config.train.rl_loss_weight * rl_loss + config.train.kl_weight * kl_regularizer_loss.mean()
                        else:
                            loss = rl_loss
                    
                    accelerator.backward(loss)
                    total_norm = None
                    if accelerator.sync_gradients:
                        total_norm = accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm) 
                        # this is working. returns the grad norm before clipping.
                    
                    loss_value = loss.cpu().item()
                    grad_value = total_norm.cpu().item() if total_norm is not None else None
                    
                    log_prob = total_prob
                    ref_log_prob = sample["log_probs"][:, t]
                    approx_kl = 0.5 * torch.mean((log_prob - ref_log_prob) ** 2)
                    clipfrac = torch.mean(
                        (torch.abs(ratio - 1.0) > config.train.eps).float()
                    )
                    
                    LossRecord[epoch][idx // config.train.batch_size].append(loss_value)
                    GradRecord[epoch][idx // config.train.batch_size].append(grad_value)
                    
                    # Log to wandb (only if enabled)
                    if wandb_run and accelerator.is_main_process and config.wandb.enabled:
                        log_dict = {
                            "train/loss": loss_value,
                            "train/epoch": epoch,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                            "train/batch_idx": idx // config.train.batch_size,
                            "train/eval_score_mean": evaluation_score.mean().cpu().item(),
                            "train/eval_score_std": evaluation_score.std().cpu().item(),
                            "train/ratio_mean": ratio.mean().cpu().item(),
                            "train/approx_kl": approx_kl.cpu().item(),
                            "train/clipfrac": clipfrac.cpu().item(),
                        }
                        if kl_regularizer_loss is not None:
                            log_dict["train/kl_regularizer_loss"] = kl_regularizer_loss.mean().cpu().item()
                            log_dict["train/rl_loss"] = rl_loss.cpu().item()
                        if grad_value is not None:
                            log_dict["train/grad_norm"] = grad_value
                        wandb_run.log(log_dict)
                    
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Log epoch summary
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1}/{config.train.num_epochs} completed")
            
            # Log epoch-level metrics (only if enabled)
            if wandb_run and config.wandb.enabled:
                epoch_losses = [item for sublist in LossRecord[epoch] for item in sublist]
                epoch_grads = [item for sublist in GradRecord[epoch] for item in sublist if item is not None]
                wandb_run.log({
                    "train/epoch_loss_mean": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
                    "train/epoch_loss_std": (sum((x - sum(epoch_losses) / len(epoch_losses))**2 for x in epoch_losses) / len(epoch_losses))**0.5 if epoch_losses else 0,
                    "train/epoch_grad_mean": sum(epoch_grads) / len(epoch_grads) if epoch_grads else 0,
                    "train/epoch_completed": epoch + 1,
                })
        
        # Save checkpoint
        if (epoch + 1) % config.train.save_interval == 0:
            accelerator.save_state()
    
    # Save final metrics
    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'loss.json'), 'w') as f:
        json.dump(LossRecord, f)
    with open(os.path.join(save_dir, 'eval', 'grad.json'), 'w') as f:
        json.dump(GradRecord, f)
    
    if external_logger:
        external_logger.info(f"Training completed for stage {stage_idx}")
    
    return save_dir
