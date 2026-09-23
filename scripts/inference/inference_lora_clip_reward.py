#!/usr/bin/env python
"""
Quick inference script to generate images from LoRA checkpoint and compute CLIP reward scores
"""

import os
import sys
import json
import gc
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
# Add project root to path for imports
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
sys.path.append(project_root)
from diffusion.ddim_with_logprob import ddim_step_with_logprob, latents_decode
from utils.utils import seed_everything
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from accelerate import Accelerator
from diffusers.loaders import AttnProcsLayers
import contextlib


def generate_and_evaluate_lora(
    checkpoint_path,
    output_dir,
    prompt_file,
    num_images=32,
    base_model="CompVis/stable-diffusion-v1-4",
    batch_size=4,
    num_inference_steps=20,
    guidance_scale=5.0,
    eta=1.0,
    seed=300,
):
    """
    Generate images from LoRA checkpoint and compute CLIP reward scores
    
    Args:
        checkpoint_path: Path to LoRA weights (.safetensors file)
        output_dir: Directory to save generated images
        prompt_file: Path to JSON file containing prompts
        num_images: Total number of images to generate
        base_model: Base Stable Diffusion model
        batch_size: Batch size for image generation
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        eta: DDIM eta parameter
        seed: Random seed
        
    Returns:
        mean_reward: Mean CLIP similarity score
        std_reward: Standard deviation of CLIP similarity scores
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    print("="*80)
    print("LORA INFERENCE + CLIP REWARD COMPUTATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images to generate: {num_images}")
    print()
    
    seed_everything(seed)
    
    # Load prompts
    print("Loading prompts...")
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================================
    # STEP 1: Load Pipeline and LoRA weights
    # ============================================================================
    

    accelerator_config = ProjectConfiguration(
        project_dir=output_dir, # os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=1,
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        project_config=accelerator_config,
        gradient_accumulation_steps=0,
        log_with="wandb",
    )
    
    print("\n[STEP 1] Load Pipeline and LoRA Weights")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    # disable safety checker
    pipeline.safety_checker = None
    print(f"✓ Base model loaded: {base_model}")
    
    # Setup DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    print(f"✓ DDIM scheduler configured")
    inference_dtype = torch.float16
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    # Setup LoRA
    lora_attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipeline.unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    pipeline.unet.set_attn_processor(lora_attn_procs)
    trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if True and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not True and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model
        
    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if True and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                base_model, revision="main", subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not True and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model
    

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # pipeline.unet.set_attn_processor(lora_attn_procs)
    # pipeline.unet.load_attn_procs(checkpoint_path)
    print(f"✓ LoRA weights loaded")
    
    # Prepare everything with our `accelerator`.
    trainable_layers = accelerator.prepare(trainable_layers)
    accelerator.load_state(checkpoint_path)
    pipeline.unet.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()
    # ============================================================================
    # STEP 2: Generate Images
    # ============================================================================
    print("\n[STEP 2] Generate Images")
    num_batches = (num_images + batch_size - 1) // batch_size
    
    # Generate negative prompt embeddings once
    neg_prompt_ids = pipeline.tokenizer(
        [""],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    neg_prompt_embeds = pipeline.text_encoder(neg_prompt_ids)[0]
    
    autocast = contextlib.nullcontext
    all_images = []
    all_prompts = []
    global_image_idx = 0
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        
        # Sample prompts (cycle through if needed)
        batch_prompts = [prompts[(batch_idx * batch_size + i) % len(prompts)]
                        for i in range(current_batch_size)]
        
        # Encode prompts
        prompt_ids1 = pipeline.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
        prompt_embeds1 = pipeline.text_encoder(prompt_ids1)[0]
        
        # Combine with negative prompts for CFG
        sample_neg_prompt_embeds = neg_prompt_embeds.repeat(current_batch_size, 1, 1)
        # combine prompt and neg_prompt
        prompt_embeds1_combine = pipeline._encode_prompt(
            None,
            accelerator.device,
            1,
            True,
            None,
            prompt_embeds=prompt_embeds1,
            negative_prompt_embeds=sample_neg_prompt_embeds
        )
        
        # Initialize latents
        noise_latents1 = pipeline.prepare_latents(
            current_batch_size, 
            pipeline.unet.config.in_channels, ## channels
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## height
            pipeline.unet.config.sample_size * pipeline.vae_scale_factor, ## width
            prompt_embeds1.dtype, 
            accelerator.device, 
            None ## generator
        )
        
        # Setup scheduler
        pipeline.scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
        timesteps = pipeline.scheduler.timesteps
        extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, eta)
        # Denoising 
        with autocast():
            with torch.no_grad():
                for t in timesteps:
                    # Expand latents for CFG
                    latent_model_input = torch.cat([noise_latents1] * 2)
                    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                    
                    # Predict noise
                    noise_pred = pipeline.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds1_combine,
                        return_dict=False,
                    )[0]
                    
                    # CFG
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    
                    # DDIM step
                    noise_latents1, _, _ = ddim_step_with_logprob(
                        pipeline.scheduler, noise_pred, t, noise_latents1, **extra_step_kwargs
                    )
        
        # Decode latents to images
        images = latents_decode(pipeline, noise_latents1, accelerator.device, prompt_embeds1.dtype)
        
        # Save images and store in memory
        for i, image_tensor in enumerate(images):
            # Save to disk
            image_array = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            img_path = os.path.join(output_dir, f"{global_image_idx:05d}.png")
            pil_image.save(img_path)
            
            # Keep in memory for evaluation
            all_images.append(image_tensor)
            all_prompts.append(batch_prompts[i])
            global_image_idx += 1
    
    print(f"✓ Generated {len(all_images)} images")
    
    # Save prompts
    with open(os.path.join(output_dir, "prompts.json"), 'w') as f:
        json.dump(all_prompts, f, indent=2)
    
    # Clean up generation models
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # ============================================================================
    # STEP 3: Compute CLIP Rewards
    # ============================================================================
    print("\n[STEP 3] Compute CLIP Rewards")
    
    # Load CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained='laion2B-s32B-b79K'
    )
    clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
    clip_model = clip_model.to(accelerator.device)
    clip_model.eval()
    print("✓ CLIP model loaded")
    
    # Compute rewards
    similarity_scores = []
    eval_batch_size = 8
    num_eval_batches = (len(all_images) + eval_batch_size - 1) // eval_batch_size
    
    for batch_idx in tqdm(range(num_eval_batches), desc="Evaluating"):
        start_idx = batch_idx * eval_batch_size
        end_idx = min(start_idx + eval_batch_size, len(all_images))
        
        # Load images from disk and preprocess
        batch_images = []
        batch_prompts = all_prompts[start_idx:end_idx]
        
        for idx in range(start_idx, end_idx):
            img_path = os.path.join(output_dir, f"{idx:05d}.png")
            img = Image.open(img_path)
            batch_images.append(clip_preprocess(img))
        
        image_input = torch.stack(batch_images).to(accelerator.device)
        text_input = clip_tokenizer(batch_prompts).to(accelerator.device)
        
        # Encode and compute similarity
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            text_features = clip_model.encode_text(text_input)
            
            # Normalize
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (diagonal elements = same image-text pairs)
            batch_size_actual = len(image_features_norm)
            similarity = (image_features_norm @ text_features_norm.T)[
                torch.arange(batch_size_actual), torch.arange(batch_size_actual)
            ]
            similarity_scores.extend(similarity.cpu().tolist())
    
    # Convert to tensor for statistics
    similarity_scores = torch.tensor(similarity_scores)
    
    mean_reward = similarity_scores.mean().item()
    std_reward = similarity_scores.std().item()
    
    print(f"✓ CLIP rewards computed")
    
    # ============================================================================
    # Results
    # ============================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Number of images: {len(all_images)}")
    print(f"CLIP Reward Mean: {mean_reward:.4f}")
    print(f"CLIP Reward Std:  {std_reward:.4f}")
    print(f"CLIP Reward Min:  {similarity_scores.min():.4f}")
    print(f"CLIP Reward Max:  {similarity_scores.max():.4f}")
    print("="*80)
    
    # Save results
    results_file = os.path.join(output_dir, "clip_rewards.json")
    results = {
        "checkpoint": checkpoint_path,
        "num_images": len(all_images),
        "clip_reward_mean": mean_reward,
        "clip_reward_std": std_reward,
        "clip_reward_min": float(similarity_scores.min()),
        "clip_reward_max": float(similarity_scores.max()),
        "all_scores": similarity_scores.tolist(),
        "prompts": all_prompts
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    return mean_reward, std_reward


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate images from LoRA and compute CLIP rewards")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/path/to/model/lora/norm_all_no_branching_no_selection_only_5_steps/stage16/checkpoints/checkpoint_1/",
        help="Path to LoRA checkpoint"
    )
    # b2diffu_try2
    # b2diffu_try2
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/output/tmp",
        help="Output directory for images and results"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="configs/prompt/template1_train.json",
        help="Path to prompt file"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=48,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    mean_reward, std_reward = generate_and_evaluate_lora(
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        prompt_file=args.prompt_file,
        num_images=args.num_images,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    from run_inception_score import get_inception_score
    get_inception_score(args.output_dir)