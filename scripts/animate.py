import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler, KDPM2AncestralDiscreteScheduler, DEISMultistepScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.pipeline import send_to_device
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import shutil


def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    if args.context_length == 0:
        args.context_length = args.L
    if args.context_overlap == -1:
        args.context_overlap = args.context_length // 2

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/"
    os.makedirs(savedir, exist_ok=True)
    inference_config = OmegaConf.load(args.inference_config)

    config  = OmegaConf.load(args.config)
    print(config)
    samples = []
    
    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
            unet_additional_kwargs = inference_config.unet_additional_kwargs
            use_trained_initial_frames = model_config.get("trained_initial_frames", False)
            if use_trained_initial_frames:
                unet_additional_kwargs.update({'trained_initial_frames': True})

            vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
            tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            unet = UNet3DConditionModel.from_pretrained_2d(
                args.pretrained_model_path, subfolder="unet", 
                unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs),
                motion_module_path=model_config.motion_module
            )

            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
            else: assert False

            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DEISMultistepScheduler.from_config(OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
                scan_inversions=not args.disable_inversions,
                scheduler_config=OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ).to("cuda")

            # 1. unet ckpt
            # 1.1 motion module
            if not use_trained_initial_frames:
              motion_module_state_dict = torch.load(model_config.motion_module, map_location="cpu")
              if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
              missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
              assert len(unexpected) == 0

            # Kaiber checkpoint loading
            if model_config.get("experiment_path", "") != "":
                if model_config.experiment_path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.experiment_path)['state_dict']
                    state_dict = dict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
                    u,m = pipeline.unet.load_state_dict(state_dict, strict=False)
                    print(f"we are missing {m} and we are expecting but don't have {u}")
                elif model_config.experiment_path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.experiment_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint, strict=False)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    
                 
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)
                    
                    # additional networks
                    if hasattr(model_config, 'additional_networks') and len(model_config.additional_networks) > 0:
                        for lora_weights in model_config.additional_networks:
                            add_state_dict = {}
                            (lora_path, lora_alpha) = lora_weights.split(':')
                            print(f"loading lora {lora_path} with weight {lora_alpha}")
                            lora_alpha = float(lora_alpha.strip())
                            with safe_open(lora_path.strip(), framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    add_state_dict[key] = f.get_tensor(key)
                            pipeline = convert_lora(pipeline, add_state_dict, alpha=lora_alpha)
                else:
                    assert False, 'Checkpoint is not loading'
            
            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)['state_dict']
                    state_dict = dict([(k.replace('module.', ''), v) for k, v in state_dict.items()])
                    u,m = pipeline.unet.load_state_dict(state_dict, strict=False)
                    print(f"we are missing {m} and we are expecting but don't have {u}")
                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
                            
                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)                
                    # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint, strict=False)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    
                 
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                    
                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)
                    
                    # additional networks
                    if hasattr(model_config, 'additional_networks') and len(model_config.additional_networks) > 0:
                        for lora_weights in model_config.additional_networks:
                            add_state_dict = {}
                            (lora_path, lora_alpha) = lora_weights.split(':')
                            print(f"loading lora {lora_path} with weight {lora_alpha}")
                            lora_alpha = float(lora_alpha.strip())
                            with safe_open(lora_path.strip(), framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    add_state_dict[key] = f.get_tensor(key)
                            pipeline = convert_lora(pipeline, add_state_dict, alpha=lora_alpha)
                else:
                    assert False, 'Checkpoint is not loading'

            send_to_device(pipeline, "cuda")
            ### <<< create validation pipeline <<< ###

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            init_image   = model_config.init_image if hasattr(model_config, 'init_image') else None
            init_image_strength = model_config.init_image_strength if hasattr(model_config, 'init_image_strength') else 0.0
            
            trained_initial_frames_input_path = model_config.trained_input_image_path if use_trained_initial_frames else None

            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                
                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())
                
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    init_image          = init_image,
                    init_image_strength = init_image_strength,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = args.W,
                    height              = args.H,
                    video_length        = args.L,
                    temporal_context    = args.context_length,
                    strides             = args.context_stride + 1,
                    overlap             = args.context_overlap,
                    scale_factor        = args.scale_factor,
                    fp16                = not args.fp32,
                    interpolate_pos_emb = args.interpolate_pos_emb,
                    init_noise_correlation     = model_config.get("init_noise_correlation", 0.0),
                    trained_initial_frames_input_path = trained_initial_frames_input_path,
                    image_guidance_scale = model_config.get("image_guidance_scale", 0.0),
                ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{config_key}.mp4")
                print(f"save to {savedir}/sample/{config_key}.mp4")
                
                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")
    if init_image is not None:
        shutil.copy(init_image, f"{savedir}/init_image.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v2.yaml")    
    parser.add_argument("--config",                type=str, required=True)

    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--disable_inversions", action="store_true",
                        help="do not scan for downloaded textual inversions")

    parser.add_argument("--context_length", type=int, default=0,
                        help="temporal transformer context length (0 for same as -L)")
    parser.add_argument("--context_stride", type=int, default=0,
                        help="max stride of motion is 2^context_stride")
    parser.add_argument("--context_overlap", type=int, default=-1,
                        help="overlap between chunks of context (-1 for half of context length)")
    parser.add_argument("--scale_factor", type=float, default=1.0,
                        help="scale factor for highres fix")
    parser.add_argument("--interpolate_pos_emb", type=int, default=0,
                        help="specifies a new maximum length aside from the current of 24.")

    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)
