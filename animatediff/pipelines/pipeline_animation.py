# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import inspect
import os
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import PIL
from torch import nn
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..utils.util import preprocess_image

from ..utils import overlap_policy
from ..utils.path import get_absolute_path

import torchvision
import math

from ..utils.model import nop_train
import cv2
from PIL import Image

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    latents: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        scan_inversions: bool = True,
        scheduler_config: Optional[dict] = None,
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.embeddings_dir = get_absolute_path("models", "embeddings")
        self.embeddings_dict = {}
        self.default_tokens = len(self.tokenizer)
        self.scan_inversions = scan_inversions
        self.scheduler_config = scheduler_config

    def freeze(self):
        logger.debug("Freezing pipeline...")
        _ = self.unet.eval()
        self.unet = self.unet.requires_grad_(False)
        self.unet.train = nop_train

        _ = self.text_encoder.eval()
        self.text_encoder = self.text_encoder.requires_grad_(False)
        self.text_encoder.train = nop_train

        _ = self.vae.eval()
        self.vae = self.vae.requires_grad_(False)
        self.vae.train = nop_train

    def reset_scheduler(self):
        scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_config)
        timesteps = self.scheduler.timesteps

        self.register_modules(
            scheduler=scheduler,
        )

    def update_embeddings(self):
        if not self.scan_inversions:
            return
        names = [p for p in os.listdir(self.embeddings_dir) if p.endswith(".pt")]
        weight = self.text_encoder.text_model.embeddings.token_embedding.weight
        added_embeddings = []
        for name in names:
            embedding_path = os.path.join(self.embeddings_dir, name)
            embedding = torch.load(embedding_path)
            key = os.path.splitext(name)[0]
            if key in self.tokenizer.encoder:
                idx = self.tokenizer.encoder[key]
            else:
                idx = len(self.tokenizer)
                self.tokenizer.add_tokens([key])
            embedding = embedding["string_to_param"]["*"]
            if idx not in self.embeddings_dict:
                added_embeddings.append(name)
                self.embeddings_dict[idx] = torch.arange(
                    weight.shape[0], weight.shape[0] + embedding.shape[0]
                )
                weight = torch.cat(
                    [weight, embedding.to(weight.device, weight.dtype)], dim=0
                )
                self.tokenizer.add_tokens([key])
        if added_embeddings:
            self.text_encoder.text_model.embeddings.token_embedding = nn.Embedding(
                weight.shape[0], weight.shape[1], _weight=weight
            )
            logger.info(f"Added {len(added_embeddings)} embeddings: {added_embeddings}")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def insert_inversions(self, ids, attention_mask):
        larger = ids >= self.default_tokens
        for idx in reversed(torch.where(larger)[1]):
            ids = torch.cat(
                [
                    ids[:, :idx],
                    self.embeddings_dict[ids[:, idx].item()].unsqueeze(0),
                    ids[:, idx + 1 :],
                ],
                1,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask[:, :idx],
                        torch.ones(
                            1,
                            1,
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                        attention_mask[:, idx + 1 :],
                    ],
                    1,
                )
        if ids.shape[1] > self.tokenizer.model_max_length:
            logger.warning(
                f"After inserting inversions, the sequence length is larger than the max length. Cutting off"
                f" {ids.shape[1] - self.tokenizer.model_max_length} tokens."
            )
            ids = torch.cat(
                [ids[:, : self.tokenizer.model_max_length - 1], ids[:, -1:]], 1
            )
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.tokenizer.model_max_length]
        return ids, attention_mask

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        self.update_embeddings()
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_input_ids, attention_mask = self.insert_inversions(
            text_input_ids, attention_mask
        )
        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_input_ids = uncond_input.input_ids
            uncond_input_ids, attention_mask = self.insert_inversions(
                uncond_input_ids, attention_mask
            )
            uncond_embeddings = self.text_encoder(
                uncond_input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        device = self._execution_device
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(
                self.vae.decode(latents[frame_idx : frame_idx + 1].to(device)).sample
            )
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def generate_correlated_noise(self, latents, init_noise_correlation):
        cloned_latents = latents.clone()
        p = init_noise_correlation
        flattened_latents = torch.flatten(cloned_latents)
        noise = torch.randn_like(flattened_latents)
        correlated_noise = flattened_latents * p + math.sqrt(1 - p**2) * noise

        return correlated_noise.reshape(cloned_latents.shape)

    def generate_correlated_latents(self, latents, init_noise_correlation):
        cloned_latents = latents.clone()
        for i in range(1, cloned_latents.shape[2]):
            p = init_noise_correlation
            flattened_latents = torch.flatten(cloned_latents[:, :, i])
            prev_flattened_latents = torch.flatten(cloned_latents[:, :, i - 1])
            correlated_latents = (
                prev_flattened_latents * p/math.sqrt((1+p**2))
                + 
                flattened_latents * math.sqrt(1/(1 + p**2))
            )
            cloned_latents[:, :, i] = correlated_latents.reshape(
                cloned_latents[:, :, i].shape
            )

        return cloned_latents

    def generate_correlated_latents_legacy(self, latents, init_noise_correlation):
        cloned_latents = latents.clone()
        for i in range(1, cloned_latents.shape[2]):
            p = init_noise_correlation
            flattened_latents = torch.flatten(cloned_latents[:, :, i])
            prev_flattened_latents = torch.flatten(cloned_latents[:, :, i - 1])
            correlated_latents = (
                prev_flattened_latents * p
                + 
                flattened_latents * math.sqrt(1 - p**2)
            )
            cloned_latents[:, :, i] = correlated_latents.reshape(
                cloned_latents[:, :, i].shape
            )

        return cloned_latents

    def generate_mixed_noise(self, noise, init_noise_correlation):
        shared_noise = torch.randn_like(noise[0, :, 0])
        for b in range(noise.shape[0]):
            for f in range(noise.shape[2]):
                p = init_noise_correlation
                flattened_latents = torch.flatten(noise[b, :, f])
                shared_latents = torch.flatten(shared_noise)
                correlated_latents = (
                    shared_latents * math.sqrt(p**2/(1+p**2)) + 
                    flattened_latents * math.sqrt(1/(1+p**2))
                )
                noise[b, :, f] = correlated_latents.reshape(noise[b, :, f].shape)

        return noise

    def prepare_latents(
        self,
        init_image,
        init_image_strength,
        init_noise_correlation,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if init_image is not None:
            start_image = (
                (
                    torchvision.transforms.functional.pil_to_tensor(
                        PIL.Image.open(init_image).resize((width, height))
                    )
                    / 255
                )[:3, :, :]
                .to("cuda")
                .to(torch.bfloat16)
                .unsqueeze(0)
            )
            start_image = (
                self.vae.encode(start_image.mul(2).sub(1))
                .latent_dist.sample()
                .view(1, 4, height // 8, width // 8)
                * 0.18215
            )
            init_latents = start_image.unsqueeze(2).repeat(1, 1, video_length, 1, 1)
        else:
            init_latents = None

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device
            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                # ignore init latents for batch model
                latents = [
                    torch.randn(
                        shape, generator=generator[i], device=rand_device, dtype=dtype
                    )
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                if init_latents is not None:
                    offset = int(
                        init_image_strength * (len(self.scheduler.timesteps) - 1)
                    )
                    noise = torch.randn_like(init_latents)
                    noise = self.generate_correlated_latents(
                        noise, init_noise_correlation
                    )

                    # Eric - some black magic here
                    # We should be only adding the noise at timestep[offset], but I noticed that
                    # we get more motion and cooler motion if we add the noise at timestep[offset - 1]
                    # or offset - 2. However, this breaks the fewer timesteps there are, so let's interpolate
                    timesteps = self.scheduler.timesteps
                    average_timestep = None
                    if offset == 0:
                        average_timestep = timesteps[0]
                    elif offset == 1:
                        average_timestep = (
                            timesteps[offset - 1] * (1 - init_image_strength)
                            + timesteps[offset] * init_image_strength
                        )
                    else:
                        average_timestep = timesteps[offset - 1]

                    latents = self.scheduler.add_noise(
                        init_latents, noise, average_timestep.long()
                    )

                    latents = self.scheduler.add_noise(
                        latents, torch.randn_like(init_latents), timesteps[-2]
                    )
                else:
                    latents = torch.randn(
                        shape, generator=generator, device=rand_device, dtype=dtype
                    ).to(device)
                    latents = self.generate_correlated_latents(
                        latents, init_noise_correlation
                    )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if init_latents is None:
            latents = latents * self.scheduler.init_noise_sigma
        elif self.unet.trained_initial_frames and init_latents is not None:
            # we only want to use this as the first frame
            init_latents[:, :, 1:] = torch.zeros_like(init_latents[:, :, 1:])
            
        latents = latents.to(device)
        return latents, init_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        init_image: str = None,
        init_image_strength: Optional[float] = 0.0,
        trained_initial_frames_input_path: str = None,
        trained_initial_frames: Optional[torch.FloatTensor] = None,
        temporal_context: Optional[int] = None,
        strides: int = 3,
        overlap: int = 4,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 2.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        seq_policy=overlap_policy.uniform,
        fp16=False,
        scale_factor=1,
        init_noise_correlation=0,
        interpolate_pos_emb=0,
        is_preview=False,
        preview_latents: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if scale_factor > 1:
            height = height // scale_factor
            width = width // scale_factor

        # round to nearest multiple of 8
        height = int(math.ceil(height / 8) * 8)
        width = int(math.ceil(width / 8) * 8)

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        cpu = torch.device("cpu")
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = (
                negative_prompt
                if isinstance(negative_prompt, list)
                else [negative_prompt] * batch_size
            )
        text_embeddings = self._encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        if do_classifier_free_guidance and image_guidance_scale:
            repeater = 3
            text_embeddings = torch.cat((text_embeddings[:text_embeddings.shape[0]//2], text_embeddings))
        elif do_classifier_free_guidance:
            repeater = 2
        else:
            repeater = 1

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents, init_latents = self.prepare_latents(
            init_image,
            init_image_strength,
            init_noise_correlation,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            torch.bfloat16,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype
        

        if preview_latents:
            print("Using preview latents from file: ", preview_latents)
            latents = torch.load(preview_latents)

        if is_preview:
            # Preview frames only should show the first 16 frames.
            latents = latents[:, :, :16]

        if trained_initial_frames is not None:
            trained_initial_frames_input = trained_initial_frames
        elif trained_initial_frames_input_path != None:
            pil_im = Image.open(trained_initial_frames_input_path).resize((width, height))
            pil_im.save("test.png")
            pixel_values = torchvision.transforms.functional.pil_to_tensor(pil_im)[:3, :, :].unsqueeze(0).to(self.vae.device, dtype=self.vae.dtype) / 255
            # Turn 0 into 255

            pixel_values = pixel_values.mul(2).sub(1)
            
            # Flip pixel values from -1 to 1

            val_init_latents = self.vae.encode(pixel_values).latent_dist
            val_init_latents = val_init_latents.sample()
            trained_initial_frames_input = val_init_latents * 0.18215
            trained_initial_frames_input = trained_initial_frames_input.unsqueeze(2).repeat(repeater, 1, latents.shape[2], 1, 1)
            print(trained_initial_frames_input.shape)
            
            trained_initial_frames_input[:, :, 1:] = torch.zeros_like(trained_initial_frames_input[:, :, 1:])
            
            if image_guidance_scale:
                trained_initial_frames_input[:trained_initial_frames_input.shape[0]//3] = torch.zeros_like(trained_initial_frames_input[:trained_initial_frames_input.shape[0]//3])
                trained_initial_frames_input[-trained_initial_frames_input.shape[0]//3:] = torch.zeros_like(trained_initial_frames_input[-trained_initial_frames_input.shape[0]//3:])

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        total = sum(
            len(
                list(
                    seq_policy(
                        i,
                        num_inference_steps,
                        latents.shape[2],
                        temporal_context,
                        strides,
                        overlap,
                    )
                )
            )
            for i in range(len(timesteps))
        )
        # Initial denoising loop
        # Will be used for preview frames or the initial pass of the higher res generation
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        offset = int(init_image_strength * (len(self.scheduler.timesteps) - 1))


        if not preview_latents:
            with self.progress_bar(total=total) as progress_bar:
                for i, t in enumerate(timesteps[offset:]):
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0]
                            * (repeater if do_classifier_free_guidance else 1),
                            *latents.shape[1:],
                        ),
                        device=latents.device,
                        dtype=latents_dtype,
                    )
                    counter = torch.zeros(
                        (1, 1, latents.shape[2], 1, 1),
                        device=latents.device,
                        dtype=latents_dtype,
                    )
                    for seq in seq_policy(
                        i,
                        num_inference_steps,
                        latents.shape[2],
                        temporal_context,
                        strides,
                        overlap,
                    ):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = (
                            latents[:, :, seq]
                            .to(device)
                            .repeat(repeater if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        )
                        # latent_model_input[0, :, 0] = trained_initial_frames_input[0, :, 0]

                        # latent_model_input[0, :, 0] = self.scheduler.add_noise(
                        #     latent_model_input[0, :, 0],
                        #     torch.randn_like(latent_model_input[0, :, 0]),
                        #     t,
                        # )
                        # print(t)

                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        in_trained_initial_input = None
                        if do_classifier_free_guidance and image_guidance_scale:
                            in_trained_initial_input = trained_initial_frames_input
                            in_trained_initial_input[:in_trained_initial_input.shape[0]//2] = torch.zeros_like(in_trained_initial_input[:in_trained_initial_input.shape[0]//2])
                        else:
                            in_trained_initial_input = trained_initial_frames_input


                        # predict the noise residual
                        with torch.cuda.amp.autocast(enabled=True):
                            pred = self.unet(
                                latent_model_input.to(self.unet.device, self.unet.dtype),
                                t,
                                encoder_hidden_states=text_embeddings,
                                interpolate_pos_emb=interpolate_pos_emb,
                                trained_initial_frames_input=in_trained_initial_input if trained_initial_frames_input_path else None
                            )
                            noise_pred[:, :, seq] += pred.sample.to(
                                dtype=latents_dtype, device=device
                            )
                            counter[:, :, seq] += 1
                            progress_bar.update()
                    # perform guidance
                    if do_classifier_free_guidance:
                        if image_guidance_scale:
                            noise_pred_uncond, noise_pred_image, noise_pred_text = (
                                noise_pred / counter
                            ).chunk(3)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            ) + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        else:
                                noise_pred_uncond, noise_pred_text = (
                                    noise_pred / counter
                                ).chunk(2)
                                noise_pred = noise_pred_uncond + guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        #### HI RES UPSCALE/INIT IMAGE RUN ####
        # We do this because the initial denoising loop is done at a lower resolution
        # However, if we have an init image, the image is already the resolution we want.
        if scale_factor > 1:
            self.reset_scheduler()
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            upscaled_width = int(width * scale_factor)
            upscaled_height = int(height * scale_factor)
            upscaled_latent_width = upscaled_width // self.vae_scale_factor
            upscaled_latent_height = upscaled_height // self.vae_scale_factor

            upscaled_latents = torch.nn.functional.interpolate(
                latents.to(torch.float16),
                size=(latents.shape[2], upscaled_latent_height, upscaled_latent_width),
                mode="trilinear",
                antialias=False,
            )
            latents = upscaled_latents
            latents = latents.to(torch.bfloat16)

            noise = torch.randn_like(latents)

            # mix in the new noise
            init_alpha = 0.4
            offset = int((num_inference_steps - 1) * init_alpha)
            latents = self.scheduler.add_noise(latents, noise, timesteps[offset])

            with self.progress_bar(total=total) as progress_bar:
                for i, t in enumerate(timesteps[offset:]):
                    noise_pred = torch.zeros(
                        (
                            latents.shape[0]
                            * (repeater if do_classifier_free_guidance else 1),
                            *latents.shape[1:],
                        ),
                        device=latents.device,
                        dtype=latents_dtype,
                    )
                    counter = torch.zeros(
                        (1, 1, latents.shape[2], 1, 1),
                        device=latents.device,
                        dtype=latents_dtype,
                    )
                    
                    for seq in seq_policy(
                        i,
                        num_inference_steps,
                        latents.shape[2],
                        temporal_context,
                        strides,
                        overlap,
                    ):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = (
                            latents[:, :, seq]
                            .to(device)
                            .repeat(repeater if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                        )
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )
                        # predict the noise residual
                        pred = self.unet(
                            latent_model_input.to(self.unet.device, self.unet.dtype),
                            t,
                            encoder_hidden_states=text_embeddings,
                            interpolate_pos_emb=interpolate_pos_emb,
                            trained_initial_frames_input=trained_initial_frames_input if trained_initial_frames_input_path != None else None,
                        )
                        noise_pred[:, :, seq] += pred.sample.to(
                            dtype=latents_dtype, device=device
                        )
                        counter[:, :, seq] += 1
                        progress_bar.update()

                    # perform guidance
                    if do_classifier_free_guidance:
                        if image_guidance_scale != None:
                            noise_pred_uncond, noise_pred_image, noise_pred_text = (
                                noise_pred / counter
                            ).chunk(3)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            ) + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                        else:
                            noise_pred_uncond, noise_pred_text = (
                                noise_pred / counter
                            ).chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (
	                            noise_pred_text - noise_pred_uncond
	                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs
                    ).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0
                    ):
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video, latents=latents)
