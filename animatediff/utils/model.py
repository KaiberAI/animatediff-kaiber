import logging
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import Optional, TypeVar
import torch

from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download, snapshot_download
from torch import nn
from tqdm.rich import tqdm

from animatediff.utils.util import path_from_cwd

logger = logging.getLogger(__name__)

IGNORE_TF = ["*.git*", "*.h5", "tf_*"]
IGNORE_FLAX = ["*.git*", "flax_*", "*.msgpack"]
IGNORE_TF_FLAX = IGNORE_TF + IGNORE_FLAX

ALLOW_ST = ["*.safetensors", "*.yaml", "*.md", "*.json"]


class DownloadTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            {
                "ncols": 100,
                "dynamic_ncols": False,
                "disable": None,
            }
        )
        super().__init__(*args, **kwargs)


# for the nop_train() monkeypatch
T = TypeVar("T", bound=nn.Module)

def nop_train(self: T, mode: bool = True) -> T:
    """No-op for monkeypatching train() call to prevent unfreezing module"""
    return self

def checkpoint_to_pipeline(
    checkpoint: Path,
    target_dir: Optional[Path] = None,
    save: bool = True,
) -> StableDiffusionPipeline:
    logger.debug(f"Converting checkpoint {path_from_cwd(checkpoint)}")

    pipeline = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=str(checkpoint.absolute()),
        local_files_only=True,
        load_safety_checker=False,
    )

    if save:
        logger.info(f"Saving pipeline to {path_from_cwd(target_dir)}")
        pipeline.save_pretrained(target_dir, safe_serialization=True)
    return pipeline


def get_checkpoint_weights(checkpoint: Path):
    temp_pipeline: StableDiffusionPipeline
    temp_pipeline = checkpoint_to_pipeline(checkpoint, save=False)
    unet_state_dict = temp_pipeline.unet.state_dict()
    tenc_state_dict = temp_pipeline.text_encoder.state_dict()
    vae_state_dict = temp_pipeline.vae.state_dict()
    return unet_state_dict, tenc_state_dict, vae_state_dict


def get_base_model(model_name_or_path: str, local_dir: Path, force: bool = False):
    model_name_or_path = Path(model_name_or_path)

    model_save_dir = local_dir.joinpath(str(model_name_or_path).split("/")[-1])
    model_is_repo_id = False if model_name_or_path.joinpath("model_index.json").exists() else True

    # if we have a HF repo ID, download it
    if model_is_repo_id:
        logger.debug("Base model is a HuggingFace repo ID")
        if model_save_dir.joinpath("model_index.json").exists():
            logger.debug(f"Base model already downloaded to: {path_from_cwd(model_save_dir)}")
        else:
            logger.info(f"Downloading base model from {model_name_or_path}...")
            _ = get_hf_pipeline(model_name_or_path, model_save_dir.absolute(), save=True)
        model_name_or_path = model_save_dir

    return model_name_or_path
