# AnimateDiff

This repository is the official implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725).

**[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)**
</br>
Yuwei Guo,
Ceyuan Yang*,
Anyi Rao,
Yaohui Wang,
Yu Qiao,
Dahua Lin,
Bo Dai

<p style="font-size: 0.8em; margin-top: -1em">*Corresponding Author</p>

[Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/)

## Todo
- [x] Code Release
- [x] Arxiv Report
- [x] GPU Memory Optimization
- [ ] Gradio Interface

## Setup for Inference

### Prepare Environment
~~Our approach takes around 60 GB GPU memory to inference. NVIDIA A100 is recommanded.~~

***We updated our inference code with xformers and a sequential decoding trick. Now AnimateDiff takes only ~12GB VRAM to inference, and run on a single RTX3090 !!***

```
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff

conda env create -f environment.yaml
conda activate animatediff
```

### Download Base T2I & Motion Module Checkpoints
We provide two versions of our Motion Module, which are trained on stable-diffusion-v1-4 and finetuned on v1-5 seperately.
It's recommanded to try both of them for best results.
```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/

**Instead of git lfs install (faster):**
wget https://storage.googleapis.com/kaiber_files/animation-StableDiffusion.zip
unzip animation-StableDiffusion.zip

bash download_bashscripts/0-MotionModule.sh

wget https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors -O models/DreamShaper_8_pruned.safetensors
```
You may also directly download the motion module checkpoints from [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing), then put them in `models/Motion_Module/` folder.

### Prepare Personalize T2I
Here we provide inference configs for 6 demo T2I on CivitAI.
You may run the following bash scripts to download these checkpoints.
```
bash download_bashscripts/1-ToonYou.sh
bash download_bashscripts/2-Lyriel.sh
bash download_bashscripts/3-RcnzCartoon.sh
bash download_bashscripts/4-MajicMix.sh
bash download_bashscripts/5-RealisticVision.sh
bash download_bashscripts/6-Tusun.sh
bash download_bashscripts/7-FilmVelvia.sh
bash download_bashscripts/8-GhibliBackground.sh
bash download_bashscripts/9-AdditionalNetworks.sh
```

### Inference
After downloading the above peronalized T2I checkpoints, run the following commands to generate animations. The results will automatically be saved to `samples/` folder.
```
python -m scripts.animate --config configs/prompts/1-ToonYou.yaml
python -m scripts.animate --config configs/prompts/2-Lyriel.yaml
python -m scripts.animate --config configs/prompts/3-RcnzCartoon.yaml
python -m scripts.animate --config configs/prompts/4-MajicMix.yaml
python -m scripts.animate --config configs/prompts/5-RealisticVision.yaml
python -m scripts.animate --config configs/prompts/6-Tusun.yaml
python -m scripts.animate --config configs/prompts/7-FilmVelvia.yaml
python -m scripts.animate --config configs/prompts/8-GhibliBackground.yaml
python -m scripts.animate --config configs/prompts/9-AdditionalNetworks.yml
```

To generate animations with a new DreamBooth/LoRA model, you may create a new config `.yaml` file in the following format:
```
NewModel:
  path: "[path to your DreamBooth/LoRA model .safetensors file]"
  base: "[path to LoRA base model .safetensors file, leave it empty string if not needed]"

  motion_module:
    - "models/Motion_Module/mm_sd_v14.ckpt"
    - "models/Motion_Module/mm_sd_v15.ckpt"
    
  steps:          25
  guidance_scale: 7.5

  prompt:
    - "[positive prompt]"

  n_prompt:
    - "[negative prompt]"
```
Then run the following commands:
```
python -m scripts.animate --config [path to the config file]
```

## Gallery
Here we demonstrate several best results we found in our experiments.

<table class="center">
    <tr>
    <td><img src="__assets__/animations/model_01/01.gif"></td>
    <td><img src="__assets__/animations/model_01/02.gif"></td>
    <td><img src="__assets__/animations/model_01/03.gif"></td>
    <td><img src="__assets__/animations/model_01/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/30240/toonyou">ToonYou</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_02/01.gif"></td>
    <td><img src="__assets__/animations/model_02/02.gif"></td>
    <td><img src="__assets__/animations/model_02/03.gif"></td>
    <td><img src="__assets__/animations/model_02/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4468/counterfeit-v30">Counterfeit V3.0</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_03/01.gif"></td>
    <td><img src="__assets__/animations/model_03/02.gif"></td>
    <td><img src="__assets__/animations/model_03/03.gif"></td>
    <td><img src="__assets__/animations/model_03/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4201/realistic-vision-v20">Realistic Vision V2.0</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_04/01.gif"></td>
    <td><img src="__assets__/animations/model_04/02.gif"></td>
    <td><img src="__assets__/animations/model_04/03.gif"></td>
    <td><img src="__assets__/animations/model_04/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model： <a href="https://civitai.com/models/43331/majicmix-realistic">majicMIX Realistic</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_05/01.gif"></td>
    <td><img src="__assets__/animations/model_05/02.gif"></td>
    <td><img src="__assets__/animations/model_05/03.gif"></td>
    <td><img src="__assets__/animations/model_05/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/66347/rcnz-cartoon-3d">RCNZ Cartoon</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_06/01.gif"></td>
    <td><img src="__assets__/animations/model_06/02.gif"></td>
    <td><img src="__assets__/animations/model_06/03.gif"></td>
    <td><img src="__assets__/animations/model_06/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/33208/filmgirl-film-grain-lora-and-loha">FilmVelvia</a></p>

### Longer generations
You can also generate longer animations by using overlapping sliding windows.
```
python -m scripts.animate --config configs/prompts/{your_config}.yaml --L 64 --context_length 16 
```
##### Sliding window related parameters:

```L``` - the length of the generated animation.

```context_length``` - the length of the sliding window (limited by motion modules capacity), default to ```L```.

```context_overlap``` - how much neighbouring contexts overlap. By default ```context_length``` / 2

```context_stride``` - (2^```context_stride```) is a max stride between 2 neighbour frames. By default 0

##### Extended this way gallery examples

<table class="center">
    <tr>
    <td><img src="__assets__/animations/model_01_4x/01.gif"></td>
    <td><img src="__assets__/animations/model_01_4x/02.gif"></td>
    <td><img src="__assets__/animations/model_01_4x/03.gif"></td>
    <td><img src="__assets__/animations/model_01_4x/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/30240/toonyou">ToonYou</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_03_4x/01.gif"></td>
    <td><img src="__assets__/animations/model_03_4x/02.gif"></td>
    <td><img src="__assets__/animations/model_03_4x/03.gif"></td>
    <td><img src="__assets__/animations/model_03_4x/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4201/realistic-vision-v20">Realistic Vision V2.0</a></p>

#### Community Cases
Here are some samples contributed by the community artists. Create a Pull Request if you would like to show your results here😚.

<table>
    <tr>
    <td><img src="__assets__/animations/model_07/init.jpg"></td>
    <td><img src="__assets__/animations/model_07/01.gif"></td>
    <td><img src="__assets__/animations/model_07/02.gif"></td>
    <td><img src="__assets__/animations/model_07/03.gif"></td>
    <td><img src="__assets__/animations/model_07/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">
Character Model：<a href="https://civitai.com/models/13237/genshen-impact-yoimiya">Yoimiya</a> 
(with an initial reference image, see <a href="https://github.com/talesofai/AnimateDiff">WIP fork</a> for the extended implementation.)

### Training Motion

First, prep your dataset - these are usually a folder of videos and a csv or parquet file with the corresponding metadata (usually just the prompt or caption of the video)

We have a Pexels dataset ready to be downloaded here: https://github.com/KaiberAI/pexels_dataset - to use this just follow the instructions in the Readme.

Training assumes you have a suitable GPU and enough memory - we typically use multiple or single A100 80GBs for larger batch sizes.

To run a training job, you must first construct a config:

    image_finetune: false
    
    output_dir: "outputs"
    pretrained_model_path: "/home/fsuser/motion/models/StableDiffusion/models/StableDiffusion"
    motion_module_checkpoint_path: "/home/fsuser/motion/models/Motion_Module/mm_sd_v14.ckpt"
    unet_checkpoint_path: ""
    dreambooth_model_checkpoint_path: "/home/fsuser/motion/models/AbsoluteReality_1.8.1_pruned.safetensors"
    noise_correlation_alpha: 0
    use_8bit_adam: true
    gradient_accumulation_steps: 1
    max_grad_norm: 1
    snr_gamma: null
    gradient_checkpointing: true
    use_progressive_noise_correlation: true
    dataloader_class_name: AnimatePexelsDataset
    
    unet_additional_kwargs:
      use_motion_module              : true
      motion_module_resolutions      : [ 1,2,4,8 ]
      unet_use_cross_frame_attention : false
      unet_use_temporal_attention    : false
      interpolate_pos_embed              : false
    
      motion_module_type: Vanilla
      motion_module_kwargs:
        num_attention_heads                : 8
        num_transformer_block              : 1
        attention_block_types              : [ "Temporal_Self", "Temporal_Self" ]
        temporal_position_encoding         : true
        temporal_position_encoding_max_len : 40
        temporal_attention_dim_div         : 1
        zero_initialize                    : true
    
    noise_scheduler_kwargs:
      num_train_timesteps: 1000
      beta_start:          0.00085
      beta_end:            0.012
      beta_schedule:       "linear"
      steps_offset:        1
    
    
    train_data:
      parquet_path:    "/home/fsuser/pexels_dataset/PexelVideos.parquet.gzip"
      folder_path:    "/home/fsuser/pexels_dataset/pexels_processed"
      sample_frame_rate:   3
      sample_n_frames: 32
      sample_size: 256
      n_image_frames: 0
    
    validation_data:
      prompts:
        - "dog running with a girl"
        - "dog running to owner"
      num_inference_steps: 25
      guidance_scale: 35
      video_length: 32
    
    trainable_modules:
      - "motion"
    
    learning_rate:    1.e-4
    train_batch_size: 8
    
    max_train_epoch:      -1
    max_train_steps:      20000
    checkpointing_epochs: -1
    checkpointing_steps:  200
    
    validation_steps:       100
    validation_steps_tuple: []
    
    global_seed: 1337
    mixed_precision_training: true
    enable_xformers_memory_efficient_attention: false
    
    is_debug: False

Here are a few tuneable params:

General
- use_motion_module: true to inject the motion module network into the unet
- motion_module_resolutions: resolutions of the unet in which to use in the motion module [1,2,4,8].
- unet_use_cross_frame_attention: Use cross-frame attention in UNet (currently not implemented)
- unet_use_temporal_attention: Use temporal attention in UNet if set to true.
- interpolate_pos_embed: Use interpolated position embedding if set to true. For more context - this refers to the positional embeddings within the motion model's temporal transformer. This is really only relevant if you are trying to fine-tune the mmv14 checkpoint and want to try interpolating its existing positional embeddings instead of recalculating them (you can imagine this like stretching the embeddings)

Motion Module
- motion_module_type: Type of motion module, specified as Vanilla. Do not change this.
- motion_module_kwargs: Keywords for the motion module.
- num_attention_heads: Number of attention heads in the temporal transformers in the motion module.
- num_transformer_block: Number of transformer blocks in the temporal transformers in the motion module.
- attention_block_types: Types of attention blocks. Set as a list ["Temporal_Self", "Temporal_Self"].
- temporal_position_encoding: Use temporal position encoding if set to true. Keep this on.
- temporal_position_encoding_max_len: Maximum length for this model and it's positional encoding.
- temporal_attention_dim_div: Dimension division for temporal attention. 
- zero_initialize: Initialize the out_projections of the motion module to zero if set to true. This lets you not alter the behavior of the unet by default if you are training from scratch.

Noise Scheduler
- These are defaults and should not be changed

Train Data
Please note - these params are specifically for use with AnimatePexelsDataset - if you switch the dataset you may need to change these as they get passed in as args into the dataset initializer.
- parquet_path: Path to the training dataset in parquet format.
- folder_path: Path to the processed training dataset folder.
- sample_frame_rate: Misleading name - this is the stride of your sampling from the videos (3 is every 3rd frame)
- sample_n_frames: Number of frames to extract in total.
- sample_size: Height and width of each sample.
- n_image_frames: Number of image frames. Currently not fully implemented - to be used for joint finetuning (appends n random images to the end of the video)

Validation Data
- prompts: List of prompts for validation.
- num_inference_steps: Number of inference steps for validation. 
- guidance_scale: Guidance scale for validation. 
- video_length: Video length for validation in number of frames.

Training Settings
- trainable_modules: List of modules that are trainable.
- learning_rate: Learning rate for training. Set to 1.e-4.
- train_batch_size: Training batch size. Set to 8.
- max_train_epoch: Maximum training epochs. Set to -1 for no limit.
- max_train_steps: Maximum training steps. Set to 20000.
- checkpointing_epochs: Epochs for checkpointing. Set to -1 for no checkpointing.
- checkpointing_steps: Steps for checkpointing. Set to 200.
- validation_steps: Validation steps. Set to 100.
- validation_steps_tuple: Tuple for validation steps. Left empty if not used. This is used for debugging, if you want to have a validation step early on and sanity check your work.

Other Settings
- global_seed: Global seed for reproducibility. Set to 1337.
- mixed_precision_training: Use mixed-precision training if set to true.
- enable_xformers_memory_efficient_attention: Use memory-efficient attention in Xformers if set to false.
- is_debug: Run in debug mode if set to True.

If you have a multi-GPU cluster, you can run training on each GPU in parallel and sync the gradients with:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 train.py --config <config_path> --wandb`

Alternatively, with a single GPU:

`CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 train.py --config <config_path> --wandb`

Note - wandb is a way for us to track training logs - please register for an account at https://wandb.ai/.

You can initialize wandb with the command `wandb init` and paste your API token in when prompted.
