

# AnimateDiff

This repository based on the official implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725).
This repository uses some refactors courtesy of https://github.com/neggles/animatediff-cli.

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

Kaiber Contributors/Thanks:

Mariya Vasileva

Ryan Murdock 

Gavin Liu

Jacky Lu

Eric Gao

## Todo
- [] Release trained init image checkpoints
- [] Release frame interpolator checkpoints
- [] Release training code

# Kaiber Upgrades

## Fixed Initial Image

The current implementation of initial images is incorrect as the amount of noise being added per step is incorrect. We have a fix in place which allows for far more range of init image strength values - see at 0.5:

Image:

![thundercat](https://github.com/KaiberAI/animatediff-kaiber/assets/6610675/792b6608-87e7-41ef-82f9-e70b3a634714)

Video:

https://github.com/KaiberAI/animatediff-kaiber/assets/6610675/ff6f39a5-9400-4e9a-840a-d4d30d1e87d0

### Params:

```
  init_image: path to your init image
  init_image_strength: <float>, between 0.0 and 1.0, how much of the initial image should be preserved. this usually comes at a tradeoff to the level of stylization and movement for higher strengths
```

Example Config & Command:

```
init_image_thundercat:
  base: null
  guidance_scale: 15
  init_image: __assets__/eval_images/thundercat.jpg
  init_image_strength: 0.5
  init_noise_correlation: 0.2
  lora_alpha: 1
  motion_module: models/Motion_Module/mm_sd_v15_v2.ckpt
  path: models/DreamBooth_LoRA/DreamShaper_7_pruned.safetensors
  n_prompt:
  - ""
  prompt:
  - 'a man in the water, in illustrated style, anime, soft lighting, beautiful painting, glowing, painterly strokes'
  seed:
  - 1337
  steps: 25
```

`python -m scripts.animate --config config.yaml --L 16`

## High Res Fix

AnimateDiff in it's default state struggles to produce coherent composition at resolutions > 512x512 since the underlying SD backbone was trained on images up to that resolution. We can now generate a low fidelity version of the video and upscale it.

### Params:
```
   scale_factor: <float>, 1.0 or greater
```

This feature uses a new script argument called `scale_factor` - it will create a downscaled version of the video first and then do a latent upscale on it.

To generate a video of 768x768, where it's upscaled from a 512x512 video, you can run, for example:

`python -m scripts.animate --config config.yaml --L 16 --H 768 --W 768 --scale_factor 1.5`

## Noise Correlation/Motion Intensity Controls

By feeding in correlated noise as the latents, we can adjust the the amount of movement within the final output. Credit goes to: https://twitter.com/cloneofsimo/status/1685476776183799808 for the original idea:

### Params:
```
  init_noise_correlation: <float>, recommended to be between 0.0 (default) and 0.25 (less motion)
```

Noise correlation closer to 0 implies more movement (left to right)
![cloneofsimo - 1685476776183799808](https://github.com/KaiberAI/animatediff-kaiber/assets/6610675/47fbb643-54f3-47f1-8be9-4475b788ed73)

## Proper Init Image

TODO: We are currently working on a number of checkpoints to do a "proper" init image start, such that the original image's style and composition is fully preserved. We can retrain with the following architecture:

![image](https://github.com/KaiberAI/animatediff-kaiber/assets/6610675/a7064fa1-7b63-4b90-8f61-e5bf44fae060)

Preliminary results:
https://github.com/KaiberAI/animatediff-kaiber/assets/6610675/1f4c7533-9706-48e0-b9eb-b60b932a60df

### Challenges:
- Currently we see a degradation of style expression within the model - this is likely because we are finetuning fully on the video data. We will be running a joint finetune soon so that the base model can preserve it's prior knowledge.

## Frame Interpolator

TODO: We are currently working on a separate frame interpolator, which will correct and generate smoother in between movements. This is still in early training stages - stay tuned for more info

## Training: 

Coming soon!


## Setup for Inference

### Prepare Environment

```
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff

conda create -n animatediff python=3.10.9
conda activate animatediff
pip install -r requirements.txt
```

### Download Base T2I & Motion Module Checkpoints
There are three versions of the Motion Module, which are trained on stable-diffusion-v1-4 and finetuned on v1-5 seperately.
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

You can now run a test with the following command:
`python -m scripts.animate --config config.yaml --pretrained_model_path models/StableDiffusion --inference_config configs/inference/inference-v2.yaml`

Replace the relevant paths with your own - remember that the v2 motion module only works with inference-v2.yaml, and the v14 and v15 original motion modules need the inference-y1.yaml to work.

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

