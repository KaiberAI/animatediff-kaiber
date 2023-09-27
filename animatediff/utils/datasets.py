import decord

decord.bridge.set_bridge("torch")

from torch.utils.data import Dataset
from einops import rearrange
import os
from glob import glob
import csv
import pandas as pd
import numpy as np

class AnimateDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        prompt: str,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(
            range(self.sample_start_idx, len(vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {"pixel_values": (video / 127.5 - 1.0), "prompt_ids": self.prompt_ids}

        return example


class AnimateWebVidDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        csv_path: str,
        tokenizer,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
    ):
        self.video_paths = [
            y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], "*.mp4"))
        ]

        self.video_to_prompt = {}
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.video_to_prompt[row["videoid"]] = row["name"]

        self.tokenizer = tokenizer
        self.prompt_cache = {}  # Cache for tokenized prompts

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        video_title = os.path.splitext(os.path.basename(video_path))[0]
        prompt = self.video_to_prompt.get(video_title, "")

        # Tokenize the prompt if not cached
        if video_title not in self.prompt_cache:
            self.prompt_cache[video_title] = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]

        prompt_ids = self.prompt_cache[video_title]

        # load and sample video frames
        vr = decord.VideoReader(video_path, width=self.width, height=self.height)
        sample_index = list(
            range(self.sample_start_idx, len(vr), self.sample_frame_rate)
        )[: self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": prompt_ids,
        }

        return example


class AnimatePexelsDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        parquet_path: str,
        tokenizer,
        width: int = 512,
        height: int = 512,
        n_sample_frames: int = 8,
        sample_start_idx: int = 0,
        sample_frame_rate: int = 1,
        should_rescale: bool = False,
        filter_text: str = None,
        verbose = False,
        randomize_index = False,
    ):
        self.video_paths = [
            y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], "*.mp4"))
        ]

        self.video_to_prompt = {}
        data = pd.read_parquet(parquet_path)

        self.filter_indices = []

        for index, row in data.iterrows():
            self.video_to_prompt[index] = row["title"].split("·")[0]

            if filter_text:
                if filter_text in row["title"].lower() and index < len(self.video_paths):
                    self.filter_indices.append(index)

        self.tokenizer = tokenizer
        self.prompt_cache = {}  # Cache for tokenized prompts
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate
        self.should_rescale = should_rescale
        self.verbose = verbose
        self.folder_path = folder_path
        self.randomize_index = randomize_index

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        try:
            if self.randomize_index:
                index = np.random.randint(0, len(self.video_paths))

            if self.filter_indices:
                index = np.random.choice(self.filter_indices)
                video_path = os.path.join(self.folder_path, f"{index}.mp4")
            else:
                video_path = self.video_paths[index]

            video_title = os.path.splitext(os.path.basename(video_path))[0]
            video_index = int(video_title)
            # load and sample video frames
            vr = decord.VideoReader(video_path)

            # make sure length is valid
            if len(vr) <= self.sample_frame_rate * self.n_sample_frames:
                return self.__getitem__(index + 1)

            prompt = self.video_to_prompt.get(video_index, "")

            if self.verbose:
                print(f"Video index: {video_index}")
                print(f"Prompt: {prompt}")

            # Tokenize the prompt if not cached
            if prompt not in self.prompt_cache:
                self.prompt_cache[prompt] = self.tokenizer(
                    prompt,
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids[0]

            prompt_ids = self.prompt_cache[prompt]

            if self.should_rescale:
                frame_height, frame_width, _ = vr[0].shape

                # Determine the scaling factor to preserve the aspect ratio
                scale = max(self.width / frame_width, self.height / frame_height)

                # Compute the new scaled dimensions
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)

                # Reopen the video with the new dimensions
                vr = decord.VideoReader(video_path, width=new_width, height=new_height)

                # Check width and height
                frame_height, frame_width, _ = vr[0].shape
                if frame_height < self.height or frame_width < self.width:
                    return self.__getitem__(index + 1)

                # Select a random start index for the chunk
                start_idx = np.random.randint(0, len(vr) - self.sample_frame_rate * self.n_sample_frames)

                # Compute the sample indices for the random chunk
                sample_index = list(range(start_idx, start_idx + self.sample_frame_rate * self.n_sample_frames))
                video = vr.get_batch(sample_index)

                # Get every other frame in video
                video = video[::self.sample_frame_rate]

                # Compute the cropping dimensions
                frame_height, frame_width, _ = vr[0].shape
                left_margin = (frame_width - self.width) // 2
                top_margin = (frame_height - self.height) // 2 
                video = video[:, top_margin:top_margin + self.height, left_margin:left_margin + self.width, :]
            else:
                # Select a random start index for the chunk
                start_idx = np.random.randint(0, len(vr) - self.sample_frame_rate * self.n_sample_frames)
                
                # Compute the sample indices for the random chunk
                sample_index = list(range(start_idx, start_idx + self.sample_frame_rate * self.n_sample_frames))
                video = vr.get_batch(sample_index)

                # Get every other frame in video
                video = video[::self.sample_frame_rate]

            video = rearrange(video, "f h w c -> f c h w")

            example = {
                "pixel_values": (video / 127.5 - 1.0),
                "prompt_ids": prompt_ids,
            }

            return example
        except Exception as e:
            print(e)
            return self.__getitem__(index + 1)
