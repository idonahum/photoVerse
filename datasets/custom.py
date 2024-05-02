from PIL import Image
from transformers import CLIPImageProcessor
import os
from torch.utils.data import Dataset
import torch
import numpy as np
from utils import arcface_utils
from datasets.utils import preprocess_image, prepare_prompt


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]


def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()


class CustomDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        img_subfolder='images',
        size=512,
        interpolation="bicubic",
        placeholder_token="*",
        template="a photo of a {}",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token

        if data_root:
            img_dir = os.path.join(data_root, img_subfolder)
            self.image_paths = [os.path.join(img_dir, file_path) for file_path in os.listdir(img_dir) if is_image(file_path)]
            self.image_paths = sorted(self.image_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
            self.num_images = len(self.image_paths)
            self._length = self.num_images

        self.interpolation = interpolation
        self.template = template
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        example = prepare_prompt(self.tokenizer, self.template, self.placeholder_token)
        example = self._prepare_image(example, idx)
        return example

    def _prepare_image(self, example, idx):
        image_path = self.image_paths[idx]
        raw_image = Image.open(image_path)
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")
        example["pixel_values"] = preprocess_image(raw_image, size=self.size, interpolation=self.interpolation)
        example["pixel_values_clip"] = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        return example
    

class CustomDatasetWithMasks(CustomDataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        img_subfolder='images',
        mask_subfolder='masks',
        size=512,
        interpolation="bicubic",
        placeholder_token="*",
        template="a photo of a {}"
    ):
        super().__init__(data_root=data_root, tokenizer=tokenizer, img_subfolder=img_subfolder,
                         size=size, interpolation=interpolation,
                         placeholder_token=placeholder_token, template=template)

        self.masks_paths = []
        mask_dir = os.path.join(data_root, mask_subfolder)
        self.masks_paths += [os.path.join(mask_dir, file_path) for file_path in os.listdir(mask_dir) if is_image(file_path)]
        self.masks_paths = sorted(self.masks_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))

    def _prepare_image(self, example: dict, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.masks_paths[idx]

        raw_image = Image.open(image_path)
        raw_mask = Image.open(mask_path)

        if not raw_image.mode == "RGB":
            raw_image = raw_image.convert("RGB")

        if not raw_mask.mode == "L":
            raw_mask = raw_mask.convert("L")

        reshaped_img = np.array(raw_image.resize(raw_mask.size))
        mask = np.where(np.array(raw_mask))
        clip_image = np.zeros_like(reshaped_img)
        clip_image[mask] = reshaped_img[mask]

        pixel_values_clip = self.clip_image_processor(images=clip_image, return_tensors="pt").pixel_values
        pixel_values = preprocess_image(raw_image, size=self.size, interpolation=self.interpolation)
        example["pixel_values"] = pixel_values
        example["pixel_values_clip"] = pixel_values_clip

        return example
    

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_values_clip = torch.cat([example["pixel_values_clip"] for example in batch], dim=0)
    input_ids = torch.stack([example["text_input_ids"] for example in batch])
    index = torch.stack([example["concept_placeholder_idx"] for example in batch])
    text = [example["text"] for example in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_values_clip": pixel_values_clip,
        "text_input_ids": input_ids,
        "concept_placeholder_idx": index,
        "text": text,
    }
