from packaging import version
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor
import os
import PIL
from torch.utils.data import Dataset
import torch


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


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }


def is_image(file):
    return 'jpg' in file.lower()  or 'png' in file.lower()  or 'jpeg' in file.lower()


class CustomDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        size=512,
        interpolation="bicubic",
        placeholder_token="*",
        template="a photo of a {}",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_token = placeholder_token

        self.image_paths = []
        self.image_paths += [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if is_image(file_path)]

        self.image_paths = sorted(self.image_paths)

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.template = template
        self.clip_image_processor = CLIPImageProcessor()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def _preprocess(self, image):
        return self.transforms(image)

    def _prepare_prompt(self, example: dict):
        text = self.template.format(self.placeholder_token)
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["text"] = text
        example["text_input_ids"] = input_ids
        example["concept_placeholder_idx"] = torch.tensor(self._find_placeholder_index(text))
        return example

    def _prepare_image(self, example: dict, image_path: str):
        raw_image = Image.open(image_path)

        if not raw_image.mode == "RGB":
            raw_image = raw_image.convert("RGB")

        pixel_values_clip = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        pixel_values = self._preprocess(raw_image)
        example["pixel_values"] = pixel_values
        example["pixel_values_clip"] = pixel_values_clip
        return example

    def _find_placeholder_index(self, text: str):
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == self.placeholder_token:
                return idx + 1
        return 0

    def __getitem__(self, idx):
        example = {}
        example = self._prepare_prompt(example)

        image_path = self.image_paths[idx]
        example = self._prepare_image(example, image_path)
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