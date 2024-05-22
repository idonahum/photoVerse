"""
Script to run inference using pre-trained models.

This script provides the ability to load models and generate images based on a provided input image and text prompt.
Various parameters can be specified via command-line arguments to customize the inference process.
"""

import torch
from models.infer import run_inference
from models.modeling_utils import load_models
from datasets.utils import preprocess_image, prepare_prompt
from transformers import CLIPImageProcessor

from PIL import Image
import os
import argparse

from utils.image_utils import to_pil, denormalize

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Run inference with pre-trained models")
parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5", help="Pretrained model name or path")
parser.add_argument("--extra_num_tokens", type=int, default=4, help="Number of additional tokens")
parser.add_argument("--encoder_layers_idx", nargs="+", type=int, default=[4, 8, 12, 16], help="Indices of image encoder layers")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
parser.add_argument("--checkpoint_path", type=str, default="exp1/40k_simple.pt", help="Path to the model checkpoint")
parser.add_argument("--input_image_path", type=str, default='/home/lab/haimzis/projects/photoVerse/CelebaHQMaskDataset/train/images/23.jpg', help="Path to the input image")
parser.add_argument("--output_image_path", type=str, default="generated_image", help="Prefix for the generated image")
parser.add_argument("--num_timesteps", type=int, default=25, help="Number of timesteps for inference")
parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the generated images")
parser.add_argument("--text", type=str, default="a photo of a {}", help="Prompt template for image generation")
parser.add_argument("--negative_prompt", type=str, default=None, help="Prompt template for negative images")


def preprocess_image_for_inference(image_path, tokenizer, template="a photo of a {}", placeholder_token="*",negative_prompt=None, size=512, interpolation="bicubic"):
    """Preprocess an image for inference.

    Args:
        image_path (str): Path to the input image.
        tokenizer: Tokenizer for the text prompt.
        template (str): Template string for the text prompt.
        placeholder_token (str): Placeholder token used in the template.
        size (int): Size of the output image.
        interpolation (str): Interpolation method for resizing.

    Returns:
        dict: Preprocessed image data for inference.
    """
    raw_image = Image.open(image_path)
    if (raw_image.mode != "RGB"):
        raw_image = raw_image.convert("RGB")
    example = prepare_prompt(tokenizer, template, placeholder_token, negative_prompt=negative_prompt)
    example["pixel_values_clip"] = CLIPImageProcessor()(images=raw_image, return_tensors="pt").pixel_values
    example["pixel_values"] = preprocess_image(raw_image, size=size, interpolation=interpolation).unsqueeze(0)
    return example


if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter, scheduler, _ = load_models(
        args.model_path, args.extra_num_tokens, args.checkpoint_path)

    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    image_encoder.to(device)
    image_adapter.to(device)
    text_adapter.to(device)

    example = preprocess_image_for_inference(args.input_image_path, tokenizer, template=args.text, negative_prompt=args.negative_prompt)

    with torch.no_grad():
        generated_images = run_inference(
            example, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae, scheduler, device,
            args.encoder_layers_idx, guidance_scale=args.guidance_scale, timesteps=args.num_timesteps)
    generated_images = [to_pil(denormalize(img)) for img in generated_images]

    os.makedirs(args.results_dir, exist_ok=True)
    for idx, img in enumerate(generated_images):
        img.save(os.path.join(args.results_dir, f"{args.output_image_path}{idx}.png"))
