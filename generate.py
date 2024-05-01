import torch
from models.infer import run_inference
from models.modeling_utils import load_models
from datasets.utils import preprocess_image, prepare_prompt
from transformers import CLIPImageProcessor

from PIL import Image
import os


def preprocess_image_for_inference(image_path, tokenizer, template="a photo of a {}", placeholder_token="*", size=512, interpolation="bicubic"):
    raw_image = Image.open(image_path)
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert("RGB")
    example = prepare_prompt(tokenizer, template, placeholder_token)
    example["pixel_values_clip"] = CLIPImageProcessor()(images=raw_image, return_tensors="pt").pixel_values
    example["pixel_values"] = preprocess_image(raw_image, size=size, interpolation=interpolation)
    return example


if __name__ == "__main__":
    # Load the models and tokenizer
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    extra_num_tokens = 4
    image_encoder_layers_idx = [4, 8, 12, 16]
    guidance_scale = 1
    photoverse_path = "exp1/40k_simple.pt"
    input_image_path = '/home/lab/haimzis/projects/photoVerse/CelebaHQMaskDataset/train/images/23.jpg'
    output_image_path = "generated_image"
    num_timestamps = 100

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter, scheduler = load_models(pretrained_model_name_or_path, extra_num_tokens, photoverse_path)

    # Transfer models and adapters to the specified device
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    image_encoder.to(device)
    image_adapter.to(device)
    text_adapter.to(device)

    # Example input (replace this with actual input)
    example = preprocess_image_for_inference(input_image_path, tokenizer)

    # Run inference
    generated_images = run_inference(example, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae, scheduler, device, image_encoder_layers_idx, guidance_scale, timesteps=num_timestamps)

    # Save or display the images
    for idx, img in enumerate(generated_images):
        img.save(os.path.join("results", f"{output_image_path}{idx}.png"))
