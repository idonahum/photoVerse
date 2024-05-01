import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, DDPMScheduler
from models.adapters import PhotoVerseAdapter
from models.clip import patch_clip_text_transformer
from utils.image_utils import denormalize, to_pil
from peft import LoraConfig,inject_adapter_in_model
from models.unet import set_visual_cross_attention_adapter
from datasets.custom import CustomDataset
from PIL import Image
from tqdm import tqdm
import os


def preprocess_image(image_path, tokenizer, template="a photo of a {}",
                     placeholder_token="*", size=512, interpolation="bicubic"):
    """
    Preprocess an image according to the CustomDataset's functionality.

    Args:
        image_path (str): Path to the image to be preprocessed.
        tokenizer (CLIPTokenizer): Tokenizer to use for text input processing.
        template (str): Template string for creating text inputs.
        placeholder_token (str): Placeholder token used in the template.
        size (int): Target size for resizing the image.
        interpolation (str): Interpolation method for resizing.

    Returns:
        dict: Preprocessed image data.
    """
    dataset = CustomDataset(
        data_root="",  # Placeholder value; not used in this context
        tokenizer=tokenizer,
        size=size,
        interpolation=interpolation,
        placeholder_token=placeholder_token,
        template=template
    )
    example = {}

    # Prepare the text prompt
    text = template.format(placeholder_token)
    input_ids = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    example["text"] = text
    example["text_input_ids"] =  input_ids
    example["concept_placeholder_idx"] = torch.tensor([dataset._find_placeholder_index(text)])

    # Prepare the image
    raw_image = Image.open(image_path)
    if not raw_image.mode == "RGB":
        raw_image = raw_image.convert("RGB")

    pixel_values_clip = dataset.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
    example["pixel_values_clip"] = pixel_values_clip

    return example


def load_photoverse_model(path, image_adapter, text_adapter, unet):
    state_dict = torch.load(path)
    if "image_adapter" in state_dict:
        image_adapter.load_state_dict(state_dict["image_adapter"])
    if "text_adapter" in state_dict:
        text_adapter.load_state_dict(state_dict["text_adapter"])
    if "cross_attention_adapter" in state_dict:
        unet.load_state_dict(state_dict["cross_attention_adapter"], strict=False)

    return image_adapter, text_adapter, unet


def load_models(pretrained_model_name_or_path, extra_num_tokens, photoverse_path, device):
    # Load models and tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")

    # Freeze models
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Create adapters
    image_adapter = PhotoVerseAdapter(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embedding_dim=image_encoder.config.hidden_size,
        num_tokens=extra_num_tokens+1
    )
    text_adapter = PhotoVerseAdapter(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embedding_dim=image_encoder.config.hidden_size,
        num_tokens=extra_num_tokens+1
    )

    # Patch the text encoder
    text_encoder = patch_clip_text_transformer(text_encoder)

    # set lora on textual cross attention layers add visual cross attention adapter
    lora_config = LoraConfig(
        lora_alpha=1,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=["attn2.to_k", "attn2.to_v", "attn2.to_q"],
    )
    unet = inject_adapter_in_model(lora_config, unet)
    unet = set_visual_cross_attention_adapter(unet, num_tokens=(extra_num_tokens + 1,))

    # Load pretrained weights into models
    image_adapter, text_adapter, unet = load_photoverse_model(photoverse_path, image_adapter, text_adapter, unet)
    
    # Transfer models and adapters to the specified device
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    image_encoder.to(device)
    image_adapter.to(device)
    text_adapter.to(device)

    return tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter


@torch.no_grad()
def run_inference(example, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae, scheduler, device, image_encoder_layers_idx, guidance_scale, token_index='full'):
    uncond_input = tokenizer(
        [''] * 1,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    
    latents = torch.randn(
        (1, unet.config.in_channels, 64, 64)
    )

    scheduler.set_timesteps(100)
    latents = latents.to(device)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["concept_placeholder_idx"].to(device)
    pixel_values_clip = example["pixel_values_clip"].to(device)

    # get conditional image embeddings and text embeddings
    image_features = image_encoder(pixel_values_clip, output_hidden_states=True)
    image_embeddings = [image_features[0]] + [image_features[2][i] for i in image_encoder_layers_idx if
                                              i < len(image_features[2])]
    
    assert len(image_embeddings) == extra_num_tokens + 1, "Entered indices are out of range for image_encoder layers."
    image_embeddings = [emb.detach() for emb in image_embeddings]
    concept_text_embeddings = text_adapter(image_embeddings)
    encoder_hidden_states_image = image_adapter(image_embeddings)

    # get unconditional image embeddings
    uncond_image_features = image_encoder(torch.zeros_like(example["pixel_values_clip"]).to(device), output_hidden_states=True)
    uncond_image_emmbedings = [uncond_image_features[0]] + [uncond_image_features[2][i] for i in image_encoder_layers_idx if i < len(uncond_image_features[2])]
    assert len(uncond_image_emmbedings) == extra_num_tokens + 1, "Entered indices are out of range for image_encoder layers."
    uncond_image_emmbedings = [emb.detach() for emb in uncond_image_emmbedings]
    uncond_encoder_hidden_states_image = image_adapter(uncond_image_emmbedings)

    if token_index != 'full':
        token_index = int(token_index)
        concept_text_embeddings = concept_text_embeddings[:, token_index:token_index + 1, :]
        encoder_hidden_states_image = encoder_hidden_states_image[:, token_index:token_index + 1, :]
        uncond_encoder_hidden_states_image = uncond_encoder_hidden_states_image[:, token_index:token_index + 1, :]

    uncond_embeddings = text_encoder({'text_input_ids': uncond_input.input_ids.to(device)})[0]
    encoder_hidden_states = text_encoder({'text_input_ids': example["text_input_ids"].to(device),
                                          "concept_text_embeddings": concept_text_embeddings,
                                          "concept_placeholder_idx": placeholder_idx.detach()})[0]

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        latent_model_input = scheduler.scale_model_input(latents, t)

        # Noise prediction based on conditional inputs (text + image)
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states=(encoder_hidden_states, encoder_hidden_states_image)
        ).sample

        # Noise prediction based on unconditional inputs
        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states=(uncond_embeddings, uncond_encoder_hidden_states_image)
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / vae.config.scaling_factor * latents.clone()
    images = vae.decode(_latents).sample
    ret_pil_images = [to_pil(denormalize(image)) for image in images]
    return ret_pil_images


if __name__ == "__main__":
    # Load the models and tokenizer
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    extra_num_tokens = 4
    image_encoder_layers_idx = [4, 8, 12, 16]
    guidance_scale = 7.5
    photoverse_path = "exp1/photoverse.pt"
    input_image_path = '/home/lab/haimzis/projects/photoVerse/CelebaHQMaskDataset/train/images/4.jpg'
    output_image_path = "generated_image"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter = load_models(pretrained_model_name_or_path, extra_num_tokens, photoverse_path, device)

    scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    # Example input (replace this with actual input)
    example = preprocess_image(input_image_path, tokenizer)

    # Run inference
    generated_images = run_inference(example, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae, scheduler, device, image_encoder_layers_idx, guidance_scale)

    # Save or display the images
    for idx, img in enumerate(generated_images):
        img.save(os.path.join("results", f"{output_image_path}{idx}.png"))
