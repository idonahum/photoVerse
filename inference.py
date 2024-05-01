import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from models.adapters import PhotoVerseAdapter
from models.clip import patch_clip_text_transformer
from utils.image_utils import denormalize, to_pil


def load_photoverse_model(path, image_adapter, text_adapter, unet):
    state_dict = torch.load(path)
    if "image_adapter" in state_dict:
        image_adapter.load_state_dict(state_dict["image_adapter"])
    if "text_adapter" in state_dict:
        text_adapter.load_state_dict(state_dict["text_adapter"])
    if "cross_attention_adapter" in state_dict:
        unet.load_state_dict(state_dict["cross_attention_adapter"], strict=False)

    return image_adapter, text_adapter, unet


def load_models(pretrained_model_name_or_path, extra_num_tokens, image_encoder_layers_idx, photoverse_path):
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

    # Load pretrained weights into models
    image_adapter, text_adapter, unet = load_photoverse_model(photoverse_path, image_adapter, text_adapter, unet)

    return tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter


@torch.no_grad()
def run_inference(example, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae, device, image_encoder_layers_idx, extra_num_tokens, guidance_scale):
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=100,
    )

    uncond_input = tokenizer(
        [''] * example["pixel_values"].shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    uncond_embeddings = text_encoder({'text_input_ids':uncond_input.input_ids.to(device)})[0]
    latents = torch.randn((example["pixel_values"].shape[0], unet.in_channels, 64, 64)).to(device)
    scheduler.set_timesteps(100)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["concept_placeholder_idx"].to(device)
    pixel_values_clip = example["pixel_values_clip"].to(device)

    image_features = image_encoder(pixel_values_clip, output_hidden_states=True)
    image_embeddings = [image_features[0]] + [image_features[2][i] for i in image_encoder_layers_idx if i < len(image_features[2])]
    image_embeddings = [emb.detach() for emb in image_embeddings]
    concept_text_embeddings = text_adapter(image_embeddings)
    encoder_hidden_states_image = image_adapter(image_embeddings)
    encoder_hidden_states = text_encoder({'text_input_ids': example["text_input_ids"].to(device),
                                          "concept_text_embeddings": concept_text_embeddings,
                                          "concept_placeholder_idx": placeholder_idx.detach()})[0]

    for t in scheduler.timesteps:
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_text = unet(
            latent_model_input,
            t,
            encoder_hidden_states=(encoder_hidden_states, encoder_hidden_states_image)
        ).sample

        noise_pred_uncond = unet(
            latent_model_input,
            t,
            encoder_hidden_states=uncond_embeddings
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    _latents = 1 / vae.config.scaling_factor * latents.clone()
    images = vae.decode(_latents).sample
    return [to_pil(denormalize(image)) for image in images]


if __name__ == "__main__":
    # Load the models and tokenizer
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    extra_num_tokens = 4
    image_encoder_layers_idx = [4, 8, 12, 16]
    guidance_scale = 7.5
    photoverse_path = "photoverse.pt"

    tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter = load_models(pretrained_model_name_or_path, extra_num_tokens, image_encoder_layers_idx, photoverse_path)

    # Example input (replace this with actual input)
    example = {
        "pixel_values": torch.randn(1, 3, 512, 512),
        "pixel_values_clip": torch.randn(1, 3, 224, 224),
        "concept_placeholder_idx": torch.tensor([0]),
        "text_input_ids": torch.tensor([[0, 1, 2, 3]])
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run inference
    generated_images = run_inference(example, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae, device, image_encoder_layers_idx, extra_num_tokens, guidance_scale)

    # Save or display the images
    for idx, img in enumerate(generated_images):
        img.save(f"generated_image_{idx}.png")
