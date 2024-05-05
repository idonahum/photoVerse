from diffusers import DPMSolverMultistepScheduler
from utils.image_utils import denormalize, to_pil
import torch

from tqdm import tqdm


@torch.no_grad()
def run_inference(example, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae, scheduler,
                  device, image_encoder_layers_idx, latent_size=64, guidance_scale=1, timesteps=100, token_index=0,
                  disable_tqdm=False):
    scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    uncond_input = tokenizer(
        [''] * example["pixel_values"].shape[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    latents = torch.randn(
        (example["pixel_values"].shape[0], unet.config.in_channels, latent_size, latent_size)
    )

    scheduler.set_timesteps(timesteps)
    latents = latents.to(device)
    latents = latents * scheduler.init_noise_sigma

    placeholder_idx = example["concept_placeholder_idx"].to(device)
    pixel_values_clip = example["pixel_values_clip"].to(device)

    # get conditional image embeddings and text embeddings
    image_features = image_encoder(pixel_values_clip, output_hidden_states=True)
    uncond_image_features = image_encoder(torch.zeros_like(example["pixel_values_clip"]).to(device),
                                          output_hidden_states=True)

    image_embeddings = [image_features[0]] + [image_features[2][i] for i in image_encoder_layers_idx if
                                              i < len(image_features[2])]
    uncond_image_emmbedings = [uncond_image_features[0]] + [uncond_image_features[2][i] for i in
                                                            image_encoder_layers_idx if
                                                            i < len(uncond_image_features[2])]

    image_embeddings = [emb.detach() for emb in image_embeddings]
    uncond_image_emmbedings = [emb.detach() for emb in uncond_image_emmbedings]

    concept_text_embeddings = text_adapter(image_embeddings, token_index=token_index)
    encoder_hidden_states_image = image_adapter(image_embeddings, token_index=token_index)
    uncond_encoder_hidden_states_image = image_adapter(uncond_image_emmbedings, token_index=token_index)

    uncond_embeddings = text_encoder({'text_input_ids': uncond_input.input_ids.to(device)})[0]
    encoder_hidden_states = text_encoder({'text_input_ids': example["text_input_ids"].to(device),
                                          "concept_text_embeddings": concept_text_embeddings,
                                          "concept_placeholder_idx": placeholder_idx.detach()})[0]

    for t in tqdm(scheduler.timesteps, desc="Denoising", disable=disable_tqdm):
        latent_model_input = scheduler.scale_model_input(latents, t)

        # TODO: fix this error, after refactor methods from this script to shared utils.
        # ERROR:
        # hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        # RuntimeError: shape '[3, 1, 1, 320]' is invalid for input of size 2880

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