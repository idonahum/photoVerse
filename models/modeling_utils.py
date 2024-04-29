import torch
import os


def load_photoverse_model(path, image_adapter, text_adapter, unet):
    state_dict = torch.load(path)
    if "image_adapter" in state_dict:
        image_adapter.load_state_dict(state_dict["image_adapter"])
    if "text_adapter" in state_dict:
        text_adapter.load_state_dict(state_dict["text_adapter"])
    if "cross_attention_adapter" in state_dict:
        unet.load_state_dict(state_dict["cross_attention_adapter"], strict=False)

    return image_adapter, text_adapter, unet


def save_progress(image_adapter, text_adapter, unet, accelerator, output_path, step=None):
    state_dict_image_adapter = accelerator.unwrap_model(image_adapter).state_dict()
    state_dict_text_adapter = accelerator.unwrap_model(text_adapter).state_dict()
    state_dict_cross_attention = {}
    unet_state_dict = accelerator.unwrap_model(unet).state_dict()
    for key, value in unet_state_dict.items():
        if "attn2" in key:
            if "processor" in key or "to_q" in key or "to_k" in key or "to_v" in key:
                state_dict_cross_attention[key] = value
    final_state_dict = {
        "image_adapter": state_dict_image_adapter,
        "text_adapter": state_dict_text_adapter,
        "cross_attention_adapter": state_dict_cross_attention
    }
    if step is not None:
        torch.save(final_state_dict, os.path.join(output_path, f"photoverse_{str(step).zfill(6)}.pt"))
    else:
        torch.save(final_state_dict, os.path.join(output_path, "photoverse.pt"))