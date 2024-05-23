import torch.nn.functional as F
import torch

from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor
from models.attention_processor import PhotoVerseAttnProcessor2_0, PhotoVerseAttnProcessor


def set_visual_cross_attention_adapter(unet, num_tokens=(5,)):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_processor_class = (
                AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
            )
            attn_procs[name] = attn_processor_class()
        else:
            attn_processor_class = (
                PhotoVerseAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else PhotoVerseAttnProcessor
            )
            attn_procs[name] = attn_processor_class(
                cross_attention_dim=cross_attention_dim,
                hidden_size=hidden_size,
                num_tokens=num_tokens,
            )
    unet.set_attn_processor(attn_procs)
    return unet


def get_visual_cross_attention_values_norm(unet):
    attn_values = []
    for name, attn_processor in unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            continue
        attn_values.append(attn_processor.to_v_ip_norm)
    cross_attn_values_norm = torch.stack(attn_values, dim=1)
    bsz = cross_attn_values_norm.shape[0]
    cross_attn_values_norm = cross_attn_values_norm.view(bsz, -1)
    return cross_attn_values_norm


def set_cross_attention_layers_to_train(unet):
    for name, module in unet.named_modules():
        if 'attn2' in name:
            module.train()
