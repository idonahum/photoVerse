import torch.nn.functional as F

from diffusers.models.attention_processor import AttnProcessor2_0, AttnProcessor
from models.attention_processor import PhotoVerseAttnProcessor2_0, \
    PhotoVerseAttnProcessor


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