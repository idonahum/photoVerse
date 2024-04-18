import torch
import torch.utils.checkpoint
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING


from typing import Optional, Tuple, Union


def _inject_concept_embeddings(inputs_embeds, concept_text_embeddings, concept_placeholder_idx):
    new_inputs_embeds = inputs_embeds.clone()
    emb_length = concept_text_embeddings.shape[1] # 5
    for bsz, idx in enumerate(concept_placeholder_idx):
        leftover_length = new_inputs_embeds.shape[1] - emb_length - idx # 77 - 5 - 5 = 67
        new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+leftover_length] # new_inputs_embeds[bsz, 10:] = inputs_embeds[bsz, 6:6+67]
        new_inputs_embeds[bsz, idx: idx+emb_length] = concept_text_embeddings[bsz] # new_inputs_embeds[bsz, 5:10] = concept_text_embeddings[bsz]
    return new_inputs_embeds


@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def clip_text_transformer_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    text_input_ids = input_ids['text_input_ids']
    concept_text_embeddings = input_ids.get('concept_text_embeddings', None)
    concept_placeholder_idx = input_ids.get('concept_placeholder_idx', None)

    input_shape = text_input_ids.size()
    text_input_ids = text_input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embeddings.token_embedding(text_input_ids)
    if concept_text_embeddings is not None:
        new_inputs_embeds = _inject_concept_embeddings(inputs_embeds,concept_text_embeddings, concept_placeholder_idx)
    else:
        new_inputs_embeds = inputs_embeds.clone()

    hidden_states = self.embeddings(input_ids=text_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=text_input_ids.device), text_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask


def patch_clip_text_transformer(text_encoder):
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = clip_text_transformer_forward
    return text_encoder
