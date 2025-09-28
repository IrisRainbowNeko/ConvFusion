import os
from typing import Optional, List, Union, Tuple

import torch
from torch import nn
from transformers import Gemma3TextConfig, Gemma3TextModel, Gemma3ForCausalLM, Gemma3PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

from cfgs.train.py.blocks import FeatScale, SimpleMLP

class Gemma3ModelScore(Gemma3TextModel):
    def __init__(self, config: Gemma3TextConfig, diffusion_dim, with_noise=False):
        super().__init__(config)
        self.with_noise = with_noise
        self.diffusion_dim = diffusion_dim

    def build_adapter(self):
        self.x0_proj = nn.Linear(self.config.hidden_size, self.diffusion_dim, bias=True)
        self.score_proj = nn.ModuleList([
            nn.Linear(self.config.hidden_size, self.diffusion_dim, bias=True) 
            for _ in range(len(self.layers))
        ])
        self.connector = SimpleMLP(self.diffusion_dim, self.diffusion_dim)

        if self.with_noise:
            self.layer_ht = nn.ModuleList([
                FeatScale((1, 1, self.config.hidden_size)) 
                for _ in range(len(self.layers))
            ])
            for ht in self.layer_ht:
                ht.alpha.data.fill_(0.01)

    def sample(self, ht, t, ht_prev=None, px=False):
        if ht_prev is None:
            return ht

        score_raw = ht - ht_prev
        score = self.score_proj[t](score_raw)

        if self.with_noise:
            noise = torch.randn_like(ht_prev)
            noise = self.layer_ht[t](noise)
            ht = ht + noise

        return ht, score

    def forward_inner(self, hidden_state, attention_mask, position_ids):
        ht_enc_prev = hidden_state

        hidden_state_list = [hidden_state]
        hidden_state_px_list = [hidden_state]
        score_list = []

        for i, module in enumerate(self.layers):
            with torch.no_grad():
                layer_outputs = module(
                    hidden_state, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids
                )
                ht = layer_outputs[0]

                B, L, C = hidden_state.shape
                ht_px = hidden_state.flatten(0, 1).unsqueeze(1)
                layer_outputs_px = module(
                    ht_px, 
                    attention_mask=None, 
                    position_ids=position_ids[:, :1]
                )
                ht_px = layer_outputs_px[0].view(B, L, C)

                ht_px[torch.isnan(ht_px)] = 0.
                ht[torch.isnan(ht)] = 0.
            
            ht_enc = ht - ht_px
            ht_enc, score = self.sample(ht_enc, i, ht_prev=ht_enc_prev)
            ht_enc_prev = ht_enc
            hidden_state = ht

            hidden_state_list.append(ht)
            hidden_state_px_list.append(ht_px)
            score_list.append(score)

        return hidden_state_list, hidden_state_px_list, score_list

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, 
                dtype=torch.long, device=device
            ).unsqueeze(0)

        # 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_state = inputs_embeds
        hidden_state_list, hidden_state_px_list, score_list = self.forward_inner(
            hidden_state, attention_mask, position_ids
        )

        last_hidden_state = self.x0_proj(hidden_state) + sum(score_list)
        last_hidden_state = self.connector(last_hidden_state)

        hidden_state_list[-1] = last_hidden_state
        return hidden_state_list, last_hidden_state


class Gemma3Encoder(Gemma3ForCausalLM):
    def __init__(self, config, diffusion_dim=1280, with_noise=True):
        Gemma3PreTrainedModel.__init__(self, config)
        self.model = Gemma3ModelScore(config, diffusion_dim=diffusion_dim, with_noise=with_noise)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        if hasattr(self, "text_model") and self.text_model is not None:
            return self.text_model.embed_tokens
        return self.model.embed_tokens

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        hidden_state_list, last_hidden_state = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            hidden_states=tuple(hidden_state_list),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], 
                       diffusion_dim=1600, *model_args, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        kwargs_mod = dict(kwargs)
        kwargs_mod.setdefault("torch_dtype", torch.float16)
        if local_rank != -1 and "device_map" not in kwargs_mod:
            kwargs_mod["device_map"] = f"cuda:{local_rank}"

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            diffusion_dim=diffusion_dim,
            **kwargs_mod,
        )
        
        model.text_model = model.model
        del model.model

        model.text_model.final_layer_norm = GemmaRMSNorm(diffusion_dim, eps=model.config.rms_norm_eps)
        model.text_model.final_layer_norm.weight.data = model.text_model.norm.weight.data[:diffusion_dim]
        del model.text_model.norm

        model.text_model.build_adapter()

        return model
