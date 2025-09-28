import os
from typing import Optional, List, Union, Tuple

import torch
from torch import nn
from transformers import Gemma3TextConfig, Gemma3TextModel, Gemma3ForCausalLM, Gemma3PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

# 复用已有的特征缩放和MLP模块
from cfgs.train.py.blocks import FeatScale, SimpleMLP

"""注意力掩码有差异，但是接口自动适配"""
class Gemma3ModelScore(Gemma3TextModel):
    def __init__(self, config: Gemma3TextConfig, diffusion_dim, with_noise=False):
        super().__init__(config)
        self.with_noise = with_noise
        self.diffusion_dim = diffusion_dim

    # 构建与扩散模型衔接的适配器
    def build_adapter(self):
        # 将初始嵌入映射到扩散模型维度
        self.x0_proj = nn.Linear(self.config.hidden_size, self.diffusion_dim, bias=True)
        # 为每一层Transformer创建分数投影层
        self.score_proj = nn.ModuleList([
            nn.Linear(self.config.hidden_size, self.diffusion_dim, bias=True) 
            for _ in range(len(self.layers))
        ])
        # 特征精炼MLP，对齐扩散模型特征分布
        self.connector = SimpleMLP(self.diffusion_dim, self.diffusion_dim)

        if self.with_noise:
            # Langevin采样中的噪声缩放因子（Gemma3隐藏层维度适配）
            self.layer_ht = nn.ModuleList([
                FeatScale((1, 1, self.config.hidden_size)) 
                for _ in range(len(self.layers))
            ])
            # 初始化噪声缩放因子为较小值
            for ht in self.layer_ht:
                ht.alpha.data.fill_(0.01)

    # Langevin采样实现（与Gemma3的前向流程适配）
    def sample(self, ht, t, ht_prev=None, px=False):
        if ht_prev is None:
            return ht

        # 计算原始分数（层间语义变化量）
        score_raw = ht - ht_prev
        # 投影到扩散模型维度
        score = self.score_proj[t](score_raw)

        # 噪声注入（Gemma3对噪声更敏感，保持与原逻辑一致）
        if self.with_noise:
            noise = torch.randn_like(ht_prev)
            noise = self.layer_ht[t](noise)
            ht = ht + noise

        return ht, score

    # Gemma3内部Transformer层的前向计算
    def forward_inner(self, hidden_state, attention_mask, position_ids):
        # 初始化上一层编码状态
        ht_enc_prev = hidden_state

        # 存储句子级和单token级隐状态
        hidden_state_list = [hidden_state]
        hidden_state_px_list = [hidden_state]
        score_list = []

        for i, module in enumerate(self.layers):
            with torch.no_grad():
                # 句子级输入的Transformer层计算（带上下文）
                layer_outputs = module(
                    hidden_state, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids
                )
                ht = layer_outputs[0]

                # 单token级输入的Transformer层计算（无上下文）
                B, L, C = hidden_state.shape
                # Gemma3的token处理方式与Llama一致，保持张量变形逻辑
                ht_px = hidden_state.flatten(0, 1).unsqueeze(1)
                layer_outputs_px = module(
                    ht_px, 
                    attention_mask=None, 
                    position_ids=position_ids[:, :1]
                )
                ht_px = layer_outputs_px[0].view(B, L, C)
                
                # 数值稳定性处理（Gemma3训练中更易出现极端值）
                ht_px[torch.isnan(ht_px)] = 0.
                ht[torch.isnan(ht)] = 0.
            
            # 提取上下文依赖的语义增量
            ht_enc = ht - ht_px
            # 执行Langevin采样
            ht_enc, score = self.sample(ht_enc, i, ht_prev=ht_enc_prev)
            ht_enc_prev = ht_enc
            # 更新下一层输入
            hidden_state = ht

            # 保存中间结果
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

        # 输入验证（复用Gemma的输入处理逻辑）
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # 处理past_key_values（Gemma3的KV缓存格式兼容）
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        # 生成位置ID（Gemma3使用0-based位置编码）
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, 
                dtype=torch.long, device=device
            ).unsqueeze(0)

        # 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 处理注意力掩码（适配Gemma3的掩码要求）
        if getattr(self.config, "_flash_attn_2_enabled", False):
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # 核心前向计算
        hidden_state = inputs_embeds
        hidden_state_list, hidden_state_px_list, score_list = self.forward_inner(
            hidden_state, attention_mask, position_ids
        )

        # 生成最终文本编码（与扩散模型衔接）
        last_hidden_state = self.x0_proj(hidden_state) + sum(score_list)
        last_hidden_state = self.connector(last_hidden_state)

        hidden_state_list[-1] = last_hidden_state
        return hidden_state_list, last_hidden_state


class Gemma3Encoder(Gemma3ForCausalLM):
    def __init__(self, config, diffusion_dim=1280, with_noise=True):
        Gemma3PreTrainedModel.__init__(self, config)
        self.model = Gemma3ModelScore(config, diffusion_dim=diffusion_dim, with_noise=with_noise)

        # 保留语言模型头（如需兼容原始Gemma3功能）
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        # 在from_pretrained加载过程中，HF会在我们尚未重定向为text_model前调用此方法
        # 此处做兼容：优先使用text_model，否则回退到model
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

        # 调用改造后的Gemma3模型
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

        # 输出扩散模型兼容的格式
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            hidden_states=tuple(hidden_state_list),
        )

    # 类似于静态方法，直接通过类名访问
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], 
                       diffusion_dim=1600, *model_args, **kwargs):
        # 设置分布式
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        # 合并并规范可选参数，避免重复关键字（如 device_map / torch_dtype）
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
        
        # 为了统一使用text_model进行访问
        model.text_model = model.model
        del model.model

        # 替换最终归一化层以匹配扩散模型维度
        model.text_model.final_layer_norm = GemmaRMSNorm(diffusion_dim, eps=model.config.rms_norm_eps)
        # text_model.norm有可能不存在？
        model.text_model.final_layer_norm.weight.data = model.text_model.norm.weight.data[:diffusion_dim]
        del model.text_model.norm

        # 构建适配器
        model.text_model.build_adapter()

        # 得到的是Gemma3Encoder实例
        return model
