from .transformers.pixart_conv2d import PixArtConv2DModel
from .unets.stable_diffusion_conv2d import (
    UNet2DConditionConvModel, 
    SD15ConvModel,
    SDXLConvModel
)
from .block_patch import BasicTransformerBlockPatch

from diffusers import PixArtTransformer2DModel, UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention import BasicTransformerBlock
from transformers import T5EncoderModel
import torch
from torch import nn
from torch.amp import autocast
from torch.nn.parallel.distributed import DistributedDataParallel

from typing import Dict, Union
from functools import partial

from hcpdiff.utils import pad_attn_bias, auto_text_encoder_cls
from hcpdiff.diffusion.sampler import BaseSampler
from hcpdiff.models import PixArtWrapper, SD15Wrapper, SDXLWrapper, CFGContext
from hcpdiff.models.wrapper import TEHookCFG, SD15_TEHookCFG, SDXL_TEHookCFG
from rainbowneko.utils import to_cpu, to_cuda

class PixArtDistWrapper(PixArtWrapper):
    def __init__(self, denoiser: UNet2DConditionModel, denoiser_T:UNet2DConditionModel, TE, vae: AutoencoderKL, 
                 noise_sampler: BaseSampler, tokenizer, dtype=torch.bfloat16, min_attnmask=0, 
                 TE_hook_cfg:TEHookCFG=TEHookCFG(clip_skip=0, clip_final_norm=False), 
                 cfg_context=CFGContext(), key_map_in=None, key_map_out=None):
        super().__init__(denoiser, TE, vae, noise_sampler, tokenizer, min_attnmask, TE_hook_cfg, 
                         cfg_context, key_map_in, key_map_out)
        self.denoiser_T = denoiser_T
        self.denoiser_T.eval()

        for name, module in self.denoiser_T.named_modules():
            if isinstance(module, BasicTransformerBlock):
                BasicTransformerBlockPatch.patch_to(module)

    def forward_denoiser(self, x_t, prompt_ids, encoder_hidden_states, timesteps, attn_mask=None, 
                         position_ids=None, resolution=None, aspect_ratio=None, plugin_input={}, **kwargs):
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         encoder_hidden_states=encoder_hidden_states, **plugin_input)
        if hasattr(self.denoiser_T, 'input_feeder'):
            for feeder in self.denoiser_T.input_feeder:
                feeder(input_all)        
        if hasattr(self.denoiser, 'input_feeder'):
            for feeder in self.denoiser.input_feeder:
                feeder(input_all)

        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            model_pred_T = self.denoiser_T(
                hidden_states=x_t,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timesteps,
                encoder_attention_mask=attn_mask,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            feat_list = []
            for block in self.denoiser_T.transformer_blocks:
                feat_list.append(block.hidden_states_attn1)

        with autocast('cuda', dtype=torch.bfloat16):
            model_pred = self.denoiser(
                hidden_states=x_t, 
                encoder_hidden_states=encoder_hidden_states, 
                timestep=timesteps, 
                encoder_attention_mask=attn_mask,
                added_cond_kwargs=added_cond_kwargs,
                feat_T=feat_list
            )
            loss_feat = model_pred.loss_feat
            model_pred = model_pred.sample

        #? remove pred vars for pixart output (see DiT for more)
        model_pred_T, _ = model_pred_T.chunk(2, dim=1)
        model_pred, _ = model_pred.chunk(2, dim=1)

        return model_pred, model_pred_T, loss_feat

    def model_forward(self, prompt_ids, image, attn_mask=None, position_ids=None, neg_prompt_ids=None, neg_attn_mask=None, neg_position_ids=None,
                      plugin_input={}, **kwargs):
        # input prepare
        x_0 = self.get_latents(image)
        x_t, noise, timesteps = self.noise_sampler.add_noise_rand_t(x_0)
        x_t_in = x_t*self.noise_sampler.sigma_scheduler.c_in(timesteps).to(dtype=x_t.dtype).view(-1,1,1,1)
        t_in = self.noise_sampler.sigma_scheduler.c_noise(timesteps)

        if neg_prompt_ids:
            prompt_ids = torch.cat([neg_prompt_ids, prompt_ids], dim=0)
            if neg_attn_mask:
                attn_mask = torch.cat([neg_attn_mask, attn_mask], dim=0)
            if neg_position_ids:
                position_ids = torch.cat([neg_position_ids, position_ids], dim=0)

        # model forward
        x_t_in, t_in = self.cfg_context.pre(x_t_in, t_in)
        encoder_hidden_states = self.forward_TE(
            prompt_ids, t_in, attn_mask=attn_mask,
            plugin_input=plugin_input, **kwargs
        )
        model_pred, model_pred_T, loss_feat = self.forward_denoiser(
            x_t_in, 
            prompt_ids, 
            encoder_hidden_states, 
            t_in, 
            attn_mask=attn_mask, 
            position_ids=position_ids,
            plugin_input=plugin_input, 
            **kwargs
        )
        model_pred = self.cfg_context.post(model_pred)
        model_pred_T = self.cfg_context.post(model_pred_T)

        return dict(
            model_pred=model_pred, 
            model_pred_T=model_pred_T, 
            loss_feat=loss_feat, 
            noise=noise, 
            timesteps=timesteps, 
            x_0=x_0, x_t=x_t, 
            noise_sampler=self.noise_sampler
        )

    @classmethod
    def from_pretrained(cls, models: Union[partial, Dict[str, nn.Module]], **kwargs):
        models = models() if isinstance(models, partial) else models
        return cls(
            models['denoiser'], 
            models['denoiser_T'], 
            models['TE'], 
            models['vae'], 
            models['noise_sampler'], 
            models['tokenizer'], 
            **kwargs
        )


class StableDiffusionDistWrapper(SD15Wrapper):
    def __init__(self, denoiser: UNet2DConditionModel, denoiser_T:UNet2DConditionModel, TE, vae: AutoencoderKL, 
                 noise_sampler: BaseSampler, tokenizer, min_attnmask=0, TE_hook_cfg:TEHookCFG=SD15_TEHookCFG, 
                 cfg_context=CFGContext(), key_map_in=None, key_map_out=None):
        super().__init__(denoiser, TE, vae, noise_sampler, tokenizer, min_attnmask, TE_hook_cfg, 
                         cfg_context, key_map_in, key_map_out)
        self.denoiser_T = denoiser_T
        self.denoiser_T.eval()

        for name, module in self.denoiser_T.named_modules():
            if isinstance(module, BasicTransformerBlock):
                BasicTransformerBlockPatch.patch_to(module)

    def forward_denoiser(self, x_t, prompt_ids, encoder_hidden_states, timesteps, attn_mask=None, position_ids=None, plugin_input={}, **kwargs):
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         encoder_hidden_states=encoder_hidden_states, **plugin_input)
        if hasattr(self.denoiser_T, 'input_feeder'):
            for feeder in self.denoiser_T.input_feeder:
                feeder(input_all)        
        if hasattr(self.denoiser, 'input_feeder'):
            for feeder in self.denoiser.input_feeder:
                feeder(input_all)

        with torch.no_grad(), autocast('cuda', dtype=torch.float16):
            model_pred_T = self.denoiser_T(
                sample=x_t, 
                timestep=timesteps, 
                encoder_hidden_states=encoder_hidden_states, 
                encoder_attention_mask=attn_mask,
            ).sample

            feat_list = []

            # down
            for down_block in self.denoiser_T.down_blocks:
                for attn in getattr(down_block, 'attentions', []):
                    for basic_block in attn.transformer_blocks:
                        assert isinstance(basic_block, BasicTransformerBlock)
                        feat_list.append(basic_block.hidden_states_attn1)
            # mid
            for attn in self.denoiser_T.mid_block.attentions:
                for basic_block in attn.transformer_blocks:
                    assert isinstance(basic_block, BasicTransformerBlock)
                    feat_list.append(basic_block.hidden_states_attn1)
            # up
            for up_block in self.denoiser_T.up_blocks:
                for attn in getattr(up_block, 'attentions', []):
                    for basic_block in attn.transformer_blocks:
                        assert isinstance(basic_block, BasicTransformerBlock)
                        feat_list.append(basic_block.hidden_states_attn1)

        with autocast('cuda', dtype=torch.float16):
            model_pred = self.denoiser(
                sample=x_t, 
                timestep=timesteps, 
                encoder_hidden_states=encoder_hidden_states, 
                encoder_attention_mask=attn_mask,
                feat_T=feat_list
            )
            loss_feat = model_pred.loss_feat
            model_pred = model_pred.sample

        return model_pred, model_pred_T, loss_feat

    def model_forward(self, prompt_ids, image, attn_mask=None, position_ids=None, neg_prompt_ids=None, neg_attn_mask=None, neg_position_ids=None,
                      plugin_input={}, **kwargs):
        # input prepare
        x_0 = self.get_latents(image)
        x_t, noise, timesteps = self.noise_sampler.add_noise_rand_t(x_0)
        x_t_in = x_t*self.noise_sampler.sigma_scheduler.c_in(timesteps).to(dtype=x_t.dtype).view(-1,1,1,1)
        t_in = self.noise_sampler.sigma_scheduler.c_noise(timesteps)

        if neg_prompt_ids:
            prompt_ids = torch.cat([neg_prompt_ids, prompt_ids], dim=0)
            if neg_attn_mask:
                attn_mask = torch.cat([neg_attn_mask, attn_mask], dim=0)
            if neg_position_ids:
                position_ids = torch.cat([neg_position_ids, position_ids], dim=0)

        # model forward
        x_t_in, t_in = self.cfg_context.pre(x_t_in, t_in)
        encoder_hidden_states = self.forward_TE(prompt_ids, t_in, attn_mask=attn_mask, position_ids=position_ids,
                                                plugin_input=plugin_input, **kwargs)
        model_pred, model_pred_T, loss_feat = self.forward_denoiser(
            x_t_in, 
            prompt_ids, 
            encoder_hidden_states, 
            t_in, 
            attn_mask=attn_mask, 
            position_ids=position_ids,
            plugin_input=plugin_input, 
            **kwargs
        )
        model_pred = self.cfg_context.post(model_pred)
        model_pred = self.cfg_context.post(model_pred_T)

        return dict(
            model_pred=model_pred,
            model_pred_T=model_pred_T, 
            loss_feat=loss_feat, 
            noise=noise, 
            timesteps=timesteps, 
            x_0=x_0, x_t=x_t, 
            noise_sampler=self.noise_sampler
        )

    @classmethod
    def from_pretrained(cls, models: Union[partial, Dict[str, nn.Module]], **kwargs):
        models = models() if isinstance(models, partial) else models
        return cls(
            models['denoiser'], 
            models['denoiser_T'], 
            models['TE'], 
            models['vae'], 
            models['noise_sampler'], 
            models['tokenizer'], 
            **kwargs
        )


class SDXLDistWrapper(SDXLWrapper):
    def __init__(self, denoiser: UNet2DConditionModel, denoiser_T:UNet2DConditionModel, TE, 
                 vae: AutoencoderKL, noise_sampler: BaseSampler, tokenizer, min_attnmask=0,
                 TE_hook_cfg:TEHookCFG=SDXL_TEHookCFG, cfg_context=CFGContext(), low_vram=False,
                 key_map_in=None, key_map_out=None):
        super().__init__(denoiser, TE, vae, noise_sampler, tokenizer, min_attnmask, TE_hook_cfg, cfg_context, key_map_in, key_map_out)
        self.denoiser_T = denoiser_T
        self.denoiser_T.eval()
        self.low_vram = low_vram

        for name, module in self.denoiser_T.named_modules():
            if isinstance(module, BasicTransformerBlock):
                BasicTransformerBlockPatch.patch_to(module)

    def forward_denoiser(self, x_t, prompt_ids, encoder_hidden_states, timesteps, added_cond_kwargs, attn_mask=None, position_ids=None,
                         plugin_input={}, **kwargs):
        if attn_mask is not None:
            attn_mask[:, :self.min_attnmask] = 1
            encoder_hidden_states, attn_mask = pad_attn_bias(encoder_hidden_states, attn_mask)

        input_all = dict(prompt_ids=prompt_ids, timesteps=timesteps, position_ids=position_ids, attn_mask=attn_mask,
                         encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, **plugin_input)
        if hasattr(self.denoiser_T, 'input_feeder'):
            for feeder in self.denoiser_T.input_feeder:
                feeder(input_all)        
        if hasattr(self.denoiser, 'input_feeder'):
            for feeder in self.denoiser.input_feeder:
                feeder(input_all)

        with torch.no_grad(), autocast('cuda', dtype=torch.float16):
            if self.low_vram:
                to_cuda(self.denoiser_T)

            model_pred_T = self.denoiser_T(
                sample=x_t,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attn_mask,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            if self.low_vram:
                to_cpu(self.denoiser_T)

            feat_list = []
            # down
            for down_block in self.denoiser_T.down_blocks:
                for attn in getattr(down_block, 'attentions', []):
                    for basic_block in attn.transformer_blocks:
                        assert isinstance(basic_block, BasicTransformerBlock)
                        feat_list.append(basic_block.hidden_states_attn1)
            # mid
            for attn in self.denoiser_T.mid_block.attentions:
                for basic_block in attn.transformer_blocks:
                    assert isinstance(basic_block, BasicTransformerBlock)
                    feat_list.append(basic_block.hidden_states_attn1)
            # up
            for up_block in self.denoiser_T.up_blocks:
                for attn in getattr(up_block, 'attentions', []):
                    for basic_block in attn.transformer_blocks:
                        assert isinstance(basic_block, BasicTransformerBlock)
                        feat_list.append(basic_block.hidden_states_attn1)

        with autocast('cuda', dtype=torch.float16):
            model_pred = self.denoiser(
                sample=x_t, 
                timestep=timesteps, 
                encoder_hidden_states=encoder_hidden_states, 
                encoder_attention_mask=attn_mask,
                added_cond_kwargs=added_cond_kwargs,
                feat_T=feat_list
            )
            loss_feat = model_pred.loss_feat
            model_pred = model_pred.sample

        return model_pred, model_pred_T, loss_feat

    def model_forward(self, prompt_ids, image, attn_mask=None, position_ids=None, neg_prompt_ids=None, neg_attn_mask=None, neg_position_ids=None,
                      crop_info=None, plugin_input={}):
        # input prepare
        x_0 = self.get_latents(image)
        x_t, noise, timesteps = self.noise_sampler.add_noise_rand_t(x_0)
        x_t_in = x_t*self.noise_sampler.sigma_scheduler.c_in(timesteps).to(dtype=x_t.dtype).view(-1,1,1,1)
        t_in = self.noise_sampler.sigma_scheduler.c_noise(timesteps)

        if neg_prompt_ids:
            prompt_ids = torch.cat([neg_prompt_ids, prompt_ids], dim=0)
            if neg_attn_mask:
                attn_mask = torch.cat([neg_attn_mask, attn_mask], dim=0)
            if neg_position_ids:
                position_ids = torch.cat([neg_position_ids, position_ids], dim=0)

        # model forward
        x_t_in, t_in = self.cfg_context.pre(x_t_in, t_in)
        with torch.no_grad():
            encoder_hidden_states, pooled_output = self.forward_TE(prompt_ids, t_in, attn_mask=attn_mask, position_ids=position_ids,
                                                               plugin_input=plugin_input)

        added_cond_kwargs = {"text_embeds":pooled_output, "time_ids":crop_info}
        model_pred, model_pred_T, loss_feat = self.forward_denoiser(
            x_t_in, 
            prompt_ids, 
            encoder_hidden_states, 
            t_in, 
            added_cond_kwargs=added_cond_kwargs,
            attn_mask=attn_mask, 
            position_ids=position_ids, 
            plugin_input=plugin_input
        )
        model_pred = self.cfg_context.post(model_pred)
        model_pred_T = self.cfg_context.post(model_pred_T)

        return dict(
            model_pred=model_pred,
            model_pred_T=model_pred_T, 
            loss_feat=loss_feat, 
            noise=noise, 
            timesteps=timesteps, 
            x_0=x_0, x_t=x_t, 
            noise_sampler=self.noise_sampler
        )

    @classmethod
    def from_pretrained(cls, models: Union[partial, Dict[str, nn.Module]], **kwargs):
        models = models() if isinstance(models, partial) else models
        return cls(
            models['denoiser'], 
            models['denoiser_T'], 
            models['TE'], 
            models['vae'], 
            models['noise_sampler'], 
            models['tokenizer'], 
            **kwargs
        )