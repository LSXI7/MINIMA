# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# codes borrowed from https://github.com/kohya-ss/sd-scripts
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.00085
BETA_END = 0.0120

UNET_PARAMS_MODEL_CHANNELS = 320
UNET_PARAMS_CHANNEL_MULT = [1, 2, 4, 4]
UNET_PARAMS_ATTENTION_RESOLUTIONS = [4, 2, 1]
UNET_PARAMS_IMAGE_SIZE = 64  # fixed from old invalid value `32`
UNET_PARAMS_IN_CHANNELS = 4
UNET_PARAMS_OUT_CHANNELS = 4
UNET_PARAMS_NUM_RES_BLOCKS = 2
UNET_PARAMS_CONTEXT_DIM = 768
UNET_PARAMS_NUM_HEADS = 8

VAE_PARAMS_Z_CHANNELS = 4
VAE_PARAMS_RESOLUTION = 256
VAE_PARAMS_IN_CHANNELS = 3
VAE_PARAMS_OUT_CH = 3
VAE_PARAMS_CH = 128
VAE_PARAMS_CH_MULT = [1, 2, 4, 4]
VAE_PARAMS_NUM_RES_BLOCKS = 2

# V2
V2_UNET_PARAMS_ATTENTION_HEAD_DIM = [5, 10, 20, 20]
V2_UNET_PARAMS_CONTEXT_DIM = 1024

DIFFUSERS_REF_MODEL_ID_V1 = 'runwayml/stable-diffusion-v1-5'
DIFFUSERS_REF_MODEL_ID_V2 = 'stabilityai/stable-diffusion-2-1'

CIVITAI_TO_SCEPTER_PARAMS_DICT = {
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'middle_block.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'middle_block.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight':
    'middle_block.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight':
    'middle_block.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'middle_block.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.3.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.4.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_1_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.5.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.6.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.7.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_2_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.8.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.9.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_1_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.10.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_q.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_q.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_k.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_k.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_v.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_v.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_out_0.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn1.to_out.0.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_ff_net_0_proj.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.ff.net.0.proj.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_ff_net_2.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.ff.net.2.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_ff_net_2.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.ff.net.2.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_q.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_q.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_q.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_q.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_k.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_k.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_k.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_k.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_v.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_v.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_v.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_v.lora_B.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.lora_A.0_SwiftLoRA.weight',
    'up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_up.weight':
    'output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.lora_B.0_SwiftLoRA.weight'
}


def convert_tuner_civitai_to_scepter(ckpt):
    params_dict = CIVITAI_TO_SCEPTER_PARAMS_DICT
    swift_lora, lora_config = {}, {}
    lora_config = {
        'alpha_pattern': {},
        'auto_mapping': None,
        'base_model_name_or_path': None,
        'bias': 'none',
        'enable_lora': None,
        'fan_in_fan_out': False,
        'inference_mode': False,
        'init_lora_weights': True,
        'layer_replication': None,
        'layers_pattern': None,
        'layers_to_transform': None,
        'loftq_config': {},
        'lora_alpha': 256,
        'lora_dropout': 0.0,
        'lora_dtype': None,
        'lorap_emb_lr': 1e-06,
        'lorap_lr_ratio': 16.0,
        'megatron_config': None,
        'megatron_core': 'megatron.core',
        'model_key_mapping': None,
        'modules_to_save': None,
        'peft_type': 'LORA',
        'r': 256,
        'rank_pattern': {},
        'revision': None,
        'swift_type': 'LORA',
        'target_modules':
        '(cond_stage_model.*(q_proj|k_proj|v_proj|out_proj|mlp.fc1|mlp.fc2))|(model.*(to_q|to_k|to_v|to_out.0|net.0.proj|net.2))$',  # noqa
        'task_type': None,
        'use_dora': False,
        'use_merged_linear': False,
        'use_qa_lora': False,
        'use_rslora': False,
    }
    alpha, r = -1, -1
    unload_params = []
    for k, v in ckpt.items():
        if k[len('lora_unet_'):] in params_dict and 'lora_unet' in k:
            swift_key = 'model.' + params_dict[k[len('lora_unet_'):]]
            swift_lora[swift_key] = v
        elif 'lora_te' in k:
            # 'lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight'
            # 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.k_proj.lora_A.0_SwiftLoRA.weight'
            swift_key = k.replace('lora_te_',
                                  'transformer.')  # .replace('_', '.')
            swift_key = swift_key.replace('lora_down', 'lora_A').replace(
                'lora_up', 'lora_B')
            swift_key = swift_key.replace('_', '.').replace(
                '.model', '_model').replace('.A', '_A').replace('.B', '_B')
            swift_key = swift_key.replace('self.',
                                          'self_').replace('.proj', '_proj')
            swift_key = 'cond_stage_model.' + swift_key.replace(
                '.weight', '.0_SwiftLoRA.weight')
            swift_lora[swift_key] = v
        elif 'alpha' not in k:
            unload_params.append(k)
        if 'alpha' in k and alpha == -1:
            alpha = v.item()
        if 'lora_down' in k and r == -1:
            r = v.size(0)
    assert r != -1 and alpha != -1
    lora_config['r'] = r
    lora_config['lora_alpha'] = alpha

    return lora_config, swift_lora, unload_params


def convert_lora_checkpoint(ckpt_text=None,
                            ckpt_unet=None,
                            lora_prefix_text='lora_te',
                            lora_prefix_unet='lora_unet'):
    model_dict = {}
    if ckpt_text is not None:
        for key, val in ckpt_text.items():
            key = (lora_prefix_text + '.' + key).replace('.', '_').replace(
                '_0_SwiftLoRA_weight', '')
            if key.endswith('_lora_A'):
                key = key.replace('_lora_A', '.lora_down.weight')
            if key.endswith('_lora_B'):
                key = key.replace('_lora_B', '.lora_up.weight')
            model_dict[key] = val
    if ckpt_unet is not None:
        for key, val in ckpt_unet.items():
            key = (lora_prefix_unet + '.' + key).replace('.', '_').replace(
                '_0_SwiftLoRA_weight', '')
            if key.endswith('_lora_A'):
                key = key.replace('_lora_A', '.lora_down.weight')
            if key.endswith('_lora_B'):
                key = key.replace('_lora_B', '.lora_up.weight')
            model_dict[key] = val

    return model_dict


def create_unet_diffusers_config(v2):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    # unet_params = original_config.model.params.unet_config.params

    block_out_channels = [
        UNET_PARAMS_MODEL_CHANNELS * mult for mult in UNET_PARAMS_CHANNEL_MULT
    ]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = 'CrossAttnDownBlock2D' if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else 'DownBlock2D'
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = 'CrossAttnUpBlock2D' if resolution in UNET_PARAMS_ATTENTION_RESOLUTIONS else 'UpBlock2D'
        up_block_types.append(block_type)
        resolution //= 2

    config = dict(
        sample_size=UNET_PARAMS_IMAGE_SIZE,
        in_channels=UNET_PARAMS_IN_CHANNELS,
        out_channels=UNET_PARAMS_OUT_CHANNELS,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=UNET_PARAMS_NUM_RES_BLOCKS,
        cross_attention_dim=UNET_PARAMS_CONTEXT_DIM
        if not v2 else V2_UNET_PARAMS_CONTEXT_DIM,
        attention_head_dim=UNET_PARAMS_NUM_HEADS
        if not v2 else V2_UNET_PARAMS_ATTENTION_HEAD_DIM,
    )

    return config


def convert_ldm_clip_checkpoint_v1(checkpoint):
    keys = list(checkpoint.keys())
    text_model_dict = {}
    for key in keys:
        if key.startswith('cond_stage_model.transformer'):
            text_model_dict[
                key[len('cond_stage_model.transformer.'):]] = checkpoint[key]
    return text_model_dict


def convert_ldm_unet_tuner_checkpoint(v2,
                                      checkpoint,
                                      config,
                                      unet_key='model.diffusion_model.'):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())
    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, '')] = checkpoint.pop(key)

    new_checkpoint = {}

    if 'time_embed.0.weight' in unet_state_dict:
        new_checkpoint['time_embedding.linear_1.weight'] = unet_state_dict[
            'time_embed.0.weight']
    if 'time_embedding.linear_1.bias' in unet_state_dict:
        new_checkpoint['time_embedding.linear_1.bias'] = unet_state_dict[
            'time_embed.0.bias']
    if 'time_embedding.linear_2.weight' in unet_state_dict:
        new_checkpoint['time_embedding.linear_2.weight'] = unet_state_dict[
            'time_embed.2.weight']
    if 'time_embedding.linear_2.bias' in unet_state_dict:
        new_checkpoint['time_embedding.linear_2.bias'] = unet_state_dict[
            'time_embed.2.bias']

    if 'conv_in.weight' in unet_state_dict:
        new_checkpoint['conv_in.weight'] = unet_state_dict[
            'input_blocks.0.0.weight']
    if 'conv_in.bias' in unet_state_dict:
        new_checkpoint['conv_in.bias'] = unet_state_dict[
            'input_blocks.0.0.bias']

    if 'conv_norm_out.weight' in unet_state_dict:
        new_checkpoint['conv_norm_out.weight'] = unet_state_dict[
            'out.0.weight']
    if 'conv_norm_out.bias' in unet_state_dict:
        new_checkpoint['conv_norm_out.bias'] = unet_state_dict['out.0.bias']
    if 'conv_out.weight' in unet_state_dict:
        new_checkpoint['conv_out.weight'] = unet_state_dict['out.2.weight']
    if 'conv_out.bias' in unet_state_dict:
        new_checkpoint['conv_out.bias'] = unet_state_dict['out.2.bias']

    # Retrieves the keys for the input blocks only
    # num_input_blocks_list = len({".".join(layer.split(".")[]) for layer in unet_state_dict
    # if "input_blocks" in layer})
    num_input_blocks_list = set([
        int(layer.split('.')[1]) for layer in unet_state_dict
        if 'input_blocks' in layer
    ])
    input_blocks = {
        layer_id:
        [key for key in unet_state_dict if f'input_blocks.{layer_id}.' in key]
        for layer_id in num_input_blocks_list
    }

    # Retrieves the keys for the middle blocks only
    # num_middle_blocks_list = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict
    # if "middle_block" in layer})
    num_middle_blocks_list = set([
        int(layer.split('.')[1]) for layer in unet_state_dict
        if 'middle_block' in layer
    ])
    middle_blocks = {
        layer_id:
        [key for key in unet_state_dict if f'middle_block.{layer_id}.' in key]
        for layer_id in num_middle_blocks_list
    }

    # Retrieves the keys for the output blocks only
    # num_output_blocks_list = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict
    # if "output_blocks" in layer})
    num_output_blocks_list = set([
        int(layer.split('.')[1]) for layer in unet_state_dict
        if 'output_blocks' in layer
    ])
    output_blocks = {
        layer_id: [
            key for key in unet_state_dict
            if f'output_blocks.{layer_id}.' in key
        ]
        for layer_id in num_output_blocks_list
    }

    for i in num_input_blocks_list:
        block_id = (i - 1) // (config['layers_per_block'] + 1)
        layer_in_block_id = (i - 1) % (config['layers_per_block'] + 1)

        resnets = [
            key for key in input_blocks[i] if f'input_blocks.{i}.0' in key
            and f'input_blocks.{i}.0.op' not in key
        ]
        attentions = [
            key for key in input_blocks[i] if f'input_blocks.{i}.1' in key
        ]

        if f'input_blocks.{i}.0.op.weight' in unet_state_dict:
            new_checkpoint[
                f'down_blocks.{block_id}.downsamplers.0.conv.weight'] = unet_state_dict.pop(
                    f'input_blocks.{i}.0.op.weight')
            new_checkpoint[
                f'down_blocks.{block_id}.downsamplers.0.conv.bias'] = unet_state_dict.pop(
                    f'input_blocks.{i}.0.op.bias')

        paths = renew_resnet_paths(resnets)
        meta_path = {
            'old': f'input_blocks.{i}.0',
            'new': f'down_blocks.{block_id}.resnets.{layer_in_block_id}'
        }
        assign_to_checkpoint(paths,
                             new_checkpoint,
                             unet_state_dict,
                             additional_replacements=[meta_path],
                             config=config)

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {
                'old': f'input_blocks.{i}.1',
                'new': f'down_blocks.{block_id}.attentions.{layer_in_block_id}'
            }
            assign_to_checkpoint(paths,
                                 new_checkpoint,
                                 unet_state_dict,
                                 additional_replacements=[meta_path],
                                 config=config)

    if 0 in middle_blocks:
        resnet_0 = middle_blocks[0]
        resnet_0_paths = renew_resnet_paths(resnet_0)
        assign_to_checkpoint(resnet_0_paths,
                             new_checkpoint,
                             unet_state_dict,
                             config=config)

    if 2 in middle_blocks:
        resnet_1 = middle_blocks[2]
        resnet_1_paths = renew_resnet_paths(resnet_1)
        assign_to_checkpoint(resnet_1_paths,
                             new_checkpoint,
                             unet_state_dict,
                             config=config)

    if 1 in middle_blocks:
        attentions = middle_blocks[1]
        attentions_paths = renew_attention_paths(attentions)
        meta_path = {'old': 'middle_block.1', 'new': 'mid_block.attentions.0'}
        assign_to_checkpoint(attentions_paths,
                             new_checkpoint,
                             unet_state_dict,
                             additional_replacements=[meta_path],
                             config=config)

    for i in num_output_blocks_list:
        block_id = i // (config['layers_per_block'] + 1)
        layer_in_block_id = i % (config['layers_per_block'] + 1)
        output_block_layers = [
            shave_segments(name, 2) for name in output_blocks[i]
        ]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split('.')[0], shave_segments(
                layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        # if len(output_block_list) > 1:
        if len(output_block_list) > 1 or (len(output_block_list) == 1
                                          and '1' in output_block_list):
            resnets = [
                key for key in output_blocks[i]
                if f'output_blocks.{i}.0' in key
            ]
            attentions = [
                key for key in output_blocks[i]
                if f'output_blocks.{i}.1' in key
            ]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {
                'old': f'output_blocks.{i}.0',
                'new': f'up_blocks.{block_id}.resnets.{layer_in_block_id}'
            }
            assign_to_checkpoint(paths,
                                 new_checkpoint,
                                 unet_state_dict,
                                 additional_replacements=[meta_path],
                                 config=config)

            # if ["conv.weight", "conv.bias"] in output_block_list.values():
            #   index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])

            for v in output_block_list.values():
                v.sort()

            if ['conv.bias', 'conv.weight'] in output_block_list.values():
                index = list(output_block_list.values()).index(
                    ['conv.bias', 'conv.weight'])
                new_checkpoint[
                    f'up_blocks.{block_id}.upsamplers.0.conv.bias'] = unet_state_dict[
                        f'output_blocks.{i}.{index}.conv.bias']
                new_checkpoint[
                    f'up_blocks.{block_id}.upsamplers.0.conv.weight'] = unet_state_dict[
                        f'output_blocks.{i}.{index}.conv.weight']

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    'old': f'output_blocks.{i}.1',
                    'new':
                    f'up_blocks.{block_id}.attentions.{layer_in_block_id}',
                }
                assign_to_checkpoint(paths,
                                     new_checkpoint,
                                     unet_state_dict,
                                     additional_replacements=[meta_path],
                                     config=config)
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers,
                                                n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = '.'.join(['output_blocks', str(i), path['old']])
                new_path = '.'.join([
                    'up_blocks',
                    str(block_id), 'resnets',
                    str(layer_in_block_id), path['new']
                ])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    if v2:
        linear_transformer_to_conv(new_checkpoint)

    return new_checkpoint


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace('in_layers.0', 'norm1')
        new_item = new_item.replace('in_layers.2', 'conv1')

        new_item = new_item.replace('out_layers.0', 'norm2')
        new_item = new_item.replace('out_layers.3', 'conv2')

        new_item = new_item.replace('emb_layers.1', 'time_emb_proj')
        new_item = new_item.replace('skip_connection', 'conv_shortcut')

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({'old': old_item, 'new': new_item})

    return mapping


def linear_transformer_to_conv(checkpoint):
    keys = list(checkpoint.keys())
    tf_keys = ['proj_in.weight', 'proj_out.weight']
    for key in keys:
        if '.'.join(key.split('.')[-2:]) in tf_keys:
            if checkpoint[key].ndim == 2:
                checkpoint[key] = checkpoint[key].unsqueeze(2).unsqueeze(2)


def assign_to_checkpoint(paths,
                         checkpoint,
                         old_checkpoint,
                         attention_paths_to_split=None,
                         additional_replacements=None,
                         config=None):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(
        paths, list
    ), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1,
                            channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config['num_head_channels'] // 3

            old_tensor = old_tensor.reshape((num_heads, 3 * channels //
                                             num_heads) + old_tensor.shape[1:])
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map['query']] = query.reshape(target_shape)
            checkpoint[path_map['key']] = key.reshape(target_shape)
            checkpoint[path_map['value']] = value.reshape(target_shape)

    for path in paths:
        new_path = path['new']

        # These have already been assigned
        if attention_paths_to_split is not None and new_path in attention_paths_to_split:
            continue

        # Global renaming happens here
        new_path = new_path.replace('middle_block.0', 'mid_block.resnets.0')
        new_path = new_path.replace('middle_block.1', 'mid_block.attentions.0')
        new_path = new_path.replace('middle_block.2', 'mid_block.resnets.1')

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement['old'],
                                            replacement['new'])

        # proj_attn.weight has to be converted from conv 1D to linear
        if 'proj_attn.weight' in new_path:
            checkpoint[new_path] = old_checkpoint[path['old']][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path['old']]


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return '.'.join(path.split('.')[n_shave_prefix_segments:])
    else:
        return '.'.join(path.split('.')[:n_shave_prefix_segments])


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
        #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

        #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
        #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

        #         new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({'old': old_item, 'new': new_item})

    return mapping
