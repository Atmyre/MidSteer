import logging
import numpy as np
import torch
import abc
from collections import defaultdict
from typing import Optional, Dict, Any

import enum
import torch.nn.functional as F
from utils import fractional_matrix_power_cov_torch


logger = logging.getLogger()

EPS = 1e-6

class VectorControlMode(enum.StrEnum):
    ATTN_OUTPUT = 'attn_output'
    ATTN_HEADS = 'attn_head'


# Define Controller for BasicTransformerBlock
class VectorControl(abc.ABC):
    def __init__(self, mode: VectorControlMode):
        self._mode = mode
        self._active = True
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        self.num_attn_layers = -1

    @property
    def active(self) -> bool:
        return self._active
    
    @active.setter
    def active(self, value: bool):
        self._active = value
    
    def reset(self):
        self._diffusion_step = 0

    def between_steps(self, last_diffusion_step: int):
        pass
    
    @abc.abstractmethod
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):
        raise NotImplementedError

    def __call__(self, vector, place_in_unet: str):
        block_index = self._current_position[place_in_unet]
        vector = self.forward(vector, self._diffusion_step, place_in_unet, block_index)
        self._current_position[place_in_unet] += 1

        self._current_attn_layer += 1
        if self._current_attn_layer == self.num_attn_layers:
            self._current_attn_layer = 0
            self._current_position = defaultdict(int)
            self.between_steps(self._diffusion_step)
            self._diffusion_step += 1
        return vector

class CrossAttentionOutputSteering(VectorControl):
    def __init__(
        self,
        mode: VectorControlMode,
        *,
        casteer_vectors=None,
        mmsteer_vectors=None,
        steer_type: str = None,
        
        mmsteer_threshold: float,
        steer_only_up=False, 
        alpha: float = 10.0,
        beta: float = 2.0, 
        steer_back: bool = False,
        device: Any,
    ):
        super().__init__(mode=mode)
        self.device = device
        
        if casteer_vectors is not None:
            self.casteer_vectors = defaultdict(lambda: defaultdict(list))
            for num_steer in casteer_vectors:
                for place_in_unet in casteer_vectors[num_steer]:
                    for block_idx in range(len(casteer_vectors[num_steer][place_in_unet])):
                        b = casteer_vectors[num_steer][place_in_unet][block_idx]
                        b = torch.tensor(b).half().to(self.device)
                        self.casteer_vectors[num_steer][place_in_unet].append(b)
        else:
            self.casteer_vectors = None

        if mmsteer_vectors is not None:
            self.mmsteer_vectors = defaultdict(lambda: defaultdict(list))
            for num_steer in mmsteer_vectors:
                for place_in_unet in mmsteer_vectors[num_steer]:
                    for block_idx in range(len(mmsteer_vectors[num_steer][place_in_unet])):
                        W, b = mmsteer_vectors[num_steer][place_in_unet][block_idx]
                        W = torch.tensor(W).half().to(self.device)
                        b = torch.tensor(b).half().to(self.device)
                        self.mmsteer_vectors[num_steer][place_in_unet].append((W, b))
        else:
            self.mmsteer_vectors = None
        
        self.mmsteer_threshold = mmsteer_threshold
        
        self.steer_only_up = steer_only_up
        self.alpha = alpha
        self.beta = beta
        self.steer_back = steer_back
        self.steer_type = steer_type

        self.steering_cache = {}


    def steer_CASteer(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        batch_size = vector.shape[0]
        hidden_dim = vector.shape[-1]
        (b,) = steering_tensors

        assert len(b.shape) in (1, 2)
        if len(b.shape) == 1:  # Old code, add a num_heads dim
            b = b.reshape(1, -1)
        num_heads = b.shape[0]

        if self.steer_back:
            # steering backward, i.e. removing notion from vector

            # computing dot products between vector components and steering vector x
            sim = (vector.reshape(-1, num_heads, hidden_dim).transpose(0, 1) @ b.unsqueeze(-1)).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)

            # we will steer back only if dot product is positive, i.e.
            # if there's positive amount of information from steering vector in the vector
            sim = torch.where(sim>0, sim, 0)

            # steer backward for beta*sim
            vector -= self.beta * sim * b
        else:
            # steer forward, i.e. add a steering vector x multiplied by self.intensity
            vector += self.alpha * b
        
        return vector

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int):

        if place_in_unet in ['up', 'mid'] or (place_in_unet == 'down' and not self.steer_only_up): 
            # if steering vectors are from turbo version, then there's only one key in self.steering_vectors, 
            # and we'll use it for all the steps of generation
            # if steering vectors are from full version, then there's a key in self.steering_vectors
            # for each of the generation steps 
            num_steer = 0 #if len(list(self.steering_vectors.keys()))==1 else diffusion_step

            norm = torch.norm(vector, dim=-1, keepdim=True)
            if self.steer_type == 'casteer':
                vector = self.steer_CASteer(vector, self.casteer_vectors[num_steer][place_in_unet][block_index])
                vector = vector / (torch.norm(vector, dim=-1, keepdim=True) + EPS)
                vector = vector * norm
            elif self.steer_type == 'mmsteer':
                pos = (num_steer, place_in_unet, block_index)
                if pos in self.steering_cache:
                    W_alpha, b_alpha = self.steering_cache[*pos]
                else:
                    (W, b) = self.mmsteer_vectors[num_steer][place_in_unet][block_index]
                    if len(W.shape) == 2:
                        W = W[None, ...]
                        b = b[None, :]

                    if self.alpha != 1.0:
                        W = W.float()
                        b = b.float()
                        I = torch.eye(W.shape[1], device=W.device)[None, ...]
                        W_alpha = fractional_matrix_power_cov_torch(W, self.alpha)
                        b_alpha = ((I - W_alpha) @ (I - W).inverse() @ b[..., None])[..., 0]
                        W_alpha = W_alpha.half()
                        b_alpha = b_alpha.half()
                    else:
                        W_alpha, b_alpha = W, b

                    self.steering_cache[*pos] = W_alpha, b_alpha

                num_heads = W_alpha.shape[0]
                hidden_dim = W_alpha.shape[1]
                batch_size = vector.shape[0]
                sequence_length = vector.shape[1]

                vector_steered = ((vector.reshape(-1, num_heads, hidden_dim).transpose(0, 1) @ W_alpha.mT) + b_alpha.unsqueeze(1)).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim) 

                # TODO: the code below was not rewritten in the batched fashion
                if self.casteer_vectors is not None:

                    b_casteer = -1 * self.casteer_vectors[num_steer][place_in_unet][block_index].view(1, 1, -1)

                    sim = torch.tensordot(vector / norm , b_casteer,
                                          dims=([2], [2])).view(vector.size()[0], vector.size()[1], 1)

#                     sim = torch.clamp(sim, min=0)
#                     sim = sim / (torch.max(sim, dim=1, keepdim=True)[0] + EPS)
                    
                    sim = torch.where(sim>self.mmsteer_threshold, 1.0, 0.0)

                    vector = sim * vector_steered + (1 - sim) * vector
                else:
                    vector = vector_steered
                    
            else:
                raise ValueError(f'Unknown steer type {self.steer_type}')
        return vector.half()


class CustomAttnProcessor:
    def __init__(self, controls: list[VectorControl], place_in_unet: str):
        self._controls = controls
        self._place_in_unet = place_in_unet
     
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )


        hidden_states = hidden_states.transpose(1, 2)  # (batch_size, sequence_length, num_heads, head_dim)

        for control in self._controls:
            if control._mode == VectorControlMode.ATTN_HEADS and control.active:
                hidden_states = control(hidden_states, self._place_in_unet)


        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)


        # -------------------------------
        # adding controller

        
        
#         size = hidden_states.shape[2] // attn.heads
# #         y = torch.tensor(np.zeros_like(hidden_states.data.cpu().numpy())).to(device)
#         for idx in range(attn.heads):
# #             x = deepcopy(y[:, :, :])
# #             x[:, :, size*idx:size*(idx+1)] = hidden_states[:, :, size*idx:size*(idx+1)]
            
#             x = hidden_states[:, :, size*idx:size*(idx+1)]
            
# #             x = attn.to_out[0](x)
# #             # dropout
# #             x = attn.to_out[1](x)
            
# #             if attn.residual_connection:
# #                 x = x + residual

# #             x = x / attn.rescale_output_factor
            
#             x = controller(x, attn.heads, idx)
#             hidden_states[:, :, size*idx:size*(idx+1)] = x
        # -------------------------------
            

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def register_vector_controls(model, *controls: VectorControl):
    def block_forward(self, place_in_unet):
        
        # overriding BasicTransformerBlock forward function
        def forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]
    
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            else:
                raise ValueError("Incorrect norm used")
    
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output
    
            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
    
            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
    
            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")
    
                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)
    
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # -------------------------------
                # adding controller
                attn_output = attn_output[..., None, :]
                for control in controls:
                    if control._mode == VectorControlMode.ATTN_OUTPUT and control.active:
                        attn_output = control(attn_output, place_in_unet)
                attn_output = attn_output[..., 0, :]
                # -------------------------------
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)
    
            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    
            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)
    
            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output
    
            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
                
                
            # print(controller.cur_att_layer-1)
            # x = torch.norm(attn_output, dim=2, keepdim=True) / torch.norm(hidden_states, dim=2, keepdim=True)
            # print('CA', place_in_unet, x.mean().item())
            
            # x = y / torch.norm(hidden_states, dim=2, keepdim=True)
            # print('SA',place_in_unet, x.mean().item())
            
            # x = torch.norm(ff_output, dim=2, keepdim=True) / torch.norm(hidden_states, dim=2, keepdim=True)
            # print('FF',place_in_unet, x.mean().item())
            
            # print()

            return hidden_states

        return forward

    
    def register_recr(net_, count: int, place_in_unet: str):
        '''
        registering controller for all the BasicTransformerBlocks in the model
        '''
        if net_.__class__.__name__ == 'BasicTransformerBlock':
            processor = CustomAttnProcessor(controls=controls, place_in_unet=place_in_unet)
            net_.attn2.set_processor(processor)
            net_.forward = block_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    block_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            block_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            block_count += register_recr(net[1], 0, "up")
        if "mid" in net[0]:
            block_count += register_recr(net[1], 0, "mid")
    for control in controls:
        control.num_attn_layers = block_count
