import logging
import warnings
import numpy as np
import torch
import abc
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
import weakref

import enum
import torch.nn.functional as F
from core.math import fractional_matrix_power_cov_torch
from core.math import convert_to_widest_dtype


logger = logging.getLogger()

EPS = 1e-6

class DiffusionVectorControlMode(enum.StrEnum):
    ATTN_OUTPUT = 'attn_output'
    ATTN_HEADS = 'attn_head'
    ATTN_KEY = 'attn_key'
    ATTN_VALUE = 'attn_value'
    ATTN_KEY_VALUE = 'attn_key_value'


class ModelToSteer(enum.StrEnum):
    UNET = 'unet'
    LLAMA = 'llama'


class VectorControlHook(abc.ABC):
    """Hook-based vector control that can be applied to attention layers"""
    
    def __init__(self, mode: DiffusionVectorControlMode = None, num_layers: int = None):
        self._mode = mode
        self._active = True
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        self.num_attn_layers = num_layers

    @property
    def active(self) -> bool:
        return self._active
    
    @active.setter
    def active(self, value: bool):
        self._active = value
    
    def reset(self):
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
    
    @abc.abstractmethod
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int, min_token_index: int = None):
        raise NotImplementedError

    def __call__(self, vector: torch.Tensor, place_in_unet: str):
        if not self.active:
            return vector
            
        block_index = self._current_position[place_in_unet]
        input_shape = vector.shape
        vector = self.forward(vector, self._diffusion_step, place_in_unet, block_index)
        assert vector.shape == input_shape
        self._current_position[place_in_unet] += 1

        self._current_attn_layer += 1
        if self._current_attn_layer == self.num_attn_layers:
            self._current_attn_layer = 0
            self._current_position = defaultdict(int)
            self._diffusion_step += 1
        return vector


class CrossAttentionOutputSteeringHook(VectorControlHook):
    """Hook-based version of CrossAttentionOutputSteering"""
    
    def __init__(
        self,
        model_to_steer: ModelToSteer,
        *,
        mode: DiffusionVectorControlMode = None,
        mmsteer_vectors=None,
        mu_pos=None,
        mu_neg=None,
        mu_neutral=None,
        cov=None,
        steer_type: str = None,
        
        mmsteer_threshold: float=0.1,
        steer_only_up=False, 
        alpha: float = None,
        beta: float = None,
        steer_back: bool = False,
        device: Any = None,
        num_layers: int = None,
        strength: float = None,
        identity_cov: bool = False, 
        zero_mu_neutral: bool = False,
        mm_normalize_centers: bool = False,
        renormalize_after_steering: bool = True,
        intermediate_clipping: bool = True,
    ):
        super().__init__(mode=mode, num_layers=num_layers)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mmsteer_threshold = mmsteer_threshold
        self.steer_only_up = steer_only_up
        self.steer_back = steer_back
        self.steer_type = steer_type
        self.renormalize_after_steering = renormalize_after_steering
        self.intermediate_clipping = intermediate_clipping

        if steer_type == 'casteer' and steer_back:
            self.strength = beta
        else:
            self.strength = alpha

        if strength is not None:
            self.strength = strength
        if self.strength is None:
            raise ValueError("Steering strength not provided, specify with `strength` parameter")
        
        if self.strength < 0:
            mu_pos, mu_neg = mu_neg, mu_pos
            self.strength = -self.strength

        # Initialize steering vectors (same logic as original)
        if steer_type in ('casteer', 'interpret'):
            self.casteer_vectors = []
            if mu_pos is not None and mu_neg is not None:
                for mu_pos_concept, mu_neg_concept in zip(mu_pos, mu_neg):
                    casteer_concept_transforms = defaultdict(lambda: defaultdict(list))
                    for num_steer in mu_pos_concept:
                        for place_in_unet in mu_pos_concept[num_steer]:
                            for block_idx in range(len(mu_pos_concept[num_steer][place_in_unet])):
                                b = mu_pos_concept[num_steer][place_in_unet][block_idx] - mu_neg_concept[num_steer][place_in_unet][block_idx]
                                if len(b.shape) == 1:
                                    b = b.unsqueeze(0)
                                b = convert_to_widest_dtype(torch.tensor(b), device=self.device).unsqueeze(-1)
                                
                                res = self.strength*(b @ torch.linalg.pinv(b))
                                P = torch.eye(res.shape[1], dtype=res.dtype).unsqueeze(0).to(self.device) - res
                                
                                casteer_concept_transforms[num_steer][place_in_unet].append((b.squeeze(-1), P))
                    self.casteer_vectors.append(casteer_concept_transforms)
        elif steer_type == 'mmsteer':
            self.mmsteer_vectors = defaultdict(lambda: defaultdict(list))
            if mmsteer_vectors is not None:
                for num_steer in mmsteer_vectors:
                    for place_in_unet in mmsteer_vectors[num_steer]:
                        for block_idx in range(len(mmsteer_vectors[num_steer][place_in_unet])):
                            W, b = mmsteer_vectors[num_steer][place_in_unet][block_idx]
                            W = torch.tensor(W).half().to(self.device)
                            b = torch.tensor(b).half().to(self.device)
                            self.mmsteer_vectors[num_steer][place_in_unet].append((W, b))
        elif steer_type in ('leace', 'mean_matching'):
            self.leace_transforms = []
            if all(x is not None for x in [mu_pos, mu_neg, mu_neutral, cov]):
                for mu_pos_concept, mu_neg_concept, mu_neutral_concept, cov_concept in zip(mu_pos, mu_neg, mu_neutral, cov):
                    concept_transforms = defaultdict(lambda: defaultdict(list))
                    for num_steer in mu_pos_concept:
                        for place_in_unet in mu_pos_concept[num_steer]:
                            for block_idx in range(len(mu_pos_concept[num_steer][place_in_unet])):
                                sigma = convert_to_widest_dtype(
                                    cov_concept[num_steer][place_in_unet][block_idx],
                                    device=self.device, force_double=False)
                                if identity_cov:
                                    sigma = torch.eye(sigma.shape[1], dtype=sigma.dtype, device=sigma.device).unsqueeze(0)
                                m_neutral = convert_to_widest_dtype(
                                    mu_neutral_concept[num_steer][place_in_unet][block_idx],
                                    device=self.device, force_double=False)
                                if zero_mu_neutral:
                                    m_neutral = torch.zeros_like(m_neutral)
                                m_pos = convert_to_widest_dtype(
                                    mu_pos_concept[num_steer][place_in_unet][block_idx],
                                    device=self.device, force_double=False) - m_neutral
                                m_neg = convert_to_widest_dtype(
                                    mu_neg_concept[num_steer][place_in_unet][block_idx],
                                    device=self.device, force_double=False) - m_neutral
                                steering_vector = m_pos - m_neg

                                sigma_minus_half = fractional_matrix_power_cov_torch(sigma, -0.5)
                                sigma_plus_half = fractional_matrix_power_cov_torch(sigma, 0.5)

                                if steer_type == 'leace':
                                    steering_vector = (sigma_minus_half @ steering_vector.unsqueeze(-1))
                                    proj_left = sigma_plus_half @ steering_vector
                                    proj_right = torch.linalg.pinv(steering_vector) @ sigma_minus_half
                                elif steer_type == 'mean_matching':
                                    if mm_normalize_centers:
                                        m_pos /= (torch.norm(m_pos, dim=-1, keepdim=True) + EPS)
                                        m_neg /= (torch.norm(m_neg, dim=-1, keepdim=True) + EPS)
                                    m_pos = (sigma_minus_half @ m_pos.unsqueeze(-1))
                                    m_neg = (sigma_minus_half @ m_neg.unsqueeze(-1))
                                    proj_left = sigma_plus_half @ (m_neg - m_pos)
                                    proj_right = torch.linalg.pinv(m_neg) @ sigma_minus_half

                                concept_transforms[num_steer][place_in_unet].append((proj_left, proj_right, m_neutral))
                    self.leace_transforms.append(concept_transforms)
        else:
            raise ValueError(f'Unknown steer_type = {steer_type}')

        self.steering_cache = {}
        self.model_to_steer = model_to_steer

    def steer_transform(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        (proj_left, proj_right, m_neutral) = steering_tensors

        num_heads = proj_left.shape[0]
        hidden_dim = proj_left.shape[1]
        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]

        vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        m_neutral_expanded = m_neutral.to(vector.device).unsqueeze(1)
        vector_centered = vector_reshaped - m_neutral_expanded
        
        projection_scores = vector_reshaped @ proj_right.mT.to(vector.device)
        
        if self.intermediate_clipping:
            projection_scores = torch.where(projection_scores > 0, projection_scores, 0.0)
        
        steering_delta = -self.strength * (projection_scores @ proj_left.mT.to(vector.device))
        
        vector_steered = (vector_centered + steering_delta + m_neutral_expanded).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim)
        return vector_steered
    
    def steer_backward_CASteer_matrix_form(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        (_,P) = steering_tensors

        vector_steered = ((
            convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1) @ P.to(vector.device).mT)).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim) 
        return vector_steered

    def steer_backward_CASteer(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        (b,_) = steering_tensors

        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)

        sim = (
            (
                convert_to_widest_dtype(vector, device=self.device)
                .reshape(-1, num_heads, hidden_dim)
                .transpose(0, 1)
            ) @ b_norm.unsqueeze(-1)
        ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)
        
        if self.intermediate_clipping:
            sim = torch.where(sim>0, sim, 0)

        return vector - self.strength * sim.to(vector.device) * b_norm.to(vector.device)
    
    def interpret(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        (b,_) = steering_tensors
        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)
        return b_norm.to(vector.device)

    def steer_forward_CASteer(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        (b,_) = steering_tensors

        assert len(b.shape) in (1, 2)
        if len(b.shape) == 1:
            b = b.reshape(1, -1)

        return vector + self.strength * b.to(vector.device) * torch.norm(vector, dim=-1, keepdim=True).to(vector.device)
    
    def renormalize(self, vector: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        if self.renormalize_after_steering:
            return vector / (torch.norm(vector, dim=-1, keepdim=True) + EPS) * norm
        else:
            return vector

    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int, min_token_index: int = None):
        batch_size = vector.shape[0]
        if batch_size > 1 and self.model_to_steer == ModelToSteer.UNET:
            batch_slice = slice(1, None)
            warnings.warn('Steering only the prompt part of SDXL classifier-free guidance (assumed the batch_idx=0 is not conditioned on the prompt)')
        else:
            batch_slice = slice(None, None)

        vector = vector.detach().clone()

        if self.model_to_steer == ModelToSteer.LLAMA or (place_in_unet in ['up', 'mid', 'joint'] or (place_in_unet == 'down' and not self.steer_only_up)): 
            num_steer = diffusion_step

            norm = torch.norm(vector, dim=-1, keepdim=True)
            if self.steer_type == 'casteer':
                if self.steer_back:
                    for casteer_vectors in self.casteer_vectors:
                        vector[batch_slice, ...] = self.steer_backward_CASteer(vector[batch_slice, ...], *casteer_vectors[num_steer][place_in_unet][block_index])
                        vector = self.renormalize(vector, norm)
                else:
                    for casteer_vectors in self.casteer_vectors:
                        vector[batch_slice, ...] = self.steer_forward_CASteer(vector[batch_slice, ...], *casteer_vectors[num_steer][place_in_unet][block_index])
                        vector = self.renormalize(vector, norm)
            elif self.steer_type == 'interpret':
                vector[batch_slice, ...] = self.interpret(vector[batch_slice, ...], *self.casteer_vectors[0][num_steer][place_in_unet][block_index])
                vector = self.renormalize(vector, norm)
            elif self.steer_type in ('leace', 'mean_matching'):
                for leace_vectors in self.leace_transforms:
                    vector[batch_slice, ...] = self.steer_transform(vector[batch_slice, ...], *leace_vectors[num_steer][place_in_unet][block_index])
                    vector = self.renormalize(vector, norm)
            elif self.steer_type == 'mmsteer':
                pos = (num_steer, place_in_unet, block_index)
                if pos in self.steering_cache:
                    W_alpha, b_alpha = self.steering_cache[pos]
                else:
                    (W, b) = self.mmsteer_vectors[num_steer][place_in_unet][block_index]
                    if len(W.shape) == 2:
                        W = W[None, ...]
                        b = b[None, :]

                    if self.strength != 1.0:
                        W = W.float()
                        b = b.float()
                        I = torch.eye(W.shape[1], device=W.device)[None, ...]
                        W_alpha = fractional_matrix_power_cov_torch(W, self.strength)
                        b_alpha = ((I - W_alpha) @ (I - W).inverse() @ b[..., None])[..., 0]
                        W_alpha = W_alpha.half()
                        b_alpha = b_alpha.half()
                    else:
                        W_alpha, b_alpha = W, b

                    self.steering_cache[pos] = W_alpha, b_alpha

                num_heads = W_alpha.shape[0]
                hidden_dim = W_alpha.shape[1]
                batch_size = vector.shape[0]
                sequence_length = vector.shape[1]

                vector_steered = ((vector.reshape(-1, num_heads, hidden_dim).transpose(0, 1) @ W_alpha.mT) + b_alpha.unsqueeze(1)).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim) 
                vector = vector_steered

            else:
                raise ValueError(f'Unknown steer type {self.steer_type}')
        return vector.half()


class HookManager:
    """Manages registration and cleanup of hooks for vector controls"""
    
    def __init__(self):
        self.hooks = []
        self.controls = []
        self.module_info = {}  # Maps modules to their place_in_unet info
        
    def register_vector_controls_with_hooks(self, model, *controls: VectorControlHook):
        """Register vector controls using PyTorch hooks instead of method overrides"""
        self.controls = list(controls)
        self._clear_hooks()
        
        # Find all attention modules and register hooks
        block_count = self._register_hooks_recursive(model)
        print("BLOCK COUNT", block_count)
        
        # Set the number of attention layers for all controls
        for control in self.controls:
            control.num_attn_layers = block_count
            
        return self.hooks
    
    def _register_hooks_recursive(self, model) -> int:
        """Recursively find transformer blocks and register hooks"""
        block_count = 0
        
        # Check if this is a FLUX model
        if self._is_flux_model(model):
            block_count += self._register_flux_model_hooks(model)
        else:
            # Traditional SD UNet structure
            for name, module in model.named_children():
                if "down" in name:
                    block_count += self._register_hooks_in_submodule(module, "down")
                elif "up" in name:
                    block_count += self._register_hooks_in_submodule(module, "up")
                elif "mid" in name:
                    block_count += self._register_hooks_in_submodule(module, "mid")
                    
        return block_count
    
    def _is_flux_model(self, model) -> bool:
        """Check if the model is a FLUX-based diffusion model"""
        # Look for characteristic FLUX model attributes
        flux_indicators = [
            hasattr(model, 'transformer'),          # FLUX uses transformer architecture
            hasattr(model, 'joint_blocks'),         # FLUX has joint blocks
            hasattr(model, 'single_blocks'),        # FLUX has single blocks
            hasattr(model, 'pos_embed'),            # FLUX uses positional embeddings
            any('flux' in str(type(module)).lower() for _, module in model.named_modules()),
            any('dit' in str(type(module)).lower() for _, module in model.named_modules()),
        ]
        
        return sum(flux_indicators) >= 2
    
    def _register_flux_model_hooks(self, model) -> int:
        """Register hooks for FLUX model architecture"""
        block_count = 0
        
        # FLUX models have a flat transformer structure
        # Look for joint_blocks and single_blocks
        if hasattr(model, 'joint_blocks'):
            for block in model.joint_blocks:
                self._register_hooks_for_flux_block(block, "joint")
                block_count += 2
                
        if hasattr(model, 'single_blocks'):
            for block in model.single_blocks:
                self._register_hooks_for_flux_block(block, "single")
                block_count += 2
        
        # Also check transformer submodule
        # if hasattr(model, 'transformer'):
        #     transformer = model.transformer
        #     if hasattr(transformer, 'joint_blocks'):
        #         for block in transformer.joint_blocks:
        #             self._register_hooks_for_flux_block(block, "joint")
        #             block_count += 1
        #     if hasattr(transformer, 'single_blocks'):
        #         for block in transformer.single_blocks:
        #             self._register_hooks_for_flux_block(block, "single")
        #             block_count += 1
        
        # Fallback: search recursively for any FLUX blocks
        if block_count == 0:
            for name, module in model.named_modules():
                class_name = module.__class__.__name__
                if class_name in ['FluxTransformerBlock']:
                    # Determine place based on block type or name
                    place = "joint" #if "joint" in class_name.lower() or "joint" in name.lower() else "single"
                    self._register_hooks_for_flux_block(module, place)
                    block_count += 2
        
        return block_count
    
    def _register_hooks_in_submodule(self, module, place_in_unet: str) -> int:
        """Register hooks in a submodule for a specific place in UNet"""
        block_count = 0
        
        for name, submodule in module.named_modules():
            class_name = submodule.__class__.__name__
            
            # Support both SD and FLUX architectures
            if class_name == 'BasicTransformerBlock':
                # Standard Stable Diffusion blocks
                self._register_hooks_for_sd_block(submodule, place_in_unet)
                block_count += 1
            elif class_name in ['JointTransformerBlock', 'SingleTransformerBlock', 'MMDiTBlock', 'FluxTransformerBlock']:
                # FLUX DiT blocks (various naming conventions)
                self._register_hooks_for_flux_block(submodule, place_in_unet)
                block_count += 1
            elif hasattr(submodule, 'attn') and hasattr(submodule.attn, 'to_q'):
                # Generic transformer block detection for FLUX variants
                if self._is_flux_attention_block(submodule):
                    self._register_hooks_for_flux_block(submodule, place_in_unet)
                    block_count += 1
                
        return block_count
    
    def _register_hooks_for_sd_block(self, block, place_in_unet: str):
        """Register hooks for a specific BasicTransformerBlock (Stable Diffusion)"""
        # Store module info for the hook callbacks
        self.module_info[id(block)] = place_in_unet
        
        # Register hooks based on control modes
        for control in self.controls:
            if control._mode == DiffusionVectorControlMode.ATTN_OUTPUT:
                # Hook into the cross-attention output
                if hasattr(block, 'attn2') and block.attn2 is not None:
                    hook = block.attn2.register_forward_hook(
                        self._create_attn_output_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode == DiffusionVectorControlMode.ATTN_HEADS:
                # Hook into attention heads - need to hook into the attention mechanism itself
                if hasattr(block, 'attn2') and block.attn2 is not None:
                    hook = block.attn2.register_forward_hook(
                        self._create_attn_heads_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode in [DiffusionVectorControlMode.ATTN_KEY, 
                                 DiffusionVectorControlMode.ATTN_VALUE,
                                 DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                # Hook into key/value computation
                if hasattr(block, 'attn2') and block.attn2 is not None:
                    # Hook into to_k and to_v modules
                    if control._mode in [DiffusionVectorControlMode.ATTN_KEY, DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                        hook = block.attn2.to_k.register_forward_hook(
                            self._create_key_hook(control, place_in_unet)
                        )
                        self.hooks.append(hook)
                    if control._mode in [DiffusionVectorControlMode.ATTN_VALUE, DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                        hook = block.attn2.to_v.register_forward_hook(
                            self._create_value_hook(control, place_in_unet)
                        )
                        self.hooks.append(hook)
    
    def _register_hooks_for_flux_block(self, block, place_in_unet: str):
        """Register hooks for FLUX DiT blocks (JointTransformerBlock, SingleTransformerBlock, etc.)"""
        # Store module info for the hook callbacks
        self.module_info[id(block)] = place_in_unet
        
        # Register hooks based on control modes for FLUX architecture
        for control in self.controls:
            if control._mode == DiffusionVectorControlMode.ATTN_OUTPUT:
                # FLUX blocks may have different attention module names
                attn_module = self._get_flux_attention_module(block)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_flux_attn_output_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode == DiffusionVectorControlMode.ATTN_HEADS:
                attn_module = self._get_flux_attention_module(block)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._create_flux_attn_heads_hook(control, place_in_unet)
                    )
                    self.hooks.append(hook)
            elif control._mode in [DiffusionVectorControlMode.ATTN_KEY, 
                                 DiffusionVectorControlMode.ATTN_VALUE,
                                 DiffusionVectorControlMode.ATTN_KEY_VALUE]:
                # Hook into FLUX key/value computation
                self._register_flux_key_value_hooks(block, control, place_in_unet)
    
    def _is_flux_attention_block(self, module) -> bool:
        """Check if a module is a FLUX-style attention block"""
        # Check for FLUX-specific attributes
        flux_indicators = [
            hasattr(module, 'norm1') and hasattr(module, 'norm2'),  # Common in DiT
            hasattr(module, 'attn') and hasattr(module, 'mlp'),     # DiT structure
            hasattr(module, 'adaLN_modulation'),                    # Adaptive layer norm
            hasattr(module, 'txt_attn'),                            # Text attention in double stream
            hasattr(module, 'img_attn'),                            # Image attention in double stream
        ]
        
        # Return True if it has multiple FLUX indicators
        return sum(flux_indicators) >= 2
    
    def _get_flux_attention_module(self, block):
        """Get the appropriate attention module from a FLUX block"""
        # Try different possible attention module names in FLUX
        # print(block)
        possible_names = ['txt_attn', 'attn']
        
        for name in possible_names:
            if hasattr(block, name):
                attn_module = getattr(block, name)
                if attn_module is not None and hasattr(attn_module, 'to_q'):
                    return attn_module
        
        return None
    
    def _register_flux_key_value_hooks(self, block, control, place_in_unet: str):
        """Register key/value hooks for FLUX blocks"""
        attn_module = self._get_flux_attention_module(block)
        if attn_module is None:
            return
            
        # Hook into FLUX key/value projections
        if control._mode in [DiffusionVectorControlMode.ATTN_KEY, DiffusionVectorControlMode.ATTN_KEY_VALUE]:
            if hasattr(attn_module, 'to_k'):
                hook = attn_module.to_k.register_forward_hook(
                    self._create_flux_key_hook(control, place_in_unet)
                )
                self.hooks.append(hook)
        
        if control._mode in [DiffusionVectorControlMode.ATTN_VALUE, DiffusionVectorControlMode.ATTN_KEY_VALUE]:
            if hasattr(attn_module, 'to_v'):
                hook = attn_module.to_v.register_forward_hook(
                    self._create_flux_value_hook(control, place_in_unet)
                )
                self.hooks.append(hook)
    
    def _create_attn_output_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for attention output"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # Apply control to the output
            # Add extra dimension for compatibility with original code
            output_expanded = output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            return controlled_output[..., 0, :]
        
        return hook_fn
    
    def _create_attn_heads_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for attention heads"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # For attention heads, we need to reshape the output appropriately
            # This assumes the output is from scaled_dot_product_attention
            if len(output.shape) == 4:  # [batch, heads, seq_len, head_dim]
                # Transpose to [batch, seq_len, heads, head_dim] for control
                output_transposed = output.transpose(1, 2)
                controlled_output = control(output_transposed, place_in_unet)
                return controlled_output.transpose(1, 2)
            else:
                return control(output, place_in_unet)
        
        return hook_fn
    
    def _create_key_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for attention keys"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
                
            # Reshape for control application
            batch_size, seq_len, hidden_dim = output.shape
            num_heads = getattr(module, 'out_features', hidden_dim) // (hidden_dim // getattr(module, 'in_features', hidden_dim))
            head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
            
            if hidden_dim % head_dim == 0:
                output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)
                controlled_output = control(output_reshaped, place_in_unet)
                return controlled_output.view(batch_size, seq_len, hidden_dim)
            else:
                return control(output, place_in_unet)
        
        return hook_fn
    
    def _create_value_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for attention values"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
                
            # Similar to key hook
            batch_size, seq_len, hidden_dim = output.shape
            num_heads = getattr(module, 'out_features', hidden_dim) // (hidden_dim // getattr(module, 'in_features', hidden_dim))
            head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
            
            if hidden_dim % head_dim == 0:
                output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)
                controlled_output = control(output_reshaped, place_in_unet)
                return controlled_output.view(batch_size, seq_len, hidden_dim)
            else:
                return control(output, place_in_unet)
        
        return hook_fn
    
    def _create_flux_attn_output_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for FLUX attention output"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # FLUX attention outputs may have different tensor structure
            # Handle both single tensor and tuple outputs
            if isinstance(output, tuple):
                attn_output = output[0]
                encoder_attn_output = output[1]
                rest = output[2:]
            else:
                attn_output = output
                encoder_attn_output = None
                rest = None
            
            # Apply control to the attention output
            # Add extra dimension for compatibility with original code
            output_expanded = attn_output[..., None, :]
            controlled_output = control(output_expanded, place_in_unet)
            controlled_output = controlled_output[..., 0, :].to(torch.bfloat16)

            if encoder_attn_output is not None:
                encoder_attn_output_expanded = encoder_attn_output[..., None, :]
                controlled_encoder_output = control(encoder_attn_output_expanded, place_in_unet)
                controlled_encoder_output = controlled_encoder_output[..., 0, :].to(torch.bfloat16)
            else:
                controlled_encoder_output = None
            
            # Return in the same format as input
            if rest is not None:
                return (controlled_output, controlled_encoder_output) + rest
            else:
                return controlled_output
        
        return hook_fn
    
    def _create_flux_attn_heads_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for FLUX attention heads"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
            
            # Handle tuple outputs from FLUX attention
            if isinstance(output, tuple):
                attn_output = output[0]
                rest = output[1:]
            else:
                attn_output = output
                rest = None
            
            # Apply control with proper tensor reshaping for FLUX
            if len(attn_output.shape) == 4:  # [batch, heads, seq_len, head_dim]
                output_transposed = attn_output.transpose(1, 2)
                controlled_output = control(output_transposed, place_in_unet)
                controlled_output = controlled_output.transpose(1, 2)
            else:
                controlled_output = control(attn_output, place_in_unet)
            
            if rest is not None:
                return (controlled_output,) + rest
            else:
                return controlled_output
        
        return hook_fn
    
    def _create_flux_key_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for FLUX attention keys"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
                
            # FLUX keys may have different dimensionality than SD
            batch_size, seq_len = output.shape[:2]
            
            if len(output.shape) == 3:  # [batch, seq_len, hidden_dim]
                hidden_dim = output.shape[2]
                # Infer head structure from module attributes if available
                num_heads = getattr(module, 'num_heads', 8)  # Default fallback
                head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
                
                if hidden_dim % head_dim == 0:
                    output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)
                    controlled_output = control(output_reshaped, place_in_unet)
                    return controlled_output.view(batch_size, seq_len, hidden_dim)
                    
            return control(output, place_in_unet)
        
        return hook_fn
    
    def _create_flux_value_hook(self, control: VectorControlHook, place_in_unet: str):
        """Create a forward hook for FLUX attention values"""
        def hook_fn(module, input, output):
            if not control.active:
                return output
                
            # Similar to FLUX key hook but for values
            batch_size, seq_len = output.shape[:2]
            
            if len(output.shape) == 3:  # [batch, seq_len, hidden_dim]
                hidden_dim = output.shape[2]
                num_heads = getattr(module, 'num_heads', 8)  # Default fallback
                head_dim = hidden_dim // num_heads if num_heads > 0 else hidden_dim
                
                if hidden_dim % head_dim == 0:
                    output_reshaped = output.view(batch_size, seq_len, num_heads, head_dim)
                    controlled_output = control(output_reshaped, place_in_unet)
                    return controlled_output.view(batch_size, seq_len, hidden_dim)
                    
            return control(output, place_in_unet)
        
        return hook_fn
    
    def _clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_info.clear()
    
    def remove_hooks(self):
        """Public method to remove all hooks"""
        self._clear_hooks()
    
    def reset_controls(self):
        """Reset all controls to initial state"""
        for control in self.controls:
            control.reset()


# Convenience functions for backward compatibility
def register_vector_controls_with_hooks(model, *controls: VectorControlHook) -> HookManager:
    """
    Register vector controls using PyTorch hooks instead of method overrides.
    
    Args:
        model: The model to register controls on
        *controls: VectorControlHook instances to register
        
    Returns:
        HookManager: Manager object that can be used to remove hooks later
    """
    manager = HookManager()
    manager.register_vector_controls_with_hooks(model, *controls)
    return manager


# Re-export the hook-based versions with different names to avoid confusion
VectorControl = VectorControlHook
CrossAttentionOutputSteering = CrossAttentionOutputSteeringHook 