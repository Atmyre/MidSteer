import logging
import warnings
import numpy as np
import torch
import abc
from collections import defaultdict
from typing import Optional, Any

import enum
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


class VectorControl(abc.ABC):
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


class CrossAttentionOutputSteering(VectorControl):
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
        device: Any,
        num_layers: int = None,
        strength: float = None,
        identity_cov: bool = False, 
        zero_mu_neutral: bool = False,
        mm_normalize_centers: bool = False,
        renormalize_after_steering: bool = False,
        intermediate_clipping: bool = True,
    ):
        super().__init__(mode=mode, num_layers=num_layers)
        self.device = device
        
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

        if steer_type in ('casteer', 'interpret'):
            self.casteer_vectors = []
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
            for num_steer in mmsteer_vectors:
                for place_in_unet in mmsteer_vectors[num_steer]:
                    for block_idx in range(len(mmsteer_vectors[num_steer][place_in_unet])):
                        W, b = mmsteer_vectors[num_steer][place_in_unet][block_idx]
                        W = torch.tensor(W).half().to(self.device)
                        b = torch.tensor(b).half().to(self.device)
                        self.mmsteer_vectors[num_steer][place_in_unet].append((W, b))
        elif steer_type in ('leace', 'mean_matching'):
            self.proj_transforms = []
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
                                steering_vector = (sigma_minus_half @ steering_vector.unsqueeze(-1))  # [#heads, dim, 1]
                                # Store projection components instead of full matrix
                                # P = I - strength * proj_left @ proj_right
                                proj_left = sigma_plus_half @ steering_vector  # [#heads, dim, 1]
                                proj_right = torch.linalg.pinv(steering_vector) @ sigma_minus_half  # [#heads, 1, dim]
                            elif steer_type == 'mean_matching':
                                if mm_normalize_centers:
                                    m_pos /= (torch.norm(m_pos, dim=-1, keepdim=True) + EPS)
                                    m_neg /= (torch.norm(m_neg, dim=-1, keepdim=True) + EPS)
                                # pinv(x) = x.T / |x|^2
                                m_pos = (sigma_minus_half @ m_pos.unsqueeze(-1))  # [#heads, dim, 1]
                                m_neg = (sigma_minus_half @ m_neg.unsqueeze(-1))  # [#heads, dim, 1]
                                # Store projection components instead of full matrix
                                # P = I - strength * proj_left @ proj_right (same sign as LEACE)
                                # Swap m_pos and m_neg to achieve the positive effect of mean_matching
                                proj_left = sigma_plus_half @ (m_neg - m_pos)  # [#heads, dim, 1]
                                proj_right = torch.linalg.pinv(m_neg) @ sigma_minus_half  # [#heads, 1, dim]

                            # Store projection components and metadata (always use -1 sign)
                            concept_transforms[num_steer][place_in_unet].append((proj_left.mT, proj_right.mT, m_neutral))
                self.proj_transforms.append(concept_transforms)

        else:
            raise ValueError(f'Unknown steer_type = {steer_type}')

        self.steering_cache = {}
        self.model_to_steer = model_to_steer

    def steer_transform(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        assert len(vector.shape) == 4
        (proj_left, proj_right, m_neutral) = steering_tensors

        num_heads = proj_left.shape[0]
        hidden_dim = proj_left.shape[2]
        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        assert vector.shape[2] == num_heads
        assert vector.shape[3] == hidden_dim

        # Assert proj_right dimensions for matrix multiplications
        assert proj_right.shape[0] == num_heads
        assert proj_right.shape[2] == proj_left.shape[1]
        assert proj_right.shape[1] == hidden_dim

        # Center the vector around m_neutral
        vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        m_neutral_expanded = m_neutral.to(vector.device).unsqueeze(1)
        vector_centered = vector_reshaped - m_neutral_expanded
        
        # Compute projection and apply steering: vector @ (I - strength * proj_right.mT @ proj_left.mT)
        projection_scores = vector_centered @ proj_right.to(vector.device)  # output shape = [num_heads, *, k]
        
        if self.intermediate_clipping:
            projection_scores = torch.where(projection_scores > 0, projection_scores, 0.0)
        
        steering_delta = -self.strength * (projection_scores @ proj_left.to(vector.device))  # output shape = [num_heads, *, hidden_dim]
        
        vector_steered = (vector_reshaped + steering_delta).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim)
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

    # steering backward, i.e. removing notion from vector
    def steer_backward_CASteer(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        assert len(vector.shape) == 4

        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        (b,_) = steering_tensors

        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)

        vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        b_norm_reshaped = b_norm.unsqueeze(-1)
        
        # computing dot products between vector components and steering vector x
        projection_scores = (
            (
                vector_reshaped
            ) @ b_norm_reshaped
        ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)
        

        # we will steer back only if dot product is positive, i.e.
        # if there's positive amount of information from steering vector in the vector
        if self.intermediate_clipping:
            projection_scores = torch.where(projection_scores>0, projection_scores, 0)

        steering_delta = - self.strength * projection_scores.to(vector.device) * b_norm.to(vector.device)

        # steer backward for beta*sim
        return vector + steering_delta
    
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

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int, min_token_index: int = None):
        batch_size = vector.shape[0]
        if batch_size > 1 and self.model_to_steer == ModelToSteer.UNET:
            # TODO: fix it properly sometime later
            # Steer only the prompt part of SDXL classifier-free guidance method
            batch_slice = slice(1, None)
            warnings.warn('Steering only the prompt part of SDXL classifier-free guidance (assumed the batch_idx=0 is not conditioned on the prompt)')
        else:
            batch_slice = slice(None, None)

        vector = vector.detach().clone()

        if self.model_to_steer == ModelToSteer.LLAMA or (place_in_unet in ['up', 'mid', 'joint', 'single', 'sana'] or (place_in_unet == 'down' and not self.steer_only_up)):
            # if steering vectors are from turbo version, then there's only one key in self.steering_vectors, 
            # and we'll use it for all the steps of generation
            # if steering vectors are from full version, then there's a key in self.steering_vectors
            # for each of the generation steps 
            # TODO: general way to handle this
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
                for transforms in self.proj_transforms:
                    vector[batch_slice, ...] = self.steer_transform(vector[batch_slice, ...], *transforms[num_steer][place_in_unet][block_index])
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




