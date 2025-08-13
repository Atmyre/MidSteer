from core.pickle import pickle_stats
from core.controller import EPS, VectorControl, DiffusionVectorControlMode
from collections import defaultdict
import torch
import numpy as np
import functools
import enum
from core.math import convert_to_widest_dtype


class TokenAggregationMode(enum.StrEnum):
    ALL = 'all'
    LAST = 'last'
    AVERAGE = 'average'

i = 0

class CrossAttentionOutputStatsCollector(VectorControl):
    def __init__(self,
                 mode: DiffusionVectorControlMode=None,
                 *,
                 token_aggregation_mode: TokenAggregationMode,
                 normalize: bool = False,
                 last_token_offset: int = -1,
                 compute_covariances: bool = True):
        super().__init__(mode=mode)

        self._cnt = defaultdict(lambda: defaultdict(list))
        self._m = defaultdict(lambda: defaultdict(list))  # running sum
        self._mm = defaultdict(lambda: defaultdict(list))  # running sum of squares
        self._s = defaultdict(lambda: defaultdict(list))  # running sum of squared differences (for Welford's algorithm)

        self._token_aggregation_mode = token_aggregation_mode
        self._last_token_offset = last_token_offset
        self._normalize = normalize
        self._compute_covariances = compute_covariances
    
    def _update_statistics(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        stat_count = vector.shape[1]
        stat_m = torch.sum(vector, dim=1)
        if self._compute_covariances:
            stat_mm = (vector.mT @ vector)

        if len(self._cnt[diffusion_step][place_in_unet]) <= block_index:
            self._cnt[diffusion_step][place_in_unet].append(stat_count)
            self._m[diffusion_step][place_in_unet].append(stat_m)
            if self._compute_covariances:
                self._mm[diffusion_step][place_in_unet].append(stat_mm)
        else:
            self._cnt[diffusion_step][place_in_unet][block_index] += stat_count
            self._m[diffusion_step][place_in_unet][block_index] += stat_m
            if self._compute_covariances:
                self._mm[diffusion_step][place_in_unet][block_index] += stat_mm
    
    def _update_statistics_welford(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        # vector shape: [num_heads, num_samples, hidden_size]
        stat_count = vector.shape[1]
        stat_m = torch.sum(vector, dim=1)

        if len(self._cnt[diffusion_step][place_in_unet]) <= block_index:
            # Initialize new block
            self._cnt[diffusion_step][place_in_unet].append(stat_count)
            self._m[diffusion_step][place_in_unet].append(stat_m)
            if self._compute_covariances:
                # Initialize running sum of squared differences using matrix product
                current_mean = torch.mean(vector, dim=1)  # [num_heads, hidden_size]
                diff = vector - current_mean.unsqueeze(1)  # [num_heads, num_samples, hidden_size]
                # Use bmm for efficient matrix multiplication: diff @ diff^T
                stat_s = diff.mH @ diff  # [num_heads, hidden_size, hidden_size]
                self._s[diffusion_step][place_in_unet].append(stat_s)
        else:
            # Update existing block using Welford's algorithm
            old_count = self._cnt[diffusion_step][place_in_unet][block_index]
            old_mean = self._m[diffusion_step][place_in_unet][block_index] / old_count
            
            # Update count
            new_count = old_count + stat_count
            self._cnt[diffusion_step][place_in_unet][block_index] = new_count
            
            # Update mean using Welford's formula
            new_sum = self._m[diffusion_step][place_in_unet][block_index] + stat_m
            self._m[diffusion_step][place_in_unet][block_index] = new_sum
            new_mean = new_sum / new_count
            
            if self._compute_covariances:
                delta_old = vector - old_mean.unsqueeze(1)  # [num_heads, num_samples, hidden_size]
                delta_new = vector - new_mean.unsqueeze(1)  # [num_heads, num_samples, hidden_size]

                # For each head, compute: S_n = S_{n-1} + sum((x_i - μ_old) * (x_i - μ_new)^T)
                self._s[diffusion_step][place_in_unet][block_index] += delta_old.mH @ delta_new

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index, min_token_index: int = None):
        batch_size = vector.shape[0]
        num_heads = vector.shape[-2]
        hidden_size = vector.shape[-1]

        if min_token_index is not None and vector.shape[1] > 1:
            vector_slices = vector[:, min_token_index:, :, :]
        else:
            vector_slices = vector


        vector_permuted = vector_slices.permute(2, 0, 1, 3)  # [num_heads, batch_size, sequence_length, head_dim]
        vec = convert_to_widest_dtype(vector_permuted.view(num_heads, -1, hidden_size), device=vector_permuted.device, force_double=self._compute_covariances)
        if self._token_aggregation_mode == TokenAggregationMode.AVERAGE:
            vec = torch.mean(vec, dim=1, keepdim=True)
        elif self._token_aggregation_mode == TokenAggregationMode.LAST:
            if batch_size > 1:
                raise ValueError("TokenAggregationMode.LAST and batch_size > 1 is not supported currently")
            start = self._last_token_offset
            end = self._last_token_offset + 1
            if end == 0:
                end = vec.shape[1]
            vec = vec[:, start:end, :]
            assert vec.shape[1] == 1

        if self._normalize:
            vec /= torch.linalg.norm(vec, dim=2, keepdim=True) + EPS

        self._update_statistics_welford(vec, diffusion_step, place_in_unet, block_index)
        
        return vector
    
    @property
    def means(self):
        result = {}
        for diffusion_step in self._m:
            result[diffusion_step] = {}
            for place_in_unet in self._m[diffusion_step]:
                result[diffusion_step][place_in_unet] = []
                for block_idx in range(len(self._m[diffusion_step][place_in_unet])):
                    count = self._cnt[diffusion_step][place_in_unet][block_idx]
                    m = self._m[diffusion_step][place_in_unet][block_idx] / count
                    result[diffusion_step][place_in_unet].append(m)
        return result

    # @property
    # def covariances(self):
    #     if not self._compute_covariances:
    #         raise ValueError('Covariances requested not computed (self._compute_covariances = False)')
    #     result = {}
    #     for diffusion_step in self._mm:
    #         result[diffusion_step] = {}
    #         for place_in_unet in self._mm[diffusion_step]:
    #             result[diffusion_step][place_in_unet] = []
    #             for block_idx in range(len(self._mm[diffusion_step][place_in_unet])):
    #                 count = self._cnt[diffusion_step][place_in_unet][block_idx]
    #                 m = self._m[diffusion_step][place_in_unet][block_idx] / count
    #                 mm = self._mm[diffusion_step][place_in_unet][block_idx] / (count - 1)
    #                 result[diffusion_step][place_in_unet].append(
    #                     mm - m[:, :, None] @ m[:, None, :]  # compute outer product
    #                 )
    #     return result

    @property
    def covariances(self):
        if not self._compute_covariances:
            raise ValueError('Covariances requested not computed (self._compute_covariances = False)')
        result = {}
        for diffusion_step in self._s:
            result[diffusion_step] = {}
            for place_in_unet in self._s[diffusion_step]:
                result[diffusion_step][place_in_unet] = []
                for block_idx in range(len(self._s[diffusion_step][place_in_unet])):
                    count = self._cnt[diffusion_step][place_in_unet][block_idx]
                    # Welford's algorithm: covariance = S / (n-1)
                    if count > 1:
                        sigma_xx = self._s[diffusion_step][place_in_unet][block_idx]
                        S_hat = (sigma_xx + sigma_xx.mH) / 2
                        cov = S_hat / (count - 1)
                    else:
                        # Handle case with only one sample
                        cov = torch.zeros_like(self._s[diffusion_step][place_in_unet][block_idx])
                    result[diffusion_step][place_in_unet].append(cov)
        return result

    def save_stats(self, *, means_path: str = None, covariances_path: str = None, use_torch_save: bool = False):
        if means_path is not None:
            if use_torch_save:
                torch.save(self.means, means_path)
            else:
                pickle_stats(self.means, means_path)
        if self._compute_covariances and covariances_path is not None:
            if use_torch_save:
                torch.save(self.covariances, covariances_path)
            else:
                pickle_stats(self.covariances, covariances_path)
