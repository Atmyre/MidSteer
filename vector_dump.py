from construct_prompts import pickle_stats
from controller import EPS, VectorControl, VectorControlMode
from collections import defaultdict
import torch
import numpy as np
import functools
import enum
from utils import convert_to_widest_dtype


class TokenAggregationMode(enum.StrEnum):
    ALL = 'all'
    LAST = 'last'
    AVERAGE = 'average'


class CrossAttentionOutputStatsCollector(VectorControl):
    def __init__(self,
                 mode: VectorControlMode,
                 *,
                 token_aggregation_mode: TokenAggregationMode,
                 normalize: bool = False,
                 last_token_offset: int = -1,
                 compute_covariances: bool = True):
        super().__init__(mode=mode)

        self._cnt = defaultdict(lambda: defaultdict(list))
        self._m = defaultdict(lambda: defaultdict(list))
        self._mm = defaultdict(lambda: defaultdict(list))

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

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        batch_size = vector.shape[0]
        num_heads = vector.shape[-2]
        hidden_size = vector.shape[-1]

        vector_permuted = vector.permute(2, 0, 1, 3)  # [num_heads, batch_size, sequence_length, head_dim]
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

        self._update_statistics(vec, diffusion_step, place_in_unet, block_index)
        
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

    @property
    def covariances(self):
        if not self._compute_covariances:
            raise ValueError('Covariances requested not computed (self._compute_covariances = False)')
        result = {}
        for diffusion_step in self._mm:
            result[diffusion_step] = {}
            for place_in_unet in self._mm[diffusion_step]:
                result[diffusion_step][place_in_unet] = []
                for block_idx in range(len(self._mm[diffusion_step][place_in_unet])):
                    count = self._cnt[diffusion_step][place_in_unet][block_idx]
                    m = self._m[diffusion_step][place_in_unet][block_idx] / count
                    mm = self._mm[diffusion_step][place_in_unet][block_idx] / (count - 1)
                    result[diffusion_step][place_in_unet].append(
                        mm - m[:, :, None] @ m[:, None, :]  # compute outer product
                    )
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
