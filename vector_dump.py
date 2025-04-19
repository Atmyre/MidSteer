from controller import EPS, VectorControl, VectorControlMode
from collections import defaultdict
import torch
import numpy as np


class CrossAttentionOutputStatsCollector(VectorControl):
    def __init__(self, mode: VectorControlMode, *, patch_average: bool = False, normalize: bool = False):
        super().__init__(mode=mode)

        self._cnt = defaultdict(lambda: defaultdict(list))
        self._m = defaultdict(lambda: defaultdict(list))
        self._mm = defaultdict(lambda: defaultdict(list))

        self._patch_average = patch_average
        self._normalize = normalize
    
    def _update_statistics(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        stat_count = vector.shape[1]
        stat_m = torch.sum(vector, dim=1)
        stat_mm = (vector.mT @ vector)

        if len(self._cnt[diffusion_step][place_in_unet]) <= block_index:
            self._cnt[diffusion_step][place_in_unet].append(stat_count)
            self._m[diffusion_step][place_in_unet].append(stat_m)
            self._mm[diffusion_step][place_in_unet].append(stat_mm)
        else:
            self._cnt[diffusion_step][place_in_unet][block_index] += stat_count
            self._m[diffusion_step][place_in_unet][block_index] += stat_m
            self._mm[diffusion_step][place_in_unet][block_index] += stat_mm

    def convert_to_dtype(self, vector: torch.Tensor):
        # float64 is needed for numerical stability
        if torch.mps.is_available():
            return vector.to('cpu').to(torch.float64)
        else:
            return vector.to(torch.float64)

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        num_heads = vector.shape[-2]
        hidden_size = vector.shape[-1]

        vector_permuted = vector.permute(2, 0, 1, 3)  # [num_heads, batch_size, sequence_length, head_dim]
        vec = self.convert_to_dtype(vector_permuted.view(num_heads, -1, hidden_size))
        if self._patch_average:
            vec = torch.mean(vec, dim=1, keepdim=True)

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

    def between_steps(self, last_diffusion_step: int):
        super().between_steps(last_diffusion_step)
