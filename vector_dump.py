from controller import VectorControl
from collections import defaultdict
import torch
import numpy as np


class CrossAttentionStatisticsHandler(VectorControl):
    def __init__(self, patch_average: bool = False, normalize: bool = False):
        super().__init__()
        # self.step_store = self.get_empty_store()
        # self.vector_store = defaultdict(dict)

        self._cnt = defaultdict(lambda: defaultdict(list))
        self._m = defaultdict(lambda: defaultdict(list))
        self._mm = defaultdict(lambda: defaultdict(list))

        self._patch_average = patch_average
        self._normalize = normalize

    # @staticmethod
    # def get_empty_store():
    #     return {"down": [], "up": [], 'mid': []}
    
    def _update_statistics(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        stat_count = vector.shape[0]
        stat_m = torch.sum(vector, dim=0)
        stat_mm = (vector.T @ vector)

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
    
    def forward(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        hidden_size = vector.shape[-1]
        vec = self.convert_to_dtype(torch.reshape(vector, (-1, hidden_size)))
        if self._patch_average:
            vec = torch.mean(vec, axis=0, keepdims=True)

        if self._normalize:
            vec /= torch.linalg.norm(vec, dim=1, keepdim=True)
            # print(torch.linalg.norm(vec, dim=1))
            # print(torch.linalg.norm(vec, dim=1).shape)

        self._update_statistics(vec, diffusion_step, place_in_unet, block_index)

        # save activation (vector) for further computing steering vectors
        # self.step_store[place_in_unet].append(vector.data.cpu().numpy()[len(vector)//2:].mean(axis=0).mean(axis=0))
        
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
                        mm - torch.outer(m, m)
                    )
        return result

    def between_steps(self, last_diffusion_step: int):
        super().between_steps(last_diffusion_step)
        # self.vector_store[last_diffusion_step] = self.step_store
        # self.step_store = self.get_empty_store()
