import pickle
import torch
import typing as tp

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def init_pipeline_for_model(model: str) -> DiffusionPipeline:
    if model == 'sd14':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16, 
            cache_dir='./cache',
            device_map='balanced',
        )
    elif model == 'sd21':
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16, 
            cache_dir='./cache',
            device_map='balanced',
        )
    elif model == 'sd21-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
        )
    elif model == 'sdxl':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
        )
    elif model == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
        )
    return pipe


def get_num_denoising_steps(model: str) -> int:
    if model in ('sd14', 'sd21'):
        return 50
    elif model in ('sd21-turbo', 'sdxl-turbo'):
        return 1
    elif model in ('sdxl',):
        return 30
    else:
        raise ValueError('Unknown model type')


def run_model(model_type: str, pipe, prompt: str, seed: int, device: torch.device):
    if model_type in ['sd14', 'sd21', 'sdxl']:
        image = pipe(prompt=prompt,
                     num_inference_steps=get_num_denoising_steps(model_type),
                     generator=torch.Generator(device=device).manual_seed(seed),
#                      guidance_scale=0.0
                    ).images[0]

    elif model_type in ['sd21-turbo', 'sdxl-turbo']:
        image = pipe(prompt=prompt,
                     num_inference_steps=get_num_denoising_steps(model_type),
                     guidance_scale=0.0,
                     generator=torch.Generator(device=device).manual_seed(seed),
                    ).images[0]

    return image


def fractional_matrix_power_cov_torch(mat: torch.Tensor, alpha: float, eps=1e-10) -> torch.Tensor:
    device = mat.device
    if mat.device.type == 'mps':  # Workaround because MPS does not yet support torch.linalg.eig
        mat = mat.cpu()

    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clip(evals, min=0, max=None)
    evals = torch.where(evals >= eps, evals ** alpha, 0.)
    return (evecs @ torch.diag_embed(evals) @ evecs.mT).to(device)


def convert_to_widest_dtype(vector: torch.Tensor, device: tp.Any, force_double: bool = False):
    # float64 is needed for numerical stability
    if device.type == 'mps':
        if force_double:
            return vector.to('cpu').to(dtype=torch.float64)
        else:
            return vector.to(device, dtype=torch.float32)
    else:
        return vector.to(device, dtype=torch.float64)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def unpickle(path: str | None):
    if path is None:
        return None
    try:
        return torch.load(path, weights_only=False)
    except:
        try:
            with open(path, 'rb') as fin:
                return pickle.load(fin)
        except:
            with open(path, 'rb') as fin:
                return CPU_Unpickler(fin).load()

def unpickle_pack(path: str | None) -> list[dict]:
    if path is None:
        return None
    result = []
    for subpath in path.split(','):
        result.append(unpickle(subpath))
    return result