import torch

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