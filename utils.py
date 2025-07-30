import torch
from diffusers import FluxPipeline

from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def init_pipeline_for_image_model(model: str) -> DiffusionPipeline:
    if model == 'sd14':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16, 
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
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
            safety_checker=None,
        )
    elif model == 'sdxl-turbo':
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16, 
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'flux':
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16,
            token='***REMOVED***',
#             device_map='balanced'
        )
        pipe.enable_model_cpu_offload()
    elif model == 'flux-schnell':
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16,
            token='***REMOVED***',
#             device_map='balanced'
        )
        pipe.enable_model_cpu_offload()
    return pipe


def get_num_denoising_steps(model: str) -> int:
    if model in ('sd14', 'sd21'):
        return 50
    elif model in ('sd21-turbo', 'sdxl-turbo'):
        return 1
    elif model in ('sdxl',):
        return 30
    elif model in ('flux',):
        return 28  # FLUX.1-dev typically uses 28 steps
    elif model in ('flux-schnell',):
        return 4   # FLUX.1-schnell is optimized for 4 steps
    else:
        raise ValueError('Unknown model type')


def run_image_model(model_type: str, pipe, prompt: str, seed: int, device: torch.device, num_images: int = 1):
    if model_type in ['sd14', 'sd21', 'sdxl']:
        images = pipe(prompt=prompt,
                     num_inference_steps=get_num_denoising_steps(model_type),
                     generator=torch.Generator(device=device).manual_seed(seed),
                     num_images_per_prompt=num_images,
                    ).images

    elif model_type in ['sd21-turbo', 'sdxl-turbo']:
        images = pipe(prompt=prompt,
                     num_inference_steps=get_num_denoising_steps(model_type),
                     guidance_scale=0.0,
                     generator=torch.Generator(device=device).manual_seed(seed),
                     num_images_per_prompt=num_images,
                    ).images
    elif model_type in ['flux']:
        images = pipe(
            prompt,
            guidance_scale=3.5,
            num_inference_steps=get_num_denoising_steps(model_type),
            max_sequence_length=512,
            generator=torch.Generator('cpu').manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images
    elif model_type in ['flux-schnell']:
        images = pipe(
            prompt,
            guidance_scale=0.0,
            num_inference_steps=get_num_denoising_steps(model_type),
            max_sequence_length=256,
            generator=torch.Generator('cpu').manual_seed(seed),
            num_images_per_prompt=num_images,
        ).images

    return images


def init_llm_model_and_tokenizer(model_name: str, cache_dir: str | None = './cache') -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if '3.1' in model_name:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    # ***REMOVED***
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map='balanced',
        token='***REMOVED***'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map='balanced',
        token='***REMOVED***'
    )
    return model, tokenizer


