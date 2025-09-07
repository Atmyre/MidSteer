import argparse
import os

from diffusers import DiffusionPipeline
import tqdm

from core.controller import DiffusionVectorControlMode
from core.dataset import RelaionDataset
from core.utils import get_device
from core.diffusion_steering import DiffusionModelType, diffusion_register_vector_controls_with_hooks
from core.utils import SUPPORTED_DIFFUSION_MODELS, init_pipeline_for_image_model, run_image_model
from core.vector_dump import CrossAttentionOutputStatsCollector, TokenAggregationMode


def main(
        pipeline: DiffusionPipeline,
        model_name: str,
        topic: str | None,
        control_mode: DiffusionVectorControlMode,
        aggregation_mode : TokenAggregationMode,
        normalize_vectors: bool,
        seed: int,
        num_samples: int, 
        output_dir: str,
):
    output_path = os.path.join(output_dir, f"{topic}.pt") if topic is not None else os.path.join(output_dir, f"all.pt")
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping generation.")
        return
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()

    dataset = RelaionDataset(
        concept=topic,
        max_samples=num_samples,
        seed=seed,
    )

    stats_handler = CrossAttentionOutputStatsCollector(
        mode=control_mode,
        token_aggregation_mode=aggregation_mode,
        normalize=normalize_vectors,
        compute_covariances=False,
    )

    # Register hooks on the appropriate model component
    model_component = getattr(pipeline, 'transformer', None) or pipeline.unet
    diffusion_register_vector_controls_with_hooks(
        model_component,
        stats_handler,
        model_type=DiffusionModelType.from_model(model_name),
    )

    for prompt in tqdm.tqdm(dataset, desc=f"Processing prompts"):
        _ = run_image_model(
            model_type=model_name,
            pipe=pipeline,
            prompt=prompt,
            seed=0,
            device=device,
        )
        stats_handler.reset()

    stats_handler.save_stats(
        means_path=output_path,
        use_torch_save=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                        help='Diffusion model name to use for generating representation means')
    parser.add_argument('--topics', type=str, nargs='+', required=False, help='List of topics to generate steering vectors for')
    parser.add_argument('--control_mode', type=DiffusionVectorControlMode, choices=[str(x) for x in DiffusionVectorControlMode],
                        default='attn_output', help='Vector control mode for diffusion model')
    parser.add_argument('--aggregation_mode', type=TokenAggregationMode, choices=[str(x) for x in TokenAggregationMode],
                        required=True, help='Patch aggregation mode for hidden representations')
    parser.add_argument('--normalize_vectors', action='store_true',
                        help='Normalize representations to have unit length before computing statistics')
    parser.add_argument('--seed', type=int, default=42, help='Seed for dataset shuffling')
    parser.add_argument('--num_samples', type=int, default=1_000, help='Number of samples used to compute statistics')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory used to write computed statistics')

    args = parser.parse_args()

    
    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)

    if args.topics is None:
        topics = [None]
    else:
        topics = args.topics

    for topic in args.topics:
        main(
            pipeline=pipeline,
            model_name=args.model_name,
            topic=topic,
            control_mode=args.control_mode,
            aggregation_mode=args.aggregation_mode,
            normalize_vectors=args.normalize_vectors,
            seed=args.seed,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
        )
