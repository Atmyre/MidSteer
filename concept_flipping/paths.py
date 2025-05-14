import os
from typing import Optional


def get_vector_path(
    model_name: str,
    layer_type: str,
    topic: str,
) -> str:
    """
    Generate the path for vector output files.
    
    Args:
        model_name: Name of the model (will be cleaned by replacing '/' with '_')
        layer_type: Type of layer (e.g., 'decoder_block', 'self_attn', etc.)
        topic: Topic name
        num_samples: Optional number of samples, if None will not be included in filename
        
    Returns:
        Path to the vector file
    """
    clean_model_name = model_name.replace('/', '_')
    filename = f"{clean_model_name}_{layer_type}_{topic}_means.pt"
    return os.path.join('concept_flipping', 'vectors', filename)


def get_results_path(
    model_name: str,
    layer_type: str,
    source_concept: str,
    target_concept: str,
    eval_num_samples: int,
    steer_type: Optional[str] = None,
    strength: Optional[float] = None,
    alpaca_eval: bool = False,
) -> str:
    """
    Generate the path for results output files.
    
    Args:
        model_name: Name of the model (will be cleaned by replacing '/' with '_')
        layer_type: Type of layer (e.g., 'decoder_block', 'self_attn', etc.)
        source_concept: Source concept name
        target_concept: Target concept name
        steer_type: Type of steering to apply (e.g., 'casteer', 'leace', 'mean_matching')
        strength: Strength of the steering
        alpaca_eval: Whether this is for alpaca evaluation results
        train_num_samples: Number of concept samples to use
        eval_num_samples: Number of eval samples to use
    Returns:
        Path to the results file
    """
    clean_model_name = model_name.replace('/', '_')
    
    if steer_type is None:
        prefix = 'None'
    else:
        prefix = f'{steer_type}_{strength}'

    filename = f"{prefix}_{clean_model_name}_{layer_type}_{source_concept}_to_{target_concept}_{eval_num_samples}.json"
    if alpaca_eval:
        return os.path.join('concept_flipping', 'results', 'alpaca_instruct', filename)
    else:
        return os.path.join('concept_flipping', 'results', 'concepts', filename) 


def get_cov_path(
    model_name: str,
    layer_type: str,
    stats_type: str,
    num_samples: int | None = None,
) -> str:
    """
    Generate the path for covariance output files.

    Args:
        model_name: Name of the model (will be cleaned by replacing '/' with '_')
        layer_type: Type of layer (e.g., 'decoder_block', 'self_attn', etc.)
        num_samples: Optional number of samples, if None will not be included in filename

    Returns:
        Path to the covariance file
    """
    clean_model_name = model_name.replace('/', '_')

    if num_samples is not None:
        filename = f"{clean_model_name}_{layer_type}_{stats_type}_{num_samples}.pt"
    else:
        filename = f"{clean_model_name}_{layer_type}_{stats_type}.pt"

    return os.path.join('concept_flipping', 'cov', filename)