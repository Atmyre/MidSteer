"""
Utilities for handling system prompts and instruction formatting.
"""
from core.dataset import ALPACA_DEFAULT_INSTRUCTION, MMLU_SYSTEM_PROMPT


def resolve_system_prompt(system_prompt: str, dataset_type: str | None = None) -> str | None:
    """
    Resolve system prompt string to actual instruction text.
    
    Args:
        system_prompt: The system prompt specification ('auto', 'alpaca', 'mmlu', 'none', 'custom:text', or direct text)
        dataset_type: The dataset type for auto-detection ('alpaca', 'mmlu', 'template')
        
    Returns:
        The resolved instruction string or None for no instruction
    """
    # Handle explicit none
    if system_prompt == 'none':
        return None
    
    # Handle auto-detection
    if system_prompt == 'auto':
        if dataset_type == 'alpaca':
            system_prompt = 'alpaca'
        elif dataset_type == 'mmlu':
            system_prompt = 'mmlu'
        elif dataset_type == 'template':
            return None  # No system prompt for template dataset
        else:
            return None  # Unknown dataset type, no prompt
    
    # Convert system_prompt string to actual prompt text
    if system_prompt == 'alpaca':
        return ALPACA_DEFAULT_INSTRUCTION
    elif system_prompt == 'mmlu':
        return MMLU_SYSTEM_PROMPT
    elif system_prompt.startswith('custom:'):
        return system_prompt[7:]  # Remove 'custom:' prefix
    else:
        # Treat as direct custom prompt
        return system_prompt


def validate_system_prompt(system_prompt: str) -> bool:
    """
    Validate that a system prompt string is in a valid format.
    
    Args:
        system_prompt: The system prompt string to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_prompts = ['auto', 'alpaca', 'mmlu', 'none']
    return (system_prompt in valid_prompts or 
            system_prompt.startswith('custom:'))
