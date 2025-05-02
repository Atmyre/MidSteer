import typing as tp
import torch
from steering_vectors import SteeringPatchHandle, guess_and_enhance_layer_config
from torch.utils.hooks import RemovableHandle
from dataclasses import dataclass
from contextlib import contextmanager

from steering_vectors.torch_utils import get_module, untuple_tensor
from steering_vectors.layer_matching import (
    collect_matching_layers,
    guess_and_enhance_layer_config,
)

from controller import VectorControl


@contextmanager
def llm_register_vector_control(
    model,
    control: list[VectorControl],
    layer_type: str,
    layers_to_steer: tp.Iterable[int] | None = None,
    layer_config = None,
    min_token_index = None,
    token_indices = None):
    """
    Patch the activations of the given model with this steering vector.
    This will modify the model in-place, and return a handle that can be used to undo the patching.
    This method does the same thing as `apply`, but requires manually undoing the patching to
    restore the model to its original state. For most cases, `apply` is easier to use. Tokens to patch
    can be selected using either `min_token_index` or `token_indices`, but not both. If neither is provided,
    all tokens will be patched.

    Args:
        model: The model to patch
        layer_config: A dictionary mapping layer types to layer matching functions.
            If not provided, this will be inferred automatically.
        operator: A function that takes the original activation and the steering vector
            and returns a modified vector that is added to the original activation.
        multiplier: A multiplier to scale the patch activations. Default is 1.0.
        min_token_index: The minimum token index to apply the patch to. Default is None.
        token_indices: Either a list of token indices to apply the patch to, a slice, or a mask tensor. Default is None.
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
        >>> steering_vector = SteeringVector(...)
        >>> handle = steering_vector.patch_activations(model)
        >>> model.forward(...)
        >>> handle.remove()
    """
    assert (min_token_index is None) or (token_indices is None), (
        "Can not pass both min_token_index and token_indices"
    )
    if isinstance(token_indices, torch.Tensor):
        assert torch.all(
            torch.logical_or(token_indices == 0, token_indices == 1)
        ), "token_indices tensor must be a mask (containing only 0s and 1s)"
    token_indices = (
        token_indices if token_indices is not None else slice(min_token_index, None)
    )
    layer_config = guess_and_enhance_layer_config(
        model, layer_config, layer_type
    )
    hooks: list[RemovableHandle] = []
    if layer_type not in layer_config:
        raise ValueError(
            f"layer_type {layer_type} not provided in layer config"
        )
    matcher = layer_config[layer_type]
    matching_layers = collect_matching_layers(model, matcher)


    layers = set(range(len(matching_layers)))

    if layers_to_steer is not None:
        layers = layers.intersection(layers_to_steer)

    for layer_num in layers:
        layer_name = matching_layers[layer_num]

        module = get_module(model, layer_name)
        handle = module.register_forward_hook(
            # create the hook via function call since python only creates new scopes on functions
            _create_vector_control_hook(control, layer_num, token_indices)
        )
        hooks.append(handle)
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def _create_vector_control_hook(
    control: list[VectorControl],
    layer_num: int,
    token_indices: list[int] | slice | torch.Tensor
) -> tp.Any:
    """Create a hook function that adds the given target_activation to the model output"""

    def hook_fn(module: tp.Any, inputs: tp.Any, outputs: tp.Any) -> tp.Any:
        original_tensor = untuple_tensor(outputs)
        t = original_tensor.unsqueeze(-2)
        for c in control:
            if c.active:
                t = c.forward(t, 0, 'LLM', layer_num)
        modified_tensor = t.squeeze(-2)

        if isinstance(token_indices, torch.Tensor):
            mask = token_indices
        else:
            mask = torch.zeros(original_tensor.shape[1])
            mask[token_indices] = 1
        mask = (
            mask.reshape(1, -1, 1)
            if len(mask.shape) == 1
            else mask.reshape(mask.shape[0], -1, 1)
        )
        mask = mask.to(original_tensor.device)

        # TODO: do it properly (we don't now if it's generation step or forward step here)
        if mask.shape[1] == 1:
            mask = torch.ones_like(mask)

        original_tensor[None] = torch.where(mask == 1, modified_tensor, original_tensor)
        return outputs

    return hook_fn

