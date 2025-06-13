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

from core.controller import VectorControl


ADDITIONAL_LAYER_CONFIG = {
    "q_proj": "model.layers.{num}.self_attn.q_proj",
    "k_proj": "model.layers.{num}.self_attn.k_proj",
    "v_proj": "model.layers.{num}.self_attn.v_proj",
    "o_proj": "model.layers.{num}.self_attn.o_proj",
}


@contextmanager
def llm_register_vector_control(
    model,
    control: list[VectorControl],
    layer_type: str | list[str],
    layers_to_steer: tp.Iterable[int] | None = None,
    min_token_index = None):
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
    layer_config = guess_and_enhance_layer_config(model, ADDITIONAL_LAYER_CONFIG)
    hooks: list[RemovableHandle] = []


    if isinstance(layer_type, str):
        layer_types = [layer_type]
    else:
        layer_types = layer_type

    for layer_type in layer_types:
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
            # print(layer_name, module)
            handle = module.register_forward_hook(
                # create the hook via function call since python only creates new scopes on functions
                _create_vector_control_hook(control, layer_type, layer_num, min_token_index)
            )
            hooks.append(handle)
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def _create_vector_control_hook(
    control: list[VectorControl],
    layer_type: str,
    layer_num: int,
    min_token_index: int | None
) -> tp.Any:
    """Create a hook function that adds the given target_activation to the model output"""

    def hook_fn(module: tp.Any, inputs: tp.Any, outputs: tp.Any) -> tp.Any:
        original_tensor = untuple_tensor(outputs)
        t = original_tensor.unsqueeze(-2)
        for c in control:
            if c.active:
                t = c.forward(t, 0, layer_type, layer_num, min_token_index)
        modified_tensor = t.squeeze(-2)

        mask = torch.zeros(original_tensor.shape[1])
        mask[slice(min_token_index, None)] = 1
        mask = mask.reshape(1, -1, 1).to(original_tensor.device)

        # TODO: do it properly (we don't now if it's generation step or forward step here)
        if mask.shape[1] == 1:
            mask = torch.ones_like(mask)

        original_tensor[None] = torch.where(mask == 1, modified_tensor, original_tensor)
        return outputs

    return hook_fn

