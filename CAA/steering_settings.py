from typing import Optional, Literal
import os
from dataclasses import dataclass
from CAA.behaviors import ALL_BEHAVIORS

@dataclass
class SteeringSettings:
    behavior: str = "sycophancy"
    type: Literal["open_ended", "ab", "truthful_qa", "mmlu"] = "ab"
    system_prompt: Optional[Literal["pos", "neg"]] = None
    override_vector: Optional[int] = None
    override_vector_model: Optional[str] = None
    use_base_model: bool = False
    model_size: str = "7b"
    override_model_weights_path: Optional[str] = None
    control_type: Literal['CAA', 'external'] = 'CAA'
    external_steer_type: Literal['casteer', 'leace'] | None = None
    external_layer_type: Literal['decoder_block', 'self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm'] | None = None
    external_layers_to_steer: list[int] | None = None
    renormalize_after_steering: bool = False

    def __post_init__(self):
        assert self.behavior in ALL_BEHAVIORS, f"Invalid behavior {self.behavior}"
        
    def make_result_save_suffix(
        self,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            'control_type': self.control_type,
            'external_steer_type': self.external_steer_type,
            'external_layer_type': self.external_layer_type,
            'renormalize_after_steering': self.renormalize_after_steering if self.control_type == 'external' and self.renormalize_after_steering else None,
            "layer": layer,
            "multiplier": multiplier,
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_size": self.model_size,
            "override_model_weights_path": self.override_model_weights_path,
        }
        return "_".join([f"{k}={str(v).replace('/', '-')}" for k, v in elements.items() if v is not None])

    def filter_result_files_by_suffix(
        self,
        directory: str,
        layer: Optional[int] = None,
        multiplier: Optional[int] = None,
    ):
        elements = {
            'control_type': self.control_type,
            'external_steer_type': self.external_steer_type,
            'external_layer_type': self.external_layer_type,
            'renormalize_after_steering': self.renormalize_after_steering if self.control_type == 'external' and self.renormalize_after_steering else None,
            "layer": str(layer)+"_",
            "multiplier": str(float(multiplier))+"_",
            "behavior": self.behavior,
            "type": self.type,
            "system_prompt": self.system_prompt,
            "override_vector": self.override_vector,
            "override_vector_model": self.override_vector_model,
            "use_base_model": self.use_base_model,
            "model_size": self.model_size,
            "override_model_weights_path": self.override_model_weights_path,
        }

        filtered_elements = {k: v for k, v in elements.items() if v is not None}
        remove_elements = {k for k, v in elements.items() if v is None}

        matching_files = []

        for filename in os.listdir(directory):
            if all(f"{k}={str(v).replace('/', '-')}" in filename for k, v in filtered_elements.items()):
                # ensure remove_elements are *not* present
                if all(f"{k}=" not in filename for k in remove_elements):
                    matching_files.append(filename)

        return [os.path.join(directory, f) for f in matching_files]
    
    def get_formatted_model_name(self):
        if self.use_base_model:
            if self.model_size == "7b":
                return "Llama 2 7B"
            else:
                return "Llama 2 13B"
        else:
            if self.model_size == "7b":
                return "Llama 2 7B Chat"
            else:
                return "Llama 2 13B Chat"
        
