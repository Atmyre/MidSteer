#!/usr/bin/env python3
"""
Configuration models for LaTeX generation using Pydantic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, field_validator
import yaml


class RenormClipTableConfig(BaseModel):
    """Configuration for renorm/clip comparison tables."""
    
    base_path: str = Field(..., description="Base path where experiments are located")
    experiment_name: str = Field(..., description="Base experiment name")
    beta_value: float = Field(2.0, description="Beta value to use for all methods")
    enable_highlighting: bool = Field(True, description="Enable bold/underline highlighting")
    task_filter: Optional[str] = Field(None, description="Filter to specific task")


class ParetoFrontierChartConfig(BaseModel):
    """Configuration for Pareto frontier charts."""
    
    width: str = Field("0.9", description="LaTeX figure width (e.g., '0.9' for 0.9\\linewidth)")
    enable_highlighting: bool = Field(True, description="Enable Pareto frontier highlighting")
    
    # Normal mode: single experiment
    experiment_path: Optional[str] = Field(None, description="Path to single experiment directory")
    
    # Mixed mode: separate experiments for different methods
    mixed_mode: bool = Field(False, description="Use mixed experiment mode")
    clip_experiment: Optional[str] = Field(None, description="Experiment path for CASteer/LEACE (with clipping)")
    no_clip_experiment: Optional[str] = Field(None, description="Experiment path for MidSteer (without clipping)")
    
    def model_post_init(self, __context) -> None:
        """Validate that the configuration is consistent."""
        if self.mixed_mode:
            if not self.clip_experiment or not self.no_clip_experiment:
                raise ValueError("Mixed mode requires both clip_experiment and no_clip_experiment")
            if self.experiment_path:
                raise ValueError("Mixed mode cannot have experiment_path set")
        else:
            if not self.experiment_path:
                raise ValueError("Normal mode requires experiment_path")
            if self.clip_experiment or self.no_clip_experiment:
                raise ValueError("Normal mode cannot have clip_experiment or no_clip_experiment set")


class CovarianceComparisonChartConfig(BaseModel):
    """Configuration for covariance comparison charts."""
    
    width: str = Field("1.1", description="LaTeX figure width")
    pattern: str = Field("midsteer_sa_{num}_last_no_renorm_no_clip", description="Experiment pattern with {num} placeholder")
    methods: List[str] = Field(["leace", "mean_matching"], description="Methods to compare")
    beta: float = Field(2.0, description="Beta value to filter by")


class SingleExperimentResultConfig(BaseModel):
    """Configuration for single experiment result tables."""
    
    experiment_path: str = Field(..., description="Path to the experiment directory relative to base_path")
    enable_highlighting: bool = Field(True, description="Enable bold/underline highlighting for best/second-best results")
    selected_betas: Optional[List[float]] = Field(None, description="List of beta values to include (None for all)")
    task_filter: str = Field(..., description="Filter to specific task (required)")


class ImplicitConceptResultConfig(BaseModel):
    """Configuration for implicit concept result tables."""
    
    source_concepts: List[str] = Field(..., description="List of implicit source concept names")
    casteer_leace_experiment: str = Field(..., description="Experiment directory for CASteer and LEACE methods (usually with clipping)")
    midsteer_experiment: str = Field(..., description="Experiment directory for MidSteer/Mean Matching method (usually no clipping)")
    enable_highlighting: bool = Field(True, description="Enable bold/underline highlighting for best/second-best results")
    selected_betas: Optional[List[float]] = Field(None, description="List of beta values to include (None for all)")


class ArtifactConfig(BaseModel):
    """Configuration for any generated artifact with oneof-style type specification."""
    
    latex_type: Literal["fig", "tab"] = Field(..., description="LaTeX type: 'fig' for figures or 'tab' for tables")
    output_filename: Optional[str] = Field(None, description="Override output filename")
    caption: str = Field(..., description="Caption for table or figure")
    
    # Oneof-style configuration - exactly one must be set
    renorm_clip_comparison_table: Optional[RenormClipTableConfig] = Field(None, description="Renorm/clip table configuration")
    pareto_frontier_chart: Optional[ParetoFrontierChartConfig] = Field(None, description="Pareto frontier chart configuration")
    covariance_comparison_chart: Optional[CovarianceComparisonChartConfig] = Field(None, description="Covariance comparison chart configuration")
    single_experiment_result: Optional[SingleExperimentResultConfig] = Field(None, description="Single experiment result table configuration")
    implicit_concept_result: Optional[ImplicitConceptResultConfig] = Field(None, description="Implicit concept result table configuration")
    
    def model_post_init(self, __context) -> None:
        """Validate that exactly one type-specific config is set."""
        # Get all the oneof fields
        oneof_fields = ['renorm_clip_comparison_table', 'pareto_frontier_chart', 'covariance_comparison_chart', 'single_experiment_result', 'implicit_concept_result']
        
        # Count how many are set
        set_fields = []
        for field_name in oneof_fields:
            if getattr(self, field_name) is not None:
                set_fields.append(field_name)
        
        if len(set_fields) == 0:
            raise ValueError("Exactly one of the artifact type configurations must be set")
        elif len(set_fields) > 1:
            raise ValueError(f"Only one artifact type can be set, but found: {set_fields}")
    
    @property
    def artifact_type(self) -> str:
        """Get the artifact type based on which configuration is set."""
        if self.renorm_clip_comparison_table is not None:
            return "renorm_clip_table"
        elif self.pareto_frontier_chart is not None:
            return "pareto_frontier_chart"
        elif self.covariance_comparison_chart is not None:
            return "covariance_comparison_chart"
        elif self.single_experiment_result is not None:
            return "single_experiment_table"
        elif self.implicit_concept_result is not None:
            return "implicit_concept_table"
        else:
            raise ValueError("No artifact type configuration is set")
    
    @property
    def type_config(self) -> Union[RenormClipTableConfig, ParetoFrontierChartConfig, CovarianceComparisonChartConfig, SingleExperimentResultConfig, ImplicitConceptResultConfig]:
        """Get the active type-specific configuration."""
        if self.renorm_clip_comparison_table is not None:
            return self.renorm_clip_comparison_table
        elif self.pareto_frontier_chart is not None:
            return self.pareto_frontier_chart
        elif self.covariance_comparison_chart is not None:
            return self.covariance_comparison_chart
        elif self.single_experiment_result is not None:
            return self.single_experiment_result
        elif self.implicit_concept_result is not None:
            return self.implicit_concept_result
        else:
            raise ValueError("No artifact type configuration is set")





class SubsectionConfig(BaseModel):
    """Configuration for a subsection."""
    
    name: str = Field(..., description="Subsection name")
    template_file: str = Field(..., description="Template file path")
    artifacts: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Dictionary of artifacts to generate")
    skip: bool = Field(False, description="Whether to skip this subsection")
    skip_reason: Optional[str] = Field(None, description="Reason for skipping")


class NamedSubsectionConfig(BaseModel):
    """Configuration for a named subsection with ordering."""
    
    config: SubsectionConfig = Field(..., description="Subsection configuration")


class AblationSectionConfig(BaseModel):
    """Configuration for the complete ablation section."""
    
    results_base_path: str = Field(..., description="Base path to experimental results")
    templates_path: str = Field(..., description="Path to LaTeX templates")
    output_dir: str = Field(..., description="Directory to save generated files")
    
    # Section metadata
    section_title: str = Field("Ablations", description="Title for the section")
    
    # Ordered list of subsections
    subsections: List[NamedSubsectionConfig] = Field(..., description="Ordered list of subsections")
    
    @field_validator('subsections')
    @classmethod
    def validate_subsections(cls, v):
        """Validate that subsections are properly configured."""
        if len(v) == 0:
            raise ValueError("At least one subsection must be defined")
        return v
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "AblationSectionConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)


def create_artifact_config(artifact_data: Dict[str, Any]) -> ArtifactConfig:
    """Create an artifact configuration from dictionary data."""
    return ArtifactConfig(**artifact_data)
