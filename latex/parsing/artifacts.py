#!/usr/bin/env python3
"""
Abstract base classes and implementations for LaTeX artifact generation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from config import ArtifactConfig


class ArtifactGenerator(ABC):
    """Abstract base class for generating LaTeX artifacts."""
    
    def __init__(self, config: ArtifactConfig, output_dir: Path, artifact_key: str, base_path: Path = None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_key = artifact_key
        self.label = f"{config.latex_type}:{config.artifact_type}_{artifact_key}"
        self.base_path = base_path
    
    @abstractmethod
    def generate(self) -> str:
        """Generate the artifact and return LaTeX content."""
        pass
    
    def get_output_path(self, filename: str) -> Path:
        """Get the output path for a file."""
        return self.output_dir / filename


class TableGenerator(ArtifactGenerator):
    """Base class for generating LaTeX tables."""
    
    def __init__(self, config: ArtifactConfig, output_dir: Path, artifact_key: str, base_path: Path = None):
        super().__init__(config, output_dir, artifact_key, base_path)
        if config.latex_type != "tab":
            raise ValueError("TableGenerator requires latex_type='tab'")
    
    def get_type_config(self, config_attr_name: str):
        """Get and validate the type-specific configuration."""
        type_config = getattr(self.config, config_attr_name, None)
        if type_config is None:
            raise ValueError(f"{self.__class__.__name__} requires {config_attr_name} config")
        return type_config
    
    def safe_generate(self, generator_func):
        """Safely execute generator function with common error handling."""
        try:
            return generator_func()
        except Exception as e:
            return f"% Error generating {self.config.artifact_type}: {e}"
    
    @abstractmethod
    def generate(self) -> str:
        """Generate the table LaTeX content."""
        pass


class ChartGenerator(ArtifactGenerator):
    """Base class for generating charts/figures."""
    
    def __init__(self, config: ArtifactConfig, output_dir: Path, artifact_key: str, base_path: Path = None):
        super().__init__(config, output_dir, artifact_key, base_path)
        if config.latex_type != "fig":
            raise ValueError("ChartGenerator requires latex_type='fig'")
    
    def get_type_config(self, config_attr_name: str):
        """Get and validate the type-specific configuration."""
        type_config = getattr(self.config, config_attr_name, None)
        if type_config is None:
            raise ValueError(f"{self.__class__.__name__} requires {config_attr_name} config")
        return type_config
    
    def safe_generate(self, generator_func):
        """Safely execute generator function with common error handling."""
        try:
            return generator_func()
        except Exception as e:
            return f"% Error generating {self.config.artifact_type}: {e}"
    
    @abstractmethod
    def generate(self) -> str:
        """Generate the chart and LaTeX content."""
        pass

















# Import generators from their respective files
from renorm_clip_table_generator import RenormClipTableGenerator
from single_experiment_table_generator import SingleExperimentTableGenerator
from implicit_concept_table_generator import ImplicitConceptTableGenerator
from pareto_frontier_chart_generator import ParetoFrontierChartGenerator
from covariance_comparison_chart_generator import CovarianceComparisonChartGenerator

# Registry for artifact generators
GENERATOR_REGISTRY: Dict[str, type] = {
    'renorm_clip_table': RenormClipTableGenerator,
    'single_experiment_table': SingleExperimentTableGenerator,
    'implicit_concept_table': ImplicitConceptTableGenerator,
    'pareto_frontier_chart': ParetoFrontierChartGenerator,
    'covariance_comparison_chart': CovarianceComparisonChartGenerator,
    'table': TableGenerator,
    'chart': ChartGenerator,
}


def register_generator(artifact_type: str, generator_class: type):
    """Register a new artifact generator."""
    GENERATOR_REGISTRY[artifact_type] = generator_class


def create_generator(config: ArtifactConfig, output_dir: Path, artifact_key: str, base_path: Path = None) -> ArtifactGenerator:
    """Create an artifact generator from configuration."""
    artifact_type = config.artifact_type
    
    if artifact_type not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    generator_class = GENERATOR_REGISTRY[artifact_type]
    return generator_class(config, output_dir, artifact_key, base_path)
