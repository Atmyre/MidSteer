#!/usr/bin/env python3
"""
Configuration-based LaTeX ablation section generator.
"""

from pathlib import Path
from typing import Dict, List
import yaml

from config import AblationSectionConfig, SubsectionConfig, NamedSubsectionConfig, create_artifact_config
from artifacts import create_generator, ArtifactGenerator


class ConfigBasedAblationGenerator:
    """Generate ablation sections based on YAML configuration."""
    
    def __init__(self, config: AblationSectionConfig):
        self.config = config
        self.results_base_path = Path(config.results_base_path).resolve()
        self.templates_path = Path(config.templates_path).resolve()
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ConfigBasedAblationGenerator":
        """Create generator from YAML configuration file."""
        config = AblationSectionConfig.from_yaml(config_path)
        return cls(config)
    
    def load_template(self, template_name: str) -> str:
        """Load a LaTeX template file."""
        template_path = self.templates_path / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_subsection(self, subsection_name: str, subsection_config: SubsectionConfig) -> str:
        """Generate a single subsection."""
        print(f"  - Generating {subsection_name}...")
        
        if subsection_config.skip:
            print(f"  - Skipping {subsection_name} ({subsection_config.skip_reason})")
            template = self.load_template(subsection_config.template_file)
            skip_text = f"% {subsection_name} would be generated here"
            if subsection_config.skip_reason:
                skip_text += f"\n% ({subsection_config.skip_reason})"
            return template + "\n\n" + skip_text
        
        # Load template first
        template = self.load_template(subsection_config.template_file)
        
        # Generate all artifacts for this subsection
        artifacts_content = []
        
        for artifact_key, artifact_data in subsection_config.artifacts.items():
            try:
                # Convert dict to proper config object
                artifact_config = create_artifact_config(artifact_data)
                
                # Use common output directory instead of subsection-specific
                # This prevents creation of individual subsection directories
                
                # Create generator and generate content
                generator = create_generator(artifact_config, self.output_dir, artifact_key, self.results_base_path)
                content = generator.generate()
                
                if content and not content.startswith('%'):
                    artifacts_content.append(content)
                    print(f"    Generated artifact '{artifact_key}' with label: {generator.label}")
                
            except Exception as e:
                error_msg = f"% Error generating artifact '{artifact_key}': {e}"
                artifacts_content.append(error_msg)
                print(f"    Error: {e}")
        
        # Combine template with artifacts at the bottom
        result_parts = [template]
        if artifacts_content:
            result_parts.extend(artifacts_content)
        
        return '\n\n'.join(result_parts)
    
    def generate_ablation_section(self) -> str:
        """Generate the complete ablation section."""
        print("Generating ablation section...")
        
        # Start with section header
        result_parts = [f"\\subsection{{{self.config.section_title}}}"]
        
        # Generate each subsection in order
        for idx, named_subsection in enumerate(self.config.subsections):
            subsection_config = named_subsection.config
            subsection_name = subsection_config.name
            
            try:
                subsection_content = self.generate_subsection(subsection_name, subsection_config)
                result_parts.append(subsection_content)
            except Exception as e:
                error_msg = f"% Error generating subsection {subsection_name}: {e}"
                result_parts.append(error_msg)
                print(f"Error generating {subsection_name}: {e}")
        
        return '\n\n'.join(result_parts)
    
    def save_ablation_section(self, output_filename: str = None) -> Path:
        """Generate and save the ablation section."""
        if output_filename is None:
            output_filename = "ablation_section_config_based.tex"
        
        content = self.generate_ablation_section()
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Ablation section saved to: {output_path}")
        return output_path


def test_config_based_generation(config_path: str):
    """Test the configuration-based generation approach."""
    
    print("Testing configuration-based ablation generation...")
    print(f"Config file: {config_path}")
    
    try:
        # Create generator from config
        generator = ConfigBasedAblationGenerator.from_yaml(config_path)
        
        print(f"Results base path: {generator.results_base_path}")
        print(f"Templates path: {generator.templates_path}")
        print(f"Output directory: {generator.output_dir}")
        
        # Test loading a template
        template_content = generator.load_template("subsection_renorm_clip.tex")
        print(f"Template loaded successfully, length: {len(template_content)} chars")
        
        # Test generating just the first subsection (renorm_clip)
        if generator.config.subsections:
            first_subsection = generator.config.subsections[0]
            renorm_clip_config = first_subsection.config
            print(f"First subsection config: {renorm_clip_config.name}")
            print(f"Skip: {renorm_clip_config.skip}")
            print(f"Artifacts: {len(renorm_clip_config.artifacts)}")
            
            # Generate the subsection (this might fail if data is missing, but that's expected)
            try:
                subsection_content = generator.generate_subsection(renorm_clip_config.name, renorm_clip_config)
                print("✅ First subsection generated successfully")
                print(f"Content length: {len(subsection_content)} chars")
            except Exception as e:
                print(f"⚠️ Expected error generating first subsection: {e}")
        else:
            print("⚠️ No subsections found")
        
        # Try to generate the full section
        try:
            full_content = generator.generate_ablation_section()
            print("✅ Full ablation section generated successfully")
            print(f"Full content length: {len(full_content)} chars")
            
            # Save to test file
            test_output = generator.output_dir / "test_output.tex"
            with open(test_output, 'w') as f:
                f.write(full_content)
            print(f"✅ Test output saved to: {test_output}")
            
        except Exception as e:
            print(f"⚠️ Expected error generating full section: {e}")
        
        print("\n🎉 Configuration-based approach test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ablation section from YAML configuration"
    )
    parser.add_argument(
        "config_file",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-filename",
        default="ablation_section_config_based.tex",
        help="Output filename (default: ablation_section_config_based.tex)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (same as test_config_approach.py)"
    )
    
    args = parser.parse_args()
    
    if args.test:
        test_config_based_generation(args.config_file)
        return 0
    
    try:
        generator = ConfigBasedAblationGenerator.from_yaml(args.config_file)
        output_path = generator.save_ablation_section(args.output_filename)
        print(f"✅ Successfully generated ablation section: {output_path}")
    except Exception as e:
        print(f"❌ Error generating ablation section: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
