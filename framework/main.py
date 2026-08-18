import argparse
from pathlib import Path
from agents.biological_agent import BiologicalAgent
from agents.technical_agent import TechnicalAgent
from agents.experimental_agent import ExperimentalDesignAgent
from agents.experimental_agent import ExperimentalDesignAgent
from agents.integration_agent import IntegrationAgent
from core.reproducibility import set_seed, get_reproducibility_info
import json
import yaml

# Load default config if available
DEFAULT_CONFIG = {
    "paths": {
        "input_dir": "./docs",
        "output_dir": "framework_output",
        "ontology_dir": "ontologies"
    },
    "agents": {
        "temperatures": [0.0],
        "validate": False
    },
    "concurrency": {
        "max_workers": 1,
        "request_timeout": 60
    }
}

try:
    with open("config.yaml", "r") as f:
        loaded = yaml.safe_load(f)
        if loaded:
            if "paths" in loaded: DEFAULT_CONFIG["paths"].update(loaded["paths"])
            if "agents" in loaded: DEFAULT_CONFIG["agents"].update(loaded["agents"])
            if "concurrency" in loaded: DEFAULT_CONFIG["concurrency"].update(loaded["concurrency"])
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description='Unified Scientific Metadata Extraction Framework')
    parser.add_argument('--config', type=str, default="config.yaml",
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('mode', choices=['biological', 'technical', 'experimental', 'all'], 
                        help='Which extraction mode to run')
    parser.add_argument('--input', type=str, default=None,
                        help=f'Input directory path')
    parser.add_argument('--output', type=str, default=None,
                        help='Base output directory path (default: defined in config.yaml)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=None,
                        help='List of temperatures to sample')
    parser.add_argument('--single-temp', type=float, default=None,
                        help='Run with a single temperature instead of multiple')
    parser.add_argument('--validate', action='store_true', default=None,
                        help='Enable the Validation Agent to critique and correct outputs')
    parser.add_argument('--runassessor-dir', type=str, default=None,
                        help='Directory containing runassessor JSON files for enrichment')
    parser.add_argument('--integrate', action='store_true',
                        help='Enable integration with runassessor data (requires --runassessor-dir)')
    parser.add_argument('--normalize', action='store_true',
                        help='Enable ontology-based term normalization')
    parser.add_argument('--ontology-dir', type=str, default=None,
                        help='Directory containing ontology files')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config)')
    parser.add_argument('--no-seed', action='store_true',
                        help='Disable seeding for non-deterministic mode')
    
    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_CONFIG.copy()
    config_path = Path(args.config)
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    if "paths" in loaded: config["paths"].update(loaded["paths"])
                    if "agents" in loaded: config["agents"].update(loaded["agents"])
                    if "concurrency" in loaded: config["concurrency"].update(loaded["concurrency"])
                    if "llm" in loaded: config["llm"] = loaded["llm"]
                    if "reproducibility" in loaded: config["reproducibility"] = loaded["reproducibility"]
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    # Initialize reproducibility / seed control
    if args.no_seed:
        seed = None
    elif args.seed is not None:
        seed = args.seed
    else:
        seed = config.get("reproducibility", {}).get("seed", None)
    
    if seed is not None:
        set_seed(seed)
        print(f"Reproducibility: seed set to {seed}")
    else:
        print("Reproducibility: running in non-deterministic mode")
    
    # Optionally log reproducibility info
    if config.get("reproducibility", {}).get("log_info", False):
        repro_info = get_reproducibility_info()
        print(f"Reproducibility info: {repro_info}")
            
    # Resolve paths
    input_dir = args.input or config["paths"]["input_dir"]
    input_path = Path(input_dir)
    
    if args.output:
        base_output = Path(args.output)
    else:
        config_out = config["paths"].get("output_dir")
        if config_out:
            base_output = Path(config_out)
        else:
            base_output = input_path.parent / "framework_output"
            
    ontology_dir = args.ontology_dir or config["paths"]["ontology_dir"]
    
    # Resolve temperatures
    if args.single_temp is not None:
        temperatures = [args.single_temp]
    elif args.temperatures is not None:
        temperatures = args.temperatures
    else:
        temperatures = config["agents"].get("temperatures", [0.0])
        
    # Validation setting
    use_validation = args.validate if args.validate is not None else config["agents"].get("validate", False)
        
    # Get concurrency settings
    max_workers = config["concurrency"].get("max_workers", 1)
    
    # Get LLM config
    llm_config = config.get("llm", {})
    
    # Retry / feedback loop settings
    max_retries = config["agents"].get("max_retries", 1)
    confidence_threshold = config["agents"].get("confidence_threshold", 0.6)
        
    agents = []
    
    if args.mode == 'biological' or args.mode == 'all':
        out = base_output / "Biological_annotations"
        agents.append(BiologicalAgent(input_path, out, temperatures, use_validation, max_workers=max_workers, llm_config=llm_config, max_retries=max_retries, confidence_threshold=confidence_threshold))
        
    if args.mode == 'technical' or args.mode == 'all':
        out = base_output / "technical_metadata_output"
        agents.append(TechnicalAgent(input_path, out, temperatures, use_validation, max_workers=max_workers, llm_config=llm_config, max_retries=max_retries, confidence_threshold=confidence_threshold))
        
    if args.mode == 'experimental' or args.mode == 'all':
        out = base_output / "experimental_design_output"
        agents.append(ExperimentalDesignAgent(input_path, out, temperatures, use_validation, max_workers=max_workers, llm_config=llm_config, max_retries=max_retries, confidence_threshold=confidence_threshold))
        
    # Store results: { 'BiologicalAgent': { temp: { file: json } } }
    pipeline_results = {}
    
    for agent in agents:
        name = agent.__class__.__name__
        print(f"Starting {name}...")
        pipeline_results[name] = agent.run()
        print(f"Finished {name}.\n")

    # Normalization step: normalize against ontologies
    if args.normalize:
        print(f"\n{'='*60}")
        print("Starting Ontology Normalization...")
        print(f"{'='*60}\n")
        
        from agents.normalization_agent import NormalizationAgent
        normalizer = NormalizationAgent(args.ontology_dir)
        
        normalized_output = base_output / "normalized_output"
        normalized_output.mkdir(parents=True, exist_ok=True)
        
        for agent_name, temp_results in pipeline_results.items():
            for temp, file_results in temp_results.items():
                normalized = normalizer.normalize_batch(file_results)
                
                # Update pipeline_results with normalized data
                pipeline_results[agent_name][temp] = normalized
                
                # Save normalized results
                out_dir = normalized_output / agent_name / f"temp_{temp:.1f}"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                for filename, data in normalized.items():
                    out_file = out_dir / (Path(filename).stem + "_normalized.json")
                    with open(out_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved normalized: {out_file}")
        
        print("\nNormalization complete.")

    # Integration step: enrich with runassessor data
    if args.integrate:
        if not args.runassessor_dir:
            print("ERROR: --integrate requires --runassessor-dir")
            return
        
        print(f"\n{'='*60}")
        print("Starting Integration Agent...")
        print(f"{'='*60}\n")
        
        integrator = IntegrationAgent(args.runassessor_dir)
        integrated_output = base_output / "integrated_output"
        integrated_output.mkdir(parents=True, exist_ok=True)
        
        # Merge all agent outputs and enrich
        for agent_name, temp_results in pipeline_results.items():
            for temp, file_results in temp_results.items():
                # Define output directory first
                out_dir = integrated_output / agent_name / f"temp_{temp:.1f}"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Enrich with runassessor data and save disagreement log
                enriched = integrator.enrich_batch(file_results, agent_name=agent_name, output_dir=str(out_dir))
                
                # Save enriched results
                for filename, data in enriched.items():
                    out_file = out_dir / (Path(filename).stem + "_enriched.json")
                    with open(out_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Saved enriched: {out_file}")
        
        print("\nIntegration complete.")

if __name__ == "__main__":
    main()

