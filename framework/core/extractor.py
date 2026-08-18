from pathlib import Path
import json
import concurrent.futures
from .llm import LLMClient
from .logging import get_logger
from validation.validator import ValidationAgent

logger = get_logger(__name__)


def normalize_output(data: dict) -> dict:
    """
    Post-processor to fix common LLM output format issues:
    1. Remove unwanted keys (THOUGHT PROCESS, FINAL JSON, raw_output)
    2. Flatten nested arrays [[val, ev]] -> [val, ev]
    3. Fix single values without evidence [val] -> [val, ""]
    4. Validate evidence contains the extracted value
    """
    if "raw_output" in data:
        # Can't normalize raw output, return as-is
        return data
    
    normalized = {}
    unwanted_keys = {"THOUGHT PROCESS", "FINAL JSON", "raw_output", "pmid"}
    
    for field, value in data.items():
        # Skip unwanted keys
        if field in unwanted_keys:
            continue
        
        # Handle different value formats
        if isinstance(value, list):
            # Fix: nested arrays [[val, ev]] → [val, ev]
            if len(value) == 1 and isinstance(value[0], list):
                value = value[0]
            
            # Fix: multiple nested arrays [[val1, ev1], [val2, ev2]] → take first
            if len(value) > 0 and isinstance(value[0], list):
                value = value[0]
            
            # Fix: single value without evidence [val] → [val, ""]
            if len(value) == 1 and isinstance(value[0], str):
                value = [value[0], ""]
            
            # Validate: evidence must contain value (for non-unknown values)
            if len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], str):
                val, evidence = value
                if val != "unknown" and val and evidence:
                    # Check if value appears in evidence
                    if val.lower() not in evidence.lower():
                        logger.warning("Evidence mismatch for '%s': '%s' not in evidence", field, val)
                        # Keep the value but note the mismatch (don't reset to unknown)
        
        normalized[field] = value
    
    return normalized


class BaseExtractor:
    def __init__(self, input_dir: str, output_dir: str, temperatures=None, 
                 use_validation=False, max_workers=1, llm_config=None,
                 max_retries: int = 1, confidence_threshold: float = 0.6):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.temperatures = temperatures or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.llm = LLMClient(config=llm_config)
        self.validator = ValidationAgent() if use_validation else None
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold

    def get_prompt(self, text: str) -> str:
        """Override this method in subclasses to return the specific prompt"""
        raise NotImplementedError

    def run(self):
        logger.info("Processing with temperatures: %s", self.temperatures)
        logger.info("Base output folder: %s", self.output_path)
        
        all_results = {}
        for temp in self.temperatures:
            logger.info("="*60)
            logger.info("Running with temperature: %.1f", temp)
            logger.info("="*60)
            
            temp_folder = self.output_path / f"temp_{temp:.1f}"
            temp_folder.mkdir(parents=True, exist_ok=True)
            
            results = self.process_files(temp_folder, temp)
            all_results[temp] = results
            
            logger.info("Completed temperature %.1f: %d files processed", temp, len(results))
            
        return all_results

    def process_files(self, output_folder: Path, temperature: float) -> dict:
        files = list(self.input_path.glob('**/*.txt'))
        if not files:
            logger.warning("No .txt files found in %s", self.input_path)
            return {}
            
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        max_workers = self.max_workers
        if max_workers > 1:
            logger.info(f"Processing {len(files)} files with {max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_single_file, file, output_folder, temperature): file 
                    for file in files
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        file_path, metadata = future.result()
                        results[file_path] = metadata
                    except Exception as exc:
                        logger.error(f"File {file} generated an exception: {exc}")
        else:
            # Sequential fallback (for debugging or single worker)
            for file in files:
                try:
                    file_path, metadata = self._process_single_file(file, output_folder, temperature)
                    results[file_path] = metadata
                except Exception as exc:
                    logger.error(f"File {file} generated an exception: {exc}")
            
        return results

    def _process_single_file(self, file: Path, output_folder: Path, temperature: float) -> tuple:
        logger.info(f"Processing: {file.name} (temperature={temperature:.1f})")
        text = self.get_text_from_docs(file)
        prompt = self.get_prompt(text)
        
        # Call LLM
        metadata = self.llm.get_completion([{"role": "user", "content": prompt}], temperature)
        
        # --- START AGENTIC PARSING ---
        # Extract JSON more robustly - find the LAST complete JSON object
        json_str = None
        
        # First try: split on "FINAL JSON:" and get the last one
        if "FINAL JSON:" in metadata:
            parts = metadata.split("FINAL JSON:")
            # Take the last part (in case there are multiple)
            json_str = parts[-1].strip()
            thought_process = parts[0].replace("THOUGHT PROCESS:", "").strip()
            logger.debug(f"Agent thoughts for {file.name}: {thought_process[:100]}...")
        
        # Extract just the JSON object using bracket matching
        if json_str:
            # Find the first '{' and match to closing '}'
            start_idx = json_str.find('{')
            if start_idx != -1:
                # Count braces to find matching close
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(json_str[start_idx:], start=start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                json_str = json_str[start_idx:end_idx]
        else:
            # Fallback: find the last complete JSON object in the entire output
            start_idx = metadata.rfind('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = len(metadata)
                for i, char in enumerate(metadata[start_idx:], start=start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                json_str = metadata[start_idx:end_idx]
        # --- END AGENTIC PARSING ---
        
        metadata_json = {}
        try:
            # Try to sanitize common JSON issues before parsing
            if json_str:
                # Fix: escaped quotes inside strings that break JSON
                # Replace backslash-quote with single quotes temporarily
                import re
                
                # First attempt: direct parse
                try:
                    metadata_json = json.loads(json_str)
                except json.JSONDecodeError:
                    # Second attempt: try to extract using regex pattern for our format
                    # Pattern: "field": ["value", "evidence"]
                    extracted = {}
                    pattern = r'"([^"]+)":\s*\["([^"]*)",\s*"([^"]*)"\]'
                    matches = re.findall(pattern, json_str, re.DOTALL)
                    
                    if matches:
                        for field, value, evidence in matches:
                            extracted[field] = [value, evidence]
                        metadata_json = extracted
                        logger.info(f"  [Recovered {len(matches)} fields using regex for {file.name}]")
                    else:
                        # Third attempt: try to fix common issues
                        fixed_json = json_str.replace('\\"', "'")  # Replace \" with '
                        try:
                            metadata_json = json.loads(fixed_json)
                        except json.JSONDecodeError:
                            raise  # Re-raise to trigger fallback
            else:
                metadata_json = {}
            
            # Run Validation if enabled
            if self.validator:
                logger.info(f"  Validating {file.name}...")
                metadata_json = self.validator.validate(text, metadata_json)
                
                # Re-extraction feedback loop
                if self.max_retries > 0:
                    critique = self.validator.get_critique(metadata_json, text)
                    
                    for attempt in range(self.max_retries):
                        if critique["overall_confidence"] >= self.confidence_threshold:
                            break
                        if critique["n_issues"] == 0:
                            break
                        
                        logger.info(
                            f"  Retry {attempt+1}/{self.max_retries} for {file.name} "
                            f"(confidence={critique['overall_confidence']:.2f}, "
                            f"{critique['n_issues']} issues)"
                        )
                        
                        # Build critique prompt and re-extract
                        retry_prompt = self._build_critique_prompt(
                            text, self.get_prompt(text), metadata_json, critique
                        )
                        retry_temp = min(temperature + 0.1, 0.5)  # Hard cap: never retry above 0.5
                        retry_raw = self.llm.get_completion(
                            [{"role": "user", "content": retry_prompt}], retry_temp
                        )
                        retry_json = self._parse_llm_json(retry_raw, file.name)
                        
                        if not retry_json or "raw_output" in retry_json:
                            logger.warning(f"  Retry {attempt+1} failed to parse, keeping original")
                            break
                        
                        # Validate the retry result
                        retry_json = self.validator.validate(text, retry_json)
                        retry_critique = self.validator.get_critique(retry_json, text)
                        
                        # Keep whichever has higher confidence
                        if retry_critique["overall_confidence"] > critique["overall_confidence"]:
                            logger.info(
                                f"  Retry improved confidence: "
                                f"{critique['overall_confidence']:.2f} -> "
                                f"{retry_critique['overall_confidence']:.2f}"
                            )
                            metadata_json = retry_json
                            critique = retry_critique
                        else:
                            logger.info(
                                f"  Retry did not improve "
                                f"({retry_critique['overall_confidence']:.2f} <= "
                                f"{critique['overall_confidence']:.2f}), keeping original"
                            )
                            break
            
            # Run post-processor to normalize output format
            metadata_json = normalize_output(metadata_json)
                
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON for {file}. Saving raw output.")
            metadata_json = {"raw_output": metadata}
        
        output_filename = file.stem + ".json"
        output_file_path = output_folder / output_filename
        
        with open(output_file_path, 'w') as f:
            json.dump(metadata_json, f, indent=2)
        
        logger.info(f"Saved: {output_file_path}")
        return str(file), metadata_json

    def get_text_from_docs(self, path) -> str:
        with open(path, 'r') as file:
            text = file.read()
        return text

    def _build_critique_prompt(self, text: str, original_prompt: str,
                                extraction: dict, critique: dict) -> str:
        """Build a refinement prompt that includes validation critique.
        
        The prompt contains:
        1. The original extraction task
        2. The previous (flawed) extraction as JSON
        3. Specific issues found by validation
        4. Instructions to fix them
        """
        # Format the previous extraction (exclude internal metadata)
        prev_json = {k: v for k, v in extraction.items() if not k.startswith('_')}
        prev_str = json.dumps(prev_json, indent=2)
        
        # Format issues as a numbered list
        issues_str = "\n".join(f"  {i+1}. {issue}" for i, issue in enumerate(critique["issues"]))
        
        return f"""You previously extracted metadata from a scientific manuscript but some issues were found. 
Please correct the following problems and return an improved extraction.

ORIGINAL TASK:
{original_prompt}

YOUR PREVIOUS EXTRACTION:
{prev_str}

VALIDATION ISSUES FOUND:
{issues_str}

INSTRUCTIONS:
- Fix each issue listed above
- For fields with "no supporting evidence quote provided": find the exact sentence or phrase in the manuscript that supports the extracted value and include it as the evidence
- For fields with "value not found in source text": re-read the manuscript carefully and either correct the value to what actually appears in the text, or set to ["unknown", ""] if the information is truly not in the manuscript
- For fields with "evidence quote not found in source text": replace with an actual quote from the manuscript
- Keep correct extractions unchanged
- Return ONLY the corrected JSON in the same format as before"""

    def _parse_llm_json(self, raw_output: str, filename: str = "") -> dict:
        """Extract and parse JSON from raw LLM output.
        
        Handles agentic format (THOUGHT PROCESS / FINAL JSON) and
        plain JSON output. Returns dict or None on failure.
        """
        import re
        
        json_str = None
        
        # Try: split on "FINAL JSON:" and get the last one
        if "FINAL JSON:" in raw_output:
            parts = raw_output.split("FINAL JSON:")
            json_str = parts[-1].strip()
        
        # Extract just the JSON object using bracket matching
        if json_str:
            start_idx = json_str.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(json_str[start_idx:], start=start_idx):
                    if char == '{': brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                json_str = json_str[start_idx:end_idx]
        else:
            # Fallback: find the last complete JSON object
            start_idx = raw_output.rfind('{')
            if start_idx != -1:
                brace_count = 0
                for i, char in enumerate(raw_output[start_idx:], start=start_idx):
                    if char == '{': brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = raw_output[start_idx:i+1]
                            break
        
        if not json_str:
            return None
        
        # Try to parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Regex fallback for our [value, evidence] format
            extracted = {}
            pattern = r'"([^"]+)":\s*\["([^"]*)",\s*"([^"]*)"\]'
            matches = re.findall(pattern, json_str, re.DOTALL)
            if matches:
                for field, value, evidence in matches:
                    extracted[field] = [value, evidence]
                logger.info(f"  [Retry recovered {len(matches)} fields via regex for {filename}]")
                return extracted
            
            # Last resort: fix escaped quotes
            try:
                return json.loads(json_str.replace('\\"', "'"))
            except json.JSONDecodeError:
                return None


