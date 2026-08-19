"""Code to assemble the prompt, consisting of instructions, schema, and collected manuscript content"""

import json
from pathlib import Path

from ..schemas import EXTRACTION_SCHEMA

INSTRUCTIONS_PATH = Path(__file__).parent / "instructions.txt"
INSTRUCTIONS_TEMPLATE = INSTRUCTIONS_PATH.read_text(encoding="utf-8")
SCHEMA_STR = json.dumps(EXTRACTION_SCHEMA, indent=2)


def build_prompt(manuscript_path: Path) -> tuple[str, dict]:
    """Build a prompt for a single PXD manuscript.

    Returns:
        tuple of (prompt_text, json_schema) ready for ollama_request.
    """
    manuscript = manuscript_path.read_text(encoding="utf-8")
    prompt = INSTRUCTIONS_TEMPLATE.format(schema=SCHEMA_STR, manuscript=manuscript)
    return prompt, EXTRACTION_SCHEMA