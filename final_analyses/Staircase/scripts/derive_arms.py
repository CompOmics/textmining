#!/usr/bin/env python3
"""Derive and freeze S1-S5 prompts from the copied original pipelines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


FILES = ("pipeline_biological.yaml", "pipeline_technical.yaml", "pipeline_experimental.yaml")


def bare_prompt(agent: str, fields: list[str]) -> str:
    field_lines = "\n".join(f"- {field}" for field in fields)
    return f"""You are the {agent} scientific metadata extraction agent.
Extract only metadata supported by the supplied proteomics or mass-spectrometry text.

FIELDS TO EXTRACT:
{field_lines}

For every field return exactly [value, evidence_sentence].
Use [\"unknown\", \"\"] when the text does not support a value.
Evidence for a non-unknown value must be a verbatim sentence from the text.

TEXT:
DOCUMENT ID: {{{{ input.id }}}}
{{{{ input.text }}}}

Return a JSON object with every listed field present.
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    frozen = args.root / "frozen_prompts"
    manifest: dict[str, object] = {"source": {}, "arms": {}}

    # S0 is the frozen single-master prompt.
    s0 = args.root / "arms" / "S0" / "pipeline_master_prompt.yaml"
    s0.write_bytes((frozen / "pipeline_master_prompt.yaml").read_bytes())
    manifest["source"]["pipeline_master_prompt.yaml"] = digest(frozen / "pipeline_master_prompt.yaml")
    manifest["arms"]["S0"] = {s0.name: digest(s0)}

    for arm in ("S1", "S2", "S3", "S4", "S5"):
        arm_hashes = {}
        for filename in FILES:
            source = frozen / filename
            cfg = yaml.safe_load(source.read_text())
            op = cfg["operations"][0]
            if arm == "S1":
                fields = list(op["output"]["schema"])
                op["prompt"] = bare_prompt(op["name"], fields)
                op.pop("validate", None)
                op.pop("num_retries_on_validate_failure", None)
                op.pop("gleaning", None)
            elif arm == "S2":
                op.pop("validate", None)
                op.pop("num_retries_on_validate_failure", None)
                op.pop("gleaning", None)
            elif arm == "S3":
                op.pop("gleaning", None)
            # S4 and S5 retain the copied original prompt/config. Normalization
            # is a runner feature flag and is enabled only for S5.
            target = args.root / "arms" / arm / filename
            target.write_text(yaml.safe_dump(cfg, sort_keys=False, width=1000), encoding="utf-8")
            arm_hashes[filename] = digest(target)
            manifest["source"][filename] = digest(source)
        manifest["arms"][arm] = arm_hashes

    (args.root / "prompt_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.root / "prompt_manifest.json")


if __name__ == "__main__":
    main()
