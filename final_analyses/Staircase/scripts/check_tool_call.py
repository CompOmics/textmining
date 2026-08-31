#!/usr/bin/env python3
"""Fail unless the local model produces one valid automatic tool call."""

import argparse
import json
from openai import OpenAI


parser = argparse.ArgumentParser()
parser.add_argument("--base-url", required=True)
parser.add_argument("--model", required=True)
parser.add_argument(
    "--tool-choice",
    choices=("auto", "required", "named"),
    default="named",
    help="Tool-selection mode; GLM-4.7's parser supports automatic selection only.",
)
args = parser.parse_args()
client = OpenAI(base_url=args.base_url, api_key="local-vllm", timeout=180)
tool_choice = (
    {"type": "function", "function": {"name": "report_result"}}
    if args.tool_choice == "named"
    else args.tool_choice
)
response = client.chat.completions.create(
    model=args.model,
    temperature=0,
    messages=[{"role": "user", "content": "Call report_result with value exactly STAIRCASE_OK."}],
    tools=[{"type": "function", "function": {
        "name": "report_result", "description": "Report a test value",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    }}],
    tool_choice=tool_choice,
)
calls = response.choices[0].message.tool_calls or []
if len(calls) != 1 or calls[0].function.name != "report_result":
    raise SystemExit(f"tool-call gate failed: {response.model_dump_json()}")
payload = json.loads(calls[0].function.arguments)
if payload.get("value") != "STAIRCASE_OK":
    raise SystemExit(f"tool-call arguments invalid: {payload}")
print(json.dumps({"model": args.model, "tool": "report_result", "arguments": payload}))
