"""Code to send and receive requests to an Ollama server"""

import json
import logging
import requests

log = logging.getLogger(__name__)


def ollama_request(
    prompt: str,
    model: str,
    base_url: str = "http://localhost:11434",
    json_schema: dict = None,
    num_ctx: int = 32768,
    temperature: float = 0.0,
    timeout: int = 300,
    max_tries: int = 3,
) -> dict | None:
    """Send a prompt to Ollama and return structured JSON output.

    Retries on timeout or malformed response up to max_tries times.
    Returns None if all retries are exhausted.
    """
    for attempt in range(1, max_tries + 1):
        # Bump temperature on retries to break deterministic failure loops
        attempt_temp = temperature if attempt == 1 else max(temperature, 0.1)
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": json_schema,
                    "options": {
                        "num_ctx": num_ctx,
                        "temperature": attempt_temp,
                    },
                },
                timeout=timeout,
            )

            r = response.json()
            formatted = _format_response(r["response"])

            eval_dur = r.get("eval_duration", 0)
            eval_cnt = r.get("eval_count", 0)

            return {
                "response": formatted,
                "total_duration_s": r.get("total_duration", 0) / 1e9,
                "gen_tokens": eval_cnt,
                "tok_s": eval_cnt / (eval_dur / 1e9) if eval_dur > 0 else 0,
            }

        except requests.exceptions.ReadTimeout:
            log.warning("  Attempt %d/%d timed out (%ds)", attempt, max_tries, timeout)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("  Attempt %d/%d failed: %s", attempt, max_tries, e)

    return None


def _format_response(raw: str) -> str:
    """Parse and re-serialize JSON response in compact ground-truth style."""
    prediction = json.loads(raw)
    lines = ["{"]
    lines.append('  "sample_groups": [')
    groups = prediction.get("sample_groups", [])
    for gi, group in enumerate(groups):
        lines.append("    {")
        fields = list(group.items())
        for fi, (key, val) in enumerate(fields):
            comma = "," if fi < len(fields) - 1 else ""
            if isinstance(val, list):
                if not val:
                    lines.append(f'      "{key}": []{comma}')
                else:
                    lines.append(f'      "{key}": [')
                    for ii, item in enumerate(val):
                        ic = "," if ii < len(val) - 1 else ""
                        lines.append(f'        {json.dumps(item, ensure_ascii=False)}{ic}')
                    lines.append(f'      ]{comma}')
            elif isinstance(val, dict):
                lines.append(f'      "{key}": {{')
                ditems = list(val.items())
                for di, (dk, dv) in enumerate(ditems):
                    dc = "," if di < len(ditems) - 1 else ""
                    lines.append(f'        "{dk}": {json.dumps(dv, ensure_ascii=False)}{dc}')
                lines.append(f'      }}{comma}')
            else:
                lines.append(f'      "{key}": {json.dumps(val, ensure_ascii=False)}{comma}')
        trail = "," if gi < len(groups) - 1 else ""
        lines.append(f"    }}{trail}")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"
