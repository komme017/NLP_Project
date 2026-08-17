"""Cost instrumentation.

Azure access for this course project is institutional/free, but the
accompanying business plan needs a real market cost estimate. Every API call
is logged with prompt/completion tokens, the model, and the pipeline stage;
totals are accumulated per contract run and appended to runs.jsonl.
"""

import json
import time
from dataclasses import dataclass, field

# Published commercial rates (USD per 1M tokens). Source: Azure OpenAI /
# OpenAI API pricing pages, as of Aug 2026. Update here if pricing changes —
# kept as one constant so the business plan's cost math stays in sync with
# whatever's shown in the app.
PRICING_USD_PER_1M = {
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
}

# NOTE: Azure deployment names are set by whoever provisions the resource
# and don't have to match the underlying model name (e.g. a deployment
# could be named "redline-dev"). If the configured deployment isn't a key
# in PRICING_USD_PER_1M, we fall back to the gpt-4.1-mini rate since that's
# the model this prototype is built against, rather than crash on a cost
# estimate that isn't the point of the demo.
DEFAULT_MODEL_KEY = "gpt-4.1-mini"


@dataclass
class CostTracker:
    calls: list = field(default_factory=list)

    def log_call(self, stage: str, model: str, prompt_tokens: int, completion_tokens: int):
        self.calls.append(
            {
                "stage": stage,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    def _rate_for(self, model: str):
        return PRICING_USD_PER_1M.get(model, PRICING_USD_PER_1M[DEFAULT_MODEL_KEY])

    def totals(self) -> dict:
        prompt_tokens = sum(c["prompt_tokens"] for c in self.calls)
        completion_tokens = sum(c["completion_tokens"] for c in self.calls)
        cost = 0.0
        for c in self.calls:
            rates = self._rate_for(c["model"])
            cost += c["prompt_tokens"] / 1_000_000 * rates["prompt"]
            cost += c["completion_tokens"] / 1_000_000 * rates["completion"]
        return {
            "num_calls": len(self.calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": round(cost, 4),
        }

    def by_stage(self) -> dict:
        stages: dict = {}
        for c in self.calls:
            s = stages.setdefault(
                c["stage"], {"num_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            )
            s["num_calls"] += 1
            s["prompt_tokens"] += c["prompt_tokens"]
            s["completion_tokens"] += c["completion_tokens"]
        return stages

    def log_run(self, contract_name: str, extra: dict = None, path: str = "runs.jsonl") -> dict:
        """Append a summary of this run to runs.jsonl and return the record."""
        record = {
            "timestamp": time.time(),
            "contract": contract_name,
            **self.totals(),
            "by_stage": self.by_stage(),
        }
        if extra:
            record.update(extra)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record
