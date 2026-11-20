"""Semantic code mutation, crossover and repair backends.

The live backend calls the OpenAI Responses API using only the standard
library. `mock` is intentionally narrow and is used for smoke tests only.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    source: str
    description: str


def strip_code_fences(response: str) -> str:
    response = response.strip()
    if response.startswith("```python") or response.startswith("```py"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0].strip()
    elif response.startswith("```"):
        response = response.split("\n", 1)[1]
        response = response.rsplit("```", 1)[0].strip()
    if not response:
        raise ValueError("generator returned empty code")
    if len(response) > 300_000:
        raise ValueError("generated file exceeds 300 KB limit")
    # Invalid syntax must reach the preflight/repair protocol rather than
    # prematurely aborting a whole generation.
    return response.rstrip() + "\n"


def clean_description(text: str) -> str:
    return re.sub(r"[\t\n\r,]+", " ", text).strip()[:180] or "unnamed change"


class OpenAIProvider:
    def __init__(self, model: str):
        self.key = os.getenv("OPENAI_API_KEY", "")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set; use --provider mock for local tests")
        if not model:
            raise RuntimeError("Choose an available model with --model or OPENAI_MODEL")
        self.model = model
        self.url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/responses"

    def _complete(self, instruction: str, context: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "instructions": instruction,
            "input": context,
            "max_output_tokens": 16000,
        }).encode()
        for attempt in range(4):
            req = urllib.request.Request(
                self.url, data=payload, method="POST",
                headers={
                    "Authorization": "Bearer " + self.key,
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as stream:
                    document = json.loads(stream.read())
                # Responses REST API commonly nests text in output[].content[].
                output = document.get("output_text")
                if not output:
                    output = "".join(
                        content.get("text", "")
                        for block in document.get("output", [])
                        for content in block.get("content", [])
                        if content.get("type") == "output_text"
                    )
                if not output:
                    raise RuntimeError("No output_text returned by model")
                return output
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                    detail = exc.read(1000).decode("utf-8", errors="replace")
                    raise RuntimeError(f"Model request HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"Model request failed: {exc}") from exc
            time.sleep(2 ** attempt)
        raise RuntimeError("model request retries exceeded")

    def propose(self, sources: list[str], method: str, intensity: str,
                history: str, hint: str = "") -> Proposal:
        assert method in ("mutation", "crossover", "repair")
        instructions = (
            "You are an expert language model training researcher. Modify ONLY train.py. "
            "Return a complete executable Python train.py source file and NOTHING else: "
            "no prose no Markdown fences. The evaluation harness prepare.py is immutable. "
            "The fixed training time is 300 seconds and val_bpb must be computed by "
            "prepare.evaluate_bpb. Never fabricate metrics or bypass evaluation. "
            "Preserve seed=42. No new imports outside Python standard library or declared "
            "pyproject.toml dependencies. Keep model(x)->logits. Avoid giant VRAM increases. "
            "Make a coherent change and retain the required metric summary. "
            "Do not modify prepare.py or create additional source files."
        )
        if method == "crossover":
            goal = "Recombine complementary proven strengths from TWO parents, resolving incompatibilities."
        elif method == "repair":
            goal = f"Repair a prior candidate after failure without changing its research idea. Error:\n{hint[-4500:]}"
        else:
            guidance = {
                "conservative": "One small hyperparameter adjustment within 2x (learning rate, batch size, weight decay, warmup).",
                "moderate": "One coherent structural modification (normalization, heads, activation, width, depth).",
                "radical": "One ambitious architectural or optimization idea. Some failures are acceptable.",
            }
            goal = guidance.get(intensity, guidance["moderate"])
        parents = "\n\n".join(f"===== PARENT {i+1} =====\n{src}" for i, src in enumerate(sources))
        context = f"RESEARCH GOAL: {goal}\n\nHISTORY TO AVOID REPEATS:\n{history[-9000:]}\n\n{parents}"
        answer = self._complete(instructions, context)
        return Proposal(strip_code_fences(answer), clean_description(f"{method} {intensity} LLM proposal"))


class MockProvider:
    """Deterministic (seeded) operator for repo tests; not an intelligent agent."""

    MUTABLE = ("LEARNING_RATE", "WEIGHT_DECAY", "DROPOUT", "DEPTH", "WIDTH", "NUM_HEADS", "BATCH_SIZE", "GRAD_ACCUM")

    def __init__(self, rng: random.Random):
        self.rng = rng

    def propose(self, sources: list[str], method: str, intensity: str,
                history: str, hint: str = "") -> Proposal:
        _ = (history, hint)
        source = sources[0]
        pattern = re.compile(r"^(LEARNING_RATE|WEIGHT_DECAY|DROPOUT|DEPTH|WIDTH|NUM_HEADS|BATCH_SIZE|GRAD_ACCUM)\s*=\s*([0-9.]+)(\s*(?:#.*)?)$", re.M)
        params = {m.group(1): float(m.group(2)) for m in pattern.finditer(source)}
        if not params:
            raise ValueError("mock generator requires an editable uppercase numeric hyperparameter")
        conservative = ("LEARNING_RATE", "WEIGHT_DECAY", "DROPOUT")
        moderate = ("DEPTH", "WIDTH", "BATCH_SIZE")
        radical = ("DEPTH", "WIDTH", "GRAD_ACCUM")
        candidates = {"conservative": conservative, "moderate": moderate, "radical": radical}[intensity]
        available = [x for x in candidates if x in params]
        if not available:
            available = list(params)
        selected = self.rng.choice(available)
        old = params[selected]
        if selected in ("WIDTH",):
            value = max(64, int(old * self.rng.choice((0.75, 1.25)) // 64) * 64)
            if "NUM_HEADS" in params:
                heads = int(params["NUM_HEADS"])
                value = max(heads * 32, ((value + heads - 1) // heads) * heads)
        elif selected in ("DEPTH", "BATCH_SIZE", "GRAD_ACCUM"):
            value = max(1, int(old) + self.rng.choice((-1, 1)))
        else:
            value = round(max(0.0, old * self.rng.choice((0.8, 1.2))), 8)
            if value == old:
                value = old + 1e-8
        literal = str(int(value)) if selected in ("DEPTH", "WIDTH", "BATCH_SIZE", "GRAD_ACCUM") else str(value)
        updated = re.sub(rf"^({selected}\s*=\s*)[0-9.]+", rf"\g<1>{literal}", source, count=1, flags=re.M)
        if method == "crossover" and len(sources) == 2:
            # For an offline demonstration, blend one orthogonal named setting.
            matches = re.findall(r"^([A-Z_]+)\s*=\s*([0-9.]+)", sources[1], re.M)
            for name, val in matches:
                if name in params and name != selected:
                    updated = re.sub(rf"^({name}\s*=\s*)[0-9.]+", rf"\g<1>{val}", updated, count=1, flags=re.M)
                    break
        if updated == source:
            raise RuntimeError("mock operator produced no change")
        return Proposal(strip_code_fences(updated), clean_description(f"{method} {intensity} {selected} {old:g} to {value:g}"))
