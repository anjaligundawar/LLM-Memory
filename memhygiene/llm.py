"""Thin Claude client used by the reader, the judge and the natural-junk
memory writer. Every call is cached on disk (keyed by model + prompt), so
re-running an analysis never pays twice.

Needs:  pip install anthropic   and an ANTHROPIC_API_KEY (or `ant auth login`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Type

DEFAULT_MODEL = "claude-opus-5-5"


class Claude:
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low", cache_dir: Path = Path("cache"),
                 max_tokens: int = 2048):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=6)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        from .embed import connect_cache
        self.db = connect_cache(Path(cache_dir) / "llm_cache.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def _key(self, *parts) -> str:
        return hashlib.sha1("\x00".join([self.model, self.effort, *map(str, parts)]).encode()).hexdigest()

    def _get(self, k: str) -> Optional[str]:
        row = self.db.execute("SELECT v FROM c WHERE k = ?", (k,)).fetchone()
        return row[0] if row else None

    def _put(self, k: str, v: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (k, v))
        self.db.commit()

    def text(self, system: str, user: str) -> str:
        """Free-text completion. Refusal fallback is enabled (server-side,
        routed by refusal category); a refused call returns ''."""
        k = self._key("text", system, user)
        hit = self._get(k)
        if hit is not None:
            return hit
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if resp.stop_reason == "refusal":
            return ""
        out = "".join(b.text for b in resp.content if b.type == "text").strip()
        self._put(k, out)
        return out

    def structured(self, system: str, user: str, schema: Type):
        """JSON output validated against a pydantic model."""
        k = self._key("json", schema.__name__, json.dumps(schema.model_json_schema(), sort_keys=True),
                      system, user)
        hit = self._get(k)
        if hit is not None:
            return schema.model_validate_json(hit)
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RuntimeError(f"no structured output (stop_reason={resp.stop_reason})")
        self._put(k, resp.parsed_output.model_dump_json())
        return resp.parsed_output
