"""The language model provider shared by issue recognition and reply writing.

Both callers share one circuit breaker: once recognition finds the provider down, reply
writing skips it too instead of making the same customer wait for a second timeout.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

import anthropic

from resolvelane.resilience import CircuitBreaker

logger = logging.getLogger("resolvelane.llm")

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_BASE_URL = "https://api.anthropic.com"
# Errors caused by the request rather than the provider; they do not count against the breaker.
REQUEST_ERRORS = (400, 413, 422)


@dataclass(frozen=True)
class LLMProvider:
    client: anthropic.Anthropic
    model: str
    breaker: CircuitBreaker


def load_llm_provider(environ: Mapping[str, str] = os.environ) -> LLMProvider | None:
    api_key = environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        logger.info("ANTHROPIC_API_KEY is not set; the language model is off")
        return None
    client = anthropic.Anthropic(
        api_key=api_key,
        # Passed explicitly: given None, the SDK reads ANTHROPIC_BASE_URL itself and would use an
        # empty value from a .env file as the URL.
        base_url=environ.get("ANTHROPIC_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        timeout=float(environ.get("RESOLVELANE_LLM_TIMEOUT_S", "10")),
        max_retries=1,
    )
    return LLMProvider(client, environ.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL,
                       CircuitBreaker("llm", failure_threshold=3, cooldown_s=30))
