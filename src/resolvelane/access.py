"""Bearer token authentication for simulated principals.

Tokens are configured as a JSON object that maps each token to a principal, for example
``{"<token>": {"subject_id": "demo-customer-101", "role": "customer"}}``. This is a
development mechanism for fictional accounts, not a production identity provider.
"""

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

TOKENS_ENV = "RESOLVELANE_API_TOKENS"
ENVIRONMENT_ENV = "RESOLVELANE_ENV"
MIN_TOKEN_LENGTH = 24


class Role(str, Enum):
    CUSTOMER = "customer"
    SUPPORT_AGENT = "support_agent"


@dataclass(frozen=True)
class Principal:
    subject_id: str
    role: Role


class AuthConfigError(ValueError):
    """Raised when the token configuration is missing or invalid."""


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class TokenRegistry:
    def __init__(self, entries: Mapping[str, Mapping[str, str]]):
        self._entries: list[tuple[bytes, Principal]] = []
        for token, account in entries.items():
            if not isinstance(token, str) or len(token) < MIN_TOKEN_LENGTH:
                raise AuthConfigError(f"API tokens must be at least {MIN_TOKEN_LENGTH} characters")
            if not isinstance(account, Mapping):
                raise AuthConfigError("Each API token must map to an object")
            subject_id = account.get("subject_id")
            if not isinstance(subject_id, str) or not subject_id:
                raise AuthConfigError("Each API token requires a non-empty subject_id")
            try:
                role = Role(account.get("role"))
            except ValueError as error:
                raise AuthConfigError("Each API token requires a known role") from error
            self._entries.append((_digest(token), Principal(subject_id, role)))

    @property
    def configured(self) -> bool:
        return bool(self._entries)

    def authenticate(self, token: str) -> Principal | None:
        # Hash first so comparisons have a fixed length, and check every entry
        # so the response time does not reveal which entry matched.
        supplied = _digest(token)
        match = None
        for digest, principal in self._entries:
            if hmac.compare_digest(digest, supplied):
                match = principal
        return match


def load_token_registry(environ: Mapping[str, str] = os.environ) -> TokenRegistry:
    raw = environ.get(TOKENS_ENV, "").strip()
    if not raw:
        if environ.get(ENVIRONMENT_ENV, "development") == "production":
            raise AuthConfigError(f"{TOKENS_ENV} must be set in production")
        return TokenRegistry({})
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AuthConfigError(f"{TOKENS_ENV} must be a JSON object") from error
    if not isinstance(entries, dict) or not entries:
        raise AuthConfigError(f"{TOKENS_ENV} must be a non-empty JSON object")
    return TokenRegistry(entries)
