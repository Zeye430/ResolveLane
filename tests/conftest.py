import os

import pytest

# Keep tests offline and independent of the developer's shell. This runs before test modules
# import resolvelane.api, which builds a module-level application from the environment.
PROVIDER_SETTINGS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
                     "RESOLVELANE_EMBEDDING_API_KEY", "RESOLVELANE_EMBEDDING_BASE_URL",
                     "RESOLVELANE_EMBEDDING_MODEL", "RESOLVELANE_EMBEDDING_THRESHOLD",
                     "RESOLVELANE_EMBEDDING_MARGIN", "RESOLVELANE_LOCAL_EMBEDDING_THRESHOLD",
                     "RESOLVELANE_LOCAL_EMBEDDING_MARGIN", "RESOLVELANE_MODEL_CACHE_DIR",
                     "RESOLVELANE_MCP_CONFIG", "RESOLVELANE_SKILLS_DIR", "RESOLVELANE_PAYMENT_FAULTS")
for _name in PROVIDER_SETTINGS:
    os.environ.pop(_name, None)
os.environ["RESOLVELANE_LOCAL_EMBEDDING_MODEL"] = "off"
# Applications built without an explicit case store keep cases in memory, never in a file.
os.environ["RESOLVELANE_DATABASE"] = ":memory:"
# Approved refunds stay in the outbox; tests send them explicitly with payments.send_due.
os.environ["RESOLVELANE_PAYMENTS"] = "paused"


@pytest.fixture(autouse=True)
def offline_recognition(monkeypatch):
    """Without provider keys and with the local model off, the application uses keywords only."""
    for name in PROVIDER_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RESOLVELANE_LOCAL_EMBEDDING_MODEL", "off")
    monkeypatch.setenv("RESOLVELANE_DATABASE", ":memory:")
    monkeypatch.setenv("RESOLVELANE_PAYMENTS", "paused")
