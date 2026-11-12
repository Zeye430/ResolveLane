import os

import pytest

# Keep tests offline and independent of the developer's shell. This runs before test modules
# import resolvelane.api, which builds a module-level application from the environment.
PROVIDER_SETTINGS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
                     "RESOLVELANE_EMBEDDING_API_KEY", "RESOLVELANE_EMBEDDING_BASE_URL",
                     "RESOLVELANE_EMBEDDING_MODEL", "RESOLVELANE_EMBEDDING_THRESHOLD",
                     "RESOLVELANE_EMBEDDING_MARGIN", "RESOLVELANE_LOCAL_EMBEDDING_THRESHOLD",
                     "RESOLVELANE_LOCAL_EMBEDDING_MARGIN", "RESOLVELANE_MODEL_CACHE_DIR")
for _name in PROVIDER_SETTINGS:
    os.environ.pop(_name, None)
os.environ["RESOLVELANE_LOCAL_EMBEDDING_MODEL"] = "off"


@pytest.fixture(autouse=True)
def offline_recognition(monkeypatch):
    """Without provider keys and with the local model off, the application uses keywords only."""
    for name in PROVIDER_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RESOLVELANE_LOCAL_EMBEDDING_MODEL", "off")
