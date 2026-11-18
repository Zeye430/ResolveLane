"""Write the API's OpenAPI document to frontend/openapi.json, from which the frontend's types
are generated (npm run api-types). Run after changing an endpoint or a schema:

    python scripts/export_openapi.py

tests/test_openapi.py fails while the committed document differs from the application's, so
the frontend cannot silently drift from the API.
"""

import json
import os
import sys
from pathlib import Path

# Set before resolvelane.api is imported, which builds the server's application: the document
# does not depend on settings, and building it must not open the database or call a provider.
os.environ["RESOLVELANE_DATABASE"] = ":memory:"
os.environ["RESOLVELANE_PAYMENTS"] = "paused"
os.environ["RESOLVELANE_LOCAL_EMBEDDING_MODEL"] = "off"
for name in ("ANTHROPIC_API_KEY", "RESOLVELANE_EMBEDDING_API_KEY", "RESOLVELANE_MCP_CONFIG"):
    os.environ.pop(name, None)

from resolvelane.api import app  # noqa: E402

DOCUMENT = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


def render(document: dict) -> str:
    return json.dumps(document, indent=1, sort_keys=True) + "\n"


if __name__ == "__main__":
    DOCUMENT.write_text(render(app.openapi()))
    print(f"Wrote {DOCUMENT}")
    sys.exit(0)
