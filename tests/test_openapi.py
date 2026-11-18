import json
from pathlib import Path

from resolvelane.access import TokenRegistry
from resolvelane.api import create_app

DOCUMENT = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


def test_frontend_contract_matches_the_api():
    # The frontend's types are generated from this file; regenerate it with
    # python scripts/export_openapi.py, then npm run api-types in frontend/.
    committed = json.loads(DOCUMENT.read_text())
    assert committed == create_app(TokenRegistry({})).openapi()
