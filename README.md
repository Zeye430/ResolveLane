# ResolveLane

ResolveLane is an AI-powered support operations platform for digital marketplaces.

## Development

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn resolvelane.api:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health` and the API documentation at `http://127.0.0.1:8000/docs`.

Run tests with `pytest`.

The health endpoint reports that the application is running; it does not check external services. The support engine and integrations have not been added yet.

## Delivery examples

The delivery domain includes fictional orders, delivery events, support policies, and customer-report scenarios for development and tests. Customer reports are not verified facts, and expected review paths are test descriptions, not implemented functionality. The examples do not contain real platform records, execute refunds, or represent the policies of an existing marketplace.
