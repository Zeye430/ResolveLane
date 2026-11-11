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

## Order lookup

`resolvelane.services.order_lookup.lookup_order` provides read-only access to an order only when the requester's customer ID matches the order owner. Lookups of another customer's order are classified internally as denied but presented to customers as not found, so order IDs cannot be probed.

`GET /api/v1/orders/{order_id}` exposes the lookup to customers. The caller is identified only by a bearer token; identity fields in the query string, headers or body are ignored. Denied and unknown orders both return `404 {"detail": "Order not found"}`. Support agents receive `403` because operator access will be scoped to assigned cases, which do not exist yet. Responses omit internal customer and merchant IDs.

Tokens are configured with `RESOLVELANE_API_TOKENS`, a JSON object mapping each token (at least 24 characters) to a fictional principal:

```bash
export RESOLVELANE_API_TOKENS='{"local-dev-token-customer-101-xyz": {"subject_id": "demo-customer-101", "role": "customer"}}'
curl -H "Authorization: Bearer local-dev-token-customer-101-xyz" http://127.0.0.1:8000/api/v1/orders/demo-order-1001
```

Without tokens, protected endpoints return `503` in development, and the application refuses to start when `RESOLVELANE_ENV=production`. Static tokens are a development mechanism for simulated accounts, not production authentication. Each lookup emits an INFO record on the `resolvelane.api` logger with its internal outcome; the default uvicorn configuration does not display it, and durable audit records are not implemented yet.
